import type { Plugin } from "@opencode-ai/plugin"
import { spawn } from "node:child_process"
import { existsSync, readFileSync } from "node:fs"
import { homedir } from "node:os"
import { join, parse } from "node:path"

// Keep in sync with package.json; logged at boot so stale plugin caches are
// visible in the opencode log.
const VERSION = "1.0.1"

//
// Generic OpenCode adapter for the Claude Code hook schema.
//
// This plugin carries NO project-specific wiring. It reads whatever hooks a
// project declares for Claude Code (`.claude/settings.json`) or Codex
// (`.codex/hooks.json`, same schema) and interprets them against OpenCode's
// event model. Drop it into any project's `.opencode/plugins/` and it runs
// that project's hooks.
//
// Event mapping (Claude -> OpenCode):
//   PreToolUse       -> tool.execute.before               (deny-wins block, updatedInput)
//   PostToolUse      -> tool.execute.after                (feedback/context appended to the
//                                                          tool output, updatedToolOutput)
//   Stop             -> event: session.idle               (soft re-prompt loop, see below)
//   UserPromptSubmit -> chat.message                      (stdout/additionalContext injected
//                                                          as a text part; block is best-effort)
//   SessionStart     -> event: session.created            (context buffered, delivered with
//                                                          the first chat.message)
//   PreCompact       -> experimental.session.compacting   (context only; custom prompt not set)
//
// Hooks matched for an event run IN PARALLEL with identical commands
// deduplicated, as under Claude Code. Every command is fed a full Claude-schema
// JSON payload on stdin (session_id, cwd, hook_event_name, snake_case
// tool_input, tool_response, ...), so both this repo's
// `hooks/adapters/claude-stdin.sh` and third-party hooks doing
// `jq -r '.tool_input.file_path'` work unmodified. `transcript_path` is present
// but always "" — transcript synthesis is deliberately not implemented.
//
// Documented deviations from Claude Code, all forced by OpenCode's model:
//   1. session.idle is NOT a blocking Stop hook; it fires AFTER the agent goes
//      idle. When a Stop-phase command fails we re-inject its stderr as a
//      prompt so the model resumes and fixes it. Faithful to Claude, re-entry
//      rounds set stop_hook_active:true (scripts own their loop guard); the
//      per-session round counter stays as a backstop for hooks that ignore it.
//   2. permissionDecision:"ask" cannot open a user prompt from
//      tool.execute.before; OpenCode's own permission flow has already run, so
//      "ask" defers to it (no-op, logged). "allow" likewise cannot bypass
//      OpenCode permissions.
//   3. continue:false cannot abort a session; it blocks (PreToolUse), appends
//      stopReason (PostToolUse), or suppresses the re-prompt chain (Stop).
//   4. UserPromptSubmit blocking relies on throwing from chat.message, which
//      OpenCode does not document as a reject channel; context injection (the
//      common case) is solid either way.
//
// Config sources, lowest precedence first; hook arrays are concatenated (later
// files ADD hooks, they don't replace): user-level Claude settings, then the
// project's `.claude/settings.json` + `.claude/settings.local.json`, then
// `.opencode/hooks.json`. If NEITHER project `.claude` file defines hooks,
// `.codex/hooks.json` is read in their place — a fallback, not an additional
// source, because a project carrying both configs would run every hook twice
// (the command strings usually differ textually, so dedup cannot catch it).
// The user-level dir honours $CLAUDE_CONFIG_DIR like Claude Code does.
// `.opencode/hooks.json` is also
// where the OpenCode-only per-hook `failClosed` flag belongs (Claude Code may
// warn on unknown keys in `.claude/settings.json`); re-declaring an identical
// command there merges the flag via dedup instead of running it twice.
// `failClosed: true` turns executor errors (spawn failure, timeout, crash)
// into a deny for PreToolUse guards; the default stays fail-open.

type ClaudeEvent =
  | "PreToolUse"
  | "PostToolUse"
  | "Stop"
  | "UserPromptSubmit"
  | "SessionStart"
  | "PreCompact"

type HookCommand = { type?: string; command: string; timeout?: number; failClosed?: boolean }
type MatcherGroup = { matcher?: string; hooks?: HookCommand[] }
type HookConfig = Partial<Record<ClaudeEvent, MatcherGroup[]>>

type RunResult = { code: number; stdout: string; stderr: string }
type Logger = (message: string, level?: "info" | "warn" | "error") => void

// Normalized shape of a hook's JSON stdout (Claude's structured protocol).
type HookJson = {
  continue?: boolean
  stopReason?: string
  decision?: string // legacy top-level: "approve" | "block"
  reason?: string
  hookSpecificOutput?: {
    permissionDecision?: "allow" | "deny" | "ask"
    permissionDecisionReason?: string
    additionalContext?: string
    updatedInput?: Record<string, unknown>
    updatedToolOutput?: unknown
  }
}

type PreDecision = {
  kind: "deny" | "ask" | "allow" | "neutral"
  reason?: string
  updatedInput?: Record<string, unknown>
}

const EVENTS: ClaudeEvent[] = [
  "PreToolUse",
  "PostToolUse",
  "Stop",
  "UserPromptSubmit",
  "SessionStart",
  "PreCompact",
]

const MAX_IDLE_FIX_ROUNDS = 3
const TIMEOUT_CODE = 124 // conventional timeout exit code; treated as failure
const HARD_KILL_GRACE_MS = 2_000

// OpenCode tool name -> Claude tool name (what matchers are written against).
// mcp__* names pass through unchanged; anything unknown is Title-cased so novel
// tools still get a stable, matchable name.
const TOOL_NAMES: Record<string, string> = {
  bash: "Bash",
  write: "Write",
  edit: "Edit",
  read: "Read",
  glob: "Glob",
  grep: "Grep",
  webfetch: "WebFetch",
  task: "Task",
  todowrite: "TodoWrite",
  todoread: "TodoRead",
  patch: "Patch",
}

function claudeToolName(tool: string): string {
  if (tool.startsWith("mcp__")) return tool
  return TOOL_NAMES[tool] ?? tool.charAt(0).toUpperCase() + tool.slice(1)
}

function matches(matcher: string | undefined, toolName: string): boolean {
  if (!matcher) return true // empty matcher = all tools (Claude semantics)
  try {
    return new RegExp(`^(?:${matcher})$`).test(toolName)
  } catch {
    return false // a bad matcher matches nothing rather than throwing
  }
}

// Shallow key reshaping only: Claude's tool_input keys are top-level snake_case
// (file_path, old_string, ...). Reshaping nested values would corrupt user data
// (e.g. todo objects), so values pass through untouched.
function toSnakeKeys(obj: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {}
  for (const [k, v] of Object.entries(obj)) {
    out[k.replace(/([a-z0-9])([A-Z])/g, "$1_$2").toLowerCase()] = v
  }
  return out
}

function toCamelKeys(obj: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {}
  for (const [k, v] of Object.entries(obj)) {
    out[k.replace(/_([a-z0-9])/g, (_, c: string) => c.toUpperCase())] = v
  }
  return out
}

// Parse a hook's JSON stdout. Tolerant of surrounding noise (a formatter that
// prints "1 file reformatted" before its JSON blob is not malformed).
function parseHookJson(stdout: string): HookJson | undefined {
  const text = stdout.trim()
  if (!text) return undefined
  const candidates = [text]
  const first = text.indexOf("{")
  const last = text.lastIndexOf("}")
  if (first > 0 && last > first) candidates.push(text.slice(first, last + 1))
  for (const c of candidates) {
    try {
      const parsed: unknown = JSON.parse(c)
      if (parsed && typeof parsed === "object") return parsed as HookJson
    } catch {
      // not JSON, try the next candidate
    }
  }
  return undefined
}

// Short display name for a hook command, for feedback labels and logs. Prefers
// the last token (the script name in the "adapter.sh script.sh" form).
function hookLabel(command: string): string {
  const tokens = command.trim().split(/\s+/)
  const last = (tokens[tokens.length - 1] ?? "").replace(/["']/g, "")
  const pick = /[/.]/.test(last) ? last : (tokens[0] ?? "").replace(/["']/g, "")
  return pick.slice(pick.lastIndexOf("/") + 1) || command
}

// Per-hook PreToolUse classification. Deny beats everything; executor errors
// (timeout, crash, spawn failure) deny only when the hook is marked failClosed,
// otherwise they stay fail-open like Claude's non-blocking errors.
function classifyPre(r: RunResult, failClosed: boolean): PreDecision {
  if (r.code === 2) return { kind: "deny", reason: r.stderr.trim() || "Blocked by hook" }
  if (r.code !== 0) {
    return failClosed
      ? { kind: "deny", reason: `hook failed closed (exit ${r.code}): ${r.stderr.trim() || "no stderr"}` }
      : { kind: "neutral" }
  }
  const json = parseHookJson(r.stdout)
  if (!json) return { kind: "neutral" }
  const out = json.hookSpecificOutput
  if (out?.permissionDecision === "deny") {
    return { kind: "deny", reason: out.permissionDecisionReason ?? "Denied by hook" }
  }
  if (json.decision === "block") return { kind: "deny", reason: json.reason ?? "Blocked by hook" }
  if (json.continue === false) return { kind: "deny", reason: json.stopReason ?? "Stopped by hook" }
  if (out?.permissionDecision === "ask") return { kind: "ask" }
  if (out?.permissionDecision === "allow") return { kind: "allow", updatedInput: out.updatedInput }
  return { kind: "neutral", updatedInput: out?.updatedInput }
}

// Claude Code deduplicates identical hook commands. Merging failClosed with OR
// (and timeout with max) lets a project re-declare a `.claude/settings.json`
// command in `.opencode/hooks.json` solely to mark it fail-closed.
function dedupeHooks(hooks: HookCommand[]): HookCommand[] {
  const byCommand = new Map<string, HookCommand>()
  for (const h of hooks) {
    const prev = byCommand.get(h.command)
    if (!prev) {
      byCommand.set(h.command, { ...h })
    } else {
      prev.failClosed = prev.failClosed || h.failClosed
      if (h.timeout && (!prev.timeout || h.timeout > prev.timeout)) prev.timeout = h.timeout
    }
  }
  return [...byCommand.values()]
}

function readHooks(path: string, log: Logger): HookConfig | undefined {
  if (!existsSync(path)) return undefined
  try {
    return (JSON.parse(readFileSync(path, "utf8")) as { hooks?: HookConfig }).hooks
  } catch (e) {
    log(`ignoring malformed hook config ${path}: ${e}`, "warn")
    return undefined
  }
}

function mergeHooks(into: HookConfig, from: HookConfig | undefined): void {
  if (!from) return
  for (const event of EVENTS) {
    const groups = from[event]
    if (Array.isArray(groups)) into[event] = [...(into[event] ?? []), ...groups]
  }
}

function hasHooks(config: HookConfig | undefined): boolean {
  return EVENTS.some((event) => (config?.[event]?.length ?? 0) > 0)
}

// See the header for the source order and the .codex fallback rule.
function loadHookConfig(projectDir: string, userConfigDir: string, log: Logger): HookConfig {
  const claudeProject = [
    readHooks(join(projectDir, ".claude/settings.json"), log),
    readHooks(join(projectDir, ".claude/settings.local.json"), log),
  ]
  const merged: HookConfig = {}
  mergeHooks(merged, readHooks(join(userConfigDir, "settings.json"), log))
  if (claudeProject.some(hasHooks)) {
    for (const c of claudeProject) mergeHooks(merged, c)
  } else {
    mergeHooks(merged, readHooks(join(projectDir, ".codex/hooks.json"), log))
  }
  mergeHooks(merged, readHooks(join(projectDir, ".opencode/hooks.json"), log))
  return merged
}

// Full Claude-schema stdin payload. The common fields are on every event;
// transcript_path is present-but-empty so `jq -r '.transcript_path'` stays
// well-formed (synthesis deliberately skipped).
function buildPayload(
  event: ClaudeEvent,
  cwd: string,
  sessionID: string | undefined,
  fields: Record<string, unknown>,
): string {
  return JSON.stringify({
    session_id: sessionID ?? "",
    transcript_path: "",
    cwd,
    hook_event_name: event,
    ...fields,
  })
}

// Run a config command string via `bash -c` so $CLAUDE_PROJECT_DIR and the
// "adapter.sh script.sh" two-token form expand. detached:true makes the child a
// process-group leader so a timeout kills the whole tree (script and everything it spawned).
function runCommand(
  command: string,
  stdin: string,
  opts: { cwd: string; env: Record<string, string>; timeoutMs?: number },
): Promise<RunResult> {
  return new Promise((resolve) => {
    const child = spawn("bash", ["-c", command], {
      cwd: opts.cwd,
      env: opts.env,
      detached: true,
    })

    let out = ""
    let err = ""
    child.stdout?.on("data", (c) => { out += c.toString() })
    child.stderr?.on("data", (c) => { err += c.toString() })

    // A closed pipe (adapter exits before reading all stdin) must not crash us.
    child.stdin?.on("error", () => {})
    try {
      child.stdin?.write(stdin)
      child.stdin?.end()
    } catch {
      // group already gone
    }

    const killGroup = (signal: string) => {
      if (child.pid === undefined) return
      try {
        process.kill(-child.pid, signal)
      } catch {
        // group already exited
      }
    }

    let timedOut = false
    let softTimer: ReturnType<typeof setTimeout> | undefined
    let hardTimer: ReturnType<typeof setTimeout> | undefined
    if (opts.timeoutMs) {
      softTimer = setTimeout(() => {
        timedOut = true
        killGroup("SIGTERM")
        hardTimer = setTimeout(() => killGroup("SIGKILL"), HARD_KILL_GRACE_MS)
      }, opts.timeoutMs)
    }

    let settled = false
    const finish = (code: number) => {
      if (settled) return
      settled = true
      if (softTimer) clearTimeout(softTimer)
      if (hardTimer) clearTimeout(hardTimer)
      if (timedOut) {
        resolve({ code: TIMEOUT_CODE, stdout: out, stderr: err || `timed out after ${opts.timeoutMs}ms (process group killed)` })
      } else {
        resolve({ code, stdout: out, stderr: err })
      }
    }
    child.on("close", (c) => finish(c ?? 0))
    child.on("error", () => finish(1))
  })
}

export const server: Plugin = async ({ directory, worktree, client }) => {
  // In a non-git directory opencode assigns the "global" project, whose
  // worktree is the filesystem root — hooks config lives where the session
  // was opened, so only trust worktree when it points at a real project.
  const projectDir = worktree && parse(worktree).root !== worktree ? worktree : directory
  const log: Logger = (message, level = "info") =>
    client.app.log({ body: { service: "claude-hooks", level, message } }).catch(() => {})

  const userConfigDir = process.env.CLAUDE_CONFIG_DIR ?? join(homedir(), ".claude")
  const config = loadHookConfig(projectDir, userConfigDir, log)
  const env = { ...process.env, CLAUDE_PROJECT_DIR: projectDir }

  // One line on every boot so a silent skip is distinguishable from a healthy
  // load (opencode itself logs nothing when a plugin registers).
  const summary = EVENTS
    .map((event) => [event, (config[event] ?? []).flatMap((g) => g.hooks ?? []).length] as const)
    .filter(([, count]) => count > 0)
    .map(([event, count]) => `${count} ${event}`)
    .join(", ")
  log(`opencode-agent-hooks ${VERSION}: ${summary ? `loaded ${summary}` : "no hooks configured"} (project: ${projectDir})`)

  // toolName undefined = lifecycle event; Claude ignores matchers there.
  const matchingHooks = (event: ClaudeEvent, toolName?: string): HookCommand[] =>
    (config[event] ?? [])
      .filter((g) => toolName === undefined || matches(g.matcher, toolName))
      .flatMap((g) => g.hooks ?? [])
      .filter((h) => !h.type || h.type === "command")

  const run = (hook: HookCommand, stdin: string) =>
    runCommand(hook.command, stdin, {
      cwd: projectDir,
      env,
      timeoutMs: hook.timeout ? hook.timeout * 1000 : undefined, // Claude timeout is seconds
    })

  // Dedupe then run in parallel (Claude behavior); results keep config order so
  // decision application stays deterministic.
  const runHooks = async (hooks: HookCommand[], stdin: string) => {
    const deduped = dedupeHooks(hooks)
    const results = await Promise.all(deduped.map((hook) => run(hook, stdin)))
    return deduped.map((hook, i) => ({ hook, result: results[i] }))
  }

  const pushText = (parts: unknown[], text: string) => parts.push({ type: "text", text })

  // Per-session Stop re-entrancy guard (see header deviation 1). `stopRunning`
  // prevents overlapping idle chains.
  const idleRounds = new Map<string, number>()
  let stopRunning = false
  // SessionStart hook output, buffered until the session's first chat.message.
  const pendingSessionContext = new Map<string, string>()

  return {
    "tool.execute.before": async (input, output) => {
      const toolName = claudeToolName(input.tool)
      const hooks = matchingHooks("PreToolUse", toolName)
      if (!hooks.length) return
      const args = (output.args ?? {}) as Record<string, unknown>
      const stdin = buildPayload("PreToolUse", projectDir, input.sessionID, {
        tool_name: toolName,
        tool_input: toSnakeKeys(args),
      })
      const results = await runHooks(hooks, stdin)
      const decisions = results.map(({ hook, result }) => ({
        hook,
        decision: classifyPre(result, hook.failClosed ?? false),
      }))

      // Deny-wins aggregation, all reasons surfaced together.
      const denies = decisions.filter((d) => d.decision.kind === "deny")
      if (denies.length) throw new Error(denies.map((d) => d.decision.reason).join("\n"))
      for (const d of decisions) {
        if (d.decision.kind === "ask" || d.decision.kind === "allow") {
          log(`${hookLabel(d.hook.command)} returned permissionDecision:"${d.decision.kind}"; deferring to OpenCode's own permission flow`)
        }
      }

      // updatedInput replaces tool_input wholesale (Claude semantics); last
      // supplier in config order wins. Mutate output.args in place — OpenCode
      // holds the object reference.
      const updated = decisions.filter((d) => d.decision.updatedInput).pop()?.decision.updatedInput
      if (updated && output.args && typeof output.args === "object") {
        const args = output.args as Record<string, unknown>
        for (const k of Object.keys(args)) delete args[k]
        Object.assign(args, toCamelKeys(updated))
      }
    },

    "tool.execute.after": async (input, output) => {
      const toolName = claudeToolName(input.tool)
      const hooks = matchingHooks("PostToolUse", toolName)
      if (!hooks.length) return
      const args = (input.args ?? {}) as Record<string, unknown>
      if ((input.tool === "write" || input.tool === "edit") && !args.filePath && !args.file_path) {
        log(`${input.tool}: no file path in args; PostToolUse hooks got an empty path`, "warn")
      }
      const stdin = buildPayload("PostToolUse", projectDir, input.sessionID, {
        tool_name: toolName,
        tool_input: toSnakeKeys(args),
        // Approximation: Claude's per-tool tool_response shapes can't be replicated.
        tool_response: {
          ...toSnakeKeys((output.metadata ?? {}) as Record<string, unknown>),
          output: output.output,
        },
      })
      const results = await runHooks(hooks, stdin)

      // Everything a hook has to say rides the tool output itself, so the model
      // sees it on the same turn (Claude's "stderr fed back" semantics).
      let replaced = 0
      for (const { hook, result } of results) {
        const json = parseHookJson(result.stdout)
        const updatedOutput = json?.hookSpecificOutput?.updatedToolOutput
        if (updatedOutput !== undefined) {
          if (replaced++) log(`multiple hooks replaced the tool output; ${hookLabel(hook.command)} wins`, "warn")
          output.output = typeof updatedOutput === "string" ? updatedOutput : JSON.stringify(updatedOutput)
        }
        if (result.code === 2 || json?.decision === "block") {
          const reason = result.code === 2 ? result.stderr.trim() : json?.reason
          output.output += `\n\n[hook feedback: ${hookLabel(hook.command)}]\n${reason || "hook reported a problem"}`
        } else if (result.stderr.trim()) {
          log(`${hook.command}: ${result.stderr.trim()}`, "warn")
        }
        const ctx = json?.hookSpecificOutput?.additionalContext
        if (typeof ctx === "string" && ctx.trim()) output.output += `\n\n[hook context]\n${ctx}`
        if (json?.continue === false) {
          output.output += `\n\n[hook stop request]\n${json.stopReason ?? "A hook requested the agent stop."}`
          log(`${hookLabel(hook.command)} requested continue:false (session abort unsupported; stopReason appended)`, "warn")
        }
      }
    },

    "chat.message": async (input, output) => {
      const pending = input.sessionID ? pendingSessionContext.get(input.sessionID) : undefined
      if (pending) {
        pendingSessionContext.delete(input.sessionID)
        pushText(output.parts, `<session-start-hook>\n${pending}\n</session-start-hook>`)
      }

      const hooks = matchingHooks("UserPromptSubmit")
      if (!hooks.length) return
      const prompt = output.parts
        .filter((p): p is { type: "text"; text: string } =>
          (p as { type?: string }).type === "text" && typeof (p as { text?: unknown }).text === "string")
        .map((p) => p.text)
        .join("\n")
      const stdin = buildPayload("UserPromptSubmit", projectDir, input.sessionID, { prompt })
      const results = await runHooks(hooks, stdin)

      const blocks: string[] = []
      const contexts: string[] = []
      for (const { hook, result } of results) {
        const json = parseHookJson(result.stdout)
        if (result.code === 2 || json?.decision === "block") {
          blocks.push((result.code === 2 ? result.stderr.trim() : json?.reason) || "Prompt blocked by hook")
          continue
        }
        if (result.code !== 0) {
          log(`${hook.command}: ${result.stderr.trim() || `exit ${result.code}`}`, "warn")
          continue
        }
        const ctx = json?.hookSpecificOutput?.additionalContext
        if (typeof ctx === "string" && ctx.trim()) contexts.push(ctx.trim())
        // UserPromptSubmit is the one event where plain stdout injects as context.
        else if (!json && result.stdout.trim()) contexts.push(result.stdout.trim())
      }
      // Best-effort block (header deviation 4): chat.message has no documented
      // reject channel. If OpenCode surfaces the throw badly, the fallback is to
      // replace the prompt text in output.parts with the reason instead.
      if (blocks.length) throw new Error(blocks.join("\n"))
      if (contexts.length) {
        pushText(output.parts, `<user-prompt-submit-hook>\n${contexts.join("\n")}\n</user-prompt-submit-hook>`)
      }
    },

    "experimental.session.compacting": async (input, output) => {
      const hooks = matchingHooks("PreCompact")
      if (!hooks.length) return
      const stdin = buildPayload("PreCompact", projectDir, input.sessionID, {
        trigger: "auto",
        custom_instructions: "",
      })
      const results = await runHooks(hooks, stdin)
      for (const { result } of results) {
        const json = parseHookJson(result.stdout)
        const ctx = json?.hookSpecificOutput?.additionalContext
        if (typeof ctx === "string" && ctx.trim()) output.context.push(ctx.trim())
        else if (!json && result.code === 0 && result.stdout.trim()) output.context.push(result.stdout.trim())
      }
    },

    event: async ({ event }) => {
      if (event.type === "session.created") {
        const hooks = matchingHooks("SessionStart")
        if (!hooks.length) return
        const info = (event.properties as { info?: { id?: string; parentID?: string } }).info
        if (!info?.id || info.parentID) return // subagent sessions don't re-run SessionStart
        const stdin = buildPayload("SessionStart", projectDir, info.id, { source: "startup" })
        const results = await runHooks(hooks, stdin)
        const pieces: string[] = []
        for (const { result } of results) {
          const json = parseHookJson(result.stdout)
          const ctx = json?.hookSpecificOutput?.additionalContext
          if (typeof ctx === "string" && ctx.trim()) pieces.push(ctx.trim())
          else if (!json && result.code === 0 && result.stdout.trim()) pieces.push(result.stdout.trim())
        }
        if (pieces.length) pendingSessionContext.set(info.id, pieces.join("\n"))
        return
      }

      if (event.type !== "session.idle") return
      const hooks = matchingHooks("Stop")
      if (!hooks.length || stopRunning) return
      const sessionID = (event.properties as { sessionID?: string }).sessionID
      if (!sessionID) return
      const round = idleRounds.get(sessionID) ?? 0
      if (round >= MAX_IDLE_FIX_ROUNDS) {
        log(`idle fix rounds exhausted (${round}); leaving session alone`, "warn")
        return
      }

      stopRunning = true
      try {
        const stdin = buildPayload("Stop", projectDir, sessionID, { stop_hook_active: round > 0 })
        const results = await runHooks(hooks, stdin)
        const failures: string[] = []
        let suppress = false
        for (const { hook, result } of results) {
          const json = parseHookJson(result.stdout)
          // Any non-zero is "not clean": 2 = the script's veto, 124 = timeout,
          // anything else = it crashed. decision:"block" vetoes even on exit 0.
          if (result.code !== 0) failures.push(result.stderr.trim() || `${hook.command} failed`)
          else if (json?.decision === "block") failures.push(json.reason ?? `${hookLabel(hook.command)} blocked completion`)
          if (json?.continue === false) suppress = true
        }
        if (suppress) {
          idleRounds.set(sessionID, MAX_IDLE_FIX_ROUNDS)
          log("a Stop hook returned continue:false; suppressing re-prompts for this chain", "warn")
        } else if (failures.length) {
          idleRounds.set(sessionID, round + 1)
          await client.session.prompt({
            path: { id: sessionID },
            body: {
              parts: [{
                type: "text",
                text: "Completion checks failed. Please fix the following before finishing:\n\n" + failures.join("\n\n"),
              }],
            },
          }).catch((e) => log(`re-prompt failed: ${e}`, "warn"))
        } else {
          idleRounds.delete(sessionID)
        }
      } finally {
        stopRunning = false
      }
    },
  }
}
