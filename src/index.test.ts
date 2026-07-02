// Tests for the OpenCode Claude-hooks bridge, driven through the exported
// server() (bare helper exports would be invoked as plugins by OpenCode's
// loader, so everything is exercised via the public surface).
//
// Run: node --test src/index.test.ts   (Node >= 23 strips types natively;
// 22.6+ needs --experimental-strip-types.)
//
// Each test boots a throwaway project dir whose .claude/settings.json declares
// inline bash one-liner hooks; CLAUDE_CONFIG_DIR points at an empty dir so the
// developer's real ~/.claude/settings.json never leaks in.

import { test } from "node:test"
import assert from "node:assert/strict"
import { existsSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { server } from "./index.ts"

process.env.CLAUDE_CONFIG_DIR = mkdtempSync(join(tmpdir(), "claude-user-"))

type AnyRecord = Record<string, any>

const cmd = (command: string, extra: AnyRecord = {}) => ({ type: "command", command, ...extra })
const group = (hooks: AnyRecord[], matcher?: string) => ({ matcher, hooks })

function makeProject(claudeHooks?: AnyRecord, opencodeHooks?: AnyRecord, codexHooks?: AnyRecord): string {
  const dir = mkdtempSync(join(tmpdir(), "hooks-test-"))
  if (claudeHooks) {
    mkdirSync(join(dir, ".claude"))
    writeFileSync(join(dir, ".claude/settings.json"), JSON.stringify({ hooks: claudeHooks }))
  }
  if (opencodeHooks) {
    mkdirSync(join(dir, ".opencode"))
    writeFileSync(join(dir, ".opencode/hooks.json"), JSON.stringify({ hooks: opencodeHooks }))
  }
  if (codexHooks) {
    mkdirSync(join(dir, ".codex"))
    writeFileSync(join(dir, ".codex/hooks.json"), JSON.stringify({ hooks: codexHooks }))
  }
  return dir
}

async function boot(dir: string, worktree?: string) {
  const prompts: AnyRecord[] = []
  const client = {
    app: { log: async () => {} },
    session: { prompt: async (p: AnyRecord) => { prompts.push(p) } },
  }
  const hooks = (await server({ directory: dir, worktree, client } as any)) as AnyRecord
  return { hooks, prompts }
}

test("PreToolUse payload carries the full Claude schema with snake_case tool_input", async () => {
  const dir = makeProject({ PreToolUse: [group([cmd("cat > pre.json")], "Bash")] })
  const { hooks } = await boot(dir)
  await hooks["tool.execute.before"](
    { tool: "bash", sessionID: "sess-1", callID: "c1" },
    { args: { command: "echo hi", someOption: true } },
  )
  const payload = JSON.parse(readFileSync(join(dir, "pre.json"), "utf8"))
  assert.equal(payload.session_id, "sess-1")
  assert.equal(payload.cwd, dir)
  assert.equal(payload.hook_event_name, "PreToolUse")
  assert.equal(payload.transcript_path, "")
  assert.equal(payload.tool_name, "Bash")
  assert.equal(payload.tool_input.command, "echo hi")
  assert.equal(payload.tool_input.some_option, true)
})

test("PreToolUse deny wins over a parallel allow", async () => {
  const dir = makeProject({
    PreToolUse: [group([
      cmd(`echo '{"hookSpecificOutput":{"permissionDecision":"allow"}}'`),
      cmd("echo blocked-reason >&2; exit 2"),
    ], "Bash")],
  })
  const { hooks } = await boot(dir)
  await assert.rejects(
    hooks["tool.execute.before"]({ tool: "bash", sessionID: "s", callID: "c" }, { args: { command: "x" } }),
    /blocked-reason/,
  )
})

test("PreToolUse updatedInput replaces args wholesale, keys camelized", async () => {
  const dir = makeProject({
    PreToolUse: [group([
      cmd(`echo '{"hookSpecificOutput":{"updatedInput":{"file_path":"/tmp/new.txt","command":"safe"}}}'`),
    ], "Write")],
  })
  const { hooks } = await boot(dir)
  const output = { args: { filePath: "/tmp/old.txt", content: "gone", command: "orig" } as AnyRecord }
  await hooks["tool.execute.before"]({ tool: "write", sessionID: "s", callID: "c" }, output)
  assert.deepEqual(output.args, { filePath: "/tmp/new.txt", command: "safe" })
})

test("failClosed from .opencode/hooks.json dedup-merges and turns a crash into a deny", async () => {
  const crash = "echo ran >> runs.txt; exit 9"
  const dir = makeProject(
    { PreToolUse: [group([cmd(crash)], "Bash")] },
    { PreToolUse: [group([cmd(crash, { failClosed: true })], "Bash")] },
  )
  const { hooks } = await boot(dir)
  await assert.rejects(
    hooks["tool.execute.before"]({ tool: "bash", sessionID: "s", callID: "c" }, { args: { command: "x" } }),
    /failed closed \(exit 9\)/,
  )
  // deduped: the identical command ran once despite appearing in two configs
  assert.equal(readFileSync(join(dir, "runs.txt"), "utf8").trim(), "ran")
})

test("PreToolUse crash without failClosed stays fail-open", async () => {
  const dir = makeProject({ PreToolUse: [group([cmd("exit 9")], "Bash")] })
  const { hooks } = await boot(dir)
  await hooks["tool.execute.before"]({ tool: "bash", sessionID: "s", callID: "c" }, { args: { command: "x" } })
})

test("failClosed timeout denies", async () => {
  const dir = makeProject(
    { PreToolUse: [group([cmd("sleep 5", { timeout: 1, failClosed: true })], "Bash")] },
  )
  const { hooks } = await boot(dir)
  await assert.rejects(
    hooks["tool.execute.before"]({ tool: "bash", sessionID: "s", callID: "c" }, { args: { command: "x" } }),
    /failed closed \(exit 124\)/,
  )
})

test("PostToolUse payload includes tool_response; context and exit-2 feedback append to output", async () => {
  const dir = makeProject({
    PostToolUse: [group([
      cmd("cat > post.json"),
      cmd(`echo '{"hookSpecificOutput":{"hookEventName":"PostToolUse","additionalContext":"watch out"}}'`),
      cmd("echo needs-work >&2; exit 2"),
    ], "Write|Edit")],
  })
  const { hooks } = await boot(dir)
  const output = { title: "t", output: "wrote file", metadata: { fileDiff: "d" } as AnyRecord }
  await hooks["tool.execute.after"](
    { tool: "edit", sessionID: "sess-2", callID: "c", args: { filePath: "/tmp/a.py" } },
    output,
  )
  const payload = JSON.parse(readFileSync(join(dir, "post.json"), "utf8"))
  assert.equal(payload.hook_event_name, "PostToolUse")
  assert.equal(payload.tool_input.file_path, "/tmp/a.py")
  assert.equal(payload.tool_response.output, "wrote file")
  assert.equal(payload.tool_response.file_diff, "d")
  assert.match(output.output, /\[hook context\]\nwatch out/)
  assert.match(output.output, /\[hook feedback: .*\]\nneeds-work/)
})

test("PostToolUse updatedToolOutput replaces the tool output", async () => {
  const dir = makeProject({
    PostToolUse: [group([
      cmd(`echo '{"hookSpecificOutput":{"updatedToolOutput":"replaced"}}'`),
    ], "Write|Edit")],
  })
  const { hooks } = await boot(dir)
  const output = { title: "t", output: "original", metadata: {} }
  await hooks["tool.execute.after"]({ tool: "write", sessionID: "s", callID: "c", args: { filePath: "/x" } }, output)
  assert.equal(output.output, "replaced")
})

test("Stop: stop_hook_active is false on the first round, true on re-entry", async () => {
  const dir = makeProject({
    Stop: [group([cmd("cat >> stops.jsonl; echo >> stops.jsonl; echo not-clean >&2; exit 2")])],
  })
  const { hooks, prompts } = await boot(dir)
  const idle = { event: { type: "session.idle", properties: { sessionID: "s-stop" } } }
  await hooks.event(idle)
  await hooks.event(idle)
  const rounds = readFileSync(join(dir, "stops.jsonl"), "utf8").trim().split("\n").map((l) => JSON.parse(l))
  assert.equal(rounds.length, 2)
  assert.equal(rounds[0].stop_hook_active, false)
  assert.equal(rounds[0].hook_event_name, "Stop")
  assert.equal(rounds[1].stop_hook_active, true)
  assert.equal(prompts.length, 2)
  assert.match(prompts[0].body.parts[0].text, /not-clean/)
})

test("Stop: decision block vetoes on exit 0; rounds cap quiesces", async () => {
  const dir = makeProject({
    Stop: [group([cmd(`echo '{"decision":"block","reason":"still dirty"}'`)])],
  })
  const { hooks, prompts } = await boot(dir)
  const idle = { event: { type: "session.idle", properties: { sessionID: "s" } } }
  for (let i = 0; i < 5; i++) await hooks.event(idle)
  assert.equal(prompts.length, 3) // MAX_IDLE_FIX_ROUNDS
  assert.match(prompts[0].body.parts[0].text, /still dirty/)
})

test("UserPromptSubmit: raw stdout injects as a context part", async () => {
  const dir = makeProject({ UserPromptSubmit: [group([cmd("echo remember-the-milk")])] })
  const { hooks } = await boot(dir)
  const output = { message: {}, parts: [{ type: "text", text: "original prompt" }] as AnyRecord[] }
  await hooks["chat.message"]({ sessionID: "s" }, output)
  assert.equal(output.parts.length, 2)
  assert.match(output.parts[1].text, /<user-prompt-submit-hook>\nremember-the-milk\n<\/user-prompt-submit-hook>/)
})

test("UserPromptSubmit: exit 2 blocks the prompt", async () => {
  const dir = makeProject({ UserPromptSubmit: [group([cmd("echo no-secrets >&2; exit 2")])] })
  const { hooks } = await boot(dir)
  await assert.rejects(
    hooks["chat.message"]({ sessionID: "s" }, { message: {}, parts: [{ type: "text", text: "hi" }] }),
    /no-secrets/,
  )
})

test("SessionStart context is buffered and delivered with the first chat.message", async () => {
  const dir = makeProject({ SessionStart: [group([cmd("echo starting-context")])] })
  const { hooks } = await boot(dir)
  await hooks.event({ event: { type: "session.created", properties: { info: { id: "s-new" } } } })
  const output = { message: {}, parts: [{ type: "text", text: "hi" }] as AnyRecord[] }
  await hooks["chat.message"]({ sessionID: "s-new" }, output)
  assert.equal(output.parts.length, 2)
  assert.match(output.parts[1].text, /<session-start-hook>\nstarting-context\n<\/session-start-hook>/)
  // drained: a second message gets nothing
  const again = { message: {}, parts: [{ type: "text", text: "more" }] as AnyRecord[] }
  await hooks["chat.message"]({ sessionID: "s-new" }, again)
  assert.equal(again.parts.length, 1)
})

test("SessionStart skips subagent (child) sessions", async () => {
  const dir = makeProject({ SessionStart: [group([cmd("echo child-context")])] })
  const { hooks } = await boot(dir)
  await hooks.event({ event: { type: "session.created", properties: { info: { id: "kid", parentID: "parent" } } } })
  const output = { message: {}, parts: [{ type: "text", text: "hi" }] as AnyRecord[] }
  await hooks["chat.message"]({ sessionID: "kid" }, output)
  assert.equal(output.parts.length, 1)
})

test("root worktree (opencode's global project) falls back to directory", async () => {
  const dir = makeProject({ PreToolUse: [group([cmd("cat > pre.json")], "Bash")] })
  const { hooks } = await boot(dir, "/")
  await hooks["tool.execute.before"]({ tool: "bash", sessionID: "s", callID: "c" }, { args: { command: "x" } })
  const payload = JSON.parse(readFileSync(join(dir, "pre.json"), "utf8"))
  assert.equal(payload.cwd, dir)
})

test("Codex-only project: .codex/hooks.json is read as the fallback source", async () => {
  const dir = makeProject(undefined, undefined, {
    PreToolUse: [group([cmd("cat > pre.json")], "Bash")],
  })
  const { hooks } = await boot(dir)
  await hooks["tool.execute.before"]({ tool: "bash", sessionID: "s", callID: "c" }, { args: { command: "x" } })
  const payload = JSON.parse(readFileSync(join(dir, "pre.json"), "utf8"))
  assert.equal(payload.hook_event_name, "PreToolUse")
})

test("project .claude hooks win over .codex/hooks.json (no double-run)", async () => {
  const dir = makeProject(
    { PreToolUse: [group([cmd("echo x > claude-ran.txt")], "Bash")] },
    undefined,
    { PreToolUse: [group([cmd("echo x > codex-ran.txt")], "Bash")] },
  )
  const { hooks } = await boot(dir)
  await hooks["tool.execute.before"]({ tool: "bash", sessionID: "s", callID: "c" }, { args: { command: "x" } })
  assert.equal(existsSync(join(dir, "claude-ran.txt")), true)
  assert.equal(existsSync(join(dir, "codex-ran.txt")), false)
})

test("a .claude/settings.json without hooks still falls back to .codex", async () => {
  const dir = makeProject(
    {}, // settings file exists but defines no hooks
    undefined,
    { PreToolUse: [group([cmd("echo x > codex-ran.txt")], "Bash")] },
  )
  const { hooks } = await boot(dir)
  await hooks["tool.execute.before"]({ tool: "bash", sessionID: "s", callID: "c" }, { args: { command: "x" } })
  assert.equal(existsSync(join(dir, "codex-ran.txt")), true)
})

test("PreCompact hook stdout lands in the compaction context", async () => {
  const dir = makeProject({ PreCompact: [group([cmd("echo keep-the-todos")])] })
  const { hooks } = await boot(dir)
  const output = { context: [] as string[] }
  await hooks["experimental.session.compacting"]({ sessionID: "s" }, output)
  assert.deepEqual(output.context, ["keep-the-todos"])
})
