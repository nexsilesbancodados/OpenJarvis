# Nightly report — voice-first foundation

Session run autonomously on branch `feat/voice-first-foundation`, cut from
`main @ 4f857b0`. Everything below is reproducible from the commands quoted.

A second agent (Codex) worked in the same tree during this session. Where
that matters — shared files, an unreviewed regression — it is called out
explicitly rather than blended in.

---

## Starting state

Nothing ran. The checkout had never been built on this machine:

| | Found | Needed |
|---|---|---|
| Python | 3.14.7 only | `>=3.10,<3.14` |
| uv | absent | required |
| Rust / MSVC | absent | needed by the native extension |
| Ollama | absent | an inference provider |
| `~/.openjarvis` | did not exist | runtime state |
| `upstream` remote | not configured | needed to stay mergeable |

The frontend was a working React 19 / Vite 6 / Tauri 2 app — not greenfield.

---

## What was done

### Environment (Phase 0)

uv 0.12.3, Python 3.13.15 pinned, CI extras synced, Rust 1.97.1 with MSVC
14.44 and Windows SDK 10.0.26100, and `openjarvis-rust==0.1.0` built from
source. `upstream` now points at `open-jarvis/OpenJarvis`.

Configured OpenAI as the provider. One credential covers the whole stack:
`gpt-4o-mini` for chat, `whisper-1` for STT, `tts-1` for TTS. The key lives
in `~/.openjarvis/credentials.toml`, outside the repository, with its ACL
restricted to the current user.

### Shipped on this branch

| Commit | What |
|---|---|
| `7753f47` | `SETUP.md` — contributor setup, the non-hermetic test suite, the Windows toolchain requirement, the AudioWorklet CSP trap |
| `ecc5ad5` | Microphone unblocked in served builds |
| `29e8c01` | SendBlue webhook registration pointed at the real handler |
| `b3bd7e3` | Tool arguments validated against the advertised JSON Schema |
| `31ff79e` | Codex CLI adapter (`agents/codex.py`) |
| `5dbe642` | External analytics made opt-in |
| `0e8fdf5` | Server tool calls gated on the approval queue |

---

## Bugs found and fixed

**The microphone was blocked by a header.** `Permissions-Policy:
microphone=()` went out on every response including the SPA's `index.html`,
so `getUserMedia` was denied in any build served by FastAPI. It only appeared
to work because the Vite dev server does not send that header. For a
voice-first product this was the foundation being quietly absent.

**Three SendBlue webhook URLs pointed nowhere.** They registered
`/v1/channels/sendblue/webhook`; the handler is `POST /webhooks/sendblue`.
A URL under `/v1/` matches no router and lands on the SPA catch-all, which
answers `200` with `index.html` — so SendBlue recorded a healthy-looking
webhook that never delivered a message. A fourth call site already used the
correct path, which is what confirmed the target.

**Server agents bypassed the approval queue entirely.** Every server path
passed `confirm_callback=lambda _prompt: True`. The risk tiers and remembered
permissions in `approval_store` were real code protecting nothing — an agent
reachable by voice could run `shell_exec` unattended. This was the blocker
for any computer-control work.

**External analytics were on by default** with a hardcoded PostHog host and
key, no `DO_NOT_TRACK`, and no environment override. Acceptable upstream;
not acceptable in a fork adding observers that read window titles and
terminal commands.

**A latent reload hazard, caught while investigating my own flaky tests.**
`ensure_registries_populated` reloads every `openjarvis.tools.*` module when
the registry empties, and it runs on a production path.
`importlib.reload` rebinds module globals in place, so after a reload the old
function object raises the *new* exception class while `tools/_stubs.py` —
excluded from the reload — still holds the old one. `except ValidationError`
would stop matching and the exception would escape `ToolExecutor.execute`.
Verified empirically, then avoided by putting the validator in `core/`, with
a regression test pinning it.

**A tiering mistake of my own,** caught by an existing test rather than by
inspection: `memory_store` was classified `low`, so the gate queued it. An
assistant that asks permission before remembering anything is unusable, and a
gate the user switches off protects nothing. The assistant's own notes are
now `trivial`; things that touch the world outside it are not.

---

## Known regression — not mine, not fixed

`tests/server/test_model_management.py::TestStreamingResilience::test_stream_without_agent_uses_direct_engine`
fails on the current working tree. It comes from the other agent's
uncommitted change to `src/openjarvis/server/routes.py`, which routes
streaming chat through the agent whenever `agent.accepts_tools` is truthy.
**It is not included in this branch.**

The intent is right — the old streaming path silently bypassed every
configured tool. Two problems with the implementation:

1. `getattr(agent, "accepts_tools", False)` is truthy for any non-boolean.
   The failing test passes a `MagicMock`, so a mock standing in for `simple`
   is treated as tool-accepting. A real agent is fine, but the check should
   be `is True`.
2. **The larger one, and the reason I did not simply fix the test.** The new
   path runs the agent to completion and then slices the finished response
   into 64-character chunks. That is not streaming. For the default
   `orchestrator` configuration it converts real token-by-token output into
   post-hoc chunking, so time-to-first-token becomes time-to-full-response.
   For a voice product that is the single most expensive latency to give up:
   time-to-first-audio is bounded by it.

The real fix is to stream tokens *and* run tools — `engine.stream_full()`
already yields `tool_calls`, and `agent_manager_routes` already does exactly
this on the managed-agent SSE path. That is a design change in a file another
agent was actively editing, so I left it alone rather than risk clobbering
work in progress.

---

## Tests

Run with `OPENJARVIS_HOME` pointed at a throwaway directory — see `SETUP.md`
for why that is not optional.

| Scope | Result |
|---|---|
| Full suite, with the Rust extension | **7382 passed**, 43 failed, 56 skipped |
| Full suite, without the Rust extension | 7143 passed, 163 failed |
| Frontend `vitest` | 21 passed |
| `npx tsc --noEmit` | exit 0 |
| `npm run build` | exit 0 |
| `ruff check` / `ruff format --check` | clean |

The Rust extension is worth ~120 tests.

**The 43 remaining failures are platform, not regressions.** Verified by
`git stash`-ing my changes, re-running, and comparing the failing sets:
every failure in the "after" set is also in the "before" set, except the
`routes.py` one described above, which is not on this branch.

| Cluster | Why it fails on Windows |
|---|---|
| `security/test_subprocess_sandbox.py` (7) | `os.setsid` / `os.killpg` are POSIX-only |
| `security/test_file_permissions.py`, `core/test_credentials.py` (6) | `os.chmod(0o600)` is a no-op; the file reports `0o666` |
| `telemetry/test_energy_rapl.py` (5) | reads Linux `/sys/class/powercap` |
| `traces/test_store_fts.py` (4) | Windows holds the SQLite handle during teardown |
| `tools/test_template_loader_security.py` (2) | asserts POSIX shell quoting |
| assorted | path separators, packaging layout, banner text |

The suite is mildly flaky on Windows under `xdist` — totals drifted between
42 and 52 across identical runs, driven by the SQLite-teardown cluster. The
*set* of failing test ids is the reliable signal; raw counts are not.

### End-to-end, against the real server

```
GET  /health                → 200 {"status":"ok"}
GET  /v1/info               → gpt-4o-mini · orchestrator · cloud
GET  /v1/speech/health      → available: true, backend openai
POST /v1/chat/completions   → real inference, correct answer
Permissions-Policy header   → camera=(), microphone=(self), geolocation=()
Content-Security-Policy     → …; media-src 'self' blob:; worker-src 'self' blob:
```

---

## Technical decisions

**Queue-and-refuse, not block.** `confirm_callback` is synchronous and
returns a bool. Blocking a voice turn until a human notices would hold the
request open indefinitely. A gated call is refused with a sentence the
assistant can say aloud, the action lands in the queue the UI already polls,
and the next attempt after approval spends that grant once.

**Everything additive.** No existing ABC signature changed, no existing route
contract changed. `ToolExecutor` gained an optional `approval_gate`; left
unset it behaves exactly as before, so the CLI path is untouched. This is to
keep merging from `upstream` cheap.

**Two upstream tests were deliberately changed** (`test_stubs.py`,
`test_client.py`). Both asserted that calling a tool with no arguments
succeeds even when its schema marks one required — incidental behaviour,
since `think` declares `thought` required and then does
`params.get("thought", "")`. This is a real contract change and is flagged as
such; it can be made opt-in if minimising divergence matters more.

**Risk is a property of the call, not the tool.** `shell_exec ls` and
`shell_exec rm -rf` classify differently. Permission keys are narrower than
tool names, so approving a message to one contact does not approve messages
to everyone.

---

## Security

- No secrets are committed. `.env.local` — which holds an OpenAI key — is
  covered by `.gitignore:41` and has never appeared in any commit; verified
  with `git log --all -- .env.local`.
- `scripts/run-openjarvis-with-local-key.ps1` (the other agent's) contains no
  key; it reads one from `.env.local` at runtime. Safe.
- The configured key lives in `~/.openjarvis/credentials.toml`, outside the
  repository.
- No security control was weakened to make anything pass. The only
  permission relaxed is `microphone=()` → `microphone=(self)`, which is the
  minimum a voice product needs and is pinned by a test.

### Two credential problems that need you

1. **The key pasted into chat should be rotated.** It works and is stored
   correctly, but it went through a conversation transcript. Generate a
   replacement and revoke it.
2. **`os.chmod` does not protect anything on Windows.**
   `core/credentials.py` and `connectors/oauth.py` both write secrets and
   call `os.chmod(path, 0o600)` — which is a no-op here, leaving the files
   readable by every account on the machine. OAuth token files hold refresh
   tokens *and* client secrets in cleartext. Mitigated by hand with `icacls`;
   the real fix is the Credential Manager, which the Tauri side already uses
   via the `keyring` crate. This is the failing `test_file_permissions`
   cluster telling the truth.

---

## Blocked on you

- **Rotating the OpenAI key** (above).
- **WhatsApp.** Baileys is blocked by Meta — upstream marks it *Blocked*, with
  405s on unofficial connections. The official path is
  `channels/whatsapp.py` (Cloud API, send-only) and needs a Meta Business
  account. Telegram, Signal and SendBlue work today.
- **Realtime API vs a local voice pipeline.** The configured key exposes the
  `gpt-realtime` family, which provides duplex audio, VAD and barge-in
  server-side — most of Phases 3 and 4 as integration rather than
  construction. It is cloud-only and metered, which cuts against local-first.
  Recommendation: adopt it behind the same `WS /v1/voice/session` interface
  the plan already specifies, keeping a local backend as the alternative.
- **No deploy target exists.** There is no configured hosting, no deploy
  workflow bound to a branch, and no production environment for this fork —
  so there was no deploy to run or validate. CI runs on push/PR only.

---

## Recommended next steps

1. Resolve the `routes.py` streaming regression — stream tokens *and* run
   tools, rather than chunking a finished response.
2. Finish the approval loop end to end: the UI can now record "always", but
   nothing in the server executes an approved action; it waits for the
   proactive cron. Wire `ExecutePendingActionsTool` to the approval routes.
3. Extend the audit log to cover tool invocations. It currently records only
   three security-scan event types, so approvals and outbound sends leave no
   trail.
4. Turn on `CapabilityPolicy` with `default_deny=True`. It exists, is
   disabled by default, and is open-by-default when enabled.
5. Then Phase 7 (desktop control) has the safety floor it requires.

---

*Plan of record: https://claude.ai/code/artifact/d3af2549-e781-49d2-82bc-806c0459215a*
