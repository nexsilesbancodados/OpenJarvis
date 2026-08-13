# Development setup

Setup notes for this fork, verified on Windows 11 with PowerShell. The
upstream [installation docs](https://open-jarvis.github.io/OpenJarvis/getting-started/install/)
cover the end-user installers; this file covers a contributor checkout and
records the two things that are easy to get wrong here.

## Prerequisites

| Tool | Version | Notes |
|---|---|---|
| Python | `>=3.10,<3.14` | **3.14 does not work.** `uv` installs a supported one for you. |
| uv | any recent | Manages the interpreter and the venv. |
| Node | `>=20` | Only needed for the frontend and the desktop app. |
| Rust | stable | **Optional for most work** — see [The Rust extension](#the-rust-extension). |

## Quick start

```powershell
# 1. uv (installs to ~/.local/bin — restart the shell or prepend it to PATH)
irm https://astral.sh/uv/install.ps1 | iex
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"

# 2. A supported interpreter. .python-version is gitignored, so this is
#    per-checkout and will not show up as a working-tree change.
uv python install 3.13
uv python pin 3.13

# 3. Dependencies. These are the same extras `make setup` uses; skipping
#    framework-comparison makes test collection fail on a polars import.
uv sync --extra dev --extra framework-comparison --extra server

# 4. Verify — note the isolated home, see "Running the tests" below
$env:OPENJARVIS_HOME = "$env:TEMP\oj-test-home"
uv run pytest tests/ -n auto -q -m "not live and not cloud and not hub"
Remove-Item Env:\OPENJARVIS_HOME
uv run jarvis --help
```

Frontend:

```powershell
cd frontend
npm ci
npx tsc --noEmit
npm run build     # emits into src/openjarvis/server/static/
```

The Vite dev server proxies `/v1`, `/api` and `/health` to `localhost:8000`,
so `jarvis serve` plus `npm run dev` is the full loop. There is no second
backend to start — in a production build FastAPI serves the SPA from
`server/static/`.

## Running the tests

**Set `OPENJARVIS_HOME` to a throwaway directory first.** Parts of the suite
read the real `~/.openjarvis/config.toml` instead of a fixture, so a machine
with a working config fails tests that pass on a clean one:

```
tests/agents/test_base_agent.py::TestBaseAgentInit::test_default_params
tests/cli/test_ask_router.py::TestAskModelResolution::test_default_model_from_config
tests/cli/test_ask_router.py::TestAskModelResolution::test_explicit_model_flag
```

They assert framework defaults (`temperature`, `max_tokens`, `default_model`)
that any real `[intelligence]` section overrides. `core/paths.py:82`
`get_config_dir()` honours `$OPENJARVIS_HOME` ahead of `~/.openjarvis`, so
pointing it elsewhere gives the suite a clean root. CI never hit this because
CI has no user config.

### Expected results on Windows

| Configuration | Result |
|---|---|
| With the Rust extension, isolated home | **7280 passed, 42 failed, 58 skipped** |
| Without the Rust extension | 7143 passed, 163 failed |

The extension is worth ~120 tests. The ~42 that remain are platform, not
regressions — check against this list before assuming you broke something:

| Cluster | Why it fails on Windows |
|---|---|
| `security/test_subprocess_sandbox.py` (7) | `os.setsid` / `os.killpg` are POSIX-only. |
| `security/test_file_permissions.py`, `core/test_credentials.py` (6) | `os.chmod(0o600)` is a no-op; the file reports `0o666`. |
| `telemetry/test_energy_rapl.py` (5) | Reads Linux `/sys/class/powercap`. |
| `traces/test_store_fts.py` (4) | `PermissionError` — Windows holds the SQLite handle during teardown. |
| `tools/test_template_loader_security.py` (2) | Asserts POSIX shell quoting. |
| assorted singles | Path separators, packaging layout, banner text. |

The permission cluster is not merely cosmetic — see
[Fork-specific defaults](#fork-specific-defaults).

## The Rust extension

`openjarvis_rust` is a PyO3 extension built from `rust/`. The docstring in
`src/openjarvis/_rust_bridge.py` calls it mandatory, and for anything that
actually reaches Rust it is — but **importing the bridge does not fail**.
`_detect_rust()` swallows the `ImportError` and sets `RUST_AVAILABLE = False`,
so a checkout without a Rust toolchain still runs most of the suite and
starts the server.

What genuinely requires it:

- `tools/storage/bm25.py` — calls `get_rust_module()` at module scope, so the
  import itself fails. `tools/__init__.py` wraps tool imports in
  `try/except ImportError`, so the tool is skipped rather than crashing.
- `security/capabilities.py` — the Python fallback is explicitly dead code.
- `security/scanner.py`, `injection_scanner.py`, `rate_limiter.py`,
  `file_policy.py`, `agents/loop_guard.py`, `tools/storage/sqlite.py`.
- `tools/git_tool.py`, `shell_exec.py`, `http_request.py`, `file_read.py`,
  `file_write.py`, `calculator.py`, `think.py` — these try Rust and fall back
  to a Python or CLI path, so they work either way.

Build it when you need those paths:

```powershell
uv sync --group desktop-native
uv run maturin develop --manifest-path rust/crates/openjarvis-python/Cargo.toml
python -c "from openjarvis._rust_bridge import RUST_AVAILABLE; print(RUST_AVAILABLE)"
```

### Windows toolchain caveat

`rustup`'s default host is `x86_64-pc-windows-msvc`, which needs the MSVC
linker and the Windows SDK — neither ships with Windows. Check before you
start:

```powershell
Get-Command link.exe -ErrorAction SilentlyContinue
Test-Path "${env:ProgramFiles(x86)}\Windows Kits\10\Lib"
```

If both come back empty, install **Visual Studio Build Tools** with the
"Desktop development with C++" workload. The GNU host is not a shortcut
here: CPython on Windows is MSVC-built, and mixing a GNU-built PyO3
extension against it is a known-fragile path.

## Fork-specific defaults

Two deliberate divergences from upstream, both about the fork's
session-awareness work reading window titles, terminal commands and
repository state.

**External analytics ships off.** `AnalyticsConfig.enabled` defaults to
`False` here; upstream defaults to `True`. The gate is
`openjarvis.analytics.identity.is_analytics_enabled`, in precedence order:

| Input | Effect |
|---|---|
| `DO_NOT_TRACK=1` | Off. Cannot be overridden. |
| `OPENJARVIS_ANALYTICS=1` / `=0` | Explicit per-run override. Unrecognised values are ignored. |
| `[analytics] enabled` in `config.toml` | Off unless set. |

Local telemetry (`~/.openjarvis/telemetry.db`) is a separate concern and is
unaffected — it never leaves the machine.

**Credential files are not actually protected on Windows.**
`core/credentials.py` writes `~/.openjarvis/credentials.toml` and calls
`os.chmod(path, 0o600)`; `connectors/oauth.py` does the same for OAuth token
files, which hold refresh tokens *and* client secrets in cleartext. On
Windows `os.chmod` cannot express POSIX bits — the file stays `0o666`, which
is what `tests/security/test_file_permissions.py` is telling you. Every
account on the machine can read your API keys. Until the backend uses the
Credential Manager (the Tauri side already does, via the `keyring` crate),
lock the files by hand:

```powershell
icacls "$env:USERPROFILE\.openjarvis\credentials.toml" /inheritance:r /grant:r "${env:USERNAME}:(R,W)"
icacls "$env:USERPROFILE\.openjarvis\connectors" /inheritance:r /grant:r "${env:USERNAME}:(OI)(CI)(R,W)"
```

**Microphone headers are relaxed for same-origin.** `server/middleware.py`
sends `Permissions-Policy: microphone=(self)` and adds
`media-src 'self' blob:; worker-src 'self' blob:` to the CSP; the Tauri CSP
mirrors the two directives. Upstream sends `microphone=()`, which makes
`getUserMedia` fail in any build served by FastAPI — the mic button only
appeared to work because the Vite dev server does not send that header.

> **Loading an AudioWorklet:** `worker-src` covers Worker / SharedWorker /
> ServiceWorker, **not** worklets. Worklet module fetches are checked
> against `script-src`, which has no `blob:` in either CSP. Ship the
> processor as a static asset from the app origin (`public/`, or Vite's
> `?worker&url`) so `'self'` covers it, rather than widening `script-src`.

## Staying mergeable with upstream

```powershell
git remote add upstream https://github.com/open-jarvis/OpenJarvis.git
git fetch upstream
git log --oneline HEAD..upstream/main    # what we are behind by
```

Fork changes should be additive: new modules, new registry entries, optional
protocol methods. Avoid changing an existing ABC signature or an existing
route contract — the registry pattern
(`@ToolRegistry.register`, `@AgentRegistry.register`, …) exists precisely so
new capability does not need edits to shared files.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `ImportError: polars is required` during collection | Missing `--extra framework-comparison`. |
| `ModuleNotFoundError: openjarvis_rust` | Expected without the extension. Only matters for the paths listed above. |
| `requires-python` resolution failure | An unsupported interpreter is active. `uv python pin 3.13`. |
| Mic denied in the browser but fine under `npm run dev` | The `Permissions-Policy` header — see above. |
| `jarvis serve` exits immediately on a non-loopback `--host` | `check_bind_safety()` refuses that without `OPENJARVIS_API_KEY`. Intended. |
