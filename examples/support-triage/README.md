# Support-triage consumer (Phase 16 portability test)

A small, independent support-ticket triage application proving that a consumer
can build on `llm-orchestrator` from a **wheel installed in a fresh virtual
environment** — outside the repository, without the library's source tree.
It prefers Groq, falls back to Gemini, retries transient failures with
deterministic backoff, validates structured output against a Pydantic schema,
and exposes a tiny CLI.

- App: `triage/` — business rules only (prompt, schema, exit codes); all
  orchestration is the library's real components.
- Tests: `tests/` — offline by default, no credentials required; the live
  test is opt-in and skipped unless explicitly enabled.
- Findings: `friction-log.md`.

## Install into a fresh environment (run outside the repository)

Windows:

```powershell
python -m venv venv-triage
venv-triage\Scripts\python -m pip install <repo>\dist\llm_orchestrator-0.1.0-py3-none-any.whl
# ...or with every optional extra (live mode + Orchestrator path):
venv-triage\Scripts\python -m pip install "llm-orchestrator[all] @ file:///<repo>/dist/llm_orchestrator-0.1.0-py3-none-any.whl"
```

POSIX uses the same commands with `venv-triage/bin/python` (the library's CI
installs the built wheel with the same syntax on Linux/macOS).

A core-only install runs every offline workflow test; three tests skip there
(the `[all]`-only credential check, the `Orchestrator` path, and the opt-in
live test). In `[all]`, three different ones skip (the two missing-extra
assertions, whose extras are present, and the live test).

## Run the offline tests (no credentials required)

Install the test runner once (it is not a library dependency), then run from
any directory outside the repository:

```powershell
venv-triage\Scripts\python -m pip install pytest
venv-triage\Scripts\python -m pytest <repo>\examples\support-triage\tests -v
```

## Classify a ticket (requires provider credentials)

Run with this consumer directory as the current working directory:

```powershell
$env:GROQ_API_KEY = "<your key>"   # set in the environment; never commit it
venv-triage\Scripts\python -m triage.app "I was charged twice this month"
```

Gemini fallback also needs `GEMINI_API_KEY`. Without credentials the command
exits with code 2 and a value-free message. Optional model overrides:
`TRIAGE_GROQ_MODEL`, `TRIAGE_GEMINI_MODEL`.

Exit codes: `0` classification JSON on stdout · `1` usage error · `2`
configuration error (missing/unsupported provider or extra) · `3` invalid
structured output · `4` upstream failure after retries and fallback.

## Live Groq round-trip (opt-in, never runs by default)

```powershell
$env:RUN_GROQ_INTEGRATION = "1"
$env:GROQ_API_KEY = "<your key>"
$env:GROQ_TEST_MODEL = "<your model id>"
venv-triage\Scripts\python -m pytest <repo>\examples\support-triage\tests\test_live.py -v
```
