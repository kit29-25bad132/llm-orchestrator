# Phase 16 friction log — examples/support-triage

Consumer: support-ticket triage CLI, built against
`llm_orchestrator-0.1.0-py3-none-any.whl` installed in two fresh virtual
environments (core-only and `[all]`), executed from working directories
outside the repository. Every entry below separates **observed** facts
(reproduced during this phase's runs) from **anticipated** risks (code-reading
only, not reproduced).

## Severity scale

- **S1 — Blocking:** cannot proceed without a change outside
  `examples/support-triage/` (owner decision required).
- **S2 — Major:** works, but with meaningful rework, duplication, or guesswork.
- **S3 — Minor:** extra friction; a documented workaround exists.
- **S4 — Cosmetic/nit:** clarity, naming, or documentation polish.

## Observed

### F-01 — Consumer package not importable without a root `conftest.py` (S3)

- **Symptom:** the suite fails at collection with
  `ModuleNotFoundError: No module named 'triage'`.
- **Repro:** from a directory outside the repository, with the wheel installed
  in a fresh venv:
  `<venv>\python -m pytest <repo>\examples\support-triage\tests`
  (run before `conftest.py` existed; exact failure captured in this phase's
  session log).
- **Root cause:** pytest's prepend import mode inserts only the test module's
  own directory (`tests/`) into `sys.path`; the repository's pytest config
  (`pythonpath = ["tests"]`, rooted at the library's `pyproject.toml`) points
  at the *library's* tests, not at an example's package root. The consumer
  package directory (`support-triage/`) never lands on `sys.path`.
- **Classification:** consumer setup / test-harness layout (not a library
  defect — the library's own harness doesn't claim to run examples).
- **Evidence:** collection traceback above; fix =
  `examples/support-triage/conftest.py` (docstring records the reason).
- **Proposed remedy:** none required for this phase. Optional (owner decision):
  mention in the library's testing docs that an example/consumer suite needs
  its own root `conftest.py` when run against the repo rootdir.
- **API change required:** no. **Owner decision needed:** optional docs note
  (would be an edit outside this stage's write boundary).
- **Acceptance test:** suite green in both venvs from external cwd (12
  passed / 3 skipped each).

### F-02 — Gating `[all]`-only behaviour: `find_spec` on dotted names raises (S4)

- **Symptom:** `importlib.util.find_spec("google.genai")` raises
  `ModuleNotFoundError: No module named 'google'` in the core-only venv
  instead of returning `None`, breaking collection of the failure-path tests.
- **Repro:** core-only venv, run the suite once before the fix (traceback in
  session log at `test_failures.py:33`).
- **Root cause:** Python's `find_spec` imports the *parent* package for dotted
  names; standard-library semantics, not library behaviour.
- **Classification:** consumer misuse/setup (Python semantics).
- **Evidence:** traceback above; fix = `_has_module()` helper with
  `except ModuleNotFoundError`.
- **Proposed remedy:** none (consumer-side). **API change required:** no.
  **Owner decision needed:** no.
- **Acceptance test:** env-aware skips behave correctly in both venvs.

### F-03 — `ErrorCategory` is exported from `errors`, not `contracts` (S4)

- **Symptom:** asserting `result.error.category` against a category imported
  from `llm_orchestrator.contracts` fails with
  `ImportError: cannot import name 'ErrorCategory' from 'llm_orchestrator.contracts'`
  (first attempt at the validation-error assertion).
- **Repro:** `from llm_orchestrator.contracts import ErrorCategory`.
- **Root cause:** `ErrorInfo` (in `contracts`) is typed with `ErrorCategory`,
  which lives in and is exported from `llm_orchestrator.errors`
  (`errors/__init__` documents that import). The type's public home is
  discoverable, but not from the module that owns `ErrorInfo`.
- **Classification:** minor discoverability / documentation gap (works as
  documented once the right module is found).
- **Evidence:** ImportError above; correct usage:
  `from llm_orchestrator.errors import ErrorCategory` (now asserted in
  `tests/test_workflow.py`).
- **Proposed remedy (optional):** re-export `ErrorCategory` from
  `llm_orchestrator.contracts` alongside `ErrorInfo`. Alternative: note the
  pairing in `docs/errors.md`.
- **API change required:** additive re-export only (optional).
  **Owner decision needed:** yes, if a change is desired; recommended default
  is leave-as-is (current behaviour is documented).
- **Acceptance test:** `test_invalid_structured_output_raises_documented_validation_error`.

## Anticipated / not yet observed

- **A-01 — `FallbackPolicy.include_validation_failures` not reachable from the
  documented post-hoc validation flow:** the consumer validates after
  `FallbackExecutor.run` returns, so a validation failure raises outside the
  executor and never sees `fallback_on`/`include_validation_failures`. Code
  reading only (`fallback.py`, `structured.py`); not reproduced this phase,
  because the consumer followed the documented
  `validate_structured_output(response, Schema)` pattern.
  Classification (provisional): feature request / documentation gap.
  Owner decision if ever pursued: how validation failures should surface to
  the executor.
- **A-02 — Provider SDK presence changes which offline tests run:** the
  consumer suite is intentionally env-aware (missing-extra assertions run only
  where the extra is absent; credential construction only where both SDKs are
  present). If a future consumer installs *some* SDKs, both classes of test
  could skip together and mask coverage. Mitigation already in place: the two
  venvs produced complementary skips, covering every check across the pair.

## Issues observed: 3 (0 blocking, 0 major, 1 minor, 2 cosmetic)

No S1/S2 issues: nothing required a change outside
`examples/support-triage/`, no library defect was reproduced, and every
acceptance test passed in both fresh environments. The optional remedies in
F-01/F-03 are owner decisions, not blockers.
