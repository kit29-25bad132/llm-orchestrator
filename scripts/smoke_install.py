"""Post-install smoke checks for an installed ``llm_orchestrator`` distribution.

Run with the interpreter of the environment under test, from a working
directory outside the repository, after installing a built wheel (or sdist)::

    python smoke_install.py                            # core-only install
    python smoke_install.py --expect-extras gemini      # single-extra install
    python smoke_install.py --expect-extras gemini groq langgraph   # [all]

Verifies that the installed package (not a source checkout) imports, that the
version/public API/``py.typed`` marker are present, that each optional feature
either imports and constructs with its extra installed or fails with the
actionable ``ConfigurationError`` naming its extra when absent, and that no
optional dependency leaks into a core-only environment.

Standard library only; never performs network calls or provider API calls.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

FEATURES: dict[str, dict] = {
    "gemini": {
        "module": "llm_orchestrator.providers.gemini",
        "root": "google",
    },
    "groq": {
        "module": "llm_orchestrator.providers.groq",
        "root": "groq",
    },
    "langgraph": {
        "module": "llm_orchestrator.orchestration",
        "root": "langgraph",
    },
}

# Synthetic credentials: constructors only, no model call is ever made.
_FAKE_GEMINI_KEY = "AIzaFakeKeyForOfflineSmokeTest012345"
_FAKE_GROQ_KEY = "gsk_FakeKeyForOfflineSmokeTest012345"


def _check_present(feature: str) -> None:
    """The extra is installed: the feature must import and construct."""
    if feature == "gemini":
        from llm_orchestrator.providers.gemini import GeminiProvider

        provider = GeminiProvider(api_key=_FAKE_GEMINI_KEY)
        assert provider.provider_id == "gemini"
    elif feature == "groq":
        from llm_orchestrator.providers.groq import GroqProvider

        provider = GroqProvider(api_key=_FAKE_GROQ_KEY)
        assert provider.provider_id == "groq"
    else:
        from llm_orchestrator.contracts import RetryPolicy
        from llm_orchestrator.orchestration import Orchestrator
        from llm_orchestrator.providers import ProviderRegistry
        from llm_orchestrator.reliability import RetryEngine

        registry = ProviderRegistry()
        engine = RetryEngine(RetryPolicy(max_attempts=1))
        orchestrator = Orchestrator(registry, engine)  # compiles the graph offline
        assert orchestrator is not None
    print(f"  {feature}: present -> import + construction OK")


def _check_missing(feature: str) -> None:
    """The extra is absent: the feature must fail with an actionable error."""
    spec = importlib.util.find_spec(FEATURES[feature]["root"])
    assert spec is None, f"{FEATURES[feature]['root']} is importable in a core install"
    module = FEATURES[feature]["module"]
    try:
        __import__(module)
    except Exception as exc:  # noqa: BLE001 - asserted precisely below
        from llm_orchestrator.errors import ConfigurationError

        assert isinstance(exc, ConfigurationError), (
            f"{module}: expected ConfigurationError, got {type(exc).__name__}: {exc}"
        )
        assert f"[{feature}]" in str(exc), f"{module}: message must name the extra: {exc}"
        assert "pip install" in str(exc), f"{module}: message must show an install command: {exc}"
    else:
        raise AssertionError(f"{module} imported although the {feature!r} extra is not installed")
    print(f"  {feature}: absent -> actionable ConfigurationError OK")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expect-extras",
        nargs="*",
        default=[],
        choices=sorted(FEATURES),
        metavar="EXTRA",
        help="optional features expected to be installed (default: none — core only)",
    )
    args = parser.parse_args(argv)
    expected = set(args.expect_extras)

    import llm_orchestrator

    origin = Path(llm_orchestrator.__file__).resolve().as_posix()
    assert "site-packages" in origin, (
        f"llm_orchestrator resolved outside site-packages: {origin} "
        "(source-checkout import — run from outside the repository)"
    )
    print(f"  resolved from: {origin}")

    from importlib.metadata import version

    assert llm_orchestrator.__version__ == version("llm-orchestrator")
    assert llm_orchestrator.__all__ == ["__version__"]

    marker = Path(llm_orchestrator.__file__).resolve().parent / "py.typed"
    assert marker.is_file(), "py.typed marker missing from the installed distribution"

    import llm_orchestrator.config
    import llm_orchestrator.contracts
    import llm_orchestrator.errors
    import llm_orchestrator.providers
    import llm_orchestrator.reliability
    import llm_orchestrator.routing
    import llm_orchestrator.structured

    assert llm_orchestrator.contracts.__all__, "contracts must expose a public API"
    assert llm_orchestrator.providers.__all__ == ["LLMProvider", "ProviderRegistry"]

    for feature in sorted(FEATURES):
        if feature in expected:
            _check_present(feature)
        else:
            _check_missing(feature)

    print(
        f"OK: smoke checks passed (version={llm_orchestrator.__version__}, "
        f"extras={sorted(expected) or 'core-only'})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
