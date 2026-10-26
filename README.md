# llm-orchestrator

### A reusable orchestration layer for LLM-powered Python applications

**One consistent interface for providers, retries, fallback, routing,
and validated output.**

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License:
MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Typing](https://img.shields.io/badge/typing-PEP%20561-blue)](https://peps.python.org/pep-0561/)
:::

------------------------------------------------------------------------

## What is `llm-orchestrator`?

`llm-orchestrator` is a Python library that helps applications call
Large Language Models (LLMs) through a consistent, reusable interface.

When an application uses AI, it often needs to handle more than just
sending a prompt. It may need to choose a model, retry a temporary
failure, switch to another provider, classify errors, validate a JSON
response, or apply configuration consistently.

This library puts those reusable engineering pieces in one place---so
each application can focus on its own product logic.

**It is infrastructure, not a complete AI application.** Your
application still owns its prompts, business rules, data, and output
schemas.

> **Project status:** Early-stage (`0.1.0`), suitable for evaluation and
> careful internal reuse. It is not yet a stable, broadly
> production-proven release. The package is not currently published to
> PyPI; install it from a source checkout or a built wheel.

## At a glance

-   **Provider-neutral contracts** --- use consistent request and
    response objects.
-   **Provider adapters** --- Gemini and Groq are implemented behind the
    same provider interface.
-   **Retry with backoff and jitter** --- retry eligible transient
    failures without changing the candidate.
-   **Candidate fallback** --- move to another configured model/provider
    when policy allows.
-   **Routing and model selection** --- choose an initial candidate
    deterministically from an explicit candidate list.
-   **Normalized errors** --- handle failures using common categories
    instead of vendor-specific exceptions.
-   **Structured-output validation** --- validate response text against
    your own Pydantic model.
-   **Optional LangGraph workflow** --- run a single candidate through a
    small, deterministic graph.
-   **Typed configuration** --- configure providers, models, policies,
    and a default timeout.
-   **Developer safeguards** --- automated tests, type/lint checks,
    packaging checks, and a dependency-audit CI job.

**Not included yet:** built-in observability dashboards/tracing,
automatic structured-output repair, native provider-enforced schema
modes, or a multi-agent framework. See the [roadmap](docs/roadmap.md).

## How it works

At a high level, the pieces have separate responsibilities:

``` text
Your application
  │  Request, prompts, business rules, output schema
  ▼
Contracts and configuration
  │
  ▼
Routing ─────────────── selects the first candidate
  │
  ▼
Retry engine ────────── retries the same candidate when eligible
  │
  ▼
Fallback executor ───── can advance to another candidate
  │
  ▼
Provider adapter ────── talks to Gemini, Groq, or your own provider
  │
  ▼
Normalized response / error
  │
  ▼
Optional Pydantic structured-output validation
```

The layers are intentionally separate:

  -----------------------------------------------------------------------
  Component               What it does            What it does **not** do
  ----------------------- ----------------------- -----------------------
  **Routing**             Selects the initial     Does not call a
                          eligible candidate      provider or perform
                                                  retries

  **Retry**               Re-attempts the same    Does not switch
                          candidate for retryable providers/models
                          failures                

  **Fallback**            Tries the next eligible Does not hide
                          candidate after a       programming errors or
                          qualifying terminal     cancellation
                          failure                 

  **Provider**            Translates the library  Does not own retry or
                          request to a provider   routing policy
                          SDK call                

  **Structured            Parses response text    Does not call the
  validation**            and validates it        model, repair output,
                          against your Pydantic   or automatically retry
                          model                   validation failures
  -----------------------------------------------------------------------

This separation makes the behavior easier to test and reason about.

## Requirements

-   Python **3.11 or newer**
-   A Gemini or Groq API key only if you want to make live requests
    through that provider
-   Network access only for installation and live provider calls

The core library can be installed without provider SDKs or LangGraph.
Those features are optional extras.

## Installation

The package is not published to PyPI yet. Use a checkout of this
repository.

### 1. Get the source

``` bash
git clone https://github.com/kit29-25bad132/llm-orchestrator.git
cd llm-orchestrator
```

### 2. Create and activate a virtual environment

**Windows PowerShell**

``` powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If you have another supported Python version installed, replace `-3.11`
with that version.

**macOS / Linux**

``` bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 3. Install what you need

From the repository root:

``` bash
python -m pip install .
```

Choose an optional extra when you need it:

``` bash
python -m pip install ".[gemini]"      # Gemini adapter
python -m pip install ".[groq]"        # Groq adapter
python -m pip install ".[langgraph]"   # LangGraph orchestration
python -m pip install ".[all]"         # All optional runtime features
```

For development and the full test suite:

``` bash
python -m pip install -e ".[dev,all]"
```

### Optional features

  -----------------------------------------------------------------------
  Extra                               Adds
  ----------------------------------- -----------------------------------
  *(none)*                            Core contracts, errors, provider
                                      abstraction, retry/fallback,
                                      routing, structured validation, and
                                      configuration

  `gemini`                            Gemini provider adapter
                                      (`google-genai`)

  `groq`                              Groq provider adapter (`groq`)

  `langgraph`                         LangGraph-based orchestration

  `all`                               All optional runtime features above

  `dev`                               Development tools such as pytest,
                                      Ruff, mypy, and build
  -----------------------------------------------------------------------

You do not need to install every extra to use the core library. If you
try to use a feature whose extra is missing, the library raises a clear
configuration error with installation guidance.

## Quickstart: understand the core without an API key

You can create requests, configure candidate models, and select a
candidate without making a network request.

``` python
from llm_orchestrator.contracts import (
    LLMRequest,
    Message,
    MessageRole,
    ModelConfig,
    RetryPolicy,
)
from llm_orchestrator.reliability import RetryEngine
from llm_orchestrator.routing import ModelRouter, StaticRoutingPolicy

# Describe what you want to send. The request is provider-neutral.
request = LLMRequest(
    messages=[
        Message(role=MessageRole.USER, content="Explain APIs in simple terms.")
    ]
)

# Supply the candidates your application is willing to use.
candidates = [
    ModelConfig(provider="gemini", model="<your-gemini-model-id>"),
    ModelConfig(provider="groq", model="<your-groq-model-id>"),
]

# Select the first candidate deterministically.
router = ModelRouter(StaticRoutingPolicy(preferred_provider="gemini"))
selected = router.select(request, candidates)

# Set a bounded retry policy for when you execute a provider request.
retry_engine = RetryEngine(
    RetryPolicy(max_attempts=3, initial_backoff_s=0.5)
)

print(selected)
```

This example does **not** call an AI model. It demonstrates the
provider-neutral request, candidate selection, and retry-policy setup.
Replace the placeholder model IDs with model IDs supported by your
chosen provider before making a live call.

## Make a live request

This example uses Gemini. Install the Gemini extra first:

``` bash
python -m pip install ".[gemini]"
```

Set your API key in the environment. Do not put a real key directly in
source code or commit it to Git.

**Windows PowerShell**

``` powershell
$env:GEMINI_API_KEY = "your-api-key"
```

**macOS / Linux**

``` bash
export GEMINI_API_KEY="your-api-key"
```

Then run this Python example from an environment where
`llm-orchestrator` is installed:

``` python
import asyncio
import os

from llm_orchestrator.contracts import LLMRequest, Message, MessageRole, ModelConfig
from llm_orchestrator.providers.gemini import GeminiProvider

async def main():
    provider = GeminiProvider(api_key=os.environ["GEMINI_API_KEY"])
    request = LLMRequest(
        messages=[Message(role=MessageRole.USER, content="Give me three tips for clear writing.")]
    )
    model = ModelConfig(provider="gemini", model="<your-gemini-model-id>")

    try:
        response = await provider.generate(request, model)
        print(response)
    finally:
        await provider.aclose()

if __name__ == "__main__":
    asyncio.run(main())
```

Replace `<your-gemini-model-id>` with a currently supported model ID for
your account. Provider/model availability, quotas, and pricing are
controlled by the provider and may change. This direct adapter example
makes one provider attempt; retry and fallback are separate components
you can compose as needed.

To use Groq instead, install `.[groq]`, set `GROQ_API_KEY`, and use
`GroqProvider` from `llm_orchestrator.providers.groq`.

## Configure providers from settings

For applications that want a central, typed configuration, the
configuration layer can build a registry from explicit settings and
environment variables.

``` python
import os

from llm_orchestrator.config import OrchestratorSettings, build_registry, load_settings
from llm_orchestrator.contracts import ModelConfig, ProviderConfig

settings = load_settings(
    explicit=OrchestratorSettings(
        providers=(
            ProviderConfig(
                name="gemini",
                credential_env_var="GEMINI_API_KEY",
            ),
        ),
        models=(
            ModelConfig(provider="gemini", model="<your-gemini-model-id>"),
        ),
    ),
    env=os.environ,
)

registry = build_registry(settings, env=os.environ)
```

This example requires the `gemini` extra and `GEMINI_API_KEY` to be set
before `build_registry` resolves the credential. The library does not
automatically load a `.env` file. Your application can choose its own
environment-loading approach.

For all supported settings and timeout behavior, read
[Configuration](docs/configuration.md).

## Add retries and fallback

Retries and fallback solve different problems:

-   **Retry:** try the same provider/model again after an eligible
    transient error.
-   **Fallback:** after the current candidate reaches a terminal
    failure, move to another candidate if the error and policy allow it.

A simplified composition looks like this:

``` python
from llm_orchestrator.reliability import FallbackExecutor, RetryEngine

retry_engine = RetryEngine(retry_policy)
executor = FallbackExecutor(
    registry=registry,
    retry_engine=retry_engine,
    fallback_policy=fallback_policy,
)

result = await executor.run(
    request,
    candidates,
    initial=selected,
)
```

Here, `retry_policy`, `fallback_policy`, `registry`, `request`,
`candidates`, and `selected` are objects your application must create
using the library's contracts and configuration. See
[Reliability](docs/reliability.md) and [Fallback](docs/fallback.md) for
the complete behavior and policy details.

## Validate structured output

If your application expects JSON-like data, define the output shape with
Pydantic and validate the model response against it.

``` python
from pydantic import BaseModel
from llm_orchestrator.structured import validate_structured_output

class SupportAnswer(BaseModel):
    answer: str
    needs_human_follow_up: bool

# `response` is an llm_orchestrator.contracts.LLMResponse
validated = validate_structured_output(response, SupportAnswer)
print(validated.answer)
```

The validator accepts JSON text or one complete fenced JSON block.
Invalid JSON, truncated output, or a payload that does not match your
schema raises a normalized `StructuredOutputValidationError`. Validation
happens after the response is generated; it does not automatically call
the model again or switch providers.

See [Structured output](docs/structured-output.md).

## Use the LangGraph orchestration layer

LangGraph support is optional. Install it with:

``` bash
python -m pip install ".[langgraph]"
```

The `Orchestrator` runs a single candidate through a deterministic
workflow and uses the existing retry engine.

``` python
from llm_orchestrator.orchestration import Orchestrator

orchestrator = Orchestrator(registry, retry_engine)
result = await orchestrator.run(request, model)
```

The registry, retry engine, request, and model must be configured by
your application. The graph does not itself perform multi-candidate
routing or fallback; use the separate routing and fallback components
for those responsibilities.

See [Orchestration](docs/orchestration.md).

## Error handling and timeouts

Provider failures are translated into a shared error hierarchy so
application code can make consistent decisions across providers.
Categories include authentication/configuration failures, invalid
requests, rate limits, timeouts, network errors, and provider
availability errors.

``` python
from llm_orchestrator.errors import ErrorCategory, LLMError

try:
    ...
except LLMError as exc:
    print(exc.category)
    print(exc.retryable)
```

Use the error category and documented retry/fallback semantics rather
than relying on provider-specific exception classes. Error messages are
sanitized to reduce the risk of leaking credentials.

Per-call `timeout_s` values must be positive and finite numbers, or
`None`. Invalid values---including booleans, zero, negative values, NaN,
and infinity---are rejected before an attempt or SDK call. A configured
default timeout can be supplied through `LLM_ORCH_DEFAULT_TIMEOUT_S`;
see [Configuration](docs/configuration.md).

## Examples and documentation

The repository includes a separate portability example:

-   [`examples/support-triage/`](examples/support-triage/) --- a small
    support-triage consumer showing how an application can use the
    library, with offline tests and an optional live-provider test.

Read the focused documentation when you need more detail:

  ----------------------------------------------------------------------------
  Guide                                    What it covers
  ---------------------------------------- -----------------------------------
  [Architecture](docs/architecture.md)     Design principles and dependency
                                           boundaries

  [Providers](docs/providers.md)           Provider interface, Gemini, and
                                           Groq adapters

  [Configuration](docs/configuration.md)   Settings, environment variables,
                                           registry construction, and timeouts

  [Reliability](docs/reliability.md)       Retry behavior, backoff, and jitter

  [Fallback](docs/fallback.md)             Candidate order and fallback rules

  [Routing](docs/routing.md)               Eligibility, deterministic
                                           selection, and cost estimates

  [Errors](docs/errors.md)                 Error categories, retry/fallback
                                           semantics, and sanitization

  [Structured                              Pydantic validation behavior and
  output](docs/structured-output.md)       limitations

  [Orchestration](docs/orchestration.md)   LangGraph workflow boundaries

  [Testing](docs/testing.md)               Test groups and quality checks

  [Roadmap](docs/roadmap.md)               Implemented, deferred, and planned
                                           work

  [Changelog](CHANGELOG.md)                Version and change history
  ----------------------------------------------------------------------------

## Development setup

Want to help improve the library? Set up a development environment.

### 1. Install development dependencies

From the repository root:

``` bash
python -m venv .venv
```

Activate the environment, then run:

``` bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev,all]"
```

### 2. Run the test suite

``` bash
python -m pytest
```

The default suite is designed to run offline. Live provider tests are
opt-in, so normal tests do not consume API quota.

### 3. Run quality checks

``` bash
ruff check .
ruff format --check .
mypy
```

### 4. Check the built package

``` bash
python -m build
python scripts/validate_dist.py dist
```

For the complete test inventory and clean-environment package smoke
test, see [Testing](docs/testing.md).

### Optional live-provider tests

Only run these if you intentionally want to make real API calls and have
valid credentials and model IDs configured.

``` bash
RUN_GEMINI_INTEGRATION=1 GEMINI_API_KEY=... GEMINI_TEST_MODEL=... python -m pytest -m live
RUN_GROQ_INTEGRATION=1 GROQ_API_KEY=... GROQ_TEST_MODEL=... python -m pytest -m live
```

These tests may use provider quota and are affected by provider
availability and account limits. Keep credentials out of source control
and logs.

## Security and responsible use

-   Never hard-code API keys in application code.
-   Never commit real credentials or `.env` files.
-   Keep provider keys in environment variables or your chosen secret
    manager.
-   Review provider pricing, rate limits, data handling, and terms
    before using a model with sensitive data.
-   Treat model output as untrusted input and validate it before using
    it in downstream systems.
-   The library normalizes errors and sanitizes messages, but your
    application remains responsible for its own logging, data retention,
    and security policies.

## Current limitations

To set clear expectations, the current project does **not** claim to
provide:

-   A published PyPI package or stable `1.x` API guarantee.
-   Live-provider validation for every model/account combination.
-   Built-in dashboards, tracing, or metrics exporters.
-   Automatic repair or retry when structured-output validation fails
    after generation.
-   Native provider-enforced structured-output modes.
-   A multi-agent system or local model runtime.

The project includes automated tests and CI checks, but provider
behavior still depends on external services. The CI dependency-audit job
should be checked in the repository's GitHub Actions results before
treating its latest run as verified.

## Contributing

Contributions and issue reports are welcome. Please:

1.  Read the [architecture guide](docs/architecture.md) and relevant
    feature guide.
2.  Keep changes focused and add or update tests.
3.  Run the test suite and quality checks before opening a pull request.
4.  Do not include API keys, credentials, or private customer data in
    commits or issues.

Please open an issue first for larger design changes so the intended
public API and scope can be discussed before implementation.

## License

This project is licensed under the [MIT License](LICENSE).

------------------------------------------------------------------------


**Built to make LLM integrations easier to reuse, test, and maintain.**

