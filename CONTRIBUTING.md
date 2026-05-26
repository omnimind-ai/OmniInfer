# Contributing to OmniInfer

Thank you for taking the time to improve OmniInfer.

## Before You Start

- Open an issue or discussion for larger changes so maintainers can confirm scope and direction.
- Work from a feature branch instead of committing directly to `main`.
- Keep changes focused. Avoid bundling unrelated refactors with bug fixes or features.
- Update tests and documentation when behavior, APIs, CLI output, build scripts, or platform support changes.

## Development Setup

Use the build guide for platform-specific runtime setup:

- [Build Guide](docs/build.md)
- [CLI Guide](docs/CLI.md)
- [API Reference](docs/API.md)
- [Android Integration](docs/android/integration.md)

From a source checkout, build or install at least one backend runtime before testing model load or chat flows.

## Validation

Run the smallest relevant test set for your change, then broaden coverage when the change affects shared behavior.

Common checks:

```bash
python3 -m unittest tests.test_runtime tests.test_http_handler tests.test_anthropic_adapter
python3 -m py_compile service_core/runtime.py service_core/service.py
bash -n scripts/install.sh
git diff --check
```

For platform-specific changes, also run the matching build-script syntax checks and focused tests documented in [docs/build.md](docs/build.md).

## Pull Requests

- Use a clear title that describes the behavior change.
- Include the user-visible impact, relevant platforms, and validation commands.
- Link related issues when available.
- Do not include generated dependency caches, local `.local/` state, benchmark scratch files, or machine-specific paths.

## Reporting Issues

Please include:

- OS, CPU/GPU, driver/runtime versions, and backend id.
- The exact command or API request.
- Relevant logs from `.local/logs/`, backend runtime logs, or installer summary files.
- Whether the issue reproduces with a fresh service restart and model reload.
