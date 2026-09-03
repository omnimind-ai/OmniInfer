# Benchmark Results

OmniInfer can measure the currently loaded text model and archive a benchmark
submission as JSON. The default destination is:

```text
.local/benchmarks/results/<benchmark-id>.json
```

This directory is local state and is ignored by Git. A result is never uploaded
automatically.

## Run a benchmark

The standard profile currently supports llama.cpp-family external runtimes. It
requires prompt caching to be explicitly disabled and a slot path so OmniInfer
can erase every runtime slot before each measured request:

```sh
mkdir -p .local/benchmarks/slots
./omniinfer load -m /models/model.gguf --ctx-size 4096 -- \
  -b 512 \
  --cache-ram 0 \
  --no-cache-idle-slots \
  --no-cache-prompt \
  --slot-prompt-similarity 0 \
  --slot-save-path .local/benchmarks/slots

./omniinfer bench run \
  --catalog-model-id <catalog-model-id> \
  --format GGUF \
  --quantization Q4_K_M \
  --model-url https://huggingface.co/owner/model/resolve/<40-character-commit>/model.gguf \
  --baseline \
  --submitter-name <name>
```

For known NVIDIA, Apple Silicon, AMD, and x86-64 devices, OmniInfer fills the
device name and catalog ID when detection is unambiguous. Managed prebuilt
runtimes also provide their pinned version and install command. For a custom
device or source build, pass `--device-name`, `--soc`, `--backend-version`, and
`--build-command` explicitly.

OmniInfer records backend identity separately from compute placement. A known
single-accelerator backend is inferred automatically. For a heterogeneous run,
declare both phases explicitly, for example
`--prefill-accelerator htp --decode-accelerator cpu`. This produces
`execution.compute_mode=mixed`; identical phase accelerators produce `single`.
Use `--privilege-level elevated` only when the archived run command retains its
`su` or `sudo` wrapper.

Use `--optimization <slug>` instead of `--baseline` when an optional method was
active. Repeat the option for multiple methods:

```sh
./omniinfer bench run <metadata-options> \
  --optimization dflash \
  --optimization turboquant-turbo4
```

OmniInfer rejects a baseline declaration when the loaded backend ID or launch
arguments contain a known DFlash or TurboQuant marker. This guard cannot prove
that every third-party optimization is active or inactive; the submitter must
check the runtime logs and declare every method that actually affected the run.

For a fixed-length benchmark, add `--ignore-eos` with the requested token
budget:

```sh
./omniinfer bench run <metadata-options> --max-tokens 256 --ignore-eos
```

This requests `ignore_eos: true` and establishes a fixed-length benchmark
contract: every measured response must report `completion_tokens` equal to
`--max-tokens`. The CLI validates each response and aborts instead of archiving
the result when a response is short or otherwise mismatches; `--ignore-eos`
does not make the exact length unconditional. The mode is recorded in the
existing `protocol.notes` field, without a schema change.

The command performs one unrecorded warmup by default, then three measured
non-streaming requests at concurrency 1. Before every measured request it asks
the gateway to enumerate and erase every runtime slot, requires a successful
acknowledgement, and also sends `cache_prompt: false`. Unsupported cache-reset
paths fail closed. Prompt and completion token counts must be consistent, and
both Prefill and Decode throughput must have a coefficient of variation no
greater than 5%; otherwise no result is written. The effective launch command is
captured from OmniInfer state and credential values are redacted. Use
`--run-command` only when the runtime cannot expose its actual launch command.
Archived commands are text evidence and are never executed by the submission
service.

Use `--json` when another program needs the complete result on stdout. Progress
and the saved path are written to stderr, so stdout remains one valid JSON value.

## Inspect local results

```sh
./omniinfer bench list
./omniinfer bench list --json
```

`--state-root` changes the root that owns `.local/benchmarks/results`, consistent
with the rest of the CLI. `--output <path>` writes one result elsewhere and
refuses to overwrite an existing file.

## Synchronize the OmniStudio contract

OmniInfer vendors the public, measurement-free OmniStudio producer contract in
`benchmarks/contract/`. The snapshot contains the submission JSON Schema, the
catalog entities and complete legal model/platform/backend/format/quantization
combinations, the upstream manifest, and deterministic provenance. It does not
contain benchmark values, submitters, review state, or audit data.

Refresh it only as an explicit, reviewable change:

```sh
python3 scripts/sync_benchmark_contract.py
python3 scripts/sync_benchmark_contract.py --check
```

Synchronization accepts HTTPS only, bounds every download, rejects duplicate
JSON keys and non-finite numbers, verifies the declared byte counts and
SHA-256 digests, and replaces changed files atomically with the manifest last.
When the upstream contract is unchanged, the command does not rewrite any
file. `--check` performs no network access and is the mode used by pull-request
CI.

Before measurements begin, `bench run` verifies the embedded snapshot and
requires the loaded backend, local platform, model format and quantization, and
device catalog identity to form one legal catalog combination. After building
the result, it validates the complete JSON against the vendored Draft-07 Schema
before selecting the output path or writing a file. Unknown IDs, incompatible
contract or Schema versions, and altered snapshots fail closed.

## Submission compatibility

OmniInfer emits benchmark Schema `1.4.0`, including its generator version,
prompt identity, backend-independent compute placement, scored token counts,
per-run durations, standard methodology profile, and cache policy. Local
contract validation is an early producer check, not an approval decision. The
website's current server-side Schema, semantic and history checks, followed by
human review, remain authoritative; generating a file does not guarantee that
it can be uploaded or published. The website accepts any JSON that satisfies
its current contract and does not require the file to have been generated by
OmniInfer.

Before upload, review the public model URL, device and backend catalog IDs,
runtime commands, optimization declaration, and submitter metadata. Commands
must not contain credential values; use `<redacted>` or an environment-variable
reference instead.
