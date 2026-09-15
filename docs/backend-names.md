# Backend names and compatibility

Use a hyphenated selector such as `llama.cpp-cpu`, `llama.cpp-cuda`, or
`llama.cpp-metal`. A selector identifies a runtime variant on the current host;
it does not guarantee that every operation runs on that accelerator.

OmniInfer keeps the existing runtime IDs for configuration, model catalogs,
installation directories, profiles, API identity fields, and benchmark history.
Do not rename these directories or rewrite old records to adopt a selector.
The registry is the source of truth for selector resolution. An exact legacy ID
wins over an alias; unsupported architectures and ambiguous aliases fail instead
of falling back to a different accelerator or engine.

## Desktop and standalone runtime mapping

| Host | Runtime ID (still accepted) | Recommended selector |
|---|---|---|
| Linux | llama.cpp-linux | llama.cpp-cpu |
| Linux | llama.cpp-linux-cuda | llama.cpp-cuda |
| Linux | llama.cpp-linux-rocm | llama.cpp-rocm |
| Linux | llama.cpp-linux-vulkan | llama.cpp-vulkan |
| Linux | llama.cpp-linux-s390x | llama.cpp-cpu-s390x |
| Linux | llama.cpp-linux-openvino | llama.cpp-openvino |
| Linux | ik_llama.cpp-linux | ik_llama.cpp-cpu |
| Linux | ik_llama.cpp-linux-cuda | ik_llama.cpp-cuda |
| Linux | stable-diffusion.cpp-linux-vulkan | stable-diffusion.cpp-vulkan |
| Linux | omniinfer-native-linux | omniinfer-native-eagle3 |
| Linux | mnn-linux | mnn-cpu |
| Linux | vllm-linux-cuda | vllm-cuda |
| Linux | freetoken-linux-cuda | freetoken-cuda |
| Linux | vla.cpp-linux | vla.cpp-cpu |
| Linux | vla.cpp-linux-cuda | vla.cpp-cuda |
| Windows | llama.cpp-cpu | llama.cpp-cpu |
| Windows | llama.cpp-cuda | llama.cpp-cuda |
| Windows | llama.cpp-vulkan | llama.cpp-vulkan |
| Windows | llama.cpp-hip | llama.cpp-hip |
| Windows | llama.cpp-sycl | llama.cpp-sycl |
| Windows ARM64 | llama.cpp-windows-arm64 | llama.cpp-cpu-arm64 |
| Windows | ik_llama.cpp-cpu | ik_llama.cpp-cpu |
| Windows | ik_llama.cpp-cuda | ik_llama.cpp-cuda |
| Windows | stable-diffusion.cpp-vulkan | stable-diffusion.cpp-vulkan |
| Windows / WSL2 | vllm-wsl2-cuda | vllm-wsl2-cuda |
| Windows / WSL2 | vllm-wsl2-rocm | vllm-wsl2-rocm |
| macOS Apple Silicon | llama.cpp-mac | llama.cpp-metal |
| macOS Intel | llama.cpp-mac-intel | llama.cpp-cpu |
| macOS | turboquant-mac | turboquant-metal |
| macOS | mlx-mac | mlx-metal |
| Android standalone | llama.cpp-android | llama.cpp-android |
| iOS | llama.cpp-ios | llama.cpp-metal |
| iOS | mlx-ios | mlx-metal |

WSL2 remains explicit because it requires a Linux distribution and different
installation prerequisites. HIP and ROCm remain distinct platform runtime builds.
Architecture-specific CPU packages retain a suffix where needed to distinguish
legacy variants. The Android standalone registry entry does not imply the AAR's
CPU/HTP load defaults; see [Android integration](android/aar-integration.md).
TurboQuant and OmniInfer Native remain distinct implementations, not aliases for
the official llama.cpp binary. `family` describes adapter compatibility and must
not be used to relabel ik_llama.cpp as official llama.cpp.

Registry JSON adds `selector` and `execution_environment`; `id` and `recommended`
remain legacy runtime IDs. `label` uses the hyphenated selector. Clients should
use `selector` for display/input and `id` for stored identity, with an `id` fallback
when talking to an older service. Registry rows also expose
`architecture_compatible`, separately from runtime hardware detection.
