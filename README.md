# OmniInfer-VLA（Jetson 部署与 Benchmark）

OmniInfer-VLA 在 Jetson CUDA 设备上运行 Pi0.5 和 GR00T N1.7 VLA 模型，
提供一个本地 ZeroMQ/Protobuf 服务端，以及固定输入的延迟测试脚本。

以下命令以解压后的项目位于 `~/OmniInfer` 为例；若实际目录不同，只需修改
`OMNIINFER_ROOT` 的值。

```text
~/OmniInfer/framework/OmniInfer-VLA
```

## 1. 目录与组件

```text
OmniInfer-VLA/
├── omniinfer_server.py             # ZeroMQ/Protobuf VLA 服务端
├── omniinfer-vla/src/               # 主 Python 包：omniinfer_vla
├── omniinfer-vla-kernel/            # CUDA/Triton 内核
├── omniinfer-vla-ext/               # TVM-FFI C++ 扩展
├── omniinfer-vla-utils/             # Pi0.5 / GR00T Processor 与工具
├── omniinfer-vla-model-optimizer/   # 模型优化工具
└── vla.proto                        # 服务协议
```

## 2. 环境配置

开始前，请确认 Jetson 已安装与当前 JetPack 匹配的 CUDA 驱动。首次部署时，先安装宿主构建工具；
Python 依赖会由后续的 `uv sync` 安装到项目隔离环境中，不会修改系统 Python：

```bash
sudo apt-get update
sudo apt-get install -y git curl build-essential cmake ninja-build pkg-config

# 仅首次安装 uv 与 Rust 工具链；安装后重新打开终端，或执行下面的 PATH 命令。
curl --proto '=https' --tlsv1.2 -LsSf https://astral.sh/uv/install.sh | sh
curl --proto '=https' --tlsv1.2 -LsSf https://sh.rustup.rs | sh -s -- -y --profile minimal
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
```

`vla_pb2.py` 已随源码提供，因此运行服务端和 benchmark 不要求额外安装系统 `protoc`。

### 2.1 编译 OmniInfer CLI

CLI 属于 OmniInfer 根仓库。先进入项目根目录，再执行 Cargo 命令：

```bash
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export OMNIINFER_ROOT="${OMNIINFER_ROOT:-$HOME/OmniInfer}"
cd "$OMNIINFER_ROOT"
cargo build -p omniinfer-cli
```

生成的可执行文件为：

```text
~/OmniInfer/target/debug/omniinfer
```

### 2.2 配置 OmniInfer-VLA Python 环境

VLA Python workspace 位于 `framework/OmniInfer-VLA`。进入该目录后执行一次 `uv sync`，
即可创建隔离环境、安装全部 workspace 依赖，并自动构建 `omniinfer-vla-ext` 的 C++ 扩展。

```bash
export OMNIINFER_ROOT="${OMNIINFER_ROOT:-$HOME/OmniInfer}"
export VLA_ROOT="$OMNIINFER_ROOT/framework/OmniInfer-VLA"
cd "$VLA_ROOT"
uv sync

# 验证 Python、PyTorch/CUDA 和服务端导入均可用。
uv run python -c 'import torch; print(torch.__version__, torch.version.cuda)'
uv run python omniinfer_server.py --help
```

`uv sync` 会创建项目隔离环境，并通过 `scikit-build-core` 自动调用 CMake/Ninja 编译 C++ 扩展；
不需要手动编译该扩展。仅在修改 `omniinfer-vla-ext` C++ 源码后，才需要在 VLA 目录执行：

```bash
uv sync --reinstall-package omniinfer-vla-ext
```


## 3. 模型与本地资源

默认 benchmark 使用以下目录，可通过环境变量覆盖：

| 资源 | 默认路径 | 用途 |
|---|---|---|
| Pi0.5 checkpoint | `~/models/pi05_libero_finetuned_v044` | Pi0.5 权重 |
| PaliGemma tokenizer | `~/models/paligemma-3b-pt-224` | Pi0.5 native Processor |
| GR00T checkpoint | `~/models/GR00T-N1.7-LIBERO/libero_object` | GR00T LIBERO 权重 |
| Cosmos/Qwen resources | `~/models/Cosmos-Reason2-2B` | GR00T native Processor |

若开发板已有 `~/vla-bench/models/GR00T-N1.7-LIBERO/libero_object` 与
`~/vla-bench/models/Cosmos-Reason2-2B`，一键 benchmark 会自动使用它们。其他目录布局可显式指定：

```bash
export GROOT_CHECKPOINT=/path/to/GR00T-N1.7-LIBERO/libero_object
export GROOT_PROCESSOR=/path/to/Cosmos-Reason2-2B
```

验证资源：

```bash
test -d "$HOME/models/pi05_libero_finetuned_v044"
test -d "$HOME/models/paligemma-3b-pt-224"
test -d "$HOME/models/GR00T-N1.7-LIBERO/libero_object"
test -d "$HOME/models/Cosmos-Reason2-2B"
```

## 4. 启动服务

服务只能绑定 loopback 地址。客户端通过 `vla.proto` 的 ZeroMQ 请求发送图像、语言和机器人
状态，返回 action chunk。

### Pi0.5：native Processor

```bash
uv run --project "$VLA_ROOT" python "$VLA_ROOT/omniinfer_server.py" \
  --bind tcp://127.0.0.1:5555 \
  --checkpoint "$HOME/models/pi05_libero_finetuned_v044" \
  --arch pi05 \
  --num-images 3 \
  --params-dtype bfloat16 \
  --vision-dtype float32 \
  --processor-mode native \
  --pi05-tokenizer "$HOME/models/paligemma-3b-pt-224"
```

### GR00T：prepared / engine-only 模式

此模式跳过原生 Processor、tokenizer 和图像预处理。客户端必须提供已准备好的输入；它适合
测 Engine 延迟，不应标为完整端到端延迟。

```bash
uv run --project "$VLA_ROOT" python "$VLA_ROOT/omniinfer_server.py" \
  --bind tcp://127.0.0.1:5556 \
  --checkpoint "$HOME/models/GR00T-N1.7-LIBERO/libero_object" \
  --arch gr00t_n17 \
  --num-images 2 \
  --params-dtype bfloat16 \
  --processor-mode prepared
```

GR00T native 模式还需要：

```text
--processor-mode native
--processor-model-name-or-path ~/models/Cosmos-Reason2-2B
--embodiment-tag LIBERO_PANDA
```

## 5. 一键 Benchmark

脚本会自动：启动服务器 → 等待 CUDA Graph / 模型 ready → 执行预热和计时 → 输出 JSON 与汇总 →
关闭它启动的服务器。

```bash
export OMNIINFER_ROOT="${OMNIINFER_ROOT:-$HOME/OmniInfer}"
cd "$OMNIINFER_ROOT"

export OMNIINFER_VLA_RUNTIME_HOME="$OMNIINFER_ROOT/framework/OmniInfer-VLA"

# 预热 10 次、正式计时 10 次，Pi0.5 + GR00T native
WARMUP=10 TIMED=10 ./scripts/benchmark_omniinfer_vla.sh

# Pi0.5 native 端到端（Processor + Engine + decode）
WARMUP=10 TIMED=10 ./scripts/benchmark_omniinfer_vla.sh pi05

# GR00T native 端到端；要求 torchvision 与 Jetson torch ABI 匹配
WARMUP=10 TIMED=10 ./scripts/benchmark_omniinfer_vla.sh gr00t

# GR00T prepared / engine-only；跳过 tokenizer 和视觉 Processor
NATIVE=0 WARMUP=10 TIMED=10 ./scripts/benchmark_omniinfer_vla.sh gr00t
```

可覆盖的常用路径和参数：

```bash
PI_CHECKPOINT=/path/to/pi05 \
PI05_TOKENIZER=/path/to/paligemma \
GROOT_CHECKPOINT=/path/to/gr00t/libero_object \
GROOT_PROCESSOR=/path/to/cosmos-qwen \
GROOT_EMBODIMENT_TAG=LIBERO_PANDA \
WARMUP=10 TIMED=10 \
./scripts/benchmark_omniinfer_vla.sh pi05
```

默认固定输入：

| 模型 | 参数 / 图像 | 文本长度 | Diffusion | 输出 |
|---|---|---:|---:|---|
| Pi0.5 | BF16 参数、BF16 vision、3 × 224 图像 | 48 tokens | 10 steps | `50 × 7` LIBERO actions |
| GR00T N1.7 | BF16、2 × 256 图像 | 156 tokens | 固定 noise | `40 × 132` 内部动作张量 |

结果及服务日志写入：

```text
~/OmniInfer/.local/benchmarks/<model>-<timestamp>.json
~/OmniInfer/.local/benchmarks/<model>-<timestamp>.server.log
```

输出指标含义：

| 指标 | 含义 |
|---|---|
| `processor` | native 模型 Processor；prepared 模式不会包含这部分 |
| `engine` | 已 CUDA 同步的核心引擎推理 |
| `postprocess` | action decode / 返回前处理 |
| `server total` | 服务端完整请求处理时间 |
| `ZMQ RTT` | 客户端到服务端往返时间 |

仅比较 engine 时，所有实现必须使用 prepared 输入；比较机器人真实部署延迟时，所有实现必须包含
同样的图像、语言、状态处理和 decode。
