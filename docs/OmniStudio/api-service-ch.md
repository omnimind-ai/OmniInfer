# OmniStudio API 服务指南

OmniStudio 是基于 OmniInfer 的多平台模型推理客户端。在对话界面加载模型后，OmniStudio 会自动在后台启动 OpenAI 兼容的 API 服务，供外部应用通过 HTTP 调用本地推理引擎。

## API 概览

API 服务运行在 `http://<host>:<port>`（默认 `http://127.0.0.1:9000`）。

主要端点：

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查与运行状态 |
| `/omni/state` | GET | 当前后端、模型及配置信息 |
| `/omni/backends` | GET | 列出可用后端 |
| `/omni/model/select` | POST | 加载模型 |
| `/omni/backend/stop` | POST | 停止当前后端 |
| `/v1/models` | GET | 列出当前已加载的模型 |
| `/v1/chat/completions` | POST | OpenAI 兼容的对话补全接口 |
| `/omni/shutdown` | POST | 关闭网关服务 |

完整 API 参考文档请见 [API.md](https://github.com/omnimind-ai/OmniInfer/blob/main/docs/API.md)。

---

## PC 客户端（Windows / macOS）

Windows 和 macOS 客户端架构一致：OmniStudio 内置对应平台的 OmniInfer 发布包，在后台管理推理网关进程。

### 前置条件

- 已安装 OmniStudio 桌面客户端
- 已下载至少一个模型文件
- 足够的内存 / 显存以加载目标模型

### 第一步：加载模型

打开 OmniStudio 对话界面，选择并加载一个模型。模型加载完成后，API 服务会自动启动。

此时后台运行的进程结构：

```
OmniStudio (UI)
  └─ OmniInfer Gateway (HTTP, 默认端口 9000)
```

### 第二步：验证服务

打开终端执行：

```bash
curl http://127.0.0.1:9000/health
```

预期响应：

```json
{
  "status": "ok",
  "omni": {
    "backend": "llama.cpp-cpu",
    "model": "Qwen3.5-0.8B-Q4_K_M.gguf",
    "backend_ready": true
  }
}
```

`backend_ready: true` 表示 API 已就绪，可以接收推理请求。

### 第三步：调用 API

**非流式请求：**

```bash
curl -X POST http://127.0.0.1:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "1+1等于几？"}
    ],
    "temperature": 0.7,
    "max_tokens": 256,
    "stream": false
  }'
```

响应：

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "model": "Qwen3.5-0.8B-Q4_K_M.gguf",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "1 + 1 = 2。"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 8,
    "total_tokens": 18
  }
}
```

**流式请求：**

```bash
curl -X POST http://127.0.0.1:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "用四行写一首关于代码的诗。"}
    ],
    "stream": true
  }'
```

响应格式为 Server-Sent Events (SSE)：

```text
data: {"choices":[{"delta":{"role":"assistant","content":""}}]}
data: {"choices":[{"delta":{"content":"代码"}}]}
data: {"choices":[{"delta":{"content":"如诗"}}]}
...
data: {"usage":{"prompt_tokens":12,"completion_tokens":18,"total_tokens":30}}
data: [DONE]
```

### 第四步：接入第三方应用

任何支持 OpenAI 兼容 API 的应用都可以连接 OmniStudio。将 Base URL 设为 `http://127.0.0.1:9000`，API Key 留空或填写任意占位符即可。

**Python (openai SDK)：**

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:9000/v1",
    api_key="not-needed",
)

response = client.chat.completions.create(
    model="local",
    messages=[{"role": "user", "content": "你好"}],
    stream=True,
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

**JavaScript/TypeScript (fetch)：**

```typescript
const response = await fetch("http://127.0.0.1:9000/v1/chat/completions", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    messages: [{ role: "user", content: "你好" }],
    stream: false,
  }),
});

const data = await response.json();
console.log(data.choices[0].message.content);
```

### 可用后端

不同平台提供不同的后端。OmniStudio 会自动选择最优后端，也可通过 API 手动切换：

**Windows：**

| 后端 | 加速方式 | 前置要求 |
|------|---------|---------|
| `llama.cpp-cpu` | CPU | 无 |
| `llama.cpp-cuda` | NVIDIA GPU | CUDA 运行时 |
| `llama.cpp-vulkan` | GPU（跨厂商） | Vulkan 驱动 |

**macOS：**

| 后端 | 加速方式 | 前置要求 |
|------|---------|---------|
| `llama.cpp-mac` | Metal (Apple Silicon) | Apple Silicon Mac |
| `mlx-mac` | MLX (Apple Silicon) | Apple Silicon Mac |

### 进阶：通过 API 切换后端

```bash
curl -X POST http://127.0.0.1:9000/omni/model/select \
  -H "Content-Type: application/json" \
  -d '{
    "model": "path/to/model.gguf",
    "backend": "llama.cpp-cuda",
    "ctx_size": 4096,
    "launch_args": ["-ngl", "999"]
  }'
```

### 停止服务

API 服务在以下情况下自动停止：
- 在 OmniStudio 中卸载模型
- 关闭 OmniStudio

也可通过 API 手动停止：

```bash
curl -X POST http://127.0.0.1:9000/omni/shutdown
```

---

## Android 客户端

Android 客户端完全在设备端运行推理引擎。OmniStudio 内嵌 OmniInfer 原生库，通过 Ktor 驱动的本地 HTTP 服务器对外暴露相同的 OpenAI 兼容 API。

### 架构

```
OmniStudio Android (UI)
  └─ OmniInferService (前台服务)
       └─ Ktor HTTP Server (127.0.0.1:9099)
            └─ JNI → 原生 C++
```

与 PC 客户端运行独立网关进程不同，Android 客户端将 HTTP 服务器作为 Android 前台服务运行在应用进程内，既保持单进程架构，又符合 Android 后台执行限制。

### 前置条件

- 已安装 OmniStudio Android，设备为 ARM64 架构（推荐骁龙 8 Gen 1 及以上）
- 已下载至少一个模型到本地存储
- 足够的可用内存

### 端点

Android API 提供以下端点：

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/v1/models` | GET | 列出已加载的模型 |
| `/v1/chat/completions` | POST | 对话补全（支持流式和非流式） |

> **注意：** Android 不提供 `/omni/*` 管理端点。模型加载和后端选择通过 OmniStudio 界面或 Kotlin API (`OmniInferServer.loadModel()`) 控制。

### 第一步：加载模型

打开 OmniStudio，选择一个模型进行对话。模型加载完成后，API 服务自动在 **9099** 端口启动。

### 第二步：验证服务

**从 PC 端（通过 adb 端口转发）：**

```bash
adb forward tcp:9099 tcp:9099
curl http://127.0.0.1:9099/health
```

**从设备上的其他应用：**

```kotlin
val url = URL("http://127.0.0.1:9099/health")
val result = url.readText()  // {"status":"ok"}
```

预期响应：

```json
{"status": "ok"}
```

### 第三步：调用 API

**非流式请求：**

```bash
curl -X POST http://127.0.0.1:9099/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "1+1等于几？"}
    ],
    "max_tokens": 256,
    "stream": false
  }'
```

响应：

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "model": "omniinfer",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "1 + 1 = 2。"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 19,
    "completion_tokens": 8,
    "total_tokens": 27,
    "prompt_tokens_details": {
      "text_tokens": 19,
      "cached_tokens": 0,
      "cache_creation_input_tokens": 19,
      "cache_type": "ephemeral"
    },
    "performance": {
      "prefill_time_ms": 150.0,
      "prefill_tokens_per_second": 126.7,
      "decode_time_ms": 280.0,
      "decode_tokens_per_second": 28.6,
      "total_time_ms": 430.0,
      "time_to_first_token_ms": 150.0
    }
  }
}
```

**流式请求：**

```bash
curl -X POST http://127.0.0.1:9099/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "用四行写一首关于代码的诗。"}
    ],
    "stream": true
  }'
```

响应格式为 Server-Sent Events (SSE)：

```text
data: {"choices":[{"delta":{"role":"assistant","content":""},"index":0,"finish_reason":null}]}
data: {"choices":[{"delta":{"content":"代码"},"index":0,"finish_reason":null}]}
data: {"choices":[{"delta":{"content":"如诗"},"index":0,"finish_reason":null}]}
...
data: {"choices":[{"delta":{},"index":0,"finish_reason":"stop"}]}
data: {"choices":[],"usage":{...}}
data: [DONE]
```

### 第四步：接入第三方应用

**从 PC 端（通过 adb）：**

PC 上任何支持 OpenAI 兼容 API 的应用都可以连接手机的推理引擎。先设置端口转发：

```bash
adb forward tcp:9099 tcp:9099
```

然后将应用的 Base URL 设为 `http://127.0.0.1:9099`。

**Python (openai SDK)：**

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:9099/v1",  # 需先 adb forward
    api_key="not-needed",
)

response = client.chat.completions.create(
    model="local",
    messages=[{"role": "user", "content": "你好"}],
    stream=True,
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

**从设备上的其他 Android 应用：**

任何 Android 应用都可以通过 OkHttp、Retrofit、Ktor client 或其他 HTTP 库调用 `http://127.0.0.1:9099/v1/chat/completions`。访问 localhost 无需特殊权限。

### 可用后端

| 后端 | 模型格式 | 加速方式 |
|------|---------|---------|
| `llama.cpp` | GGUF (`.gguf`) | CPU (ARM NEON, i8mm, dotprod) |
| `mnn` | MNN (`config.json` + `.mnn` 权重) | CPU (ARM82) + 可选 OpenCL |

两个后端均支持：
- 多轮对话
- KV 缓存前缀复用（自动生效，无需配置）
- 多模态 / 视觉（模型需包含视觉编码器文件）
- 思考 / 推理模式（`enable_thinking` 或 `reasoning_effort` 参数）
- 工具调用（llama.cpp：所有含工具模板的模型；MNN：Qwen3.5、Qwen3、Hunyuan 系列）

### 停止服务

API 服务在以下情况下自动停止：
- 在 OmniStudio 中卸载模型
- 关闭 OmniStudio 或取消前台服务通知

Android 不提供 HTTP 关闭端点，通过 OmniStudio 界面管理服务生命周期。

---

## iOS 客户端

iOS 客户端与 Android 类似，完全在设备端运行推理引擎。OmniStudio 内嵌 OmniInfer Swift Package，通过基于 Hummingbird (swift-nio) 的进程内 HTTP 服务器提供与桌面端相同的 OpenAI 兼容 API。

### 架构

```
OmniStudio iOS (SwiftUI)
  └─ OmniInferServer（进程内）
       └─ Hummingbird HTTP Server (127.0.0.1:9099)
```

与 Android 使用前台服务不同，iOS 客户端以进程内 NIO 事件循环方式运行 HTTP 服务器。由于 iOS 不支持后台服务，API 仅在应用处于前台时可用。

### 前提条件

- 已安装 OmniStudio iOS（需要 iOS 17+）
- 至少已下载一个模型

### 端点

iOS API 提供与 Android 相同的端点子集：

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/v1/models` | GET | 列出已加载模型 |
| `/v1/chat/completions` | POST | 聊天补全（流式和非流式） |

### 第一步：加载模型

打开 OmniStudio 选择一个模型进行对话。模型加载完成后，API 服务自动在 **9099** 端口启动。

### 第二步：验证服务

iOS API 绑定 `127.0.0.1`（仅本地回环），只能从 app 进程内或通过 USB 调试工具访问。

**从 app 内以编程方式访问：**

```swift
import OmniInferServer

// 加载模型
await OmniInferServer.shared.loadModel(
    modelPath: "/path/to/model.gguf",
    backend: "llama.cpp"  // 或 "mlx"
)

// 服务器已在 http://127.0.0.1:9099 运行
let url = URL(string: "http://127.0.0.1:9099/health")!
let (data, _) = try await URLSession.shared.data(from: url)
// {"status":"ok"}
```

### 第三步：调用 API

**非流式请求（app 内调用）：**

```swift
let url = URL(string: "http://127.0.0.1:9099/v1/chat/completions")!
var request = URLRequest(url: url)
request.httpMethod = "POST"
request.setValue("application/json", forHTTPHeaderField: "Content-Type")
request.httpBody = try JSONSerialization.data(withJSONObject: [
    "messages": [["role": "user", "content": "2+2等于多少？"]],
    "stream": false
])

let (data, _) = try await URLSession.shared.data(for: request)
```

响应：

```json
{
  "object": "chat.completion",
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "2 + 2 = 4。"
      },
      "index": 0,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prefill_tokens_per_second": 45.2,
    "decode_tokens_per_second": 21.8
  }
}
```

**流式请求：**

```swift
let (bytes, _) = try await URLSession.shared.bytes(for: request)
for try await line in bytes.lines {
    guard line.hasPrefix("data: ") else { continue }
    let payload = String(line.dropFirst(6))
    if payload == "[DONE]" { break }
    // 解析 SSE chunk...
}
```

### 第四步：接入第三方应用

同一设备上的其他 iOS 应用可以通过 URLSession 或任何 HTTP 库调用 `http://127.0.0.1:9099/v1/chat/completions`。访问 localhost 无需特殊权限。

**通过 Swift Package 直接集成：**

无需经过 HTTP，直接添加 OmniInferServer Swift Package 作为依赖：

```swift
// Package.swift
dependencies: [
    .package(path: "Vendor/OmniInfer/ios/OmniInferServer")
]
```

然后直接使用 OmniInferServer API：

```swift
import OmniInferServer

await OmniInferServer.shared.loadModel(
    modelPath: modelDir,
    backend: "llama.cpp",  // 或 "mlx"
    port: 9099,
    nCtx: 4096
)
// 服务器已就绪
```

### 停止服务

API 服务在以下情况下自动停止：
- 在 OmniStudio 中卸载模型
- 关闭 OmniStudio 或应用进入后台

通过代码停止：

```swift
await OmniInferServer.shared.stop()
```

iOS 不提供 HTTP 关闭端点，通过 OmniInferServer Swift API 管理服务生命周期。
