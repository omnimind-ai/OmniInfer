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

> **TODO** — Android API 服务文档待补充。

---

## iOS 客户端

> **TODO** — iOS API 服务文档待补充。
