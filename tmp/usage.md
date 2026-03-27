# OmniInfer 后端接入教程

文中的 `<repo-root>` 表示别人把仓库下载到本机后的根目录。

## 1. 启动服务

源码模式，默认隐藏终端窗口：

```powershell
cd <repo-root>
python .\omniinfer_gateway.py
```

源码模式，显示终端窗口：

```powershell
cd <repo-root>
python .\omniinfer_gateway.py --window-mode visible
```

发布版，默认隐藏终端窗口：

```powershell
cd <repo-root>\release\portable\OmniInfer
.\OmniInfer.exe
```

发布版，显示终端窗口：

```powershell
cd <repo-root>\release\portable\OmniInfer
.\OmniInfer.exe --window-mode visible
```

默认对外地址：

- `http://127.0.0.1:9000`

## 2. 隐藏启动后如何退出

如果你是默认隐藏方式启动的，看不到终端窗口，推荐直接调用关闭接口：

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:9000/omni/shutdown" `
  | ConvertTo-Json
```

如果服务已经无响应，再用任务管理器结束 `OmniInfer.exe`，或源码模式下结束对应的 `python.exe`。

## 3. 健康检查

```powershell
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:9000/health" | ConvertTo-Json -Depth 8
```

## 4. 查看当前后端状态

查看所有可用后端：

```powershell
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:9000/omni/backends" | ConvertTo-Json -Depth 8
```

查看当前状态：

```powershell
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:9000/omni/state" | ConvertTo-Json -Depth 10
```

手动停止当前已拉起的后端：

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:9000/omni/backend/stop"
```

## 5. 查询平台支持模型

### 5.1 原始后端视角

`GET /omni/supported-models` 会：

- 每次都从远端重新下载最新的 `model_list.json`
- `system=windows` 时读取 `https://omnimind-model.oss-cn-beijing.aliyuncs.com/backend/windows/model_list.json`
- `system=mac` 时读取 `https://omnimind-model.oss-cn-beijing.aliyuncs.com/backend/mac/model_list.json`
- 保留远端 JSON 的原始 backend 分组结构
- 在每个量化项中补两个字段：
  - `required_memory_gib`
  - `suitable`

判断规则：

- `llama.cpp-cpu`：看当前设备可用系统内存，安全余量 `1.0 GiB`
- `llama.cpp-cuda`：看当前设备可用 GPU 显存，安全余量 `0.5 GiB`
- 纯文本模型：只看文本权重大小
- 图文模型：文本权重大小 + `vision.size`

Windows：

```powershell
Invoke-RestMethod `
  -Method Get `
  -Uri "http://127.0.0.1:9000/omni/supported-models?system=windows" `
  | ConvertTo-Json -Depth 20
```

Mac：

```powershell
Invoke-RestMethod `
  -Method Get `
  -Uri "http://127.0.0.1:9000/omni/supported-models?system=mac" `
  | ConvertTo-Json -Depth 20
```

### 5.2 自动选最优 backend 的视角

`GET /omni/supported-models/best` 会：

- 每次都从远端重新下载最新的模型目录
- 去掉最外层 backend 分组
- 保留模型原始结构
- 在每个量化项中补：
  - `required_memory_gib`
  - `suitable`
  - `backend`

当前 backend 优先级：

- `llama.cpp-cuda > llama.cpp-cpu`

如果至少有一个 backend 可用，就选最优的那个。  
如果没有任何 backend 可用，模型也不会被过滤掉，而是：

- `suitable=false`
- `backend=""`
- `required_memory_gib` 仍然保留最佳候选 backend 对应的需求值

Windows：

```powershell
Invoke-RestMethod `
  -Method Get `
  -Uri "http://127.0.0.1:9000/omni/supported-models/best?system=windows" `
  | ConvertTo-Json -Depth 20
```

Mac：

```powershell
Invoke-RestMethod `
  -Method Get `
  -Uri "http://127.0.0.1:9000/omni/supported-models/best?system=mac" `
  | ConvertTo-Json -Depth 20
```

## 6. 选择并加载模型

`POST /omni/model/select` 现在通常不再需要传 `backend`。  
OmniInfer 会根据当前系统支持目录自动选择最优 backend。

如果当前设备没有任何合适 backend，会直接返回错误，不会盲目尝试启动。

文本模型：

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:9000/omni/model/select" `
  -ContentType "application/json" `
  -Body '{"model":"D:\\models\\Qwen3.5-0.8B-Q4_K_M.gguf"}'
```

图文模型：

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:9000/omni/model/select" `
  -ContentType "application/json" `
  -Body '{"model":"D:\\models\\Qwen3.5-0.8B-Q4_K_M.gguf","mmproj":"D:\\models\\mmproj-F32.gguf"}'
```

## 7. think 模式开关

查看默认值：

```powershell
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:9000/omni/thinking" | ConvertTo-Json
```

打开：

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:9000/omni/thinking/select" `
  -ContentType "application/json" `
  -Body '{"enabled":true}'
```

关闭：

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:9000/omni/thinking/select" `
  -ContentType "application/json" `
  -Body '{"enabled":false}'
```

## 8. 文本推理

### 8.1 推荐用法：先加载模型，再推理

如果已经调用过 `/omni/model/select`，通常不需要再传：

- `backend`
- `model`

```powershell
$Body = @{
  think = $false
  messages = @(
    @{
      role = "user"
      content = "请用一句中文介绍你自己。"
    }
  )
  temperature = 0.2
  max_tokens = 128
} | ConvertTo-Json -Depth 10

Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:9000/v1/chat/completions" `
  -ContentType "application/json" `
  -Body $Body `
  | ConvertTo-Json -Depth 20
```

### 8.2 一步完成：在推理请求里直接带模型

如果还没预加载模型，也可以直接在请求里传 `model`。  
OmniInfer 会自动选择最优 backend 并加载模型。

```powershell
$Body = @{
  model = "D:\models\Qwen3.5-0.8B-Q4_K_M.gguf"
  think = $false
  messages = @(
    @{
      role = "user"
      content = "请用一句中文介绍你自己。"
    }
  )
  temperature = 0.2
  max_tokens = 128
} | ConvertTo-Json -Depth 10

Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:9000/v1/chat/completions" `
  -ContentType "application/json" `
  -Body $Body `
  | ConvertTo-Json -Depth 20
```

## 9. 图文推理

同样推荐先调用 `/omni/model/select` 预加载图文模型。  
真正推理时通常不再需要传 `backend`，也不需要重复传 `model/mmproj`。

```powershell
$ImagePath = "<repo-root>\tests\pictures\test1.png"
$Bytes = [System.IO.File]::ReadAllBytes($ImagePath)
$Base64 = [System.Convert]::ToBase64String($Bytes)
$DataUrl = "data:image/png;base64,$Base64"

$Body = @{
  think = $false
  messages = @(
    @{
      role = "user"
      content = @(
        @{
          type = "text"
          text = "请用中文简要描述这张图片。"
        },
        @{
          type = "image_url"
          image_url = @{
            url = $DataUrl
          }
        }
      )
    }
  )
  temperature = 0.2
  max_tokens = 256
} | ConvertTo-Json -Depth 12

Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:9000/v1/chat/completions" `
  -ContentType "application/json" `
  -Body $Body `
  | ConvertTo-Json -Depth 20
```

如果还没预加载模型，也可以在请求里直接带 `model` 和 `mmproj`，但仍然不需要传 `backend`。

## 10. 流式推理

流式请求同样遵循相同逻辑：

- 如果模型已预加载，通常不需要传 `backend`
- 通常也不需要重复传 `model`

推荐使用 `curl.exe`，因为它更适合实时查看 SSE 输出。

```powershell
$RequestFile = Join-Path $PWD "stream-request.json"

$Body = @{
  think = $false
  messages = @(
    @{
      role = "user"
      content = "请连续输出三句短句，每句单独成行。"
    }
  )
  stream = $true
  stream_options = @{
    include_usage = $true
  }
  temperature = 0.2
  max_tokens = 64
} | ConvertTo-Json -Depth 10

$Body | Set-Content -Path $RequestFile -Encoding UTF8

curl.exe -N -sS -X POST "http://127.0.0.1:9000/v1/chat/completions" `
  -H "Content-Type: application/json" `
  --data-binary "@$RequestFile"
```

如果还没预加载模型，也可以把 `model` 放进 `$Body`，但仍然不需要传 `backend`。

## 11. 接口说明

### 11.1 GET /health

- 健康检查
- 查看当前后端、当前模型、默认 think 状态

### 11.2 GET /omni/state

- 查看当前服务完整状态

### 11.3 GET /omni/backends

- 查看所有可用后端

### 11.4 GET /omni/supported-models

- 返回按 backend 分组的原始支持目录
- 每个量化项补 `required_memory_gib` 和 `suitable`

查询参数：

- `system`
  - 必填
  - `windows` 或 `mac`

### 11.5 GET /omni/supported-models/best

- 返回自动选最优 backend 后的支持目录
- 去掉最外层 backend 分组
- 每个量化项补 `required_memory_gib`、`suitable`、`backend`

查询参数：

- `system`
  - 必填
  - `windows` 或 `mac`

### 11.6 POST /omni/model/select

- 自动选择最优 backend 并加载模型

请求字段：

- `model`
- `mmproj`

### 11.7 GET /omni/thinking

- 查看默认 think 开关

### 11.8 POST /omni/thinking/select

- 修改默认 think 开关

### 11.9 POST /omni/backend/stop

- 停止当前已启动的后端进程

### 11.10 POST /omni/shutdown

- 关闭 OmniInfer 服务本身
- 隐藏启动时推荐用这个接口退出

### 11.11 GET /v1/models

- 当前不维护

### 11.12 POST /v1/chat/completions

- OpenAI 风格聊天推理接口

标准字段：

- `model`
- `messages`
- `temperature`
- `max_tokens`
- `stream`

OmniInfer 扩展字段：

- `mmproj`
- `think`

`backend` 现在通常不需要再传。
