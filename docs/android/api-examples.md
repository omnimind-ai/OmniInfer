# Android HTTP API Examples

OmniInfer Android exposes the same local OpenAI-compatible HTTP API for every backend.

Base URL:

```text
http://127.0.0.1:<port>
```

Default port:

```text
9099
```

## Endpoints

| Endpoint | Description |
|---|---|
| `GET /health` | Returns `{"status":"ok"}` when the local server is alive |
| `GET /v1/models` | Lists loaded model names |
| `POST /v1/chat/completions` | Chat inference, streaming or non-streaming |
| `POST /v1/cancel` | Gracefully cancel an active streaming generation |

## Text Request

```json
{
  "model": "local",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "stream": false,
  "max_tokens": 128,
  "temperature": 0.7,
  "top_p": 0.9
}
```

Sampling parameters are optional:

| Parameter | Type | Default |
|---|---|---|
| `temperature` | float | Backend default |
| `top_p` | float | Backend default |
| `top_k` | int | Backend default |
| `repetition_penalty` | float | `1.0` |
| `frequency_penalty` | float | `0.0` |
| `presence_penalty` | float | `0.0` |

## OkHttp Non-Streaming Example

```kotlin
import com.omniinfer.server.OmniInferServer
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.util.concurrent.TimeUnit

private val client = OkHttpClient.Builder()
    .connectTimeout(10, TimeUnit.SECONDS)
    .readTimeout(300, TimeUnit.SECONDS)
    .build()

suspend fun chatOnce(prompt: String): String = withContext(Dispatchers.IO) {
    val body = """
    {
      "model": "local",
      "messages": [{"role": "user", "content": ${JSONObject.quote(prompt)}}],
      "stream": false,
      "max_tokens": 128
    }
    """.trimIndent()

    val request = Request.Builder()
        .url("http://127.0.0.1:${OmniInferServer.getPort()}/v1/chat/completions")
        .post(body.toRequestBody("application/json".toMediaType()))
        .build()

    client.newCall(request).execute().use { response ->
        check(response.isSuccessful) { response.body?.string() ?: response.message }
        val json = JSONObject(response.body!!.string())
        json.getJSONArray("choices")
            .getJSONObject(0)
            .getJSONObject("message")
            .getString("content")
    }
}
```

## adb / curl Test

```bash
adb forward tcp:9099 tcp:9099
curl -sS -H "Content-Type: application/json" \
  --data-binary @request.json \
  http://127.0.0.1:9099/v1/chat/completions
```

Use `--data-binary @request.json` for repeatable tests. It avoids shell quoting problems and prevents accidentally sending an empty request body.

## Tool Calling

OmniInfer accepts OpenAI-compatible `tools` requests on supported backends, including llama.cpp, MNN, and LiteRT-LM.

```json
{
  "model": "local",
  "messages": [
    {"role": "user", "content": "What is the product of 12.34 and 98.76?"}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "product",
        "description": "Get the product of a list of numbers.",
        "parameters": {
          "type": "object",
          "properties": {
            "numbers": {
              "type": "array",
              "items": {"type": "number"}
            }
          },
          "required": ["numbers"]
        }
      }
    }
  ],
  "tool_choice": "auto",
  "stream": false
}
```

When the model chooses a tool, the response contains structured tool calls:

```json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "id": "call_0",
            "type": "function",
            "function": {
              "name": "product",
              "arguments": {"numbers": [12.34, 98.76]}
            }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ]
}
```

Run the tool in the host app, then send the tool result back:

```json
{
  "model": "local",
  "messages": [
    {"role": "user", "content": "What is the product of 12.34 and 98.76?"},
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [
        {
          "id": "call_0",
          "type": "function",
          "function": {
            "name": "product",
            "arguments": "{\"numbers\":[12.34,98.76]}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_0",
      "name": "product",
      "content": "1218.6984"
    }
  ],
  "tool_choice": "none",
  "stream": false
}
```

`tool_choice` may be `"none"`, `"auto"`, `"required"`, or the OpenAI object form `{"type":"function","function":{"name":"product"}}`.

## Streaming Metrics

The final SSE chunk includes `usage` and `performance` fields:

```json
{
  "usage": {
    "prompt_tokens": 460,
    "completion_tokens": 18,
    "total_tokens": 478,
    "completion_tokens_details": {
      "reasoning_tokens": 120,
      "text_tokens": 29
    },
    "prompt_tokens_details": {
      "image_tokens": 253,
      "text_tokens": 207,
      "cached_tokens": 438,
      "cache_creation_input_tokens": 22,
      "cache_type": "ephemeral"
    },
    "performance": {
      "prefill_time_ms": 230.6,
      "prefill_tokens_per_second": 1995.1,
      "decode_time_ms": 1021.8,
      "decode_tokens_per_second": 17.6,
      "total_time_ms": 1252.4,
      "time_to_first_token_ms": 230.6
    }
  }
}
```

Important fields:

| Field | Meaning |
|---|---|
| `cached_tokens` | KV cache prefix reused from previous request |
| `cache_creation_input_tokens` | Tokens actually prefilled this request |
| `prefill_tokens_per_second` | Prompt processing throughput |
| `decode_tokens_per_second` | Generation throughput |
| `time_to_first_token_ms` | Time before first generated token |

## Cancel Behavior

Graceful cancel:

```bash
curl -s -X POST http://127.0.0.1:9099/v1/cancel
```

| Scenario | Current response | KV cache | Next request |
|---|---|---|---|
| `POST /v1/cancel` | Finishes normally with final chunk | Preserved | Can reuse prefix |
| Client disconnect | Interrupted | Cleared | Full prefill |

Use `/v1/cancel` for user-facing stop buttons. Treat client disconnect as a hard cancellation path.
