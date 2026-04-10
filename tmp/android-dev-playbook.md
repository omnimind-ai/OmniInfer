# Android Dev Playbook

Operational reference for Claude Code sessions working on `android/omniinfer-server`.

## Device & ADB Setup

```bash
# Git Bash on Windows mangles /data/... paths. Always prefix:
MSYS_NO_PATHCONV=1 adb shell am start ...

# ADB version mismatch between system and Gradle SDK is common.
# Gradle installDebug often fails on first try after ADB daemon restart.
# Workaround: run `adb install -r` manually with the built APK.
adb install -r tmp/OmniInferServerTest/app/build/outputs/apk/debug/app-debug.apk

# Port forwarding (server default port 9099):
adb forward tcp:9099 tcp:9099
```

## Test App

**Location:** `tmp/OmniInferServerTest/`

An Android app that depends on `:omniinfer-server` (via `settings.gradle.kts` pointing to `../../android/omniinfer-server`). It loads a model, starts the Ktor HTTP server, and exposes the OpenAI-compatible API at `http://127.0.0.1:9099`.

**Package:** `com.test.omniinfer.server`
**Activity:** `com.test.server.MainActivity`

### Launch via Intent (auto-load model, no manual UI tap needed)

```bash
# llama.cpp backend
MSYS_NO_PATHCONV=1 adb shell am start \
  -n com.test.omniinfer.server/com.test.server.MainActivity \
  --es backend "llama.cpp" \
  --es model "/data/local/tmp/gemma-4-E2B-gguf/gemma-4-E2B-it-Q4_K_M.gguf"

# MNN backend
MSYS_NO_PATHCONV=1 adb shell am start \
  -n com.test.omniinfer.server/com.test.server.MainActivity \
  --es backend "mnn" \
  --es model "/data/local/tmp/Qwen3.5-0.8B-MNN/config.json"
```

Intent extras: `backend` (required), `model` (required), `prompt` (optional — empty = load only, non-empty = load + auto-test).

### Build & Install

```bash
cd tmp/OmniInferServerTest
./gradlew :app:installDebug          # build + install (may fail on ADB mismatch, retry or use adb install)
./gradlew :omniinfer-server:clean :app:clean   # clean CMake cache (needed after changing CMake options)
```

First native build takes ~3 min. Incremental Kotlin-only changes take ~3s.

## Models on Device

```
/data/local/tmp/gemma-4-E2B-gguf/
  ├── gemma-4-E2B-it-Q4_K_M.gguf     (llama.cpp, VLM — has mmproj)
  └── mmproj-F16.gguf

/data/local/tmp/Qwen3.5-0.8B-Q4_K_M.gguf   (llama.cpp, text-only)

/data/local/tmp/Qwen3.5-0.8B-MNN/
  ├── config.json                     (MNN, VLM — has visual.mnn)
  ├── llm.mnn / llm.mnn.weight
  └── visual.mnn / visual.mnn.weight

/data/local/tmp/mnn-models/gemma-3-1b-it-MNN/
  ├── config.json                     (MNN, text-only, Gemma 3 1B QAT Q4_0)
  ├── llm.mnn / llm.mnn.weight / llm.mnn.json
  └── tokenizer.txt

/data/local/tmp/mnn-models/MiniCPM4-0.5B-MNN/
  ├── config.json                     (MNN, text-only, MiniCPM4 0.5B)
  ├── llm.mnn / llm.mnn.weight
  └── tokenizer.txt                   (SentencePiece — outputs ▁ markers, see gotcha #12)
```

## API Testing (curl)

After model loaded and port forwarded:

```bash
# Health check
curl -s http://127.0.0.1:9099/health

# Text inference (non-streaming)
curl -s -X POST http://127.0.0.1:9099/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"test","messages":[{"role":"user","content":"What is 2+3?"}],"stream":false,"reasoning_effort":"none","max_tokens":50}'

# Text inference (streaming)
curl -s -X POST http://127.0.0.1:9099/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"test","messages":[{"role":"user","content":"Count 1 to 5."}],"stream":true,"reasoning_effort":"none","max_tokens":50}'

# Multimodal (image as base64 data URI)
IMG_B64=$(base64 -w 0 /path/to/image.jpg)
curl -s -X POST http://127.0.0.1:9099/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"test\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"Describe this image.\"},{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/jpeg;base64,${IMG_B64}\"}}]}],\"stream\":false,\"reasoning_effort\":\"none\",\"max_tokens\":100}"

# Thinking mode (explicitly enabled)
curl -s ... -d '{"model":"test","messages":[...],"stream":true,"enable_thinking":true,"max_tokens":30}'
```

**Tips:**
- Use `"reasoning_effort":"none"` to disable thinking (default is off on mobile).
- Use `"max_tokens":N` to cap output length — small models like Qwen3.5-0.8B tend to ramble.
- For thinking mode verification, keep `max_tokens` low (e.g. 30) to avoid waiting for long output.

## Debugging

```bash
# App logs (OmniInfer tags)
adb logcat -s OmniInferJni:* OmniInferServer:* OmniInferService:*

# Full app process logs
adb logcat --pid=$(adb shell pidof com.test.omniinfer.server)

# MNN-specific logs
adb logcat -s MNNJNI:*

# Force stop
adb shell am force-stop com.test.omniinfer.server
```

## Key Gotchas (Lessons Learned)

1. **MSYS path conversion** — Git Bash converts `/data/local/tmp/...` to `D:/Program/.../data/local/tmp/...`. Always use `MSYS_NO_PATHCONV=1` before adb commands with Android paths.

2. **App file permissions** — The app process cannot write to `/data/local/tmp/` (owned by shell). Writable locations: app's `cacheDir` (passed via config JSON as `cache_dir`), or app's internal storage.

3. **MNN vision requires `MNN_BUILD_LLM_OMNI=ON`** — Without this CMake flag, the Omni class (vision/audio) is not compiled. This flag auto-enables `MNN_BUILD_OPENCV` and `MNN_IMGCODECS`.

4. **MNN `<img>` tag placement** — MNN's multimodal tokenizer uses regex `<(img|audio)>(.*?)</\1>` on the final prompt string. The `<img>` tag must be inserted *after* chat template apply (into the formatted prompt), not before (the jinja template would ignore it).

5. **llama.cpp `<__media__>` marker** — Must be inserted into user message content *before* chat template apply. The template embeds it into the formatted prompt, and `mtmd_tokenize` replaces it with image tokens.

6. **llama.cpp `<think>` tag injection** — Only prepend the thinking start tag when `thinking_enabled == true`. Small models (Qwen3.5-0.8B) don't reliably output `</think>` when thinking is off, causing the Kotlin SSE layer to misclassify content as `reasoning_content`.

7. **Non-streaming imageData** — Both streaming and non-streaming paths in `OmniInferService.kt` must pass `imageData` to `OmniInferBridge.generate()`. Easy to miss when only one path is updated.

8. **MNN `messages_json` parsing** — The MNN backend must parse the `messages_json` parameter (JSON array of role/content pairs). The JNI layer sets `system_prompt` and `user_prompt` to empty strings when `messages_json` is present.

9. **MNN thinking control** — Use `llm_->set_config(R"({"jinja":{"context":{"enable_thinking":false}}})")` before `apply_chat_template()`. The old `/no_think` system prompt hack does not work reliably. Only models whose jinja template references `enable_thinking` (e.g. Qwen3.5) are affected; others ignore it silently.

10. **MNN `bos_token` not injected by MNN framework** — `setChatTemplate()` in MNN passes `eos` but not `bos` from `llm_config.json`'s jinja section. Models whose jinja templates use `{{ bos_token }}` (e.g. Gemma) render it as empty, causing immediate EOS. Workaround: read `llm_config.json` at load time and inject `bos_token` via `set_config` into the jinja context. See `inject_jinja_special_tokens()` in `backend_mnn.h`.

11. **KV cache prefix reuse** — Both llama.cpp and MNN backends track previous prompt tokens and reuse the KV cache prefix on multi-turn conversations. Three cases: (a) no match → full reset + full prefill, (b) partial prefix match → trim old suffix KV + prefill new suffix only, (c) exact match → erase last KV entry + re-decode 1 token to restore logits. Multimodal requests always invalidate cache.

12. **MNN SentencePiece `▁` markers** — Some MNN models (e.g. MiniCPM4) use SentencePiece tokenizers where MNN's detokenizer does not convert `▁` (U+2581) to spaces. The raw `▁` appears in API output. This is an MNN model packaging issue, not a backend code bug.

13. **Non-streaming thinking output** — `OmniInferService.kt` parses `<think>...</think>` tags in non-streaming results to split `reasoning_content` and `content`. See `parseThinkingTags()`. The streaming path handles this via incremental SSE delta detection.

## File Map (Android Module)

```
android/omniinfer-server/
├── build.gradle.kts                         # Gradle config (compileSdk 35, arm64-v8a, CMake args)
├── src/main/
│   ├── java/com/omniinfer/server/
│   │   ├── OmniInferService.kt              # Ktor HTTP server, SSE streaming, request routing
│   │   ├── OmniInferServer.kt               # Facade: init/loadModel/unloadModel/stop
│   │   └── OmniInferBridge.kt               # JNI bridge, native method declarations
│   └── cpp/omniinfer-jni/
│       ├── CMakeLists.txt                   # Backend selection, llama.cpp + MNN build config
│       ├── omniinfer_jni.cpp                # JNI entry points, session management
│       ├── inference_backend.h              # Abstract backend interface
│       ├── backend_llama_cpp.h              # llama.cpp: text + multimodal (mtmd)
│       └── backend_mnn.h                   # MNN: text + multimodal (Omni class)
```
