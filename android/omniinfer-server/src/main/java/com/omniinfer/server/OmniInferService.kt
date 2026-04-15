package com.omniinfer.server

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import android.util.Log
import io.ktor.http.*
import io.ktor.server.cio.*
import io.ktor.server.engine.*
import io.ktor.server.engine.connector
import io.ktor.server.request.*
import io.ktor.server.response.*
import io.ktor.server.routing.*
import io.ktor.server.application.*
import kotlinx.coroutines.*
import kotlinx.serialization.json.*
import java.util.UUID
import kotlin.coroutines.CoroutineContext

class OmniInferService : Service() {
    companion object {
        private const val TAG = "OmniInferService"
        private const val NOTIFICATION_ID = 9099
        private const val CHANNEL_ID = "omniinfer_server"
    }

    private var server: EmbeddedServer<CIOApplicationEngine, CIOApplicationEngine.Configuration>? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val port = intent?.getIntExtra("port", 0) ?: 0
        if (port > 0 && server == null) {
            promoteToForeground(port)
            startServer(port)
        }
        return START_NOT_STICKY
    }

    private fun promoteToForeground(port: Int) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID, "OmniInfer Server", NotificationManager.IMPORTANCE_LOW
            ).apply { description = "Local inference server" }
            getSystemService(NotificationManager::class.java).createNotificationChannel(channel)
        }
        val notification = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            Notification.Builder(this, CHANNEL_ID)
        } else {
            @Suppress("DEPRECATION") Notification.Builder(this)
        }
            .setContentTitle("OmniInfer Server")
            .setContentText("Running on port $port")
            .setSmallIcon(android.R.drawable.ic_menu_manage)
            .setOngoing(true)
            .build()

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            startForeground(NOTIFICATION_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE)
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }
    }

    private fun startServer(port: Int) {
        try {
            // CoroutineExceptionHandler prevents Ktor's async BindException from
            // propagating to the Android framework as FATAL EXCEPTION.
            val handler = CoroutineExceptionHandler { _, throwable ->
                Log.e(TAG, "Server engine error on port $port: ${throwable.message}")
            }
            val config = serverConfig {
                parentCoroutineContext = handler
                module {
                    routing {
                        get("/health") {
                            call.respondText("{\"status\":\"ok\"}", ContentType.Application.Json)
                        }

                        get("/v1/models") {
                            val models = OmniInferServer.getLoadedModels()
                            val json = buildJsonObject {
                                put("object", "list")
                                putJsonArray("data") {
                                    models.forEach { m ->
                                        addJsonObject {
                                            put("id", m)
                                            put("object", "model")
                                        }
                                    }
                                }
                            }
                            call.respondText(json.toString(), ContentType.Application.Json)
                        }

                        post("/v1/chat/completions") {
                            try {
                                handleChatCompletion(call)
                            } catch (e: Exception) {
                                Log.e(TAG, "chat/completions error", e)
                                call.respondText(
                                    buildJsonObject { put("error", e.message ?: "internal error") }.toString(),
                                    ContentType.Application.Json,
                                    HttpStatusCode.InternalServerError
                                )
                            }
                        }

                        post("/v1/cancel") {
                            val handle = OmniInferServer.currentHandle
                            if (handle != 0L) {
                                OmniInferBridge.gracefulStop(handle)
                                call.respondText("""{"status":"ok"}""", ContentType.Application.Json)
                            } else {
                                call.respondText("""{"error":"no model loaded"}""",
                                    ContentType.Application.Json, HttpStatusCode.ServiceUnavailable)
                            }
                        }
                    }
                }
            }
            server = embeddedServer(CIO, config) {
                connector { this.port = port; this.host = "127.0.0.1" }
            }.also { it.start(wait = false) }
            Log.i(TAG, "Server started on port $port")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start server on port $port: ${e.message}")
            server = null
        }
    }

    private suspend fun handleChatCompletion(call: ApplicationCall) {
        val body = call.receiveText()
        val req = Json.parseToJsonElement(body).jsonObject

        val messages = req["messages"]?.jsonArray ?: run {
            call.respondText("{\"error\":\"missing messages\"}", ContentType.Application.Json, HttpStatusCode.BadRequest)
            return
        }
        val stream = req["stream"]?.jsonPrimitive?.booleanOrNull ?: false
        val requestModel = req["model"]?.jsonPrimitive?.contentOrNull ?: "omniinfer"
        val includeUsage = req["stream_options"]?.jsonObject?.get("include_usage")?.jsonPrimitive?.booleanOrNull ?: true

        // Thinking mode: support both reasoning_effort and enable_thinking (mutually exclusive).
        val reasoningEffort = req["reasoning_effort"]?.jsonPrimitive?.contentOrNull
        val enableThinking = req["enable_thinking"]?.jsonPrimitive?.booleanOrNull
        if (reasoningEffort != null && enableThinking != null) {
            call.respondText("{\"error\":\"reasoning_effort and enable_thinking are mutually exclusive\"}", ContentType.Application.Json, HttpStatusCode.BadRequest)
            return
        }
        val thinkEnabled = when {
            enableThinking != null -> enableThinking
            reasoningEffort != null -> reasoningEffort != "none"
            else -> false // default off on mobile for speed
        }

        // Max tokens (OpenAI uses max_completion_tokens, we also accept max_tokens).
        val maxTokens = req["max_completion_tokens"]?.jsonPrimitive?.intOrNull
            ?: req["max_tokens"]?.jsonPrimitive?.intOrNull

        // Tools support.
        val toolsJson = req["tools"]?.jsonArray?.toString()
        val toolChoice = req["tool_choice"]?.jsonPrimitive?.contentOrNull

        // Build normalized messages and extract all images in one pass.
        // Image positions are preserved as <image> placeholders in the content text.
        // The C++ backend replaces each <image> with its native marker.
        val imageDataList = mutableListOf<ByteArray>()
        val normalizedMessages = buildJsonArray {
            for (msg in messages) {
                val role = msg.jsonObject["role"]?.jsonPrimitive?.contentOrNull ?: continue
                val contentEl = msg.jsonObject["content"]
                // For array content (may contain text + image_url parts):
                // extract text and insert <image> placeholder for each image.
                val content = if (contentEl is JsonArray) {
                    buildString {
                        for (part in contentEl) {
                            when (part.jsonObject["type"]?.jsonPrimitive?.contentOrNull) {
                                "text" -> append(part.jsonObject["text"]?.jsonPrimitive?.contentOrNull ?: "")
                                "image_url" -> {
                                    val url = part.jsonObject["image_url"]?.jsonObject?.get("url")?.jsonPrimitive?.contentOrNull
                                    if (url != null && url.startsWith("data:")) {
                                        val base64Part = url.substringAfter("base64,", "")
                                        if (base64Part.isNotEmpty()) {
                                            imageDataList.add(android.util.Base64.decode(base64Part, android.util.Base64.DEFAULT))
                                            append("<image>")
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else {
                    extractTextContent(contentEl) ?: ""
                }
                addJsonObject {
                    put("role", role)
                    put("content", content)
                    // Preserve tool_calls array on assistant messages.
                    msg.jsonObject["tool_calls"]?.jsonArray?.let { put("tool_calls", it) }
                    // Preserve tool_call_id and name on tool-result messages.
                    msg.jsonObject["tool_call_id"]?.jsonPrimitive?.let { put("tool_call_id", it) }
                    if (role == "tool") {
                        msg.jsonObject["name"]?.jsonPrimitive?.let { put("name", it) }
                    }
                }
            }
        }
        val imageDataArray: Array<ByteArray>? = if (imageDataList.isNotEmpty()) imageDataList.toTypedArray() else null

        val hasUser = messages.any { it.jsonObject["role"]?.jsonPrimitive?.contentOrNull == "user" }
        if (!hasUser) {
            call.respondText("{\"error\":\"no user message\"}", ContentType.Application.Json, HttpStatusCode.BadRequest)
            return
        }

        val handle = OmniInferServer.currentHandle
        if (handle == 0L) {
            call.respondText("{\"error\":\"no model loaded\"}", ContentType.Application.Json, HttpStatusCode.ServiceUnavailable)
            return
        }

        val completionId = "chatcmpl-${UUID.randomUUID()}"
        val created = System.currentTimeMillis() / 1000

        // Shared state across callbacks.
        var metricsStr: String? = null

        if (stream) {
            call.respondTextWriter(contentType = ContentType.Text.EventStream) {
                var isFirst = true
                var inReasoning = false
                val thinkEndBuf = StringBuilder()
                val hasTools = toolsJson != null
                var connectionAlive = true
                var inToolCall = false  // true once tool call marker detected in stream
                val contentBuf = StringBuilder()  // small lookback buffer for marker detection

                val result = try {
                    OmniInferBridge.generate(
                    handle = handle,
                    messagesJson = normalizedMessages.toString(),
                    imageDataArray = imageDataArray,
                    thinkEnabled = thinkEnabled,
                    toolsJson = toolsJson,
                    toolChoice = toolChoice,
                    maxTokens = maxTokens,
                    callback = OmniInferStreamCallback(
                        onTokenHandler = { token ->
                            if (connectionAlive) {
                                try { runBlocking {
                                    if (isFirst) {
                                        val initChunk = buildChunk(completionId, requestModel, created) {
                                            put("role", "assistant")
                                            put("content", "")
                                        }
                                        write("data: $initChunk\n\n")
                                        flush()
                                        isFirst = false
                                    }

                                    if (token.trim() == "<think>") {
                                        inReasoning = true
                                        return@runBlocking
                                    }

                                    if (inReasoning) {
                                        thinkEndBuf.append(token)
                                        val buf = thinkEndBuf.toString()
                                        val endIdx = buf.indexOf("</think>")
                                        if (endIdx >= 0) {
                                            val reasonPart = buf.substring(0, endIdx)
                                            if (reasonPart.isNotEmpty()) {
                                                val chunk = buildChunk(completionId, requestModel, created) {
                                                    put("reasoning_content", reasonPart)
                                                    put("content", JsonNull)
                                                }
                                                write("data: $chunk\n\n")
                                            }
                                            inReasoning = false
                                            val contentPart = buf.substring(endIdx + "</think>".length)
                                            if (contentPart.isNotBlank()) {
                                                val chunk = buildChunk(completionId, requestModel, created) {
                                                    put("content", contentPart)
                                                }
                                                write("data: $chunk\n\n")
                                            }
                                            thinkEndBuf.clear()
                                        } else {
                                            val splitAt = safeSplit(buf, if (buf.length > 10) buf.length - 10 else 0)
                                            if (splitAt > 0) {
                                                val chunk = buildChunk(completionId, requestModel, created) {
                                                    put("reasoning_content", buf.substring(0, splitAt))
                                                    put("content", JsonNull)
                                                }
                                                write("data: $chunk\n\n")
                                                thinkEndBuf.clear()
                                                thinkEndBuf.append(buf.substring(splitAt))
                                            }
                                        }
                                    } else if (!inToolCall) {
                                        // Detect tool call start markers in content stream.
                                        if (hasTools) {
                                            contentBuf.append(token)
                                            // Check for known tool call markers.
                                            if (contentBuf.contains("<|tool_call") || contentBuf.contains("<tool_call")) {
                                                inToolCall = true
                                                return@runBlocking
                                            }
                                            // Flush content up to a safe point (keep tail for marker detection).
                                            val buf = contentBuf.toString()
                                            val splitAt = safeSplit(buf, if (buf.length > 15) buf.length - 15 else 0)
                                            if (splitAt > 0) {
                                                val chunk = buildChunk(completionId, requestModel, created) {
                                                    put("content", buf.substring(0, splitAt))
                                                }
                                                write("data: $chunk\n\n")
                                                contentBuf.clear()
                                                contentBuf.append(buf.substring(splitAt))
                                            }
                                        } else {
                                            val chunk = buildChunk(completionId, requestModel, created) {
                                                put("content", token)
                                            }
                                            write("data: $chunk\n\n")
                                        }
                                    }
                                    // inToolCall == true: suppress content, C++ will parse tool calls after generate()
                                    flush()
                                } } catch (_: Exception) {
                                    // Client disconnected — cancel backend generation.
                                    connectionAlive = false
                                    OmniInferBridge.cancel(handle)
                                }
                            }
                        },
                        onMetricsHandler = { metrics ->
                            metricsStr = metrics
                        }
                    )
                )
                } catch (e: Exception) {
                    // Client disconnected or write failed — cancel the running generate.
                    Log.w(TAG, "Stream interrupted: ${e.message}")
                    OmniInferBridge.cancel(handle)
                    connectionAlive = false
                    ""
                }

                if (!connectionAlive) return@respondTextWriter

                // Check for backend error (e.g. prompt too long).
                if (result.startsWith("{\"error\":")) {
                    write("data: ${buildJsonObject {
                        put("id", completionId)
                        put("object", "chat.completion.chunk")
                        put("model", requestModel)
                        put("error", Json.parseToJsonElement(result).jsonObject)
                    }}\n\n")
                    write("data: [DONE]\n\n")
                    flush()
                    return@respondTextWriter
                }

                // Flush remaining reasoning buffer as content if model ended without </think>.
                if (inReasoning && thinkEndBuf.isNotEmpty()) {
                    val chunk = buildChunk(completionId, requestModel, created) {
                        put("content", thinkEndBuf.toString())
                    }
                    write("data: $chunk\n\n")
                    flush()
                    thinkEndBuf.clear()
                }

                // Flush remaining content buffer (when tools present but no tool call detected).
                if (!inToolCall && contentBuf.isNotEmpty()) {
                    val chunk = buildChunk(completionId, requestModel, created) {
                        put("content", contentBuf.toString())
                    }
                    write("data: $chunk\n\n")
                    flush()
                    contentBuf.clear()
                }

                // After generate: check if result contains tool calls.
                val streamToolCalls = if (hasTools) {
                    runCatching {
                        val parsed = Json.parseToJsonElement(result).jsonObject
                        parsed["tool_calls"]?.jsonArray
                    }.getOrNull()
                } else null

                // If tool calls detected, emit incremental tool_calls delta chunks.
                if (streamToolCalls != null) {
                    for ((idx, tc) in streamToolCalls.withIndex()) {
                        val tcObj = tc.jsonObject
                        val fn = tcObj["function"]?.jsonObject
                        val tcId = tcObj["id"]?.jsonPrimitive?.contentOrNull ?: "call_$idx"
                        val tcName = fn?.get("name")?.jsonPrimitive?.contentOrNull ?: ""
                        val tcArgs = fn?.get("arguments")?.toString() ?: "{}"

                        // First chunk: id + name + empty arguments.
                        write("data: ${buildToolCallChunk(completionId, requestModel, created, idx, tcId, tcName, "")}\n\n")

                        // Incremental argument chunks (split into small pieces).
                        var i = 0
                        while (i < tcArgs.length) {
                            val end = minOf(i + 8, tcArgs.length)
                            val fragment = tcArgs.substring(i, end)
                            write("data: ${buildToolCallChunk(completionId, requestModel, created, idx, "", null, fragment)}\n\n")
                            i = end
                        }

                        // Final empty chunk for this tool call.
                        write("data: ${buildToolCallChunk(completionId, requestModel, created, idx, "", null, "")}\n\n")
                    }
                }

                // Phase 4: FINISH chunk.
                val finishReason = if (streamToolCalls != null) "tool_calls" else "stop"
                val finishChunk = buildJsonObject {
                    put("id", completionId)
                    put("object", "chat.completion.chunk")
                    put("model", requestModel)
                    put("created", created)
                    putJsonArray("choices") {
                        addJsonObject {
                            putJsonObject("delta") {}
                            put("index", 0)
                            put("finish_reason", finishReason)
                        }
                    }
                }
                write("data: $finishChunk\n\n")

                // Phase 5: USAGE chunk (choices=[]).
                if (includeUsage) {
                    val usageChunk = buildJsonObject {
                        put("id", completionId)
                        put("object", "chat.completion.chunk")
                        put("model", requestModel)
                        put("created", created)
                        putJsonArray("choices") {} // empty
                        buildUsageObject(handle, metricsStr)?.let { put("usage", it) }
                    }
                    write("data: $usageChunk\n\n")
                }

                write("data: [DONE]\n\n")
                flush()
            }
        } else {
            val result = OmniInferBridge.generate(
                handle = handle,
                messagesJson = normalizedMessages.toString(),
                imageDataArray = imageDataArray,
                thinkEnabled = thinkEnabled,
                toolsJson = toolsJson,
                toolChoice = toolChoice,
                maxTokens = maxTokens,
                callback = OmniInferStreamCallback(
                    onMetricsHandler = { metrics ->
                        metricsStr = metrics
                    }
                )
            )

            // Check for backend error (e.g. prompt too long).
            if (result.startsWith("{\"error\":")) {
                call.respondText(result, ContentType.Application.Json, HttpStatusCode.BadRequest)
                return
            }

            val toolCallResult = runCatching {
                val parsed = Json.parseToJsonElement(result).jsonObject
                parsed["tool_calls"]?.jsonArray
            }.getOrNull()

            // Parse <think>...</think> tags for non-streaming thinking output.
            val (reasoningContent, contentText) = if (toolCallResult == null) {
                parseThinkingTags(result)
            } else {
                null to null
            }

            val resp = buildJsonObject {
                put("id", completionId)
                put("object", "chat.completion")
                put("model", requestModel)
                put("created", created)
                putJsonArray("choices") {
                    addJsonObject {
                        putJsonObject("message") {
                            put("role", "assistant")
                            if (toolCallResult != null) {
                                put("content", JsonNull)
                                put("tool_calls", toolCallResult)
                            } else if (reasoningContent != null) {
                                put("reasoning_content", reasoningContent)
                                put("content", contentText ?: "")
                            } else {
                                put("content", result)
                            }
                        }
                        put("index", 0)
                        put("finish_reason", if (toolCallResult != null) "tool_calls" else "stop")
                    }
                }
                buildUsageObject(handle, metricsStr)?.let { put("usage", it) }
            }
            call.respondText(resp.toString(), ContentType.Application.Json)
        }
    }

    private fun buildChunk(
        id: String, model: String, created: Long,
        deltaBuilder: JsonObjectBuilder.() -> Unit
    ): JsonObject {
        return buildJsonObject {
            put("id", id)
            put("object", "chat.completion.chunk")
            put("model", model)
            put("created", created)
            putJsonArray("choices") {
                addJsonObject {
                    putJsonObject("delta", deltaBuilder)
                    put("index", 0)
                    put("finish_reason", JsonNull)
                }
            }
        }
    }

    private fun buildToolCallChunk(
        id: String, model: String, created: Long,
        index: Int, tcId: String, name: String?, arguments: String
    ): JsonObject {
        return buildJsonObject {
            put("id", id)
            put("object", "chat.completion.chunk")
            put("model", model)
            put("created", created)
            putJsonArray("choices") {
                addJsonObject {
                    putJsonObject("delta") {
                        put("content", JsonNull)
                        put("reasoning_content", JsonNull)
                        putJsonArray("tool_calls") {
                            addJsonObject {
                                put("index", index)
                                put("id", tcId)
                                put("type", "function")
                                putJsonObject("function") {
                                    if (name != null) put("name", name)
                                    put("arguments", arguments)
                                }
                            }
                        }
                    }
                    put("index", 0)
                    put("finish_reason", JsonNull)
                }
            }
        }
    }

    private fun parseMetrics(raw: String?): Map<String, Double> {
        if (raw.isNullOrBlank()) return emptyMap()
        return raw.split(",").mapNotNull { part ->
            val kv = part.trim().split("=", limit = 2)
            if (kv.size == 2) kv[0].trim() to (kv[1].trim().toDoubleOrNull() ?: 0.0) else null
        }.toMap()
    }

    private fun buildUsageObject(handle: Long, @Suppress("UNUSED_PARAMETER") metricsStr: String?): JsonObject? {
        val diag = OmniInferBridge.collectDiagnostics(handle)
        val promptTokens = diag["prompt_tokens"]?.toIntOrNull() ?: 0
        val completionTokens = diag["generated_tokens"]?.toIntOrNull() ?: 0
        if (promptTokens == 0 && completionTokens == 0) return null

        val reasoningTokens = diag["reasoning_tokens"]?.toIntOrNull() ?: 0
        val imageTokens = diag["image_tokens"]?.toIntOrNull() ?: 0
        val cachedTokens = diag["cached_tokens"]?.toIntOrNull() ?: 0
        val prefillUs = diag["prefill_us"]?.toLongOrNull() ?: 0L
        val decodeUs = diag["decode_us"]?.toLongOrNull() ?: 0L

        return buildJsonObject {
            put("prompt_tokens", promptTokens)
            put("completion_tokens", completionTokens)
            put("total_tokens", promptTokens + completionTokens)

            if (reasoningTokens > 0) {
                putJsonObject("completion_tokens_details") {
                    put("reasoning_tokens", reasoningTokens)
                    put("text_tokens", completionTokens - reasoningTokens)
                }
            }
            putJsonObject("prompt_tokens_details") {
                if (imageTokens > 0) put("image_tokens", imageTokens)
                put("text_tokens", promptTokens - imageTokens)
                put("cached_tokens", cachedTokens)
                put("cache_creation_input_tokens", promptTokens - cachedTokens)
                put("cache_type", "ephemeral")
            }

            val prefillMs = prefillUs / 1000.0
            val decodeMs = decodeUs / 1000.0
            putJsonObject("performance") {
                put("prefill_time_ms", "%.1f".format(prefillMs).toDouble())
                if (prefillUs > 0 && promptTokens > 0) {
                    put("prefill_tokens_per_second", "%.1f".format(promptTokens / (prefillUs / 1e6)).toDouble())
                }
                put("decode_time_ms", "%.1f".format(decodeMs).toDouble())
                if (decodeUs > 0 && completionTokens > 0) {
                    put("decode_tokens_per_second", "%.1f".format(completionTokens / (decodeUs / 1e6)).toDouble())
                }
                put("total_time_ms", "%.1f".format(prefillMs + decodeMs).toDouble())
                put("time_to_first_token_ms", "%.1f".format(prefillMs).toDouble())
            }
        }
    }

    /**
     * Parse <think>...</think> from raw model output.
     * Returns (reasoningContent, content) if tags found, or (null, null) if not.
     */
    private fun parseThinkingTags(raw: String): Pair<String?, String?> {
        val thinkStart = raw.indexOf("<think>")
        if (thinkStart < 0) return null to null
        val contentAfterTag = raw.substring(thinkStart + "<think>".length)
        val thinkEnd = contentAfterTag.indexOf("</think>")
        return if (thinkEnd >= 0) {
            val reasoning = contentAfterTag.substring(0, thinkEnd).trim()
            val content = contentAfterTag.substring(thinkEnd + "</think>".length).trim()
            reasoning to content
        } else {
            // No closing tag — treat entire remainder as reasoning.
            contentAfterTag.trim() to ""
        }
    }

    /** Adjust a tentative split index so it doesn't land between a surrogate pair. */
    private fun safeSplit(s: String, index: Int): Int {
        if (index <= 0 || index >= s.length) return index
        return if (Character.isLowSurrogate(s[index])) index - 1 else index
    }

    private fun extractTextContent(content: JsonElement?): String? {
        if (content == null || content is JsonNull) return null
        if (content is JsonPrimitive) return content.contentOrNull
        if (content is JsonArray) {
            return content
                .filter { it.jsonObject["type"]?.jsonPrimitive?.contentOrNull == "text" }
                .mapNotNull { it.jsonObject["text"]?.jsonPrimitive?.contentOrNull }
                .joinToString("")
                .ifEmpty { null }
        }
        return null
    }

    override fun onDestroy() {
        server?.stop(gracePeriodMillis = 500, timeoutMillis = 1000)
        server = null
        Log.i(TAG, "Server stopped")
        super.onDestroy()
    }
}
