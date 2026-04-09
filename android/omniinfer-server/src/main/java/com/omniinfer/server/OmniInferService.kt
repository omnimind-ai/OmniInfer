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
import io.ktor.server.request.*
import io.ktor.server.response.*
import io.ktor.server.routing.*
import io.ktor.server.application.*
import kotlinx.coroutines.*
import kotlinx.serialization.json.*
import java.util.UUID

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
        server = embeddedServer(CIO, port = port, host = "127.0.0.1") {
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
            }
        }.also { it.start(wait = false) }
        Log.i(TAG, "Server started on port $port")
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
            else -> true // model default
        }

        // Tools support.
        val toolsJson = req["tools"]?.jsonArray?.toString()
        val toolChoice = req["tool_choice"]?.jsonPrimitive?.contentOrNull

        // Build normalized messages array for the backend.
        // Preserve tool_calls on assistant messages and tool role messages for multi-turn tool use.
        val normalizedMessages = buildJsonArray {
            for (msg in messages) {
                val role = msg.jsonObject["role"]?.jsonPrimitive?.contentOrNull ?: continue
                val content = extractTextContent(msg.jsonObject["content"]) ?: ""
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

                val result = try {
                    OmniInferBridge.generate(
                    handle = handle,
                    messagesJson = normalizedMessages.toString(),
                    thinkEnabled = thinkEnabled,
                    toolsJson = toolsJson,
                    toolChoice = toolChoice,
                    callback = object {
                        @Suppress("unused")
                        fun onToken(token: String) {
                            if (!connectionAlive) return

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
                                        val safe = if (buf.length > 10) buf.substring(0, buf.length - 10) else ""
                                        if (safe.isNotEmpty()) {
                                            val chunk = buildChunk(completionId, requestModel, created) {
                                                put("reasoning_content", safe)
                                                put("content", JsonNull)
                                            }
                                            write("data: $chunk\n\n")
                                            thinkEndBuf.clear()
                                            thinkEndBuf.append(buf.substring(buf.length - 10))
                                        }
                                    }
                                } else {
                                    val chunk = buildChunk(completionId, requestModel, created) {
                                        put("content", token)
                                    }
                                    write("data: $chunk\n\n")
                                }
                                flush()
                            } } catch (_: Exception) {
                                // Client disconnected — cancel backend generation.
                                connectionAlive = false
                                OmniInferBridge.cancel(handle)
                            }
                        }
                        @Suppress("unused")
                        fun onMetrics(metrics: String) {
                            metricsStr = metrics
                        }
                    }
                )
                } catch (e: Exception) {
                    // Client disconnected or write failed — cancel the running generate.
                    Log.w(TAG, "Stream interrupted: ${e.message}")
                    OmniInferBridge.cancel(handle)
                    connectionAlive = false
                    ""
                }

                if (!connectionAlive) return@respondTextWriter

                // Flush remaining reasoning buffer as content if model ended without </think>.
                if (inReasoning && thinkEndBuf.isNotEmpty()) {
                    val chunk = buildChunk(completionId, requestModel, created) {
                        put("content", thinkEndBuf.toString())
                    }
                    write("data: $chunk\n\n")
                    flush()
                    thinkEndBuf.clear()
                }

                // After generate: check if result contains tool calls.
                val streamToolCalls = if (hasTools) {
                    runCatching {
                        val parsed = Json.parseToJsonElement(result).jsonObject
                        parsed["tool_calls"]?.jsonArray
                    }.getOrNull()
                } else null

                // If tool calls detected, emit tool_calls delta chunks.
                if (streamToolCalls != null) {
                    for ((idx, tc) in streamToolCalls.withIndex()) {
                        val tcObj = tc.jsonObject
                        val fn = tcObj["function"]?.jsonObject
                        val toolChunk = buildJsonObject {
                            put("id", completionId)
                            put("object", "chat.completion.chunk")
                            put("model", requestModel)
                            put("created", created)
                            putJsonArray("choices") {
                                addJsonObject {
                                    putJsonObject("delta") {
                                        put("content", JsonNull)
                                        putJsonArray("tool_calls") {
                                            addJsonObject {
                                                put("index", idx)
                                                put("id", tcObj["id"]?.jsonPrimitive?.contentOrNull ?: "call_$idx")
                                                put("type", "function")
                                                putJsonObject("function") {
                                                    put("name", fn?.get("name")?.jsonPrimitive?.contentOrNull ?: "")
                                                    put("arguments", fn?.get("arguments")?.toString() ?: "{}")
                                                }
                                            }
                                        }
                                    }
                                    put("index", 0)
                                    put("finish_reason", JsonNull)
                                }
                            }
                        }
                        write("data: $toolChunk\n\n")
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
                thinkEnabled = thinkEnabled,
                toolsJson = toolsJson,
                toolChoice = toolChoice,
                callback = object {
                    @Suppress("unused")
                    fun onMetrics(metrics: String) {
                        metricsStr = metrics
                    }
                }
            )
            val toolCallResult = runCatching {
                val parsed = Json.parseToJsonElement(result).jsonObject
                parsed["tool_calls"]?.jsonArray
            }.getOrNull()

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

    private fun parseMetrics(raw: String?): Map<String, Double> {
        if (raw.isNullOrBlank()) return emptyMap()
        return raw.split(",").mapNotNull { part ->
            val kv = part.trim().split("=", limit = 2)
            if (kv.size == 2) kv[0].trim() to (kv[1].trim().toDoubleOrNull() ?: 0.0) else null
        }.toMap()
    }

    private fun buildUsageObject(handle: Long, metricsStr: String?): JsonObject? {
        val diag = OmniInferBridge.collectDiagnostics(handle)
        val metrics = parseMetrics(metricsStr)
        val promptTokens = diag["prompt_tokens"]?.toIntOrNull() ?: 0
        val completionTokens = diag["generated_tokens"]?.toIntOrNull() ?: 0
        if (promptTokens == 0 && completionTokens == 0) return null
        return buildJsonObject {
            put("prompt_tokens", promptTokens)
            put("completion_tokens", completionTokens)
            put("total_tokens", promptTokens + completionTokens)
            metrics["prefill_tps"]?.let { put("prefill_tokens_per_second", "%.1f".format(it).toDouble()) }
            metrics["decode_tps"]?.let { put("decode_tokens_per_second", "%.1f".format(it).toDouble()) }
        }
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
