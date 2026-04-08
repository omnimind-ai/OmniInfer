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

        // reasoning_effort: "none" disables thinking, anything else enables it.
        val reasoningEffort = req["reasoning_effort"]?.jsonPrimitive?.contentOrNull
        val thinkEnabled = reasoningEffort == null || reasoningEffort != "none"

        // Extract system prompt and last user message.
        var systemPrompt: String? = null
        var userPrompt = ""
        for (msg in messages) {
            val role = msg.jsonObject["role"]?.jsonPrimitive?.contentOrNull ?: continue
            val content = extractTextContent(msg.jsonObject["content"]) ?: continue
            when (role) {
                "system" -> systemPrompt = content
                "user" -> userPrompt = content
            }
        }

        if (userPrompt.isEmpty()) {
            call.respondText("{\"error\":\"no user message\"}", ContentType.Application.Json, HttpStatusCode.BadRequest)
            return
        }

        val handle = OmniInferServer.currentHandle
        if (handle == 0L) {
            call.respondText("{\"error\":\"no model loaded\"}", ContentType.Application.Json, HttpStatusCode.ServiceUnavailable)
            return
        }

        // Shared metrics holder — filled by onMetrics callback from JNI.
        var metricsStr: String? = null

        if (stream) {
            call.respondTextWriter(contentType = ContentType.Text.EventStream) {
                OmniInferBridge.generate(
                    handle = handle,
                    systemPrompt = systemPrompt,
                    prompt = userPrompt,
                    thinkEnabled = thinkEnabled,
                    callback = object {
                        @Suppress("unused")
                        fun onToken(token: String) {
                            val chunk = buildJsonObject {
                                put("object", "chat.completion.chunk")
                                putJsonArray("choices") {
                                    addJsonObject {
                                        putJsonObject("delta") { put("content", token) }
                                        put("index", 0)
                                    }
                                }
                            }
                            runBlocking {
                                write("data: $chunk\n\n")
                                flush()
                            }
                        }
                        @Suppress("unused")
                        fun onMetrics(metrics: String) {
                            metricsStr = metrics
                        }
                    }
                )
                // Final chunk with usage and performance data.
                val finalChunk = buildUsageChunk(handle, metricsStr)
                write("data: $finalChunk\n\n")
                write("data: [DONE]\n\n")
                flush()
            }
        } else {
            val result = OmniInferBridge.generate(
                handle = handle,
                systemPrompt = systemPrompt,
                prompt = userPrompt,
                thinkEnabled = thinkEnabled,
                callback = object {
                    @Suppress("unused")
                    fun onMetrics(metrics: String) {
                        metricsStr = metrics
                    }
                }
            )
            val resp = buildJsonObject {
                put("object", "chat.completion")
                putJsonArray("choices") {
                    addJsonObject {
                        putJsonObject("message") {
                            put("role", "assistant")
                            put("content", result)
                        }
                        put("index", 0)
                        put("finish_reason", "stop")
                    }
                }
                buildUsageObject(handle, metricsStr)?.let { put("usage", it) }
            }
            call.respondText(resp.toString(), ContentType.Application.Json)
        }
    }

    /**
     * Parse "prefill_tps=123.4, decode_tps=22.7" into a map.
     */
    private fun parseMetrics(raw: String?): Map<String, Double> {
        if (raw.isNullOrBlank()) return emptyMap()
        return raw.split(",").mapNotNull { part ->
            val kv = part.trim().split("=", limit = 2)
            if (kv.size == 2) kv[0].trim() to (kv[1].trim().toDoubleOrNull() ?: 0.0) else null
        }.toMap()
    }

    /**
     * Build a usage JsonObject from diagnostics + metrics.
     */
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

    /**
     * Build a final SSE chunk with finish_reason=stop and usage data (OpenAI streaming convention).
     */
    private fun buildUsageChunk(handle: Long, metricsStr: String?): JsonObject {
        return buildJsonObject {
            put("object", "chat.completion.chunk")
            putJsonArray("choices") {
                addJsonObject {
                    putJsonObject("delta") {}
                    put("index", 0)
                    put("finish_reason", "stop")
                }
            }
            buildUsageObject(handle, metricsStr)?.let { put("usage", it) }
        }
    }

    /**
     * Extract text from OpenAI message content which can be either:
     * - A plain string: "content": "hello"
     * - An array of parts: "content": [{"type":"text","text":"hello"}, ...]
     */
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
