package com.omniinfer.server

import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.IBinder
import android.util.Log
import io.ktor.http.*
import io.ktor.server.application.*
import io.ktor.server.cio.*
import io.ktor.server.engine.*
import io.ktor.server.request.*
import io.ktor.server.response.*
import io.ktor.server.routing.*
import kotlinx.coroutines.*
import kotlinx.serialization.json.*
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference

class OmniInferService : Service() {
    companion object {
        private const val TAG = "OmniInferService"
    }

    private var server: ApplicationEngine? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val port = intent?.getIntExtra("port", 0) ?: 0
        if (port > 0 && server == null) {
            startServer(port)
        }
        return START_NOT_STICKY
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
                    val body = call.receiveText()
                    val req = Json.parseToJsonElement(body).jsonObject

                    val messages = req["messages"]?.jsonArray ?: run {
                        call.respondText("{\"error\":\"missing messages\"}", ContentType.Application.Json, HttpStatusCode.BadRequest)
                        return@post
                    }
                    val stream = req["stream"]?.jsonPrimitive?.booleanOrNull ?: false

                    // Extract system prompt and last user message.
                    var systemPrompt: String? = null
                    var userPrompt = ""
                    for (msg in messages) {
                        val role = msg.jsonObject["role"]?.jsonPrimitive?.contentOrNull ?: continue
                        val content = msg.jsonObject["content"]?.jsonPrimitive?.contentOrNull ?: continue
                        when (role) {
                            "system" -> systemPrompt = content
                            "user" -> userPrompt = content
                        }
                    }

                    if (userPrompt.isEmpty()) {
                        call.respondText("{\"error\":\"no user message\"}", ContentType.Application.Json, HttpStatusCode.BadRequest)
                        return@post
                    }

                    val handle = OmniInferServer.currentHandle
                    if (handle == 0L) {
                        call.respondText("{\"error\":\"no model loaded\"}", ContentType.Application.Json, HttpStatusCode.ServiceUnavailable)
                        return@post
                    }

                    if (stream) {
                        call.respondTextWriter(contentType = ContentType.Text.EventStream) {
                            val result = OmniInferBridge.generate(
                                handle = handle,
                                systemPrompt = systemPrompt,
                                prompt = userPrompt,
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
                                }
                            )
                            write("data: [DONE]\n\n")
                            flush()
                        }
                    } else {
                        val result = OmniInferBridge.generate(
                            handle = handle,
                            systemPrompt = systemPrompt,
                            prompt = userPrompt
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
                        }
                        call.respondText(resp.toString(), ContentType.Application.Json)
                    }
                }
            }
        }.also { it.start(wait = false) }
        Log.i(TAG, "Server started on port $port")
    }

    override fun onDestroy() {
        server?.stop(500, 1000, TimeUnit.MILLISECONDS)
        server = null
        Log.i(TAG, "Server stopped")
        super.onDestroy()
    }
}
