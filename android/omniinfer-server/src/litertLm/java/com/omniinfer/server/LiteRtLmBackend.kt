package com.omniinfer.server

import android.os.SystemClock
import android.util.Log
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.BenchmarkInfo
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.ExperimentalApi
import com.google.ai.edge.litertlm.ExperimentalFlags
import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.MessageCallback
import com.google.ai.edge.litertlm.SamplerConfig
import org.json.JSONArray
import org.json.JSONObject
import java.util.concurrent.CancellationException
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.Locale
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicReference

@OptIn(ExperimentalApi::class)
internal class LiteRtLmBackend private constructor(
    private val modelPath: String,
    private val backendName: String,
    private val nThreads: Int,
    private val nCtx: Int,
    private val cacheDir: String?,
    private val visionBackendName: String?,
    private val maxImages: Int,
    private val speculativeDecoding: Boolean,
    private val engine: Engine,
    private val engineInitMs: Double,
) : LiteRtLmSession {
    private val lock = Any()
    private var activeConversation: Conversation? = null
    private var lastDiagnostics: Map<String, String> = baseDiagnostics()

    override fun generate(
        messagesJson: String,
        imageDataArray: Array<ByteArray>?,
        requestJson: String,
        callback: OmniInferStreamCallback?,
    ): String {
        Log.i(TAG, "generate_start backend=$backendName messagesChars=${messagesJson.length} requestChars=${requestJson.length}")
        val imageCount = imageDataArray?.size ?: 0
        if (imageCount > 0 && visionBackendName == null) {
            return """{"error":{"message":"LiteRT-LM image input requires loading the model with extraConfig vision_backend=cpu|gpu|npu"}}"""
        }
        if (imageCount > maxImages) {
            return """{"error":{"message":"LiteRT-LM image count $imageCount exceeds max_images=$maxImages"}}"""
        }

        return synchronized(lock) {
            val totalStartNs = SystemClock.elapsedRealtimeNanos()
            var conversation: Conversation? = null
            try {
                val request = JSONObject(requestJson)
                val messages = parseMessages(messagesJson, imageDataArray)
                if (messages.isEmpty()) {
                    return@synchronized """{"error":{"message":"missing messages"}}"""
                }
                Log.i(TAG, "generate_parsed messages=${messages.size}")

                val sampler = samplerConfig(request)
                Log.i(
                    TAG,
                    "sampler_config source=${sampler.source} top_k=${sampler.topK ?: ""} " +
                        "top_p=${sampler.topP ?: ""} temperature=${sampler.temperature ?: ""}"
                )
                val conversationStartNs = SystemClock.elapsedRealtimeNanos()
                Log.i(TAG, "conversation_create_start initialMessages=${(messages.size - 1).coerceAtLeast(0)}")
                conversation = engine.createConversation(
                    ConversationConfig(
                        initialMessages = messages.dropLast(1),
                        samplerConfig = sampler.config,
                    )
                )
                activeConversation = conversation
                val conversationCreateMs = elapsedMs(conversationStartNs)
                Log.i(TAG, "conversation_create_done wallMs=${"%.3f".format(Locale.US, conversationCreateMs)}")

                val sendStartNs = SystemClock.elapsedRealtimeNanos()
                val lastMessage = messages.last()
                val maxTokens = request.optInt("max_tokens", DEFAULT_MAX_DECODE_TOKENS).takeIf { it > 0 }
                    ?: DEFAULT_MAX_DECODE_TOKENS
                Log.i(TAG, "send_message_start lastRole=${lastMessage.role.value} contentChars=${lastMessage.toString().length}")
                val result = sendMessageWithLimit(
                    conversation = conversation,
                    message = lastMessage,
                    maxTokens = maxTokens,
                    callback = callback,
                )
                Log.i(TAG, "send_message_done outputChars=${result.length}")
                val generateMs = elapsedMs(sendStartNs)
                val totalMs = elapsedMs(totalStartNs)
                val benchmarkInfo = runCatching { conversation.getBenchmarkInfo() }.getOrNull()
                lastDiagnostics = buildDiagnostics(
                    benchmarkInfo = benchmarkInfo,
                    conversationCreateMs = conversationCreateMs,
                    generateMs = generateMs,
                    totalMs = totalMs,
                    sampler = sampler,
                    imageDataArray = imageDataArray,
                )
                callback?.onMetrics(metricsString(lastDiagnostics))
                Log.i(TAG, "generate_done totalMs=${"%.3f".format(Locale.US, totalMs)}")
                result
            } catch (t: Throwable) {
                Log.e(TAG, "LiteRT-LM generate failed", t)
                lastDiagnostics = baseDiagnostics() + mapOf(
                    "last_error" to (t.message ?: t::class.java.simpleName),
                    "total_wall_ms" to "%.3f".format(Locale.US, elapsedMs(totalStartNs)),
                )
                """{"error":{"message":${JSONObject.quote(t.message ?: t::class.java.simpleName)}}}"""
            } finally {
                activeConversation = null
                runCatching { conversation?.close() }
            }
        }
    }

    override fun cancel() {
        synchronized(lock) {
            runCatching { activeConversation?.cancelProcess() }
        }
    }

    override fun reset() {
        synchronized(lock) {
            runCatching { activeConversation?.close() }
            activeConversation = null
            lastDiagnostics = baseDiagnostics()
        }
    }

    override fun close() {
        synchronized(lock) {
            runCatching { activeConversation?.close() }
            activeConversation = null
            engine.close()
        }
    }

    override fun diagnostics(): Map<String, String> = synchronized(lock) { lastDiagnostics }

    private fun parseMessages(messagesJson: String, imageDataArray: Array<ByteArray>?): List<Message> {
        val array = JSONArray(messagesJson)
        var imageIndex = 0
        return buildList {
            for (i in 0 until array.length()) {
                val obj = array.optJSONObject(i) ?: continue
                val role = obj.optString("role", "user")
                val text = obj.optString("content", "")
                val contents = contentsFromTextAndImages(text, imageDataArray) { imageIndex++ }
                add(
                    when (role) {
                        "system" -> Message.system(contents)
                        "assistant", "model" -> Message.model(contents)
                        "tool" -> Message.tool(contents)
                        else -> Message.user(contents)
                    }
                )
            }
        }
    }

    private fun contentsFromTextAndImages(
        text: String,
        imageDataArray: Array<ByteArray>?,
        nextImageIndex: () -> Int,
    ): Contents {
        if (imageDataArray.isNullOrEmpty() || !text.contains(IMAGE_MARKER)) {
            return Contents.of(text)
        }
        val parts = mutableListOf<Content>()
        var cursor = 0
        while (cursor < text.length) {
            val markerIndex = text.indexOf(IMAGE_MARKER, cursor)
            if (markerIndex < 0) {
                parts.add(Content.Text(text.substring(cursor)))
                break
            }
            if (markerIndex > cursor) {
                parts.add(Content.Text(text.substring(cursor, markerIndex)))
            }
            val imageIndex = nextImageIndex()
            if (imageIndex < imageDataArray.size) {
                parts.add(Content.ImageBytes(imageDataArray[imageIndex]))
            }
            cursor = markerIndex + IMAGE_MARKER.length
        }
        return Contents.of(parts)
    }

    private data class EffectiveSamplerConfig(
        val config: SamplerConfig?,
        val source: String,
        val topK: Int?,
        val topP: Double?,
        val temperature: Double?,
    )

    private fun samplerConfig(request: JSONObject): EffectiveSamplerConfig {
        val topK = request.optInt("top_k", 0)
        val topP = if (request.has("top_p")) request.optDouble("top_p") else Double.NaN
        val temperature = if (request.has("temperature")) request.optDouble("temperature") else Double.NaN
        if (topK <= 0 && topP.isNaN() && temperature.isNaN()) {
            return EffectiveSamplerConfig(
                config = null,
                source = "litert-default",
                topK = null,
                topP = null,
                temperature = null,
            )
        }
        val effectiveTopK = if (topK > 0) topK else DEFAULT_TOP_K
        val effectiveTopP = if (!topP.isNaN()) topP else DEFAULT_TOP_P
        val effectiveTemperature = if (!temperature.isNaN()) temperature else DEFAULT_TEMPERATURE
        return EffectiveSamplerConfig(
            config = SamplerConfig(
                topK = effectiveTopK,
                topP = effectiveTopP,
                temperature = effectiveTemperature,
            ),
            source = "request",
            topK = effectiveTopK,
            topP = effectiveTopP,
            temperature = effectiveTemperature,
        )
    }

    private fun sendMessageWithLimit(
        conversation: Conversation,
        message: Message,
        maxTokens: Int,
        callback: OmniInferStreamCallback?,
    ): String {
        val done = CountDownLatch(1)
        val output = StringBuilder()
        val chunkCount = AtomicInteger(0)
        val errorRef = AtomicReference<Throwable?>(null)
        conversation.sendMessageAsync(
            message,
            object : MessageCallback {
                override fun onMessage(message: Message) {
                    val chunk = message.toString()
                    if (chunk.isNotEmpty()) {
                        synchronized(output) { output.append(chunk) }
                        callback?.onToken(chunk)
                        if (chunkCount.incrementAndGet() >= maxTokens) {
                            Thread { runCatching { conversation.cancelProcess() } }.start()
                        }
                    }
                }

                override fun onDone() {
                    done.countDown()
                }

                override fun onError(throwable: Throwable) {
                    if (throwable !is CancellationException) {
                        errorRef.set(throwable)
                    }
                    done.countDown()
                }
            },
        )

        val timeoutSeconds = maxOf(DEFAULT_GENERATION_TIMEOUT_SECONDS, maxTokens * 2L)
        if (!done.await(timeoutSeconds, TimeUnit.SECONDS)) {
            runCatching { conversation.cancelProcess() }
            throw IllegalStateException("LiteRT-LM generation timed out after ${timeoutSeconds}s")
        }
        errorRef.get()?.let { throw it }
        return synchronized(output) { output.toString() }
    }

    private fun buildDiagnostics(
        benchmarkInfo: BenchmarkInfo?,
        conversationCreateMs: Double,
        generateMs: Double,
        totalMs: Double,
        sampler: EffectiveSamplerConfig,
        imageDataArray: Array<ByteArray>?,
    ): Map<String, String> {
        val prefillTokens = benchmarkInfo?.lastPrefillTokenCount ?: 0
        val decodeTokens = benchmarkInfo?.lastDecodeTokenCount ?: 0
        val prefillTps = benchmarkInfo?.lastPrefillTokensPerSecond ?: 0.0
        val decodeTps = benchmarkInfo?.lastDecodeTokensPerSecond ?: 0.0
        val prefillUs = secondsToMicros(if (prefillTps > 0.0) prefillTokens / prefillTps else 0.0)
        val decodeUs = secondsToMicros(if (decodeTps > 0.0) decodeTokens / decodeTps else 0.0)
        return baseDiagnostics() + mapOf(
            "prompt_tokens" to prefillTokens.toString(),
            "generated_tokens" to decodeTokens.toString(),
            "image_count" to (imageDataArray?.size ?: 0).toString(),
            "image_bytes_total" to (imageDataArray?.sumOf { it.size } ?: 0).toString(),
            "prefill_us" to prefillUs.toString(),
            "decode_us" to decodeUs.toString(),
            "cached_tokens" to "0",
            "litert_init_s" to (benchmarkInfo?.initTimeInSecond?.toString() ?: ""),
            "litert_ttft_s" to (benchmarkInfo?.timeToFirstTokenInSecond?.toString() ?: ""),
            "litert_prefill_tps" to prefillTps.toString(),
            "litert_decode_tps" to decodeTps.toString(),
            "engine_init_ms" to "%.3f".format(Locale.US, engineInitMs),
            "conversation_create_ms" to "%.3f".format(Locale.US, conversationCreateMs),
            "generate_wall_ms" to "%.3f".format(Locale.US, generateMs),
            "total_wall_ms" to "%.3f".format(Locale.US, totalMs),
            "sampler_source" to sampler.source,
            "sampler_top_k" to (sampler.topK?.toString() ?: ""),
            "sampler_top_p" to (sampler.topP?.toString() ?: ""),
            "sampler_temperature" to (sampler.temperature?.toString() ?: ""),
        )
    }

    private fun baseDiagnostics(): Map<String, String> = mapOf(
        "backend" to backendName,
        "model_path" to modelPath,
        "n_threads" to nThreads.toString(),
        "n_ctx" to nCtx.toString(),
        "cache_dir" to (cacheDir ?: ""),
        "vision_backend" to (visionBackendName ?: ""),
        "max_images" to maxImages.toString(),
        "speculative_decoding" to speculativeDecoding.toString(),
    )

    companion object {
        private const val TAG = "LiteRtLmBackend"
        private const val IMAGE_MARKER = "<image>"
        private const val DEFAULT_TOP_K = 64
        private const val DEFAULT_TOP_P = 0.95
        private const val DEFAULT_TEMPERATURE = 1.0
        private const val DEFAULT_MAX_DECODE_TOKENS = 512
        private const val DEFAULT_GENERATION_TIMEOUT_SECONDS = 120L

        fun create(
            modelPath: String,
            backend: String,
            nThreads: Int,
            nCtx: Int,
            nativeLibDir: String?,
            cacheDir: String?,
            extraConfig: Map<String, String>?,
        ): LiteRtLmBackend {
            ExperimentalFlags.enableBenchmark = true
            val enableSpeculativeDecoding = extraConfig.booleanConfig(
                "enable_speculative_decoding",
                "speculative_decoding",
                "litert_enable_speculative_decoding",
                default = false,
            )
            ExperimentalFlags.enableSpeculativeDecoding = enableSpeculativeDecoding
            val liteRtBackend = selectBackend(backend, nThreads, nativeLibDir, extraConfig)
            val visionBackend = selectVisionBackend(liteRtBackend, nThreads, nativeLibDir, extraConfig)
            val maxImages = extraConfig.intConfig("max_images", "litert_max_images", default = 1).coerceAtLeast(1)
            val effectiveCacheDir = prepareCacheDir(cacheDir)
            val effectiveBackendName = liteRtBackend.name
            val effectiveVisionBackendName = visionBackend?.name
            val initStartNs = SystemClock.elapsedRealtimeNanos()
            val engine = Engine(
                EngineConfig(
                    modelPath = modelPath,
                    backend = liteRtBackend,
                    visionBackend = visionBackend,
                    maxNumTokens = nCtx.takeIf { it > 0 },
                    maxNumImages = if (visionBackend != null) maxImages else null,
                    cacheDir = effectiveCacheDir,
                )
            )
            try {
                engine.initialize()
            } finally {
                ExperimentalFlags.enableSpeculativeDecoding = false
            }
            val initMs = elapsedMs(initStartNs)
            Log.i(
                TAG,
                "created backend=litert-lm/$effectiveBackendName visionBackend=${effectiveVisionBackendName ?: "none"} " +
                    "nCtx=$nCtx cacheDir=${effectiveCacheDir ?: ""} speculativeDecoding=$enableSpeculativeDecoding"
            )
            return LiteRtLmBackend(
                modelPath = modelPath,
                backendName = "litert-lm/$effectiveBackendName",
                nThreads = if (liteRtBackend is Backend.CPU) nThreads else 0,
                nCtx = nCtx,
                cacheDir = effectiveCacheDir,
                visionBackendName = effectiveVisionBackendName,
                maxImages = maxImages,
                speculativeDecoding = enableSpeculativeDecoding,
                engine = engine,
                engineInitMs = initMs,
            )
        }

        fun isLiteRtBackend(backend: String): Boolean {
            val normalized = backend.lowercase(Locale.US)
            return normalized == "litert" ||
                normalized == "litertlm" ||
                normalized == "litert-lm" ||
                normalized == "litert-gpu" ||
                normalized == "litertlm-gpu" ||
                normalized == "litert-lm-gpu"
        }

        private fun selectBackend(
            backend: String,
            nThreads: Int,
            nativeLibDir: String?,
            extraConfig: Map<String, String>?,
        ): Backend {
            val explicitType = extraConfig?.get("backend_type")
                ?: extraConfig?.get("litert_backend")
                ?: extraConfig?.get("gpu_mode")
            val normalized = (explicitType ?: backend).lowercase(Locale.US)
            return when {
                normalized.contains("gpu") -> Backend.GPU()
                normalized.contains("npu") -> Backend.NPU(nativeLibDir ?: "")
                else -> Backend.CPU(numOfThreads = nThreads.takeIf { it > 0 })
            }
        }

        private fun selectVisionBackend(
            defaultBackend: Backend,
            nThreads: Int,
            nativeLibDir: String?,
            extraConfig: Map<String, String>?,
        ): Backend? {
            val explicitType = extraConfig?.get("vision_backend")
                ?: extraConfig?.get("litert_vision_backend")
            if (explicitType.isNullOrBlank()) {
                val enabled = extraConfig.booleanConfig(
                    "enable_vision",
                    "enable_multimodal",
                    "litert_enable_vision",
                    default = false,
                )
                return if (enabled) defaultBackend else null
            }
            return when (explicitType.lowercase(Locale.US)) {
                "none", "off", "false", "disabled" -> null
                "gpu" -> Backend.GPU()
                "npu" -> Backend.NPU(nativeLibDir ?: "")
                else -> Backend.CPU(numOfThreads = nThreads.takeIf { it > 0 })
            }
        }

        private fun prepareCacheDir(cacheDir: String?): String? {
            if (cacheDir.isNullOrBlank() || cacheDir == ":nocache") {
                return cacheDir
            }
            val dir = java.io.File(cacheDir)
            val ready = dir.exists() || dir.mkdirs()
            Log.i(
                TAG,
                "cache_dir_ready path=$cacheDir exists=${dir.exists()} mkdirsResult=$ready " +
                    "canRead=${dir.canRead()} canWrite=${dir.canWrite()}"
            )
            return cacheDir
        }

        private fun Map<String, String>?.booleanConfig(vararg keys: String, default: Boolean): Boolean {
            val value = keys.firstNotNullOfOrNull { this?.get(it) } ?: return default
            return when (value.lowercase(Locale.US)) {
                "1", "true", "yes", "on", "enabled" -> true
                "0", "false", "no", "off", "disabled" -> false
                else -> default
            }
        }

        private fun Map<String, String>?.intConfig(vararg keys: String, default: Int): Int {
            return keys.firstNotNullOfOrNull { this?.get(it)?.toIntOrNull() } ?: default
        }

        private fun metricsString(values: Map<String, String>): String =
            values.entries.joinToString(",") { "${it.key}=${it.value}" }

        private fun secondsToMicros(seconds: Double): Long = (seconds * 1_000_000.0).toLong()

        private fun elapsedMs(startNs: Long): Double =
            (SystemClock.elapsedRealtimeNanos() - startNs) / 1_000_000.0
    }
}
