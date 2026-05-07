package com.omniinfer.server

import android.util.Log
import org.json.JSONObject
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicLong

object OmniInferBridge {
    private const val TAG = "OmniInferBridge"
    private const val LIB_NAME = "omniinfer-jni"
    private val thinkModes = ConcurrentHashMap<Long, Boolean>()
    private val liteRtHandles = ConcurrentHashMap<Long, LiteRtLmSession>()
    private val nextLiteRtHandle = AtomicLong(-1L)
    @Volatile private var lastError: String = ""

    val isNativeLibraryLoaded: Boolean by lazy {
        runCatching {
            System.loadLibrary(LIB_NAME)
            true
        }.getOrElse { error ->
            Log.w(TAG, "Failed to load $LIB_NAME: ${error.message}")
            false
        }
    }

    /**
     * Initialize a backend session.
     * @param extraConfig additional key-value pairs merged into the config JSON.
     *   Useful for backend-specific settings like "qnn_lib_dir", "decoder_model_version".
     */
    fun init(
        modelPath: String,
        backend: String = "llama.cpp",
        nThreads: Int = 0,
        nCtx: Int = 4096,
        nativeLibDir: String? = null,
        cacheDir: String? = null,
        extraConfig: Map<String, String>? = null
    ): Long {
        lastError = ""
        if (LiteRtLmBackendSupport.isLiteRtBackend(backend)) {
            return runCatching {
                val session = LiteRtLmBackendSupport.create(
                    modelPath = modelPath,
                    backend = backend,
                    nThreads = nThreads,
                    nCtx = nCtx,
                    nativeLibDir = nativeLibDir,
                    cacheDir = cacheDir,
                    extraConfig = extraConfig,
                )
                val handle = nextLiteRtHandle.getAndDecrement()
                liteRtHandles[handle] = session
                handle
            }.getOrElse { error ->
                lastError = error.message ?: "Failed to initialize LiteRT-LM backend"
                Log.e(TAG, "Failed to initialize LiteRT-LM backend: $lastError", error)
                0L
            }
        }
        if (!isNativeLibraryLoaded) {
            lastError = "Native backend library '$LIB_NAME' is unavailable."
            return 0L
        }
        val configJson = JSONObject()
            .put("backend", backend)
            .put("model_path", modelPath)
            .put("native_lib_dir", nativeLibDir ?: "")
            .put("cache_dir", cacheDir ?: "")
            .put("n_threads", nThreads)
            .put("n_ctx", nCtx)
        extraConfig?.forEach { (k, v) -> configJson.put(k, v) }
        val handle = nativeInit(configJson.toString())
        if (handle == 0L) {
            lastError = nativeGetLastError().ifBlank {
                "Failed to initialize backend '$backend'."
            }
        }
        // Don't set default think mode — let each request's thinkEnabled param decide.
        return handle
    }

    fun getLastError(): String = lastError

    fun generate(
        handle: Long,
        messagesJson: String,
        imageDataArray: Array<ByteArray>? = null,
        thinkEnabled: Boolean = false,
        toolsJson: String? = null,
        toolChoice: String? = null,
        maxTokens: Int? = null,
        temperature: Float? = null,
        topP: Float? = null,
        topK: Int? = null,
        repetitionPenalty: Float? = null,
        frequencyPenalty: Float? = null,
        presencePenalty: Float? = null,
        callback: OmniInferStreamCallback? = null
    ): String {
        liteRtHandles[handle]?.let { liteRt ->
            val sb = StringBuilder()
            sb.append("{\"thinking_enabled\":").append(thinkEnabled)
            sb.append(",\"messages\":").append(messagesJson)
            if (toolsJson != null) {
                sb.append(",\"tools\":").append(toolsJson)
                if (toolChoice != null) sb.append(",\"tool_choice\":\"").append(toolChoice).append("\"")
            }
            if (maxTokens != null && maxTokens > 0) sb.append(",\"max_tokens\":").append(maxTokens)
            if (temperature != null) sb.append(",\"temperature\":").append(temperature)
            if (topP != null) sb.append(",\"top_p\":").append(topP)
            if (topK != null) sb.append(",\"top_k\":").append(topK)
            if (repetitionPenalty != null) sb.append(",\"repetition_penalty\":").append(repetitionPenalty)
            if (frequencyPenalty != null) sb.append(",\"frequency_penalty\":").append(frequencyPenalty)
            if (presencePenalty != null) sb.append(",\"presence_penalty\":").append(presencePenalty)
            sb.append("}")
            return liteRt.generate(messagesJson, imageDataArray, sb.toString(), callback)
        }
        if (!isNativeLibraryLoaded) return ""
        // Build request JSON by string concatenation to avoid org.json re-serialization
        // which can corrupt non-ASCII characters (Chinese, emoji) through its parse/toString cycle.
        val thinking = if (thinkModes.containsKey(handle)) thinkModes[handle] else thinkEnabled
        val sb = StringBuilder()
        sb.append("{\"thinking_enabled\":").append(thinking)
        sb.append(",\"messages\":").append(messagesJson)
        if (toolsJson != null) {
            sb.append(",\"tools\":").append(toolsJson)
            if (toolChoice != null) sb.append(",\"tool_choice\":\"").append(toolChoice).append("\"")
        }
        if (maxTokens != null && maxTokens > 0) sb.append(",\"max_tokens\":").append(maxTokens)
        if (temperature != null) sb.append(",\"temperature\":").append(temperature)
        if (topP != null) sb.append(",\"top_p\":").append(topP)
        if (topK != null) sb.append(",\"top_k\":").append(topK)
        if (repetitionPenalty != null) sb.append(",\"repetition_penalty\":").append(repetitionPenalty)
        if (frequencyPenalty != null) sb.append(",\"frequency_penalty\":").append(frequencyPenalty)
        if (presencePenalty != null) sb.append(",\"presence_penalty\":").append(presencePenalty)
        sb.append("}")
        return nativeGenerate(handle, "", "", sb.toString(), imageDataArray, callback)
    }

    fun loadHistory(handle: Long, roles: Array<String>, contents: Array<String>): Boolean {
        if (liteRtHandles.containsKey(handle)) return false
        if (!isNativeLibraryLoaded) return false
        return nativeLoadHistory(handle, roles, contents)
    }

    fun setThinkMode(handle: Long, enabled: Boolean) {
        if (liteRtHandles.containsKey(handle)) {
            thinkModes[handle] = enabled
            return
        }
        if (!isNativeLibraryLoaded) return
        thinkModes[handle] = enabled
        nativeSetThinkMode(handle, enabled)
    }

    fun reset(handle: Long) {
        liteRtHandles[handle]?.let { it.reset(); return }
        if (isNativeLibraryLoaded) nativeReset(handle)
    }
    fun cancel(handle: Long) {
        liteRtHandles[handle]?.let { it.cancel(); return }
        if (isNativeLibraryLoaded) nativeCancel(handle)
    }
    fun gracefulStop(handle: Long) {
        liteRtHandles[handle]?.let { it.cancel(); return }
        if (isNativeLibraryLoaded) nativeGracefulStop(handle)
    }

    fun free(handle: Long) {
        liteRtHandles.remove(handle)?.let {
            thinkModes.remove(handle)
            it.close()
            return
        }
        if (!isNativeLibraryLoaded) return
        thinkModes.remove(handle)
        nativeFree(handle)
    }

    fun collectDiagnostics(handle: Long): Map<String, String> {
        liteRtHandles[handle]?.let { return it.diagnostics() }
        if (!isNativeLibraryLoaded) return emptyMap()
        val json = runCatching { JSONObject(nativeCollectDiagnosticsJson(handle)) }.getOrNull() ?: return emptyMap()
        return buildMap { json.keys().forEach { key -> put(key, json.optString(key)) } }
    }

    private external fun nativeInit(configJson: String): Long
    private external fun nativeGetLastError(): String
    private external fun nativeGenerate(handle: Long, systemPrompt: String?, prompt: String, requestJson: String, imageDataArray: Array<ByteArray>?, callback: OmniInferStreamCallback?): String
    private external fun nativeLoadHistory(handle: Long, roles: Array<String>, contents: Array<String>): Boolean
    private external fun nativePrewarmImage(handle: Long, imageData: ByteArray?, nThreads: Int): Boolean
    private external fun nativeSetThinkMode(handle: Long, enabled: Boolean)
    private external fun nativeReset(handle: Long)
    private external fun nativeCancel(handle: Long)
    private external fun nativeGracefulStop(handle: Long)
    private external fun nativeFree(handle: Long)
    private external fun nativeCollectDiagnosticsJson(handle: Long): String
}
