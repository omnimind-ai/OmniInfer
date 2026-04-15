package com.omniinfer.server

import android.util.Log
import org.json.JSONObject
import java.util.concurrent.ConcurrentHashMap

object OmniInferBridge {
    private const val TAG = "OmniInferBridge"
    private const val LIB_NAME = "omniinfer-jni"
    private val thinkModes = ConcurrentHashMap<Long, Boolean>()

    val isNativeLibraryLoaded: Boolean by lazy {
        runCatching {
            System.loadLibrary(LIB_NAME)
            true
        }.getOrElse { error ->
            Log.w(TAG, "Failed to load $LIB_NAME: ${error.message}")
            false
        }
    }

    fun init(
        modelPath: String,
        backend: String = "llama.cpp",
        nThreads: Int = 0,
        nCtx: Int = 4096,
        nativeLibDir: String? = null,
        cacheDir: String? = null
    ): Long {
        if (!isNativeLibraryLoaded) return 0L
        val configJson = JSONObject()
            .put("backend", backend)
            .put("model_path", modelPath)
            .put("native_lib_dir", nativeLibDir ?: "")
            .put("cache_dir", cacheDir ?: "")
            .put("n_threads", nThreads)
            .put("n_ctx", nCtx)
        val handle = nativeInit(configJson.toString())
        // Don't set default think mode — let each request's thinkEnabled param decide.
        return handle
    }

    fun generate(
        handle: Long,
        messagesJson: String,
        imageDataArray: Array<ByteArray>? = null,
        thinkEnabled: Boolean = false,
        toolsJson: String? = null,
        toolChoice: String? = null,
        maxTokens: Int? = null,
        callback: OmniInferStreamCallback? = null
    ): String {
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
        sb.append("}")
        return nativeGenerate(handle, "", "", sb.toString(), imageDataArray, callback)
    }

    fun loadHistory(handle: Long, roles: Array<String>, contents: Array<String>): Boolean {
        if (!isNativeLibraryLoaded) return false
        return nativeLoadHistory(handle, roles, contents)
    }

    fun setThinkMode(handle: Long, enabled: Boolean) {
        if (!isNativeLibraryLoaded) return
        thinkModes[handle] = enabled
        nativeSetThinkMode(handle, enabled)
    }

    fun reset(handle: Long) { if (isNativeLibraryLoaded) nativeReset(handle) }
    fun cancel(handle: Long) { if (isNativeLibraryLoaded) nativeCancel(handle) }
    fun gracefulStop(handle: Long) { if (isNativeLibraryLoaded) nativeGracefulStop(handle) }

    fun free(handle: Long) {
        if (!isNativeLibraryLoaded) return
        thinkModes.remove(handle)
        nativeFree(handle)
    }

    fun collectDiagnostics(handle: Long): Map<String, String> {
        if (!isNativeLibraryLoaded) return emptyMap()
        val json = runCatching { JSONObject(nativeCollectDiagnosticsJson(handle)) }.getOrNull() ?: return emptyMap()
        return buildMap { json.keys().forEach { key -> put(key, json.optString(key)) } }
    }

    private external fun nativeInit(configJson: String): Long
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
