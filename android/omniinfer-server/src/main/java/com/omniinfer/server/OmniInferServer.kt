package com.omniinfer.server

import android.content.Context
import android.content.Intent
import android.util.Log
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.util.Locale

data class OmniInferLoadOptions(
    val backend: String = OmniInferBackend.AUTO,
    val port: Int = 9099,
    val nThreads: Int? = null,
    val nCtx: Int? = null,
    val extraConfig: Map<String, String> = emptyMap(),
)

/**
 * OmniInfer local inference server facade.
 *
 * Usage:
 *   OmniInferServer.init(context)
 *   OmniInferServer.loadModel("/path/to/model.gguf")
 *   // Server is now ready at http://127.0.0.1:PORT/v1/chat/completions
 *   OmniInferServer.unloadModel()
 *   OmniInferServer.stop()
 */
object OmniInferServer {
    private const val TAG = "OmniInferServer"

    private var appContext: Context? = null
    private var serverPort: Int = 9099
    var currentHandle: Long = 0L
        private set
    private var currentBackend: String = ""
    private var currentModelPath: String = ""
    private var serverRunning = false
    private var currentLoadKey: String = ""
    @Volatile private var lastError: String = ""

    fun init(context: Context) {
        appContext = context.applicationContext
        Log.i(TAG, "Initialized")
    }

    /**
     * Customize the foreground service notification. Call before [loadModel].
     * @param title notification title (default "OmniInfer Server")
     * @param channelName notification channel name (default "OmniInfer Server")
     * @param smallIcon resource ID for the notification icon (default system icon)
     * @param textFormat lambda that receives the port and returns the notification text
     */
    fun configureNotification(
        title: String = "OmniInfer Server",
        channelName: String = "OmniInfer Server",
        smallIcon: Int = android.R.drawable.ic_menu_manage,
        textFormat: ((port: Int) -> String) = { "Running on port $it" }
    ) {
        OmniInferService.notifTitle = title
        OmniInferService.notifChannelName = channelName
        OmniInferService.notifSmallIcon = smallIcon
        OmniInferService.notifTextFormat = textFormat
    }

    fun getPort(): Int = serverPort

    fun isReady(): Boolean = currentHandle != 0L && serverRunning

    fun listModelCatalogs(): List<String> {
        val ctx = appContext ?: return emptyList()
        return OmniInferModelCatalog.listCatalogs(ctx)
    }

    fun getModelCatalogJson(
        catalogId: String = OmniInferModelCatalog.ANDROID_DEFAULT
    ): String {
        val ctx = appContext ?: return "{}"
        return OmniInferModelCatalog.readCatalogJson(ctx, catalogId)
    }

    fun listCatalogModels(
        catalogId: String = OmniInferModelCatalog.ANDROID_DEFAULT
    ): List<OmniInferCatalogModel> {
        val ctx = appContext ?: return emptyList()
        return OmniInferModelCatalog.listModels(ctx, catalogId)
    }

    fun getRecommendedLoadConfig(
        modelId: String,
        catalogId: String = OmniInferModelCatalog.ANDROID_DEFAULT
    ): OmniInferModelLoadConfig? {
        val ctx = appContext ?: return null
        return OmniInferModelCatalog.recommendedLoadConfig(ctx, modelId, catalogId)
    }

    /**
     * Load a model and start the server.
     * This overload infers a standard backend from the bundled catalog or the
     * model file extension.
     */
    fun loadModel(modelPath: String): Boolean {
        return loadModel(modelPath, OmniInferLoadOptions())
    }

    /**
     * Load a model with a compact options object.
     * Null thread/context values use the selected backend's standard defaults.
     */
    fun loadModel(
        modelPath: String,
        options: OmniInferLoadOptions,
    ): Boolean {
        return loadModelInternal(
            modelPath = modelPath,
            backendSelector = options.backend,
            port = options.port,
            nThreads = options.nThreads,
            nCtx = options.nCtx,
            extraConfig = options.extraConfig,
            preferCatalogDefaults = true,
        )
    }

    /**
     * Load a model and start the server.
     * @param modelPath absolute path to model file (GGUF or .litertlm for auto backend inference)
     * @param backend backend selector: "auto", "llama.cpp-cpu", "llama.cpp-htp", "litert-lm-gpu",
     *   or a legacy framework name such as "llama.cpp" / "litert".
     * @param port local server port (default 9099)
     * @param nThreads CPU threads (0 = auto)
     * @param nCtx context window size
     * @param extraConfig backend-specific config (e.g. "qnn_lib_dir", "decoder_model_version")
     * @return true if model loaded and server started successfully
     */
    fun loadModel(
        modelPath: String,
        backend: String = OmniInferBackend.AUTO,
        port: Int = 9099,
        nThreads: Int = 0,
        nCtx: Int = 16384,
        extraConfig: Map<String, String>? = null
    ): Boolean {
        return loadModelInternal(
            modelPath = modelPath,
            backendSelector = backend,
            port = port,
            nThreads = nThreads,
            nCtx = nCtx,
            extraConfig = extraConfig.orEmpty(),
            preferCatalogDefaults = false,
        )
    }

    private fun loadModelInternal(
        modelPath: String,
        backendSelector: String,
        port: Int,
        nThreads: Int?,
        nCtx: Int?,
        extraConfig: Map<String, String>,
        preferCatalogDefaults: Boolean,
    ): Boolean {
        val ctx = appContext ?: run {
            lastError = "Not initialized. Call init(context) first."
            Log.e(TAG, lastError)
            return false
        }
        val resolved = resolveLoadConfig(
            ctx = ctx,
            modelPath = modelPath,
            backendSelector = backendSelector,
            nThreads = nThreads,
            nCtx = nCtx,
            extraConfig = extraConfig,
            preferCatalogDefaults = preferCatalogDefaults,
        )
        if (resolved.selector == BackendSelectors.UNSUPPORTED_AUTO_BACKEND) {
            lastError = "Auto backend inference supports catalog models, .gguf, and .litertlm/.litert files in this AAR. " +
                "Pass a supported backend selector explicitly for other formats."
            Log.e(TAG, lastError)
            return false
        }
        val loadKey = resolved.loadKey(modelPath, port)

        if (serverRunning && serverPort != port) {
            ctx.stopService(Intent(ctx, OmniInferService::class.java))
            serverRunning = false
        }

        // Unload previous model if different.
        if (currentHandle != 0L && currentLoadKey != loadKey) {
            unloadModel()
        }

        if (currentHandle != 0L) {
            Log.i(TAG, "Model already loaded: $currentModelPath ($currentBackend)")
            return true
        }

        val nativeLibDir = OmniInferEngineLoader.installedNativeLibDir
            ?: ctx.applicationInfo.nativeLibraryDir
        val handle = OmniInferBridge.init(
            modelPath = modelPath,
            backend = resolved.bridgeBackend,
            nThreads = resolved.nThreads,
            nCtx = resolved.nCtx,
            nativeLibDir = nativeLibDir,
            cacheDir = ctx.cacheDir.absolutePath,
            extraConfig = resolved.extraConfig,
        )

        if (handle == 0L) {
            lastError = OmniInferBridge.getLastError().ifBlank {
                "Failed to load model: $modelPath (${resolved.selector})"
            }
            Log.e(TAG, lastError)
            return false
        }

        currentHandle = handle
        currentBackend = resolved.selector
        currentModelPath = modelPath
        currentLoadKey = loadKey
        serverPort = port

        // Start HTTP server and verify it's reachable.
        if (!serverRunning) {
            val intent = Intent(ctx, OmniInferService::class.java).apply {
                putExtra("port", port)
            }
            if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
                ctx.startForegroundService(intent)
            } else {
                ctx.startService(intent)
            }

            // Wait for the server to become reachable. This catches port-in-use
            // and other startup failures instead of returning a false success.
            if (!waitForHealth(port, timeoutMs = 5000)) {
                Log.e(TAG, "Server failed to start on port $port (port may be in use)")
                lastError = "Server failed to start on port $port (port may be in use)."
                unloadModel()
                return false
            }
            serverRunning = true
        }

        Log.i(
            TAG,
            "Model loaded: $modelPath (${resolved.selector} -> ${resolved.bridgeBackend}), " +
                "threads=${resolved.nThreads}, ctx=${resolved.nCtx}, server on port $port",
        )
        lastError = ""
        return true
    }

    fun unloadModel() {
        if (currentHandle != 0L) {
            OmniInferBridge.free(currentHandle)
            currentHandle = 0L
            currentBackend = ""
            currentModelPath = ""
            currentLoadKey = ""
            Log.i(TAG, "Model unloaded")
        }
    }

    fun stop() {
        unloadModel()
        val ctx = appContext ?: return
        ctx.stopService(Intent(ctx, OmniInferService::class.java))
        serverRunning = false
        Log.i(TAG, "Server stopped")
    }

    fun getLoadedModels(): List<String> {
        if (currentHandle == 0L) return emptyList()
        return listOf(currentModelPath.substringAfterLast("/"))
    }

    fun getDiagnostics(): Map<String, String> {
        if (currentHandle == 0L) return emptyMap()
        return OmniInferBridge.collectDiagnostics(currentHandle)
    }

    fun getLastError(): String = lastError

    private data class ResolvedLoadConfig(
        val selector: String,
        val bridgeBackend: String,
        val nThreads: Int,
        val nCtx: Int,
        val extraConfig: Map<String, String>,
    ) {
        fun loadKey(modelPath: String, port: Int): String {
            val configKey = extraConfig.toSortedMap().entries.joinToString(";") {
                "${it.key}=${it.value}"
            }
            return listOf(modelPath, selector, bridgeBackend, port, nThreads, nCtx, configKey)
                .joinToString("|")
        }
    }

    private fun resolveLoadConfig(
        ctx: Context,
        modelPath: String,
        backendSelector: String,
        nThreads: Int?,
        nCtx: Int?,
        extraConfig: Map<String, String>,
        preferCatalogDefaults: Boolean,
    ): ResolvedLoadConfig {
        val catalogConfig = findCatalogLoadConfig(ctx, modelPath)
        val initialSelector = normalizeBackendSelector(
            selector = backendSelector,
            modelPath = modelPath,
            catalogConfig = catalogConfig,
        )
        val catalogDefaults = if (
            BackendSelectors.shouldApplyCatalogDefaults(preferCatalogDefaults, backendSelector)
        ) catalogConfig else null
        val initialExtra = mergedExtraConfig(initialSelector, catalogDefaults, extraConfig)
        val normalizedSelector = BackendSelectors.refineSelectorWithExtra(initialSelector, initialExtra)
        val baseExtra = if (normalizedSelector == initialSelector) {
            initialExtra
        } else {
            mergedExtraConfig(normalizedSelector, catalogDefaults, extraConfig)
        }

        val bridgeBackend = BackendSelectors.bridgeBackendFor(normalizedSelector)
        val defaultThreads = BackendSelectors.defaultThreadsFor(normalizedSelector)
        val defaultCtx = BackendSelectors.defaultCtxFor(normalizedSelector)

        return ResolvedLoadConfig(
            selector = normalizedSelector,
            bridgeBackend = bridgeBackend,
            nThreads = nThreads ?: catalogDefaults?.nThreads ?: defaultThreads,
            nCtx = nCtx ?: catalogDefaults?.nCtx ?: defaultCtx,
            extraConfig = baseExtra,
        )
    }

    private fun findCatalogLoadConfig(
        ctx: Context,
        modelPath: String,
    ): OmniInferModelLoadConfig? {
        val fileName = File(modelPath).name
        return runCatching {
            OmniInferModelCatalog.listCatalogs(ctx).asSequence()
                .flatMap { catalogId -> OmniInferModelCatalog.listModels(ctx, catalogId).asSequence() }
                .firstOrNull { model ->
                    model.sources.any { source ->
                        source.fileName == fileName || modelPath.endsWith("/${source.fileName}")
                    }
                }
                ?.loadConfig
        }.getOrElse { error ->
            Log.w(TAG, "Failed to inspect model catalog for $fileName: ${error.message}")
            null
        }
    }

    private fun normalizeBackendSelector(
        selector: String,
        modelPath: String,
        catalogConfig: OmniInferModelLoadConfig?,
    ): String {
        val raw = selector.trim().lowercase(Locale.US).replace('_', '-')
        if (raw.isBlank() || raw == OmniInferBackend.AUTO) {
            val fromCatalog = catalogConfig?.let { config ->
                val accelerator = config.extraConfig["accelerator"]
                    ?: config.extraConfig["backend_type"]
                    ?: config.extraConfig["litert_backend"]
                BackendSelectors.selectorFor(config.backend, accelerator)
            }
            return fromCatalog ?: BackendSelectors.inferBackendFromPath(modelPath)
        }
        return BackendSelectors.normalizeExplicit(selector)
    }

    private fun mergedExtraConfig(
        selector: String,
        catalogDefaults: OmniInferModelLoadConfig?,
        extraConfig: Map<String, String>,
    ): Map<String, String> {
        val result = BackendSelectors.defaultExtraConfig(selector).toMutableMap()
        catalogDefaults?.extraConfig?.let { result.putAll(it) }
        result.putAll(extraConfig)
        return result
    }

    private fun waitForHealth(port: Int, timeoutMs: Long = 5000): Boolean {
        val deadline = System.currentTimeMillis() + timeoutMs
        while (System.currentTimeMillis() < deadline) {
            try {
                val conn = URL("http://127.0.0.1:$port/health").openConnection() as HttpURLConnection
                conn.connectTimeout = 500
                conn.readTimeout = 500
                conn.requestMethod = "GET"
                if (conn.responseCode == 200) {
                    conn.disconnect()
                    return true
                }
                conn.disconnect()
            } catch (_: Exception) {
                // Server not ready yet.
            }
            Thread.sleep(200)
        }
        return false
    }
}
