package com.omniinfer.server

import android.content.Context
import android.content.Intent
import android.util.Log
import java.net.HttpURLConnection
import java.net.URL

/**
 * OmniInfer local inference server facade.
 *
 * Usage:
 *   OmniInferServer.init(context)
 *   OmniInferServer.loadModel("/path/to/model.gguf", backend = "llama.cpp")
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

    fun init(context: Context) {
        appContext = context.applicationContext
        Log.i(TAG, "Initialized")
    }

    fun getPort(): Int = serverPort

    fun isReady(): Boolean = currentHandle != 0L && serverRunning

    /**
     * Load a model and start the server.
     * @param modelPath absolute path to model file (GGUF) or config.json (MNN)
     * @param backend "llama.cpp" or "mnn"
     * @param port local server port (default 9099)
     * @param nThreads CPU threads (0 = auto)
     * @param nCtx context window size
     * @return true if model loaded and server started successfully
     */
    fun loadModel(
        modelPath: String,
        backend: String = "llama.cpp",
        port: Int = 9099,
        nThreads: Int = 0,
        nCtx: Int = 16384
    ): Boolean {
        val ctx = appContext ?: run {
            Log.e(TAG, "Not initialized. Call init(context) first.")
            return false
        }

        // Unload previous model if different.
        if (currentHandle != 0L && (currentModelPath != modelPath || currentBackend != backend)) {
            unloadModel()
        }

        if (currentHandle != 0L) {
            Log.i(TAG, "Model already loaded: $currentModelPath ($currentBackend)")
            return true
        }

        val nativeLibDir = ctx.applicationInfo.nativeLibraryDir
        val handle = OmniInferBridge.init(
            modelPath = modelPath,
            backend = backend,
            nThreads = nThreads,
            nCtx = nCtx,
            nativeLibDir = nativeLibDir,
            cacheDir = ctx.cacheDir.absolutePath
        )

        if (handle == 0L) {
            Log.e(TAG, "Failed to load model: $modelPath ($backend)")
            return false
        }

        currentHandle = handle
        currentBackend = backend
        currentModelPath = modelPath
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
                unloadModel()
                return false
            }
            serverRunning = true
        }

        Log.i(TAG, "Model loaded: $modelPath ($backend), server on port $port")
        return true
    }

    fun unloadModel() {
        if (currentHandle != 0L) {
            OmniInferBridge.free(currentHandle)
            currentHandle = 0L
            currentBackend = ""
            currentModelPath = ""
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
