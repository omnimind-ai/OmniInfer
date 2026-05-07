package com.omniinfer.server

object LiteRtLmBackendFactory {
    @JvmStatic
    fun create(
        modelPath: String,
        backend: String,
        nThreads: Int,
        nCtx: Int,
        nativeLibDir: String?,
        cacheDir: String?,
        extraConfig: Map<String, String>?,
    ): LiteRtLmSession {
        return LiteRtLmBackend.create(
            modelPath = modelPath,
            backend = backend,
            nThreads = nThreads,
            nCtx = nCtx,
            nativeLibDir = nativeLibDir,
            cacheDir = cacheDir,
            extraConfig = extraConfig,
        )
    }
}
