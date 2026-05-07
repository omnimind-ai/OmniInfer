package com.omniinfer.server

import java.lang.reflect.InvocationTargetException
import java.util.Locale

internal object LiteRtLmBackendSupport {
    private const val FACTORY_CLASS = "com.omniinfer.server.LiteRtLmBackendFactory"

    fun isLiteRtBackend(backend: String): Boolean {
        val normalized = backend.lowercase(Locale.US)
        return normalized == "litert" ||
            normalized == "litertlm" ||
            normalized == "litert-lm" ||
            normalized == "litert-gpu" ||
            normalized == "litertlm-gpu" ||
            normalized == "litert-lm-gpu"
    }

    fun create(
        modelPath: String,
        backend: String,
        nThreads: Int,
        nCtx: Int,
        nativeLibDir: String?,
        cacheDir: String?,
        extraConfig: Map<String, String>?,
    ): LiteRtLmSession {
        val factory = runCatching { Class.forName(FACTORY_CLASS) }.getOrElse {
            throw IllegalStateException(
                "LiteRT-LM backend is disabled. Enable -Pomniinfer.backend.litert_lm=true."
            )
        }
        val method = factory.getMethod(
            "create",
            String::class.java,
            String::class.java,
            Int::class.javaPrimitiveType,
            Int::class.javaPrimitiveType,
            String::class.java,
            String::class.java,
            Map::class.java,
        )
        return try {
            method.invoke(null, modelPath, backend, nThreads, nCtx, nativeLibDir, cacheDir, extraConfig)
                as LiteRtLmSession
        } catch (error: InvocationTargetException) {
            throw error.targetException
        }
    }
}
