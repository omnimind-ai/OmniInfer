package com.omniinfer.server

import java.util.Locale

object OmniInferBackend {
    const val AUTO = "auto"
    const val LLAMA_CPP_CPU = "llama.cpp-cpu"
    const val LLAMA_CPP_HTP = "llama.cpp-htp"
    const val LITERT_CPU = "litert-lm-cpu"
    const val LITERT_GPU = "litert-lm-gpu"
}

/** Public names do not change JNI engine identities or legacy acceleration defaults. */
internal object BackendSelectors {
    const val UNSUPPORTED_AUTO_BACKEND = "__unsupported_auto_backend__"

    fun isAuto(selector: String): Boolean =
        selector.isBlank() || selector.trim().equals(OmniInferBackend.AUTO, ignoreCase = true)

    fun shouldApplyCatalogDefaults(preferCatalogDefaults: Boolean, selector: String): Boolean =
        preferCatalogDefaults && isAuto(selector)

    fun normalizeExplicit(selector: String): String {
        val raw = selector.trim().lowercase(Locale.US).replace('_', '-')
        return when (raw) {
            "llama", "llamacpp", "llama-cpp", "llama.cpp", "llama.cpp/cpu",
            "llama.cpp-cpu", "llamacpp-cpu", "llama-cpp-cpu" -> OmniInferBackend.LLAMA_CPP_CPU

            "llama.cpp-htp", "llama.cpp-npu", "llama.cpp/htp", "llama.cpp/npu", "llamacpp-htp", "llama-cpp-htp",
            "llamacpp-npu", "llama-cpp-npu", "llama-htp", "llama-npu" ->
                OmniInferBackend.LLAMA_CPP_HTP

            "litert-lm-cpu", "litert", "litert-lm", "litertlm", "litert/cpu", "litert-lm/cpu" -> OmniInferBackend.LITERT_CPU
            "litert-lm-gpu", "litert/gpu", "litert-lm/gpu", "litertlm-gpu", "litert-gpu" ->
                OmniInferBackend.LITERT_GPU

            "mnn-cpu", "mnn/cpu" -> "mnn-cpu"
            "mnn-opencl", "mnn/opencl" -> "mnn-opencl"
            "mnn-vulkan", "mnn/vulkan" -> "mnn-vulkan"
            else -> selector
        }
    }

    fun refineSelectorWithExtra(
        selector: String,
        extraConfig: Map<String, String>,
    ): String {
        val accelerator = extraConfig["accelerator"]?.lowercase(Locale.US)
        val backendType = extraConfig["backend_type"]?.lowercase(Locale.US)
        val liteRtBackend = extraConfig["litert_backend"]?.lowercase(Locale.US)
        return when {
            selector == OmniInferBackend.LLAMA_CPP_CPU &&
                (accelerator == "htp" || accelerator == "npu" || backendType == "npu") ->
                OmniInferBackend.LLAMA_CPP_HTP

            selector == OmniInferBackend.LITERT_CPU && (backendType == "gpu" || liteRtBackend == "gpu") ->
                OmniInferBackend.LITERT_GPU

            selector == "mnn-cpu" && backendType == "opencl" -> "mnn-opencl"
            selector == "mnn-cpu" && backendType == "vulkan" -> "mnn-vulkan"
            else -> selector
        }
    }

    fun selectorFor(backend: String, accelerator: String?): String {
        val normalizedBackend = backend.lowercase(Locale.US)
        val normalizedAccelerator = accelerator?.lowercase(Locale.US)
        return when {
            normalizedBackend == "llama.cpp" && normalizedAccelerator == "htp" ->
                OmniInferBackend.LLAMA_CPP_HTP
            normalizedBackend == "llama.cpp" && normalizedAccelerator == "npu" ->
                OmniInferBackend.LLAMA_CPP_HTP
            normalizedBackend == "litert" && normalizedAccelerator == "gpu" ->
                OmniInferBackend.LITERT_GPU
            normalizedBackend == "litert" -> OmniInferBackend.LITERT_CPU
            normalizedBackend == "llama.cpp" -> OmniInferBackend.LLAMA_CPP_CPU
            else -> normalizeExplicit(backend)
        }
    }

    fun inferBackendFromPath(modelPath: String): String {
        val lower = modelPath.lowercase(Locale.US)
        return when {
            lower.endsWith(".litertlm") || lower.endsWith(".litert") -> OmniInferBackend.LITERT_GPU
            lower.endsWith(".gguf") -> OmniInferBackend.LLAMA_CPP_CPU
            else -> UNSUPPORTED_AUTO_BACKEND
        }
    }

    fun bridgeBackendFor(selector: String): String {
        return when (selector.lowercase(Locale.US)) {
            OmniInferBackend.LLAMA_CPP_CPU, OmniInferBackend.LLAMA_CPP_HTP -> "llama.cpp"
            OmniInferBackend.LITERT_CPU, OmniInferBackend.LITERT_GPU -> "litert"
            "mnn-cpu", "mnn-opencl", "mnn-vulkan" -> "mnn"
            else -> selector
        }
    }

    fun defaultThreadsFor(selector: String): Int {
        return when (selector.lowercase(Locale.US)) {
            OmniInferBackend.LLAMA_CPP_HTP -> 6
            OmniInferBackend.LITERT_CPU -> 4
            else -> 0
        }
    }

    fun defaultCtxFor(selector: String): Int {
        return when (selector.lowercase(Locale.US)) {
            else -> 8192
        }
    }

    fun defaultExtraConfig(selector: String): Map<String, String> {
        return when (selector.lowercase(Locale.US)) {
            OmniInferBackend.LLAMA_CPP_HTP -> mapOf(
                "accelerator" to "htp",
                "backend_type" to "npu",
                "llama_device" to "HTP0",
                "n_gpu_layers" to "99",
                "batch_size" to "1024",
                "ubatch_size" to "1024",
                "hexagon_opfilter" to "SSM_CONV",
            )
            OmniInferBackend.LITERT_GPU -> mapOf(
                "backend_type" to "gpu",
                "litert_backend" to "gpu",
            )
            OmniInferBackend.LITERT_CPU -> mapOf("backend_type" to "cpu")
            "mnn-opencl" -> mapOf("backend_type" to "opencl")
            "mnn-vulkan" -> mapOf("backend_type" to "vulkan")
            else -> emptyMap()
        }
    }

}
