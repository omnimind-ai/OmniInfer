package com.omniinfer.server

// Host JVM contract: no Android SDK, JNI runtime or model assets required.
fun main() {
    val aliases = mapOf(
        "llama.cpp-cpu" to listOf("llama.cpp/cpu", "llama.cpp", "llama", "llamacpp-cpu"),
        "llama.cpp-htp" to listOf("llama.cpp/htp", "llama.cpp/npu", "llama-htp", "llama.cpp-npu"),
        "litert-lm-cpu" to listOf("litert/cpu", "litert-lm/cpu", "litert", "litert-lm", "litertlm"),
        "litert-lm-gpu" to listOf("litert/gpu", "litert-lm/gpu", "litert-gpu", "litertlm-gpu"),
        "mnn-cpu" to listOf("mnn/cpu"),
        "mnn-opencl" to listOf("mnn/opencl"),
        "mnn-vulkan" to listOf("mnn/vulkan"),
    )
    for ((canonical, legacyNames) in aliases) {
        for (input in legacyNames + canonical) {
            val resolved = BackendSelectors.normalizeExplicit(input)
            check(resolved == canonical) { "$input resolved to $resolved" }
            check(BackendSelectors.defaultExtraConfig(resolved) == BackendSelectors.defaultExtraConfig(canonical))
            check(BackendSelectors.bridgeBackendFor(resolved) == BackendSelectors.bridgeBackendFor(canonical))
        }
    }
    check(BackendSelectors.bridgeBackendFor("llama.cpp-htp") == "llama.cpp")
    check(BackendSelectors.bridgeBackendFor("litert-lm-gpu") == "litert")
    check(BackendSelectors.defaultThreadsFor("llama.cpp-htp") == 6)
    check(BackendSelectors.defaultExtraConfig("llama.cpp-htp") == mapOf(
        "accelerator" to "htp", "backend_type" to "npu", "llama_device" to "HTP0",
        "n_gpu_layers" to "99", "batch_size" to "1024", "ubatch_size" to "1024",
        "hexagon_opfilter" to "SSM_CONV",
    ))
    check(BackendSelectors.inferBackendFromPath("model.litertlm") == "litert-lm-gpu")
    check(BackendSelectors.inferBackendFromPath("model.gguf") == "llama.cpp-cpu")
    check(BackendSelectors.inferBackendFromPath("model.unknown") == BackendSelectors.UNSUPPORTED_AUTO_BACKEND)
    check(BackendSelectors.refineSelectorWithExtra("llama.cpp-cpu", mapOf("accelerator" to "htp")) == "llama.cpp-htp")
    check(BackendSelectors.refineSelectorWithExtra("litert-lm-cpu", mapOf("backend_type" to "gpu")) == "litert-lm-gpu")
    check(BackendSelectors.refineSelectorWithExtra("mnn-cpu", mapOf("backend_type" to "opencl")) == "mnn-opencl")
    check(BackendSelectors.selectorFor("llama.cpp", "htp") == "llama.cpp-htp")
    check(BackendSelectors.normalizeExplicit("future-engine") == "future-engine")
    check(BackendSelectors.normalizeExplicit(" CUSTOM_ENGINE ") == " CUSTOM_ENGINE ")
    check(BackendSelectors.normalizeExplicit(" LLAMA.CPP_CPU ") == "llama.cpp-cpu")
    check(BackendSelectors.selectorFor("llama.cpp/htp", null) == "llama.cpp-htp")
    println("Android backend selector contracts passed")
}
