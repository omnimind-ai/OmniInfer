package com.omniinfer.server

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject

data class OmniInferModelSource(
    val provider: String,
    val url: String,
    val fileName: String,
    val sha256: String,
    val sizeBytes: Long,
)

data class OmniInferModelLoadConfig(
    val modelId: String,
    val backend: String,
    val nThreads: Int,
    val nCtx: Int,
    val extraConfig: Map<String, String>,
)

data class OmniInferCatalogModel(
    val id: String,
    val displayName: String,
    val backend: String,
    val format: String,
    val quantization: String,
    val accelerator: String,
    val sources: List<OmniInferModelSource>,
    val loadConfig: OmniInferModelLoadConfig,
)

object OmniInferModelCatalog {
    const val ANDROID_DEFAULT = "android-default"

    @Deprecated("Use ANDROID_DEFAULT. The default catalog now includes CPU, HTP, and LiteRT GPU entries.")
    const val ANDROID_LLAMA_CPP_HTP = ANDROID_DEFAULT

    private const val ASSET_DIR = "model-catalog"
    private const val LEGACY_LLAMA_CPP_HTP = "android-llamacpp-htp"

    fun listCatalogs(context: Context): List<String> {
        return context.assets.list(ASSET_DIR)
            ?.filter { it.endsWith(".json") }
            ?.map { it.removeSuffix(".json") }
            ?.sorted()
            .orEmpty()
    }

    fun readCatalogJson(
        context: Context,
        catalogId: String = ANDROID_DEFAULT,
    ): String {
        val path = "$ASSET_DIR/${normalizeCatalogId(catalogId)}.json"
        return context.assets.open(path).bufferedReader().use { it.readText() }
    }

    fun listModels(
        context: Context,
        catalogId: String = ANDROID_DEFAULT,
    ): List<OmniInferCatalogModel> {
        val catalog = JSONObject(readCatalogJson(context, catalogId))
        val defaults = catalog.optJSONObject("defaults") ?: JSONObject()
        val models = catalog.optJSONArray("models") ?: JSONArray()
        return buildList {
            for (i in 0 until models.length()) {
                add(parseModel(models.getJSONObject(i), defaults))
            }
        }
    }

    fun findModel(
        context: Context,
        modelId: String,
        catalogId: String = ANDROID_DEFAULT,
    ): OmniInferCatalogModel? {
        return listModels(context, catalogId).firstOrNull { it.id == modelId }
    }

    fun recommendedLoadConfig(
        context: Context,
        modelId: String,
        catalogId: String = ANDROID_DEFAULT,
    ): OmniInferModelLoadConfig? {
        return findModel(context, modelId, catalogId)?.loadConfig
    }

    private fun normalizeCatalogId(catalogId: String): String {
        return if (catalogId == LEGACY_LLAMA_CPP_HTP) ANDROID_DEFAULT else catalogId
    }

    private fun parseModel(model: JSONObject, defaults: JSONObject): OmniInferCatalogModel {
        val backend = model.optJSONObject("backend") ?: JSONObject()
        val defaultBackend = defaults.optJSONObject("backend") ?: JSONObject()
        val runtime = model.optJSONObject("runtime") ?: JSONObject()
        val defaultRuntime = defaults.optJSONObject("runtime") ?: JSONObject()

        val framework = backend.optString(
            "framework",
            defaultBackend.optString("framework", "llama.cpp"),
        )
        val format = backend.optString(
            "format",
            defaultBackend.optString("format", "gguf"),
        )
        val quantization = backend.optString("quantization", "")
        val accelerator = runtime.optString(
            "accelerator",
            defaultRuntime.optString("accelerator", "cpu"),
        )
        val nThreads = runtime.optInt(
            "n_threads",
            defaultRuntime.optInt("n_threads", 0),
        )
        val nCtx = runtime.optInt(
            "n_ctx",
            defaultRuntime.optInt("n_ctx", 8192),
        )
        val loadOptions = mergeStringMap(
            defaultRuntime.optJSONObject("load_options"),
            runtime.optJSONObject("load_options"),
        )
        val id = model.getString("id")

        return OmniInferCatalogModel(
            id = id,
            displayName = model.optString("display_name", id),
            backend = framework,
            format = format,
            quantization = quantization,
            accelerator = accelerator,
            sources = parseSources(model.optJSONArray("sources")),
            loadConfig = OmniInferModelLoadConfig(
                modelId = id,
                backend = framework,
                nThreads = nThreads,
                nCtx = nCtx,
                extraConfig = loadOptions,
            ),
        )
    }

    private fun parseSources(sources: JSONArray?): List<OmniInferModelSource> {
        if (sources == null) return emptyList()
        return buildList {
            for (i in 0 until sources.length()) {
                val item = sources.getJSONObject(i)
                add(
                    OmniInferModelSource(
                        provider = item.optString("provider"),
                        url = item.optString("url"),
                        fileName = item.optString("file_name"),
                        sha256 = item.optString("sha256"),
                        sizeBytes = item.optLong("size_bytes", -1L),
                    ),
                )
            }
        }
    }

    private fun mergeStringMap(base: JSONObject?, override: JSONObject?): Map<String, String> {
        val result = linkedMapOf<String, String>()
        fun putAll(source: JSONObject?) {
            if (source == null) return
            val keys = source.keys()
            while (keys.hasNext()) {
                val key = keys.next()
                result[key] = source.optString(key)
            }
        }
        putAll(base)
        putAll(override)
        return result
    }
}
