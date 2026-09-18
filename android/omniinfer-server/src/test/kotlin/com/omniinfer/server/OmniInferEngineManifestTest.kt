package com.omniinfer.server

// Host JVM contract: no Android SDK, JNI runtime or engine assets required.

private const val DEVICE_ABI = "arm64-v8a"
private const val SDK_INT = 34
private val FAKE_LIBS = listOf(
    "libggml-base.so" to 1000L,
    "libggml.so" to 2000L,
    "libllama-common.so" to 3000L,
    "libllama.so" to 4000L,
    "libmtmd.so" to 5000L,
    "libomniinfer-jni.so" to 6000L,
    "libggml-cpu.so" to 7000L,
)
private const val FAKE_SHA = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

private fun allLibsJson(): String = FAKE_LIBS.joinToString(",") { (name, size) ->
    """{"name":"$name","sha256":"$FAKE_SHA","sizeBytes":$size}"""
}

private fun manifestJson(
    formatVersion: Int? = 1,
    engineVersion: String? = "0.3.31-engine.1",
    interfaceVersion: Int? = 1,
    abi: String? = DEVICE_ABI,
    minSdk: Int? = 26,
    backends: String? = "\"llama.cpp-cpu\",\"llama.cpp-htp\"",
    coreLibs: String? = "\"libggml-base.so\",\"libggml.so\",\"libllama-common.so\",\"libllama.so\",\"libmtmd.so\",\"libomniinfer-jni.so\"",
    libs: String = allLibsJson(),
    extra: String = "",
): String = """
    {
      "formatVersion": ${formatVersion ?: "null"},
      "engineVersion": ${engineVersion?.let { "\"$it\"" } ?: "null"},
      "interfaceVersion": ${interfaceVersion ?: "null"},
      "abi": ${abi?.let { "\"$it\"" } ?: "null"},
      "minSdk": ${minSdk ?: "null"},
      "backends": [$backends],
      "coreLibs": [$coreLibs],
      "libs": [$libs]$extra
    }
""".trimIndent()

private fun expectEngineError(message: String, block: () -> Unit) {
    try {
        block()
    } catch (error: OmniInferEngineException) {
        check(error.message?.contains(message) == true) {
            "Expected error containing '$message' but got '${error.message}'"
        }
        return
    }
    throw IllegalStateException("Expected OmniInferEngineException containing '$message' but nothing was thrown")
}

private fun validManifest(): OmniInferEngineManifest =
    OmniInferEngineManifest.fromJson(manifestJson())

fun main() {
    // Parsing + validation happy path.
    val manifest = validManifest()
    manifest.validate(deviceAbis = listOf(DEVICE_ABI, "x86_64"), sdkInt = SDK_INT)
    check(manifest.libDirRelativePath == "lib/$DEVICE_ABI")
    check(manifest.libs.size == FAKE_LIBS.size)
    check(manifest.coreLibs.last() == OmniInferEngineManifest.JNI_BRIDGE_LIB)
    check(manifest.libs["libggml-base.so"] == OmniInferEngineLibEntry("libggml-base.so", FAKE_SHA, 1000L))

    // JSON parsing details: escapes, numbers, nested value types.
    // Raw string payload q\" r\\\\ s\/ t\\t tests: \" -> ", \\\\ -> \\, \/ -> /, \\t -> \t
    val tricky = MinimalJsonParser.parseObject(
        """
        {"a":"q\" r\\\\ s\/ t\\t","b":-42,"c":9007199254740993,
         "d":[true,false,null],"e":{"nested":"obj"}}
        """.trimIndent(),
    )
    check(tricky["a"] == "q\" r\\\\ s/ t\\t")
    check(tricky["b"] == -42L)
    check(tricky["c"] == 9007199254740993L)
    check(tricky["d"] == listOf(true, false, null))
    @Suppress("UNCHECKED_CAST")
    check((tricky["e"] as Map<String, Any?>)["nested"] == "obj")

    // Malformed JSON.
    expectEngineError("root must be a JSON object") { MinimalJsonParser.parseObject("[1,2]") }
    expectEngineError("Unexpected trailing content") { MinimalJsonParser.parseObject("{} {}") }
    expectEngineError("Expected ',' or '}'") { MinimalJsonParser.parseObject("""{"a":1 "b":2}""") }
    expectEngineError("Unterminated string") { MinimalJsonParser.parseObject("""{"a":"open""") }
    expectEngineError("Invalid escape") { MinimalJsonParser.parseObject("""{"a":"\q"}""") }
    expectEngineError("Invalid number") { MinimalJsonParser.parseObject("""{"a":--3}""") }
    expectEngineError("Invalid token") { OmniInferEngineManifest.fromJson("not json") }

    // Missing required fields.
    expectEngineError("missing formatVersion") { OmniInferEngineManifest.fromJson(manifestJson(formatVersion = null)) }
    expectEngineError("missing engineVersion") { OmniInferEngineManifest.fromJson(manifestJson(engineVersion = null)) }
    expectEngineError("missing interfaceVersion") { OmniInferEngineManifest.fromJson(manifestJson(interfaceVersion = null)) }
    expectEngineError("missing abi") { OmniInferEngineManifest.fromJson(manifestJson(abi = null)) }
    expectEngineError("missing minSdk") { OmniInferEngineManifest.fromJson(manifestJson(minSdk = null)) }
    // Array fields with null elements are rejected by the string list coercion.
    expectEngineError("backends' must contain strings") { OmniInferEngineManifest.fromJson(manifestJson(backends = null)) }
    expectEngineError("coreLibs' must contain strings") { OmniInferEngineManifest.fromJson(manifestJson(coreLibs = null)) }
    // Fields entirely absent are rejected as missing.
    expectEngineError("missing backends") {
        OmniInferEngineManifest.fromJson("""{"formatVersion":1,"engineVersion":"v","interfaceVersion":1,"abi":"arm64-v8a","minSdk":26,"coreLibs":["libomniinfer-jni.so"],"libs":[]}""")
    }
    expectEngineError("missing coreLibs") {
        OmniInferEngineManifest.fromJson("""{"formatVersion":1,"engineVersion":"v","interfaceVersion":1,"abi":"arm64-v8a","minSdk":26,"backends":["llama.cpp-cpu"],"libs":[]}""")
    }
    expectEngineError("missing libs") { OmniInferEngineManifest.fromJson("""{"formatVersion":1,"engineVersion":"v","interfaceVersion":1,"abi":"arm64-v8a","minSdk":26,"backends":["b"],"coreLibs":["libomniinfer-jni.so"]}""") }

    // Duplicate lib entries are rejected.
    val duplicateLibs = FAKE_LIBS.joinToString(",") { (name, size) ->
        """{"name":"$name","sha256":"$FAKE_SHA","sizeBytes":$size}"""
    } + "," + """{"name":"libggml-base.so","sha256":"$FAKE_SHA","sizeBytes":1000}"""
    expectEngineError("twice") { OmniInferEngineManifest.fromJson(manifestJson(libs = duplicateLibs)) }

    // Validation failures against the device.
    expectEngineError("formatVersion 1 is not supported") {
        validManifest().validate(deviceAbis = listOf(DEVICE_ABI), sdkInt = SDK_INT, supportedFormatVersion = 99)
    }
    expectEngineError("interfaceVersion 1 does not match") {
        validManifest().validate(deviceAbis = listOf(DEVICE_ABI), sdkInt = SDK_INT, supportedInterfaceVersion = 99)
    }
    expectEngineError("ABI 'x86_64' is not supported") {
        OmniInferEngineManifest.fromJson(manifestJson(abi = "x86_64"))
            .validate(deviceAbis = listOf(DEVICE_ABI), sdkInt = SDK_INT)
    }
    expectEngineError("requires minSdk 99 but this device reports $SDK_INT") {
        OmniInferEngineManifest.fromJson(manifestJson(minSdk = 99))
            .validate(deviceAbis = listOf(DEVICE_ABI), sdkInt = SDK_INT)
    }
    expectEngineError("at least one backend") {
        OmniInferEngineManifest.fromJson(manifestJson(backends = ""))
            .validate(deviceAbis = listOf(DEVICE_ABI), sdkInt = SDK_INT)
    }
    expectEngineError("must list coreLibs") {
        OmniInferEngineManifest.fromJson(manifestJson(coreLibs = ""))
            .validate(deviceAbis = listOf(DEVICE_ABI), sdkInt = SDK_INT)
    }
    expectEngineError("missing from libs") {
        OmniInferEngineManifest.fromJson(
            manifestJson(coreLibs = "\"libggml-base.so\",\"libmissing.so\",\"libomniinfer-jni.so\""),
        ).validate(deviceAbis = listOf(DEVICE_ABI), sdkInt = SDK_INT)
    }
    expectEngineError("must load the JNI bridge library libomniinfer-jni.so last") {
        OmniInferEngineManifest.fromJson(
            manifestJson(coreLibs = "\"libomniinfer-jni.so\",\"libggml-base.so\""),
        ).validate(deviceAbis = listOf(DEVICE_ABI), sdkInt = SDK_INT)
    }
    expectEngineError("must end with the JNI bridge") {
        OmniInferEngineManifest.fromJson(manifestJson(coreLibs = "\"libggml-base.so\",\"libggml.so\""))
            .validate(deviceAbis = listOf(DEVICE_ABI), sdkInt = SDK_INT)
    }

    // Lib entry sanity rules (invalid entry appended to an otherwise valid manifest).
    fun libEntryJson(name: String, sha: String = FAKE_SHA, size: Long = 1000L) =
        """{"name":"$name","sha256":"$sha","sizeBytes":$size}"""

    fun libsWithInvalid(entry: String): String = allLibsJson() + "," + entry

    expectEngineError("invalid lib entries") {
        OmniInferEngineManifest.fromJson(manifestJson(libs = libsWithInvalid(libEntryJson("../escape.so"))))
            .validate(deviceAbis = listOf(DEVICE_ABI), sdkInt = SDK_INT)
    }
    expectEngineError("invalid lib entries") {
        OmniInferEngineManifest.fromJson(manifestJson(libs = libsWithInvalid(libEntryJson("lib/x.so"))))
            .validate(deviceAbis = listOf(DEVICE_ABI), sdkInt = SDK_INT)
    }
    expectEngineError("invalid lib entries") {
        OmniInferEngineManifest.fromJson(manifestJson(libs = libsWithInvalid(libEntryJson("libextra.so", sha = "abc123"))))
            .validate(deviceAbis = listOf(DEVICE_ABI), sdkInt = SDK_INT)
    }
    expectEngineError("invalid lib entries") {
        OmniInferEngineManifest.fromJson(manifestJson(libs = libsWithInvalid(libEntryJson("libextra.so", sha = "z".repeat(64)))))
            .validate(deviceAbis = listOf(DEVICE_ABI), sdkInt = SDK_INT)
    }
    expectEngineError("invalid lib entries") {
        OmniInferEngineManifest.fromJson(manifestJson(libs = libsWithInvalid(libEntryJson("libextra.so", size = 0L))))
            .validate(deviceAbis = listOf(DEVICE_ABI), sdkInt = SDK_INT)
    }

    println("OmniInferEngineManifestTest: all checks passed")
}
