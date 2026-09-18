package com.omniinfer.server

/**
 * Raised when an engine package fails manifest parsing, verification, or loading.
 * Host apps should catch this around [OmniInferEngineLoader.install] and offer the
 * user a re-download of the engine package.
 */
class OmniInferEngineException(message: String) : IllegalStateException(message)

data class OmniInferEngineLibEntry(
    val name: String,
    val sha256: String,
    val sizeBytes: Long,
)

/**
 * Parsed `manifest.json` of a downloadable OmniInfer native engine package
 * (see docs/android/engine-download.md). Pure Kotlin: no Android SDK imports so the
 * parsing and validation rules stay testable on the host JVM.
 */
data class OmniInferEngineManifest(
    val formatVersion: Int,
    val engineVersion: String,
    val interfaceVersion: Int,
    val abi: String,
    val minSdk: Int,
    val backends: List<String>,
    /** Native libs loaded with System.load in order before any backend init (the DT_NEEDED chain ending in libomniinfer-jni.so). */
    val coreLibs: List<String>,
    /** All packaged native libs by file name. */
    val libs: Map<String, OmniInferEngineLibEntry>,
) {
    /** Directory inside the engine package that holds the native libs (mirrors APK jniLibs layout). */
    val libDirRelativePath: String get() = "lib/$abi"

    /**
     * Validate this manifest against the current device.
     * @param deviceAbis supported ABIs of the device (Build.SUPPORTED_ABIS on Android)
     * @param sdkInt device SDK int (Build.VERSION.SDK_INT on Android)
     */
    fun validate(
        deviceAbis: List<String>,
        sdkInt: Int,
        supportedFormatVersion: Int = CURRENT_FORMAT_VERSION,
        supportedInterfaceVersion: Int = CURRENT_INTERFACE_VERSION,
    ) {
        if (formatVersion != supportedFormatVersion) {
            throw OmniInferEngineException(
                "Engine manifest formatVersion $formatVersion is not supported (expected $supportedFormatVersion). " +
                    "Re-download a current engine package."
            )
        }
        if (interfaceVersion != supportedInterfaceVersion) {
            throw OmniInferEngineException(
                "Engine interfaceVersion $interfaceVersion does not match this SDK " +
                    "(expected $supportedInterfaceVersion). Update the OmniInfer SDK and engine together."
            )
        }
        if (engineVersion.isBlank()) {
            throw OmniInferEngineException("Engine manifest engineVersion must not be blank.")
        }
        if (abi !in deviceAbis) {
            throw OmniInferEngineException(
                "Engine package ABI '$abi' is not supported by this device (supported: $deviceAbis)."
            )
        }
        if (minSdk > sdkInt) {
            throw OmniInferEngineException(
                "Engine package requires minSdk $minSdk but this device reports $sdkInt."
            )
        }
        if (backends.isEmpty()) {
            throw OmniInferEngineException("Engine manifest must list at least one backend.")
        }
        if (coreLibs.isEmpty()) {
            throw OmniInferEngineException("Engine manifest must list coreLibs in load order.")
        }
        if (JNI_BRIDGE_LIB !in coreLibs) {
            throw OmniInferEngineException(
                "Engine manifest coreLibs must end with the JNI bridge library $JNI_BRIDGE_LIB."
            )
        }
        if (coreLibs.last() != JNI_BRIDGE_LIB) {
            throw OmniInferEngineException(
                "Engine manifest coreLibs must load the JNI bridge library $JNI_BRIDGE_LIB last."
            )
        }
        for (lib in coreLibs) {
            if (lib !in libs) {
                throw OmniInferEngineException("Engine manifest coreLib '$lib' is missing from libs.")
            }
        }
        val missingDescriptions = libs.values.filter { !isValidLibEntry(it) }
        if (missingDescriptions.isNotEmpty()) {
            throw OmniInferEngineException(
                "Engine manifest has invalid lib entries: ${missingDescriptions.map { it.name }}."
            )
        }
    }

    private fun isValidLibEntry(entry: OmniInferEngineLibEntry): Boolean {
        if (entry.name.isBlank() || entry.name.contains('/') || entry.name.contains('\\') ||
            entry.name.contains("..")
        ) {
            return false
        }
        if (entry.sizeBytes <= 0L) return false
        if (entry.sha256.length != 64) return false
        return entry.sha256.all { it.isDigit() || it in 'a'..'f' }
    }

    companion object {
        const val CURRENT_FORMAT_VERSION = 1
        const val CURRENT_INTERFACE_VERSION = 1
        const val JNI_BRIDGE_LIB = "libomniinfer-jni.so"

        /** Parse a manifest.json payload. Throws [OmniInferEngineException] on malformed input. */
        fun fromJson(json: String): OmniInferEngineManifest {
            val root = try {
                MinimalJsonParser.parseObject(json)
            } catch (error: OmniInferEngineException) {
                throw error
            } catch (error: Exception) {
                throw OmniInferEngineException("Engine manifest is not valid JSON: ${error.message}")
            }

            val formatVersion = root.int("formatVersion")
                ?: throw OmniInferEngineException("Engine manifest is missing formatVersion.")
            val engineVersion = root.string("engineVersion")
                ?: throw OmniInferEngineException("Engine manifest is missing engineVersion.")
            val interfaceVersion = root.int("interfaceVersion")
                ?: throw OmniInferEngineException("Engine manifest is missing interfaceVersion.")
            val abi = root.string("abi")
                ?: throw OmniInferEngineException("Engine manifest is missing abi.")
            val minSdk = root.int("minSdk")
                ?: throw OmniInferEngineException("Engine manifest is missing minSdk.")
            val backends = root.stringList("backends")
                ?: throw OmniInferEngineException("Engine manifest is missing backends.")
            val coreLibs = root.stringList("coreLibs")
                ?: throw OmniInferEngineException("Engine manifest is missing coreLibs.")

            val libsJson = root.array("libs")
                ?: throw OmniInferEngineException("Engine manifest is missing libs.")
            val libs = LinkedHashMap<String, OmniInferEngineLibEntry>()
            for (entry in libsJson) {
                val name = entry.string("name")
                    ?: throw OmniInferEngineException("Engine manifest lib entry is missing name.")
                if (name in libs) {
                    throw OmniInferEngineException("Engine manifest lists lib '$name' twice.")
                }
                libs[name] = OmniInferEngineLibEntry(
                    name = name,
                    sha256 = entry.string("sha256")
                        ?: throw OmniInferEngineException("Engine manifest lib '$name' is missing sha256."),
                    sizeBytes = entry.long("sizeBytes")
                        ?: throw OmniInferEngineException("Engine manifest lib '$name' is missing sizeBytes."),
                )
            }

            return OmniInferEngineManifest(
                formatVersion = formatVersion,
                engineVersion = engineVersion,
                interfaceVersion = interfaceVersion,
                abi = abi,
                minSdk = minSdk,
                backends = backends,
                coreLibs = coreLibs,
                libs = libs,
            )
        }

        private fun Map<String, Any?>.string(key: String): String? = this[key] as? String

        private fun Map<String, Any?>.int(key: String): Int? = (this[key] as? Number)?.toInt()

        private fun Map<String, Any?>.long(key: String): Long? = this[key] as? Long ?: (this[key] as? Number)?.toLong()

        private fun Map<String, Any?>.array(key: String): List<Map<String, Any?>>? {
            val value = this[key] as? List<*> ?: return null
            return value.map { element ->
                @Suppress("UNCHECKED_CAST")
                element as? Map<String, Any?>
                    ?: throw OmniInferEngineException("Engine manifest field '$key' must contain objects.")
            }
        }

        private fun Map<String, Any?>.stringList(key: String): List<String>? {
            val value = this[key] as? List<*> ?: return null
            return value.map { element ->
                element as? String
                    ?: throw OmniInferEngineException("Engine manifest field '$key' must contain strings.")
            }
        }
    }
}

/**
 * Restricted JSON parser for engine manifests: supports objects, arrays, strings
 * (with escapes), integers, booleans and null. Deliberately minimal so it can be
 * unit-tested on the host JVM without pulling any JSON dependency into the AAR.
 */
internal object MinimalJsonParser {
    fun parseObject(text: String): Map<String, Any?> {
        val parser = Reader(text.trim())
        parser.skipWhitespace()
        val value = parser.parseValue()
        parser.skipWhitespace()
        if (!parser.atEnd()) throw OmniInferEngineException("Unexpected trailing content at offset ${parser.pos}.")
        @Suppress("UNCHECKED_CAST")
        return value as? Map<String, Any?>
            ?: throw OmniInferEngineException("Engine manifest root must be a JSON object.")
    }

    private class Reader(val text: String) {
        var pos = 0

        fun atEnd(): Boolean = pos >= text.length

        fun skipWhitespace() {
            while (pos < text.length && text[pos].isWhitespace()) pos++
        }

        fun parseValue(): Any? {
            if (atEnd()) throw OmniInferEngineException("Unexpected end of manifest JSON.")
            return when (text[pos]) {
                '{' -> parseObjectValue()
                '[' -> parseArrayValue()
                '"' -> parseString()
                't' -> parseLiteral("true", true)
                'f' -> parseLiteral("false", false)
                'n' -> parseLiteral("null", null)
                else -> parseNumber()
            }
        }

        private fun parseObjectValue(): Map<String, Any?> {
            expect('{')
            val result = LinkedHashMap<String, Any?>()
            skipWhitespace()
            if (peek() == '}') {
                pos++
                return result
            }
            while (true) {
                skipWhitespace()
                val key = parseString()
                skipWhitespace()
                expect(':')
                skipWhitespace()
                result[key] = parseValue()
                skipWhitespace()
                when (peek()) {
                    ',' -> pos++
                    '}' -> {
                        pos++
                        return result
                    }
                    else -> throw OmniInferEngineException("Expected ',' or '}' at offset $pos.")
                }
            }
        }

        private fun parseArrayValue(): List<Any?> {
            expect('[')
            val result = ArrayList<Any?>()
            skipWhitespace()
            if (peek() == ']') {
                pos++
                return result
            }
            while (true) {
                skipWhitespace()
                result.add(parseValue())
                skipWhitespace()
                when (peek()) {
                    ',' -> pos++
                    ']' -> {
                        pos++
                        return result
                    }
                    else -> throw OmniInferEngineException("Expected ',' or ']' at offset $pos.")
                }
            }
        }

        private fun parseString(): String {
            expect('"')
            val builder = StringBuilder()
            while (true) {
                if (atEnd()) throw OmniInferEngineException("Unterminated string in manifest JSON.")
                when (val ch = text[pos++]) {
                    '"' -> return builder.toString()
                    '\\' -> builder.append(parseEscape())
                    else -> builder.append(ch)
                }
            }
        }

        private fun parseEscape(): Char {
            if (atEnd()) throw OmniInferEngineException("Unterminated escape in manifest JSON.")
            return when (val ch = text[pos++]) {
                '"' -> '"'
                '\\' -> '\\'
                '/' -> '/'
                'b' -> '\b'
                'f' -> '\u000C'
                'n' -> '\n'
                'r' -> '\r'
                't' -> '\t'
                'u' -> {
                    if (pos + 4 > text.length) throw OmniInferEngineException("Invalid unicode escape in manifest JSON.")
                    val hex = text.substring(pos, pos + 4)
                    pos += 4
                    hex.toIntOrNull(16)?.toChar()
                        ?: throw OmniInferEngineException("Invalid unicode escape '\\u$hex' in manifest JSON.")
                }
                else -> throw OmniInferEngineException("Invalid escape '\\$ch' in manifest JSON.")
            }
        }

        private fun parseNumber(): Any {
            val start = pos
            if (peek() == '-') pos++
            while (!atEnd() && (text[pos].isDigit() || text[pos] == '.' || text[pos] == 'e' || text[pos] == 'E' ||
                    text[pos] == '+' || text[pos] == '-')
            ) {
                pos++
            }
            val raw = text.substring(start, pos)
            return raw.toLongOrNull() ?: raw.toDoubleOrNull()
                ?: throw OmniInferEngineException("Invalid number '$raw' in manifest JSON.")
        }

        private fun parseLiteral(literal: String, value: Any?): Any? {
            if (text.regionMatches(pos, literal, 0, literal.length)) {
                pos += literal.length
                return value
            }
            throw OmniInferEngineException("Invalid token at offset $pos in manifest JSON.")
        }

        private fun peek(): Char {
            if (atEnd()) throw OmniInferEngineException("Unexpected end of manifest JSON.")
            return text[pos]
        }

        private fun expect(ch: Char) {
            if (atEnd() || text[pos] != ch) {
                throw OmniInferEngineException("Expected '$ch' at offset $pos in manifest JSON.")
            }
            pos++
        }
    }
}
