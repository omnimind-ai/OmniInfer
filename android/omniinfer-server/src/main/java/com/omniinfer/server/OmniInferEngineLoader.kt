package com.omniinfer.server

import android.os.Build
import android.util.Log
import java.io.File
import java.security.MessageDigest

/**
 * Installs and loads an OmniInfer native engine package that was downloaded at
 * runtime instead of being bundled inside the host APK (see docs/android/engine-download.md).
 *
 * Expected engine package layout after extraction:
 * ```
 * <engineDir>/manifest.json
 * <engineDir>/lib/arm64-v8a/&lt;lib&gt;.so
 * ```
 *
 * Process lifecycle: an engine package can only be installed once per process.
 * Switching engine versions requires restarting the app because loaded native
 * libraries cannot be safely unloaded while ggml backends may have registered.
 */
object OmniInferEngineLoader {
    private const val TAG = "OmniInferEngineLoader"
    const val MANIFEST_FILE = "manifest.json"

    @Volatile private var installedDir: File? = null

    @Volatile private var installedManifest: OmniInferEngineManifest? = null

    /**
     * Native lib dir to pass to the JNI backend once an engine package is loaded,
     * or `null` when no engine package is installed (the host falls back to
     * `applicationInfo.nativeLibraryDir`).
     */
    val installedNativeLibDir: String?
        get() {
            val dir = installedDir ?: return null
            val manifest = installedManifest ?: return null
            return File(dir, manifest.libDirRelativePath).absolutePath
        }

    /**
     * Verify an extracted engine package without loading it. Useful to validate a
     * download before committing to it (e.g. before swapping a version pointer).
     */
    fun verify(engineDir: File, verifyHashes: Boolean = true): OmniInferEngineManifest {
        val manifestFile = File(engineDir, MANIFEST_FILE)
        if (!manifestFile.isFile) {
            throw OmniInferEngineException("Engine manifest not found: ${manifestFile.absolutePath}")
        }
        val manifest = OmniInferEngineManifest.fromJson(manifestFile.readText())
        manifest.validate(
            deviceAbis = Build.SUPPORTED_ABIS.toList(),
            sdkInt = Build.VERSION.SDK_INT,
        )
        val libDir = File(engineDir, manifest.libDirRelativePath)
        if (!libDir.isDirectory) {
            throw OmniInferEngineException("Engine package is missing ${manifest.libDirRelativePath}: ${engineDir.absolutePath}")
        }

        for (entry in manifest.libs.values) {
            val file = File(libDir, entry.name)
            if (!file.isFile) {
                throw OmniInferEngineException("Engine package is missing lib '${entry.name}'. Re-download the engine package.")
            }
            if (file.length() != entry.sizeBytes) {
                throw OmniInferEngineException(
                    "Engine lib '${entry.name}' has unexpected size ${file.length()} (expected ${entry.sizeBytes}). " +
                        "Re-download the engine package."
                )
            }
            if (verifyHashes && file.sha256Hex() != entry.sha256) {
                throw OmniInferEngineException("Engine lib '${entry.name}' failed SHA-256 verification. Re-download the engine package.")
            }
        }

        val unlisted = libDir.listFiles { file -> file.isFile && file.extension == "so" }
            .orEmpty()
            .map { it.name }
            .filter { it !in manifest.libs }
        if (unlisted.isNotEmpty()) {
            throw OmniInferEngineException(
                "Engine package contains libs not listed in the manifest: $unlisted. " +
                    "The package is corrupted or tampered with."
            )
        }
        return manifest
    }

    /**
     * Verify an extracted engine package and load its core native libs into the
     * process. After this returns [OmniInferBridge] resolves through the engine
     * package and [OmniInferServer] passes [installedNativeLibDir] to the JNI
     * backend so ggml loads accelerators from the same package.
     *
     * @param verifyHashes set to `false` only to skip per-lib SHA-256 verification
     *   when the zip integrity was already checked by the downloader.
     * @throws OmniInferEngineException on any verification or loading failure.
     */
    fun install(engineDir: File, verifyHashes: Boolean = true): OmniInferEngineManifest {
        installedDir?.let { existing ->
            if (existing.absolutePath == engineDir.absolutePath && installedManifest != null) {
                return installedManifest!!
            }
            throw OmniInferEngineException(
                "OmniInfer engine ${installedManifest?.engineVersion ?: "?"} is already installed in this process " +
                    "from $existing. Restart the app to install a different engine package."
            )
        }
        val manifest = verify(engineDir, verifyHashes)
        val libDir = File(engineDir, manifest.libDirRelativePath)

        synchronized(this) {
            installedDir?.let { existing ->
                if (existing.absolutePath == engineDir.absolutePath && installedManifest != null) {
                    return installedManifest!!
                }
                throw OmniInferEngineException("Engine already installed in this process from $existing.")
            }
            for (lib in manifest.coreLibs) {
                val file = File(libDir, lib)
                try {
                    System.load(file.absolutePath)
                    Log.i(TAG, "Loaded engine core lib $lib from ${file.parent}")
                } catch (error: Throwable) {
                    throw OmniInferEngineException("Failed to load engine lib '$lib': ${error.message}")
                }
            }
            installedDir = engineDir
            installedManifest = manifest
        }
        OmniInferBridge.markEngineNativeLibsLoaded()
        Log.i(TAG, "OmniInfer engine ${manifest.engineVersion} installed (${manifest.backends}) from ${engineDir.absolutePath}")
        return manifest
    }

    private fun File.sha256Hex(): String {
        val digest = MessageDigest.getInstance("SHA-256")
        inputStream().use { input ->
            val buffer = ByteArray(1 shl 16)
            while (true) {
                val read = input.read(buffer)
                if (read <= 0) break
                digest.update(buffer, 0, read)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }
}
