plugins {
    id("com.android.library")
    id("org.jetbrains.kotlin.android")
}

val ktorVersion: String = findProperty("omniinfer.ktor.version")?.toString() ?: "3.1.3"
val enableExecutorchQnn: Boolean = findProperty("omniinfer.backend.executorch_qnn")?.toString()?.toBoolean() ?: false

// --- ExecuTorch QNN: auto-download pre-built binaries ---
if (enableExecutorchQnn) {
    val etQnnVersion = 2  // bump when uploading new binaries to OSS
    val baseUrl = "https://omnimind-model.oss-cn-beijing.aliyuncs.com/omniinfer-android/arm64-v8a"
    val jniDir = file("src/main/jniLibs/arm64-v8a")
    val versionFile = File(jniDir, ".etqnn_version")

    val etQnnFiles = listOf(
        // Universal
        "libetqnn_runner.so",
        "libqnn_executorch_backend.so",
        "libQnnHtp.so",
        "libQnnHtpPrepare.so",
        "libQnnSystem.so",
        "libQnnHtpNetRunExtensions.so",
        // Chip-specific skel/stub
        "libQnnHtpV75Skel.so", "libQnnHtpV75Stub.so",  // SM8650 (8 Gen 3)
        "libQnnHtpV79Skel.so", "libQnnHtpV79Stub.so",  // SM8750 (8 Elite)
        "libQnnHtpV81Skel.so", "libQnnHtpV81Stub.so",  // SM8850 (8 Elite Gen 5)
    )

    val downloadEtQnnLibs by tasks.registering {
        description = "Download ExecuTorch QNN pre-built binaries (v$etQnnVersion)"
        outputs.upToDateWhen {
            versionFile.exists()
                && versionFile.readText().trim() == etQnnVersion.toString()
                && etQnnFiles.all { File(jniDir, it).exists() }
        }
        doLast {
            jniDir.mkdirs()
            val outdated = !versionFile.exists()
                || versionFile.readText().trim() != etQnnVersion.toString()
            if (outdated) logger.lifecycle("ET QNN prebuilt v$etQnnVersion — updating all binaries")
            etQnnFiles.forEach { name ->
                val target = File(jniDir, name)
                if (!target.exists() || outdated) {
                    logger.lifecycle("  Downloading $name ...")
                    uri("$baseUrl/$name").toURL().openStream().use { input ->
                        target.outputStream().use { output -> input.copyTo(output) }
                    }
                }
            }
            versionFile.writeText(etQnnVersion.toString())
        }
    }

    tasks.matching { it.name.startsWith("buildCMake") || it.name.startsWith("merge") }
        .configureEach { dependsOn(downloadEtQnnLibs) }
}

android {
    namespace = "com.omniinfer.server"
    compileSdk = 35
    ndkVersion = "28.2.13676358"

    defaultConfig {
        minSdk = 26
        consumerProguardFiles("consumer-rules.pro")

        ndk {
            abiFilters += "arm64-v8a"
        }

        externalNativeBuild {
            cmake {
                arguments += "-DCMAKE_BUILD_TYPE=Release"
                arguments += "-DGGML_NATIVE=OFF"
                arguments += "-DGGML_LLAMAFILE=OFF"
                arguments += "-DLLAMA_BUILD_COMMON=ON"
                arguments += "-DBUILD_SHARED_LIBS=ON"
                arguments += "-DGGML_BACKEND_DL=ON"
                arguments += "-DGGML_CPU_ALL_VARIANTS=ON"
                arguments += "-DOMNIINFER_BACKEND_MNN=ON"
                if (enableExecutorchQnn) {
                    arguments += "-DOMNIINFER_BACKEND_EXECUTORCH_QNN=ON"
                }
            }
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/omniinfer-jni/CMakeLists.txt")
        }
    }

    // Shorten .cxx build path on Windows to avoid MAX_PATH (260 char) limit.
    // Set -Pomniinfer.cxx.dir=D:/.cxx/omniinfer to override (e.g. to save C: drive space).
    if (org.gradle.internal.os.OperatingSystem.current().isWindows) {
        val cxxDir = findProperty("omniinfer.cxx.dir")?.toString()
            ?: "${System.getProperty("user.home")}/.cxx/omniinfer"
        externalNativeBuild.cmake.buildStagingDirectory = file(cxxDir)
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }

    kotlinOptions {
        jvmTarget = "11"
    }
}

dependencies {
    compileOnly("io.ktor:ktor-server-core:$ktorVersion")
    compileOnly("io.ktor:ktor-server-cio:$ktorVersion")
    compileOnly("io.ktor:ktor-server-content-negotiation:$ktorVersion")
    compileOnly("io.ktor:ktor-serialization-kotlinx-json:$ktorVersion")
    compileOnly("org.jetbrains.kotlinx:kotlinx-serialization-json:1.6.3")
    compileOnly("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")
    implementation("androidx.core:core-ktx:1.12.0")
}
