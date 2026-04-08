plugins {
    id("com.android.library")
    id("org.jetbrains.kotlin.android")
}

val ktorVersion: String = findProperty("omniinfer.ktor.version")?.toString() ?: "3.1.3"

android {
    namespace = "com.omniinfer.server"
    compileSdk = 35

    defaultConfig {
        minSdk = 26

        ndk {
            abiFilters += "arm64-v8a"
        }

        externalNativeBuild {
            cmake {
                arguments += "-DCMAKE_BUILD_TYPE=Release"
                arguments += "-DGGML_NATIVE=OFF"
                arguments += "-DGGML_LLAMAFILE=OFF"
                arguments += "-DLLAMA_BUILD_COMMON=ON"
                arguments += "-DGGML_CPU_ARM_ARCH=armv8.2-a+fp16+dotprod+i8mm"
                arguments += "-DOMNIINFER_BACKEND_MNN=ON"
            }
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/omniinfer-jni/CMakeLists.txt")
        }
    }

    // Shorten .cxx build path on Windows to avoid MAX_PATH (260 char) limit.
    // Host projects on Windows should set android.externalNativeBuild.buildStagingDirectory
    // to a short path (e.g. C:/tmp/.cxx) if they encounter path-length build failures.

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
