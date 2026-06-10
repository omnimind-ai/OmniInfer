import org.gradle.jvm.tasks.Jar
import org.gradle.api.tasks.bundling.Zip
import org.jetbrains.kotlin.gradle.dsl.JvmTarget
import java.util.zip.ZipFile
import javax.xml.parsers.DocumentBuilderFactory
import org.w3c.dom.Element

plugins {
    id("com.android.library")
    id("org.jetbrains.kotlin.android")
    id("maven-publish")
    id("signing")
}

val ktorVersion: String = findProperty("omniinfer.ktor.version")?.toString() ?: "3.1.3"
val liteRtLmVersion: String = findProperty("omniinfer.litertlm.version")?.toString() ?: "0.11.0"
val omniInferMavenGroup: String =
    findProperty("omniinfer.maven.group")?.toString() ?: "io.github.omnimind-ai"
val omniInferMavenArtifact: String =
    findProperty("omniinfer.maven.artifact")?.toString() ?: "omniinfer"
val omniInferMavenVersion: String =
    findProperty("omniinfer.maven.version")?.toString() ?: "0.1.0-SNAPSHOT"
val omniInferMavenRepo: String =
    findProperty("omniinfer.maven.repo")?.toString() ?: layout.buildDirectory.dir("repo").get().asFile.absolutePath
val omniInferRepoDir: File = projectDir.parentFile.parentFile
val omniInferMavenScmUrl: String =
    findProperty("omniinfer.maven.scm.url")?.toString() ?: "https://github.com/omnimind-ai/OmniInfer"
val omniInferMavenScmConnection: String =
    findProperty("omniinfer.maven.scm.connection")?.toString()
        ?: "scm:git:https://github.com/omnimind-ai/OmniInfer.git"
val omniInferMavenScmDeveloperConnection: String =
    findProperty("omniinfer.maven.scm.developerConnection")?.toString()
        ?: "scm:git:ssh://git@github.com/omnimind-ai/OmniInfer.git"
val omniInferMavenDeveloperId: String =
    findProperty("omniinfer.maven.developer.id")?.toString() ?: "omnimind-ai"
val omniInferMavenDeveloperName: String =
    findProperty("omniinfer.maven.developer.name")?.toString() ?: "OmniMind AI"
val omniInferMavenDeveloperUrl: String =
    findProperty("omniinfer.maven.developer.url")?.toString() ?: "https://github.com/omnimind-ai"
val hasInMemorySigningKey: Boolean =
    !findProperty("signingInMemoryKey")?.toString().isNullOrBlank()

group = omniInferMavenGroup
version = omniInferMavenVersion

fun boolProperty(name: String, default: Boolean = true): Boolean =
    findProperty(name)?.toString()?.toBooleanStrictOrNull() ?: default

fun isDynamicDependencyVersion(version: String): Boolean =
    version.contains("+") ||
        version.startsWith("latest.", ignoreCase = true) ||
        version.contains("[") ||
        version.contains("]") ||
        version.contains("(") ||
        version.contains(")")

val enableLlamaCpp: Boolean = boolProperty("omniinfer.backend.llama_cpp")
val enableMnn: Boolean = boolProperty("omniinfer.backend.mnn")
val enableExecutorchQnn: Boolean = boolProperty("omniinfer.backend.executorch_qnn")
val enableLiteRtLm: Boolean = boolProperty("omniinfer.backend.litert_lm")
val requireLiteRtLmInPublication: Boolean = boolProperty("omniinfer.publication.require_litert_lm")
if (enableLiteRtLm && isDynamicDependencyVersion(liteRtLmVersion)) {
    throw GradleException(
        "omniinfer.litertlm.version must be a pinned release version, got '$liteRtLmVersion'. " +
            "Publish a new OmniInfer version when upgrading LiteRT-LM."
    )
}
val enableMnnThreadPool: Boolean = boolProperty("omniinfer.mnn.thread_pool")
val llamaCppHtpPrebuiltDir: File =
    findProperty("omniinfer.llama_cpp.htp_prebuilt_dir")?.toString()?.let(::File)
        ?: File(omniInferRepoDir, "tmp/llama-cpp-submodule-snapdragon-minpkg-8a091c47/lib")
val enableLlamaCppHtp: Boolean =
    boolProperty("omniinfer.backend.llama_cpp_htp", enableLlamaCpp && llamaCppHtpPrebuiltDir.isDirectory)
val llamaCppRuntimeJniDir = layout.buildDirectory.dir("generated/llamaCppRuntimeJniLibs")
val llamaCppHtpRuntimeFiles = listOf(
    "libggml-opencl.so",
    "libggml-hexagon.so",
    "libggml-htp-v68.so",
    "libggml-htp-v69.so",
    "libggml-htp-v73.so",
    "libggml-htp-v75.so",
    "libggml-htp-v79.so",
    "libggml-htp-v81.so",
)

val syncLlamaCppHtpJniLibs by tasks.registering {
    description = "Collect llama.cpp Snapdragon HTP runtime libraries for AAR packaging"
    onlyIf { enableLlamaCppHtp }
    outputs.dir(llamaCppRuntimeJniDir)
    doLast {
        val outputDir = llamaCppRuntimeJniDir.get().dir("arm64-v8a").asFile
        outputDir.mkdirs()
        llamaCppHtpRuntimeFiles.forEach { name ->
            val prebuiltLib = File(llamaCppHtpPrebuiltDir, name).takeIf { it.isFile }
            val source = prebuiltLib ?: throw GradleException(
                "Missing llama.cpp HTP runtime library $name. " +
                    "Pass -Pomniinfer.llama_cpp.htp_prebuilt_dir=/path/to/llama.cpp/lib " +
                    "or disable -Pomniinfer.backend.llama_cpp_htp=false."
            )
            source.copyTo(File(outputDir, name), overwrite = true)
        }
    }
}

// --- ExecuTorch QNN: auto-download pre-built binaries ---
if (enableExecutorchQnn) {
    val etQnnVersion = 3  // bump when uploading new binaries to OSS
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
                arguments += "-DBUILD_SHARED_LIBS=ON"
                if (enableLlamaCpp) {
                    arguments += "-DGGML_NATIVE=OFF"
                    arguments += "-DGGML_LLAMAFILE=OFF"
                    arguments += "-DLLAMA_BUILD_COMMON=ON"
                    arguments += "-DGGML_BACKEND_DL=ON"
                    arguments += "-DGGML_CPU_ALL_VARIANTS=ON"
                }
                arguments += "-DOMNIINFER_BACKEND_LLAMA_CPP=${if (enableLlamaCpp) "ON" else "OFF"}"
                arguments += "-DOMNIINFER_BACKEND_MNN=${if (enableMnn) "ON" else "OFF"}"
                arguments += "-DOMNIINFER_BACKEND_EXECUTORCH_QNN=${if (enableExecutorchQnn) "ON" else "OFF"}"
                if (enableMnn) {
                    arguments += "-DMNN_USE_THREAD_POOL=${if (enableMnnThreadPool) "ON" else "OFF"}"
                }
            }
        }
    }

    sourceSets {
        getByName("main") {
            if (enableLiteRtLm) {
                java.srcDir("src/litertLm/java")
            }
            val jniLibDirs = mutableListOf<File>()
            if (enableLlamaCppHtp) {
                jniLibDirs += llamaCppRuntimeJniDir.get().asFile
            }
            if (enableExecutorchQnn) {
                jniLibDirs += file("src/main/jniLibs")
            }
            jniLibs.setSrcDirs(jniLibDirs)
        }
    }

    publishing {
        singleVariant("release") {
            withSourcesJar()
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
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

kotlin {
    compilerOptions {
        jvmTarget.set(JvmTarget.JVM_17)
    }
}

dependencies {
    implementation("io.ktor:ktor-server-core:$ktorVersion")
    implementation("io.ktor:ktor-server-cio:$ktorVersion")
    implementation("io.ktor:ktor-server-content-negotiation:$ktorVersion")
    implementation("io.ktor:ktor-serialization-kotlinx-json:$ktorVersion")
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.6.3")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")
    implementation("androidx.core:core-ktx:1.12.0")
    if (enableLiteRtLm) {
        implementation("com.google.ai.edge.litertlm:litertlm-android:$liteRtLmVersion")
    }
}

tasks.matching { it.name.startsWith("merge") && it.name.contains("JniLibFolders") }
    .configureEach { dependsOn(syncLlamaCppHtpJniLibs) }

val mavenCentralJavadocDir = layout.buildDirectory.dir("generated/mavenCentralJavadoc")

val generateMavenCentralJavadoc by tasks.registering {
    description = "Generate placeholder API documentation for Maven Central publication"
    outputs.dir(mavenCentralJavadocDir)
    doLast {
        val outputFile = mavenCentralJavadocDir.get().file("README.md").asFile
        outputFile.parentFile.mkdirs()
        outputFile.writeText(
            """
            # OmniInfer Android

            Android local inference server for llama.cpp CPU/HTP and LiteRT-LM GPU.
            API entry point: com.omniinfer.server.OmniInferServer.
            """.trimIndent() + "\n",
        )
    }
}

val mavenCentralJavadocJar by tasks.registering(Jar::class) {
    description = "Package API documentation for Maven Central publication"
    archiveClassifier.set("javadoc")
    dependsOn(generateMavenCentralJavadoc)
    from(mavenCentralJavadocDir)
}

afterEvaluate {
    publishing {
        publications {
            create<MavenPublication>("release") {
                from(components["release"])
                groupId = omniInferMavenGroup
                artifactId = omniInferMavenArtifact
                version = omniInferMavenVersion
                artifact(mavenCentralJavadocJar)
                pom {
                    name.set("OmniInfer Android")
                    description.set("Android local inference server for llama.cpp CPU/HTP and LiteRT-LM GPU.")
                    inceptionYear.set("2026")
                    url.set(omniInferMavenScmUrl)
                    licenses {
                        license {
                            name.set("The Apache License, Version 2.0")
                            url.set("https://www.apache.org/licenses/LICENSE-2.0.txt")
                            distribution.set("repo")
                        }
                    }
                    developers {
                        developer {
                            id.set(omniInferMavenDeveloperId)
                            name.set(omniInferMavenDeveloperName)
                            url.set(omniInferMavenDeveloperUrl)
                        }
                    }
                    scm {
                        url.set(omniInferMavenScmUrl)
                        connection.set(omniInferMavenScmConnection)
                        developerConnection.set(omniInferMavenScmDeveloperConnection)
                    }
                }
            }
        }
        repositories {
            maven {
                name = "omniInferLocal"
                url = uri(omniInferMavenRepo)
            }
        }
    }

    if (hasInMemorySigningKey) {
        signing {
            useInMemoryPgpKeys(
                findProperty("signingInMemoryKey")?.toString(),
                findProperty("signingInMemoryKeyPassword")?.toString(),
            )
            sign(publishing.publications["release"])
        }
    }
}

tasks.register<Zip>("bundleMavenCentralPublication") {
    description = "Create a Maven Central upload bundle from the local Maven repository"
    group = "publishing"
    dependsOn("publishReleasePublicationToOmniInferLocalRepository")
    archiveFileName.set("${omniInferMavenArtifact}-${omniInferMavenVersion}-maven-central-bundle.zip")
    destinationDirectory.set(layout.buildDirectory.dir("distributions"))
    val publicationPath = "${omniInferMavenGroup.replace('.', '/')}/$omniInferMavenArtifact/$omniInferMavenVersion"
    from(File(omniInferMavenRepo, publicationPath)) {
        into(publicationPath)
    }
}

val verifyAarDependencyMetadata by tasks.registering {
    description = "Verify Maven metadata keeps runtime dependencies transitive and pinned"
    group = "verification"
    dependsOn("publishReleasePublicationToOmniInferLocalRepository")
    doLast {
        val publicationPath = "${omniInferMavenGroup.replace('.', '/')}/$omniInferMavenArtifact/$omniInferMavenVersion"
        val publicationDir = File(omniInferMavenRepo, publicationPath)
        val pomFile = File(publicationDir, "$omniInferMavenArtifact-$omniInferMavenVersion.pom")
        val aarFile = File(publicationDir, "$omniInferMavenArtifact-$omniInferMavenVersion.aar")

        if (!pomFile.isFile) {
            throw GradleException("Missing generated POM: ${pomFile.absolutePath}")
        }
        if (!aarFile.isFile) {
            throw GradleException("Missing generated AAR: ${aarFile.absolutePath}")
        }

        fun dependencyVersion(groupId: String, artifactId: String): String? {
            val document = DocumentBuilderFactory.newInstance().newDocumentBuilder().parse(pomFile)
            val dependencies = document.getElementsByTagName("dependency")
            for (index in 0 until dependencies.length) {
                val dependency = dependencies.item(index) as? Element ?: continue
                fun childText(name: String): String? {
                    val nodes = dependency.getElementsByTagName(name)
                    return if (nodes.length > 0) nodes.item(0).textContent.trim() else null
                }
                if (childText("groupId") == groupId && childText("artifactId") == artifactId) {
                    return childText("version")
                }
            }
            return null
        }

        if (requireLiteRtLmInPublication && !enableLiteRtLm) {
            throw GradleException(
                "This publication is expected to include LiteRT-LM. " +
                    "Set -Pomniinfer.backend.litert_lm=true, or explicitly pass " +
                    "-Pomniinfer.publication.require_litert_lm=false for a custom trimmed artifact."
            )
        }

        if (enableLiteRtLm || requireLiteRtLmInPublication) {
            val publishedLiteRtVersion = dependencyVersion(
                "com.google.ai.edge.litertlm",
                "litertlm-android",
            )
            if (publishedLiteRtVersion == null) {
                throw GradleException(
                    "Generated POM does not declare com.google.ai.edge.litertlm:litertlm-android. " +
                        "Third-party apps would have to add LiteRT-LM manually."
                )
            }
            if (publishedLiteRtVersion != liteRtLmVersion) {
                throw GradleException(
                    "Generated POM declares LiteRT-LM $publishedLiteRtVersion, expected $liteRtLmVersion."
                )
            }
            if (isDynamicDependencyVersion(publishedLiteRtVersion)) {
                throw GradleException("Generated POM uses dynamic LiteRT-LM version: $publishedLiteRtVersion")
            }
        }

        ZipFile(aarFile).use { zip ->
            val x86Entries = zip.entries().asSequence()
                .map { it.name }
                .filter { it.startsWith("jni/x86_64/") }
                .toList()
            if (x86Entries.isNotEmpty()) {
                throw GradleException(
                    "OmniInfer AAR must not package x86_64 native libraries: ${x86Entries.joinToString()}"
                )
            }
        }
    }
}

tasks.named("bundleMavenCentralPublication") {
    dependsOn(verifyAarDependencyMetadata)
}
