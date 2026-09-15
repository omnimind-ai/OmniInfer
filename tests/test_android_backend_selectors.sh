#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_root="$(mktemp -d)"
trap 'rm -rf -- "${test_root}"' EXIT
sources=(
    "${repo_root}/android/omniinfer-server/src/main/java/com/omniinfer/server/BackendSelectors.kt"
    "${repo_root}/android/omniinfer-server/src/test/kotlin/com/omniinfer/server/BackendSelectorsTest.kt"
)
if [[ -n "${KOTLIN_COMPILER_CLASSPATH:-}" ]]; then
    # Reuse an existing Gradle compiler installation without Android/Gradle builds.
    java -cp "${KOTLIN_COMPILER_CLASSPATH}" org.jetbrains.kotlin.cli.jvm.K2JVMCompiler \
        -no-stdlib -no-reflect -classpath "${KOTLIN_STDLIB:?}" \
        "${sources[@]}" -d "${test_root}/tests.jar"
    java -cp "${test_root}/tests.jar:${KOTLIN_COMPILER_CLASSPATH}" com.omniinfer.server.BackendSelectorsTestKt
else
    kotlinc "${sources[@]}" -include-runtime -d "${test_root}/tests.jar"
    java -jar "${test_root}/tests.jar"
fi
