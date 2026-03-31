param(
    [string]$BuildType = "Release"
)

$ErrorActionPreference = "Stop"

$ScriptRoot = $PSScriptRoot
$RepoRoot = Resolve-Path (Join-Path $ScriptRoot "..\..\..\..")
$PackageRoot = Join-Path $RepoRoot ".local\runtime\windows\llama.cpp-sycl"
$LlamaRoot = Resolve-Path (Join-Path $RepoRoot "framework\llama.cpp")
$BuildRoot = Join-Path $PackageRoot "build\llama.cpp-sycl"
$BinRoot = Join-Path $PackageRoot "bin"
$OneApiRoot = if ($env:ONEAPI_ROOT) { $env:ONEAPI_ROOT } else { "C:\Program Files (x86)\Intel\oneAPI" }

function Require-Command {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command '$Name' was not found in PATH."
    }
}

function Copy-OneApiRuntime {
    param(
        [string]$OneApiRootPath,
        [string]$Destination
    )

    $runtimeFiles = @(
        "mkl\latest\bin\mkl_sycl_blas.5.dll",
        "mkl\latest\bin\mkl_core.2.dll",
        "mkl\latest\bin\mkl_tbb_thread.2.dll",
        "compiler\latest\bin\ur_adapter_level_zero.dll",
        "compiler\latest\bin\ur_adapter_level_zero_v2.dll",
        "compiler\latest\bin\ur_adapter_opencl.dll",
        "compiler\latest\bin\ur_loader.dll",
        "compiler\latest\bin\ur_win_proxy_loader.dll",
        "compiler\latest\bin\sycl8.dll",
        "compiler\latest\bin\svml_dispmd.dll",
        "compiler\latest\bin\libmmd.dll",
        "compiler\latest\bin\libiomp5md.dll",
        "compiler\latest\bin\sycl-ls.exe",
        "compiler\latest\bin\libsycl-fallback-bfloat16.spv",
        "compiler\latest\bin\libsycl-native-bfloat16.spv",
        "dnnl\latest\bin\dnnl.dll",
        "tbb\latest\bin\tbb12.dll",
        "tcm\latest\bin\tcm.dll",
        "tcm\latest\bin\libhwloc-15.dll",
        "umf\latest\bin\umf.dll"
    )

    foreach ($relativePath in $runtimeFiles) {
        $source = Join-Path $OneApiRootPath $relativePath
        if (Test-Path -LiteralPath $source) {
            Copy-Item -LiteralPath $source -Destination (Join-Path $Destination (Split-Path $source -Leaf)) -Force
        }
    }
}

Require-Command cmake
Require-Command ninja
Require-Command cl
Require-Command icx

if (-not (Test-Path -LiteralPath $OneApiRoot)) {
    throw "Intel oneAPI root was not found at $OneApiRoot. Open the Intel oneAPI command prompt or set ONEAPI_ROOT."
}

$configureArgs = @(
    "-S", $LlamaRoot,
    "-B", $BuildRoot,
    "-G", "Ninja",
    "-DCMAKE_C_COMPILER=cl",
    "-DCMAKE_CXX_COMPILER=icx",
    "-DCMAKE_BUILD_TYPE=$BuildType",
    "-DBUILD_SHARED_LIBS=ON",
    "-DGGML_BACKEND_DL=ON",
    "-DGGML_CPU=OFF",
    "-DGGML_NATIVE=OFF",
    "-DGGML_SYCL=ON",
    "-DLLAMA_BUILD_BORINGSSL=ON",
    "-DLLAMA_BUILD_TESTS=OFF",
    "-DLLAMA_BUILD_EXAMPLES=OFF",
    "-DLLAMA_BUILD_SERVER=ON"
)

New-Item -ItemType Directory -Force -Path $BuildRoot, $BinRoot | Out-Null

Write-Host "Configuring llama.cpp SYCL build..."
cmake @configureArgs

Write-Host "Building SYCL llama-server.exe..."
cmake --build $BuildRoot --target llama-server --config $BuildType -j

$buildBinCandidates = @(
    (Join-Path $BuildRoot "bin\$BuildType"),
    (Join-Path $BuildRoot "bin")
)

$buildBin = $buildBinCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $buildBin) {
    throw "Build finished but no build output directory was found under $BuildRoot\bin."
}

Get-ChildItem $buildBin -File | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $BinRoot $_.Name) -Force
}

Copy-OneApiRuntime -OneApiRootPath $OneApiRoot -Destination $BinRoot

if (-not (Test-Path (Join-Path $BinRoot "llama-server.exe"))) {
    throw "Build finished but llama-server.exe was not copied into $BinRoot."
}

Write-Host ""
Write-Host "SYCL build complete."
Write-Host "Binary package location: $BinRoot"
