param(
    [string]$BuildType = "Release",
    [string]$GpuTargets = "",
    [string]$RocwmmaInclude = ""
)

$ErrorActionPreference = "Stop"

$ScriptRoot = $PSScriptRoot
$RepoRoot = Resolve-Path (Join-Path $ScriptRoot "..\..\..\..")
$PackageRoot = Join-Path $RepoRoot ".local\runtime\windows\llama.cpp-hip"
$LlamaRoot = Resolve-Path (Join-Path $RepoRoot "framework\llama.cpp")
$BuildRoot = Join-Path $PackageRoot "build\llama.cpp-hip"
$BinRoot = Join-Path $PackageRoot "bin"

function Require-Command {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command '$Name' was not found in PATH."
    }
}

function Resolve-HipRoot {
    if ($env:HIP_PATH -and (Test-Path -LiteralPath $env:HIP_PATH)) {
        return $env:HIP_PATH
    }

    $clangPath = Get-ChildItem 'C:\Program Files\AMD\ROCm\*\bin\clang.exe' -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($clangPath) {
        return (Split-Path $clangPath.FullName -Parent | Split-Path -Parent)
    }

    throw "AMD HIP SDK was not found. Install ROCm for Windows or set HIP_PATH before running this script."
}

function Copy-DirectoryContents {
    param(
        [string]$Source,
        [string]$Destination
    )

    if (-not (Test-Path -LiteralPath $Source)) {
        return
    }

    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    Copy-Item -Path (Join-Path $Source "*") -Destination $Destination -Recurse -Force
}

Require-Command cmake
Require-Command ninja

$HipRoot = Resolve-HipRoot
$HipClang = Join-Path $HipRoot "bin\clang.exe"
$HipClangxx = Join-Path $HipRoot "bin\clang++.exe"

if (-not (Test-Path -LiteralPath $HipClang) -or -not (Test-Path -LiteralPath $HipClangxx)) {
    throw "HIP compiler binaries were not found under $HipRoot\bin."
}

$env:CMAKE_PREFIX_PATH = $HipRoot

$configureArgs = @(
    "-S", $LlamaRoot,
    "-B", $BuildRoot,
    "-G", "Ninja",
    "-DCMAKE_BUILD_TYPE=$BuildType",
    "-DBUILD_SHARED_LIBS=ON",
    "-DGGML_BACKEND_DL=ON",
    "-DGGML_CPU=OFF",
    "-DGGML_NATIVE=OFF",
    "-DGGML_HIP=ON",
    "-DLLAMA_BUILD_BORINGSSL=ON",
    "-DLLAMA_BUILD_TESTS=OFF",
    "-DLLAMA_BUILD_EXAMPLES=OFF",
    "-DLLAMA_BUILD_SERVER=ON",
    "-DCMAKE_C_COMPILER=$($HipClang.Replace('\','/'))",
    "-DCMAKE_CXX_COMPILER=$($HipClangxx.Replace('\','/'))"
)

if ($GpuTargets) {
    $configureArgs += "-DGPU_TARGETS=$GpuTargets"
}

if ($RocwmmaInclude) {
    $normalized = $RocwmmaInclude.Replace('\', '/')
    $configureArgs += "-DCMAKE_CXX_FLAGS=-I${normalized} -Wno-ignored-attributes -Wno-nested-anon-types"
}

New-Item -ItemType Directory -Force -Path $BuildRoot, $BinRoot | Out-Null

Write-Host "Configuring llama.cpp HIP build..."
cmake @configureArgs

Write-Host "Building HIP llama-server.exe..."
cmake --build $BuildRoot --target llama-server --config $BuildType -j $env:NUMBER_OF_PROCESSORS

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

@(
    "libhipblas.dll",
    "libhipblaslt.dll",
    "rocblas.dll"
) | ForEach-Object {
    $source = Join-Path $HipRoot "bin\$_"
    if (Test-Path -LiteralPath $source) {
        Copy-Item -LiteralPath $source -Destination (Join-Path $BinRoot $_) -Force
    }
}

Copy-DirectoryContents -Source (Join-Path $HipRoot "bin\rocblas\library") -Destination (Join-Path $BinRoot "rocblas\library")
Copy-DirectoryContents -Source (Join-Path $HipRoot "bin\hipblaslt\library") -Destination (Join-Path $BinRoot "hipblaslt\library")

if (-not (Test-Path (Join-Path $BinRoot "llama-server.exe"))) {
    throw "Build finished but llama-server.exe was not copied into $BinRoot."
}

Write-Host ""
Write-Host "HIP build complete."
Write-Host "Binary package location: $BinRoot"
