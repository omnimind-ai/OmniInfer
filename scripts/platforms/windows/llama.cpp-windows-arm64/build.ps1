param(
    [string]$BuildType = "Release"
)

$ErrorActionPreference = "Stop"

$ScriptRoot = $PSScriptRoot
$RepoRoot = Resolve-Path (Join-Path $ScriptRoot "..\..\..\..")
$PackageRoot = Join-Path $RepoRoot ".local\runtime\windows\llama.cpp-windows-arm64"
$LlamaRoot = Resolve-Path (Join-Path $RepoRoot "framework\llama.cpp")
$BuildRoot = Join-Path $PackageRoot "build\llama.cpp-windows-arm64"
$BinRoot = Join-Path $PackageRoot "bin"
$ToolchainFile = Join-Path $LlamaRoot "cmake\arm64-windows-llvm.cmake"

function Require-Command {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command '$Name' was not found in PATH."
    }
}

function Copy-LibompRuntime {
    param([string]$Destination)

    $patterns = @(
        "C:\Program Files\Microsoft Visual Studio\2022\*\VC\Redist\MSVC\*\debug_nonredist\arm64\Microsoft.VC143.OpenMP.LLVM\libomp140.aarch64.dll",
        "C:\Program Files\Microsoft Visual Studio\2022\*\VC\Redist\MSVC\*\arm64\Microsoft.VC143.OpenMP.LLVM\libomp140.aarch64.dll"
    )

    foreach ($pattern in $patterns) {
        Get-ChildItem -Path $pattern -ErrorAction SilentlyContinue | ForEach-Object {
            Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $Destination $_.Name) -Force
        }
    }
}

Require-Command cmake
Require-Command ninja
Require-Command cl

if (-not (Test-Path -LiteralPath $ToolchainFile)) {
    throw "Required llama.cpp toolchain file was not found: $ToolchainFile"
}

$configureArgs = @(
    "-S", $LlamaRoot,
    "-B", $BuildRoot,
    "-G", "Ninja Multi-Config",
    "-DCMAKE_TOOLCHAIN_FILE=$($ToolchainFile.Replace('\','/'))",
    "-DLLAMA_BUILD_BORINGSSL=ON",
    "-DLLAMA_BUILD_TESTS=OFF",
    "-DLLAMA_BUILD_EXAMPLES=OFF",
    "-DLLAMA_BUILD_SERVER=ON",
    "-DGGML_NATIVE=OFF",
    "-DGGML_BACKEND_DL=ON",
    "-DGGML_CPU_ALL_VARIANTS=OFF",
    "-DGGML_OPENMP=ON"
)

New-Item -ItemType Directory -Force -Path $BuildRoot, $BinRoot | Out-Null

Write-Host "Configuring llama.cpp Windows arm64 build..."
cmake @configureArgs

Write-Host "Building arm64 llama-server.exe..."
cmake --build $BuildRoot --config $BuildType --target llama-server

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

Copy-LibompRuntime -Destination $BinRoot

if (-not (Test-Path (Join-Path $BinRoot "llama-server.exe"))) {
    throw "Build finished but llama-server.exe was not copied into $BinRoot."
}

Write-Host ""
Write-Host "Windows arm64 build complete."
Write-Host "Binary package location: $BinRoot"
