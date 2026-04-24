param(
    [string]$BuildType = "Release",
    [string]$CudaArchitectures = ""
)

$ErrorActionPreference = "Stop"

$ScriptRoot = $PSScriptRoot
$RepoRoot = Resolve-Path (Join-Path $ScriptRoot "..\..\..\..")
$PackageRoot = Join-Path $RepoRoot ".local\runtime\windows\ik_llama.cpp-cuda"
$LlamaRoot = Resolve-Path (Join-Path $RepoRoot "framework\ik_llama.cpp")
$BuildRoot = Join-Path $PackageRoot "build\ik_llama.cpp-cuda"
$BinRoot = Join-Path $PackageRoot "bin"

function Require-Command {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command '$Name' was not found in PATH."
    }
}

function Find-Msys2Ucrt64Toolchain {
    $candidates = @()
    if ($env:MSYS2_ROOT) { $candidates += $env:MSYS2_ROOT }
    foreach ($key in @("HKLM:\SOFTWARE\MSYS2","HKCU:\SOFTWARE\MSYS2","HKLM:\SOFTWARE\WOW6432Node\MSYS2")) {
        try { $loc = (Get-ItemProperty -Path $key -ErrorAction SilentlyContinue).InstallLocation; if ($loc) { $candidates += $loc } } catch {}
    }
    $gccInPath = Get-Command gcc.exe -ErrorAction SilentlyContinue
    if ($gccInPath) { $binDir = Split-Path $gccInPath.Source; if ($binDir -match 'ucrt64\\bin$') { $candidates += (Split-Path (Split-Path $binDir)) } }
    foreach ($drive in (Get-PSDrive -PSProvider FileSystem -ErrorAction SilentlyContinue)) {
        $candidates += Join-Path $drive.Root "msys64"; $candidates += Join-Path $drive.Root "msys2"
    }
    if ($env:ChocolateyInstall) { $candidates += Join-Path $env:ChocolateyInstall "lib\msys2\msys64" }
    if ($env:SCOOP) { $candidates += Join-Path $env:SCOOP "apps\msys2\current" }

    foreach ($msys2Root in $candidates) {
        $ucrt64Bin = Join-Path $msys2Root "ucrt64\bin"
        if (-not (Test-Path $ucrt64Bin)) { continue }
        $gcc   = Join-Path $ucrt64Bin "gcc.exe"
        $gpp   = Join-Path $ucrt64Bin "g++.exe"
        $ninja = Join-Path $ucrt64Bin "ninja.exe"
        if ((Test-Path $gcc) -and (Test-Path $gpp) -and (Test-Path $ninja)) {
            return @{ Root = $ucrt64Bin; Gcc = $gcc; Gpp = $gpp; Ninja = $ninja }
        }
    }
    return $null
}

Require-Command cmake
Require-Command nvcc

if (-not (Get-Command cl -ErrorAction SilentlyContinue)) {
    throw "CUDA builds on Windows require MSVC cl.exe in PATH. Please open a Visual Studio 2022 Developer PowerShell (or install Build Tools with the C++ workload) and retry."
}

$configureArgs = @(
    "-S", $LlamaRoot,
    "-B", $BuildRoot,
    "-DCMAKE_BUILD_TYPE=$BuildType",
    "-DBUILD_SHARED_LIBS=OFF",
    "-DLLAMA_BUILD_TESTS=OFF",
    "-DLLAMA_BUILD_EXAMPLES=ON",
    "-DLLAMA_BUILD_SERVER=ON",
    "-DLLAMA_OPENSSL=OFF",
    "-DGGML_CUDA=ON",
    "-DGGML_NATIVE=OFF",
    "-DGGML_WIN_VER=0x0A00"
)

if ($CudaArchitectures) {
    $configureArgs += "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchitectures"
}

$buildArgs = @()

if ((Get-Command cl -ErrorAction SilentlyContinue) -and (Get-Command nmake -ErrorAction SilentlyContinue)) {
    $configureArgs += @("-G", "NMake Makefiles")
    $buildArgs = @("--", "/NOLOGO")
    $toolchainKind = "msvc-nmake"
} else {
    $msys2 = Find-Msys2Ucrt64Toolchain
    if ($msys2) {
        $env:PATH = "$($msys2.Root);$env:PATH"
        $configureArgs += @(
            "-G", "Ninja",
            "-DCMAKE_MAKE_PROGRAM=$($msys2.Ninja.Replace('\','/'))"
        )
        $toolchainKind = "msys2-ucrt64-ninja"
    } else {
        Require-Command ninja
        $configureArgs += @("-G", "Ninja")
        $toolchainKind = "ninja"
    }
}

New-Item -ItemType Directory -Force -Path $BuildRoot, $BinRoot | Out-Null

Write-Host "Configuring ik_llama.cpp CUDA build..."
cmake @configureArgs
if ($LASTEXITCODE -ne 0) {
    throw "CMake configure failed for ik_llama.cpp CUDA build."
}

Write-Host "Building ik_llama.cpp CUDA llama-server.exe..."
cmake --build $BuildRoot --target llama-server --config $BuildType @buildArgs
if ($LASTEXITCODE -ne 0) {
    throw "CMake build failed for ik_llama.cpp CUDA build."
}

Get-ChildItem (Join-Path $BuildRoot "bin") -File | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $BinRoot $_.Name) -Force
}

if ($env:CUDA_PATH) {
    $cudaBin = Join-Path $env:CUDA_PATH "bin"
    Get-ChildItem $cudaBin -Filter "cudart64*.dll" -ErrorAction SilentlyContinue | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $BinRoot $_.Name) -Force
    }
    Get-ChildItem $cudaBin -Filter "cublas64*.dll" -ErrorAction SilentlyContinue | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $BinRoot $_.Name) -Force
    }
    Get-ChildItem $cudaBin -Filter "cublasLt64*.dll" -ErrorAction SilentlyContinue | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $BinRoot $_.Name) -Force
    }
}

if (-not (Test-Path (Join-Path $BinRoot "llama-server.exe"))) {
    throw "Build finished but llama-server.exe was not copied into $BinRoot."
}

Write-Host ""
Write-Host "ik_llama.cpp CUDA build complete."
Write-Host "Toolchain: $toolchainKind"
Write-Host "Binary package location: $BinRoot"
