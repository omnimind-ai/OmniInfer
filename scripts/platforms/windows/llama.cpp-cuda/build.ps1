param(
    [string]$BuildType = "Release",
    [string]$CudaArchitectures = "",
    [switch]$BuildWebUI
)

$ErrorActionPreference = "Stop"

$ScriptRoot = $PSScriptRoot
$RepoRoot = Resolve-Path (Join-Path $ScriptRoot "..\..\..\..")
$PackageRoot = Join-Path $RepoRoot ".local\runtime\windows\llama.cpp-cuda"
$LlamaRoot = Resolve-Path (Join-Path $RepoRoot "framework\llama.cpp")
$BuildRoot = Join-Path $PackageRoot "build\llama.cpp-cuda"
$BinRoot = Join-Path $PackageRoot "bin"

function Require-Command {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command '$Name' was not found in PATH."
    }
}

function Get-CMakeCacheValue {
    param([string]$CachePath, [string]$Name)
    if (-not (Test-Path -LiteralPath $CachePath)) {
        return $null
    }
    $pattern = "^$([regex]::Escape($Name)):[^=]*=(.*)$"
    foreach ($line in Get-Content -LiteralPath $CachePath) {
        if ($line -match $pattern) {
            return $matches[1]
        }
    }
    return $null
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

function Test-MsvcLinkEnvironment {
    foreach ($libDir in (($env:LIB -split ';') | Where-Object { $_ })) {
        if (Test-Path -LiteralPath (Join-Path $libDir "kernel32.lib")) {
            return $true
        }
    }
    return $false
}

function Find-NinjaExecutable {
    $ninjaInPath = Get-Command ninja.exe -ErrorAction SilentlyContinue
    if ($ninjaInPath) {
        return $ninjaInPath.Source
    }

    $msys2 = Find-Msys2Ucrt64Toolchain
    if ($msys2) {
        return $msys2.Ninja
    }

    return $null
}

function Find-And-Load-Msvc {
    if ((Get-Command cl -ErrorAction SilentlyContinue) -and (Test-MsvcLinkEnvironment)) { return $true }

    $vcvarsallCandidates = @()

    # Try vswhere first
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $installPath = & $vswhere -latest -property installationPath 2>$null
        if ($installPath) {
            $vcvarsallCandidates += Join-Path $installPath "VC\Auxiliary\Build\vcvarsall.bat"
        }
    }

    # Search common directories on all drives
    foreach ($drive in (Get-PSDrive -PSProvider FileSystem -ErrorAction SilentlyContinue)) {
        $root = $drive.Root
        $vcvarsallCandidates += Join-Path $root "Coding\Tools\VS\BuildTools2022\VC\Auxiliary\Build\vcvarsall.bat"
        $vcvarsallCandidates += Join-Path $root "BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
        foreach ($year in @("2022","2019")) {
            $vcvarsallCandidates += Join-Path $root "Program Files\Microsoft Visual Studio\$year\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
            $vcvarsallCandidates += Join-Path $root "Program Files\Microsoft Visual Studio\$year\Community\VC\Auxiliary\Build\vcvarsall.bat"
            $vcvarsallCandidates += Join-Path $root "Program Files (x86)\Microsoft Visual Studio\$year\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
        }
    }

    # Determine MSVC toolset version constraint from CUDA version.
    # CUDA 11.x requires MSVC <= 14.29; CUDA 12.x+ works with newer MSVC.
    $vcvarsVerArg = ""
    $nvccOutput = nvcc --version 2>&1 | Out-String
    if ($nvccOutput -match 'release (\d+)\.(\d+)') {
        $cudaMajor = [int]$matches[1]
        if ($cudaMajor -lt 12) {
            $vcvarsVerArg = "-vcvars_ver=14.29"
            Write-Host "CUDA $cudaMajor detected: requesting MSVC toolset 14.29 for compatibility"
        }
    }

    foreach ($vcvarsall in $vcvarsallCandidates) {
        if (-not (Test-Path $vcvarsall)) { continue }
        Write-Host "Found MSVC: $vcvarsall"
        $vcvarsCmd = "x64 $vcvarsVerArg".Trim()
        Write-Host "Loading MSVC environment: vcvarsall.bat $vcvarsCmd"
        $envLines = cmd /c "`"$vcvarsall`" $vcvarsCmd >nul 2>&1 && set" 2>&1
        foreach ($line in $envLines) {
            if ($line -match '^([^=]+)=(.*)$') {
                [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
            }
        }
        if ((Get-Command cl -ErrorAction SilentlyContinue) -and (Test-MsvcLinkEnvironment)) {
            $clPath = (Get-Command cl).Source
            Write-Host "MSVC loaded: $clPath"
            return $true
        }
    }
    return $false
}

function Find-MsvcRedistRoot {
    $candidates = @()
    if ($env:VCToolsRedistDir) {
        $candidates += $env:VCToolsRedistDir
    }

    $vcInstallRoots = @()
    if ($env:VCINSTALLDIR) {
        $vcInstallRoots += $env:VCINSTALLDIR
    }

    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path -LiteralPath $vswhere) {
        $installPath = & $vswhere -latest -property installationPath 2>$null
        if ($installPath) {
            $vcInstallRoots += Join-Path $installPath "VC"
        }
    }

    foreach ($drive in (Get-PSDrive -PSProvider FileSystem -ErrorAction SilentlyContinue)) {
        $root = $drive.Root
        $vcInstallRoots += Join-Path $root "Coding\Tools\VS\BuildTools2022\VC"
        foreach ($year in @("2022","2019")) {
            $vcInstallRoots += Join-Path $root "Program Files\Microsoft Visual Studio\$year\BuildTools\VC"
            $vcInstallRoots += Join-Path $root "Program Files\Microsoft Visual Studio\$year\Community\VC"
            $vcInstallRoots += Join-Path $root "Program Files (x86)\Microsoft Visual Studio\$year\BuildTools\VC"
        }
    }

    foreach ($vcRoot in ($vcInstallRoots | Where-Object { $_ } | Select-Object -Unique)) {
        $redistRoot = Join-Path $vcRoot "Redist\MSVC"
        if (Test-Path -LiteralPath $redistRoot) {
            Get-ChildItem -LiteralPath $redistRoot -Directory -ErrorAction SilentlyContinue |
                Where-Object { $_.Name -match '^\d+\.\d+\.\d+$' } |
                Sort-Object Name -Descending |
                ForEach-Object { $candidates += $_.FullName }
        }
    }

    foreach ($candidate in ($candidates | Where-Object { $_ } | Select-Object -Unique)) {
        $crtDir = Get-ChildItem -LiteralPath (Join-Path $candidate "x64") -Directory -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -match '^Microsoft\.VC\d+\.CRT$' } |
            Select-Object -First 1
        if ($crtDir) {
            return $candidate
        }
    }

    return $null
}

function Copy-MsvcRuntimeDlls {
    param([string]$Destination)

    $redistRoot = Find-MsvcRedistRoot
    if (-not $redistRoot) {
        throw "MSVC redistributable DLLs were not found. Install Visual Studio Build Tools with the MSVC redistributable components."
    }

    $x64Root = Join-Path $redistRoot "x64"
    $runtimeDirs = @()
    $runtimeDirs += Get-ChildItem -LiteralPath $x64Root -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^Microsoft\.VC\d+\.CRT$' }
    $runtimeDirs += Get-ChildItem -LiteralPath $x64Root -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^Microsoft\.VC\d+\.OpenMP$' }

    if (-not $runtimeDirs) {
        throw "MSVC x64 runtime directories were not found under $x64Root."
    }

    foreach ($runtimeDir in $runtimeDirs) {
        Get-ChildItem -LiteralPath $runtimeDir.FullName -Filter "*.dll" -File -ErrorAction SilentlyContinue |
            ForEach-Object {
                Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $Destination $_.Name) -Force
            }
    }
}

if (-not (Find-And-Load-Msvc)) {
    throw "CUDA builds require MSVC cl.exe. Install Visual Studio Build Tools or set vcvarsall.bat path."
}

$cachePath = Join-Path $BuildRoot "CMakeCache.txt"
$cachedCudaCompiler = Get-CMakeCacheValue -CachePath $cachePath -Name "CMAKE_CUDA_COMPILER"
$currentCudaCompiler = (Get-Command nvcc).Source
if ($cachedCudaCompiler) {
    $normalizedCachedCudaCompiler = [System.IO.Path]::GetFullPath($cachedCudaCompiler)
    $normalizedCurrentCudaCompiler = [System.IO.Path]::GetFullPath($currentCudaCompiler)
    if (
        -not (Test-Path -LiteralPath $normalizedCachedCudaCompiler) -or
        -not [string]::Equals($normalizedCachedCudaCompiler, $normalizedCurrentCudaCompiler, [System.StringComparison]::OrdinalIgnoreCase)
    ) {
        Write-Host "Cached CUDA compiler is stale. Resetting build directory:"
        Write-Host "  cached:  $cachedCudaCompiler"
        Write-Host "  current: $currentCudaCompiler"
        Remove-Item -LiteralPath $BuildRoot -Recurse -Force
    }
}

$configureArgs = @(
    "-S", $LlamaRoot,
    "-B", $BuildRoot,
    "-DCMAKE_BUILD_TYPE=$BuildType",
    "-DBUILD_SHARED_LIBS=OFF",
    "-DLLAMA_BUILD_TESTS=OFF",
    "-DLLAMA_BUILD_EXAMPLES=OFF",
    "-DLLAMA_BUILD_SERVER=ON",
    "-DLLAMA_BUILD_WEBUI=$(if ($BuildWebUI) { 'ON' } else { 'OFF' })",
    "-DLLAMA_OPENSSL=OFF",
    "-DGGML_CUDA=ON",
    "-DGGML_NATIVE=OFF"
)

if ($CudaArchitectures) {
    $configureArgs += "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchitectures"
}

$buildArgs = @()
$cachedGenerator = Get-CMakeCacheValue -CachePath $cachePath -Name "CMAKE_GENERATOR"

if ($cachedGenerator -eq "Ninja") {
    $ninja = Find-NinjaExecutable
    if ((Get-Command cl -ErrorAction SilentlyContinue) -and (Test-MsvcLinkEnvironment) -and $ninja) {
        $configureArgs += @(
            "-G", "Ninja",
            "-DCMAKE_MAKE_PROGRAM=$($ninja.Replace('\','/'))",
            "-DCMAKE_C_COMPILER=cl",
            "-DCMAKE_CXX_COMPILER=cl"
        )
        $toolchainKind = "msvc-ninja"
    }
    else {
        $msys2 = Find-Msys2Ucrt64Toolchain
        if ($msys2) {
            $env:PATH = "$($msys2.Root);$env:PATH"
            $configureArgs += @(
                "-G", "Ninja",
                "-DCMAKE_MAKE_PROGRAM=$($msys2.Ninja.Replace('\','/'))"
            )
        }
        else {
            Require-Command ninja
            $configureArgs += @("-G", "Ninja")
        }
        $toolchainKind = "ninja"
    }
}
elseif ($cachedGenerator -eq "NMake Makefiles") {
    $configureArgs += @("-G", "NMake Makefiles")
    $buildArgs = @("--", "/NOLOGO")
    $toolchainKind = "msvc-nmake"
}
elseif ((Get-Command cl -ErrorAction SilentlyContinue) -and (Get-Command nmake -ErrorAction SilentlyContinue)) {
    $ninja = Find-NinjaExecutable
    if ($ninja) {
        $configureArgs += @(
            "-G", "Ninja",
            "-DCMAKE_MAKE_PROGRAM=$($ninja.Replace('\','/'))",
            "-DCMAKE_C_COMPILER=cl",
            "-DCMAKE_CXX_COMPILER=cl"
        )
        $toolchainKind = "msvc-ninja"
    }
    else {
        $configureArgs += @("-G", "NMake Makefiles")
        $buildArgs = @("--", "/NOLOGO")
        $toolchainKind = "msvc-nmake"
    }
}
else {
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

if ($toolchainKind -like "msvc-*") {
    $configureArgs += "-DMATH_LIBRARY="
}

Write-Host "Configuring llama.cpp CUDA build..."
cmake @configureArgs
if ($LASTEXITCODE -ne 0) {
    throw "CMake configure failed for llama.cpp CUDA build."
}

Write-Host "Building CUDA llama-server.exe..."
cmake --build $BuildRoot --target llama-server --config $BuildType @buildArgs
if ($LASTEXITCODE -ne 0) {
    throw "CMake build failed for llama.cpp CUDA build."
}

$cudaCompiler = Get-CMakeCacheValue -CachePath $cachePath -Name "CMAKE_CUDA_COMPILER"
$cudaBin = $null
if ($cudaCompiler) {
    $cudaBin = Split-Path -Parent $cudaCompiler
}
elseif ($env:CUDA_PATH) {
    $cudaBin = Join-Path $env:CUDA_PATH "bin"
}

Get-ChildItem -LiteralPath $BinRoot -File -ErrorAction SilentlyContinue | Remove-Item -Force

Get-ChildItem (Join-Path $BuildRoot "bin") -File | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $BinRoot $_.Name) -Force
}

if ($cudaBin -and (Test-Path -LiteralPath $cudaBin)) {
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

if ($toolchainKind -like "msvc-*") {
    Copy-MsvcRuntimeDlls -Destination $BinRoot
}

$requiredRuntimePatterns = @(
    "cudart64*.dll",
    "cublas64*.dll",
    "cublasLt64*.dll",
    "msvcp140.dll",
    "vcruntime140.dll",
    "vcruntime140_1.dll",
    "vcomp140.dll"
)
foreach ($pattern in $requiredRuntimePatterns) {
    if (-not (Get-ChildItem -LiteralPath $BinRoot -Filter $pattern -File -ErrorAction SilentlyContinue)) {
        throw "Self-contained CUDA package is missing $pattern in $BinRoot."
    }
}

if (-not (Test-Path (Join-Path $BinRoot "llama-server.exe"))) {
    throw "Build finished but llama-server.exe was not copied into $BinRoot."
}

Write-Host ""
Write-Host "GPU build complete."
Write-Host "Toolchain: $toolchainKind"
Write-Host "Binary package location: $BinRoot"
