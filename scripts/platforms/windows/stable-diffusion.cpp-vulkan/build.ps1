param(
    [string]$BuildType = "Release",
    [switch]$Prebuilt,
    [switch]$Clean,
    [switch]$Native,
    [switch]$Lto,
    [switch]$SmokeTest
)

$ErrorActionPreference = "Stop"
$MinimumVulkanHeaderVersion = 301

if ($Prebuilt) {
    throw "No prebuilt install path is configured for stable-diffusion.cpp-vulkan. Build from the pinned source submodule."
}

$ScriptRoot = $PSScriptRoot
$RepoRoot = Resolve-Path (Join-Path $ScriptRoot "..\..\..\..")
$BackendId = "stable-diffusion.cpp-vulkan"
$PackageRoot = Join-Path $RepoRoot ".local\runtime\windows\$BackendId"
$SourceRoot = Join-Path $RepoRoot "framework\stable-diffusion.cpp"
$BuildRoot = Join-Path $PackageRoot "build\$BackendId"
$BinRoot = Join-Path $PackageRoot "bin"
$LogRoot = Join-Path $PackageRoot "logs"

function Require-Command {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command '$Name' was not found in PATH."
    }
}

function Get-VulkanHeaderVersion {
    param([string]$HeaderPath)
    if (-not (Test-Path -LiteralPath $HeaderPath)) { return 0 }
    $match = [regex]::Match(
        [System.IO.File]::ReadAllText($HeaderPath),
        '(?m)^#define\s+VK_HEADER_VERSION\s+(\d+)\s*$'
    )
    if (-not $match.Success) { return 0 }
    return [int]$match.Groups[1].Value
}

function Find-Msys2Ucrt64Toolchain {
    $candidates = @()
    if ($env:MSYS2_ROOT) { $candidates += $env:MSYS2_ROOT }
    foreach ($key in @("HKLM:\SOFTWARE\MSYS2", "HKCU:\SOFTWARE\MSYS2", "HKLM:\SOFTWARE\WOW6432Node\MSYS2")) {
        try {
            $location = (Get-ItemProperty -Path $key -ErrorAction SilentlyContinue).InstallLocation
            if ($location) { $candidates += $location }
        } catch {}
    }
    foreach ($drive in (Get-PSDrive -PSProvider FileSystem -ErrorAction SilentlyContinue)) {
        $candidates += Join-Path $drive.Root "msys64"
        $candidates += Join-Path $drive.Root "msys2"
    }
    if ($env:SCOOP) { $candidates += Join-Path $env:SCOOP "apps\msys2\current" }

    foreach ($root in ($candidates | Select-Object -Unique)) {
        $bin = Join-Path $root "ucrt64\bin"
        $gcc = Join-Path $bin "gcc.exe"
        $gpp = Join-Path $bin "g++.exe"
        $ninja = Join-Path $bin "ninja.exe"
        $glslc = Join-Path $bin "glslc.exe"
        $prefix = Split-Path $bin
        $vulkanHeader = Join-Path $prefix "include\vulkan\vulkan_core.h"
        $spirvConfig = Join-Path $prefix "share\cmake\SPIRV-Headers\SPIRV-HeadersConfig.cmake"
        $headerVersion = Get-VulkanHeaderVersion $vulkanHeader
        if ((Test-Path $gcc) -and (Test-Path $gpp) -and (Test-Path $ninja) -and
            (Test-Path $glslc) -and (Test-Path $spirvConfig) -and
            ($headerVersion -ge $MinimumVulkanHeaderVersion)) {
            return @{
                Bin = $bin
                Prefix = $prefix
                Gcc = $gcc
                Gpp = $gpp
                Ninja = $ninja
                Glslc = $glslc
                HeaderVersion = $headerVersion
            }
        }
    }
    return $null
}

function Find-VulkanSdk {
    $candidates = @()
    if ($env:VULKAN_SDK) { $candidates += $env:VULKAN_SDK }
    $candidates += "C:\VulkanSDK", "C:\Program Files\VulkanSDK"
    foreach ($candidate in $candidates) {
        if (-not (Test-Path -LiteralPath $candidate)) { continue }
        $roots = if (Test-Path (Join-Path $candidate "Bin")) {
            @($candidate)
        } else {
            @(Get-ChildItem -Path $candidate -Directory -ErrorAction SilentlyContinue | Sort-Object Name -Descending | Select-Object -ExpandProperty FullName)
        }
        foreach ($root in $roots) {
            $bin = Join-Path $root "Bin"
            $vulkanHeader = Join-Path $root "Include\vulkan\vulkan_core.h"
            $spirvConfig = Join-Path $root "share\cmake\SPIRV-Headers\SPIRV-HeadersConfig.cmake"
            $headerVersion = Get-VulkanHeaderVersion $vulkanHeader
            if ((Test-Path (Join-Path $bin "glslc.exe")) -and
                (Test-Path $spirvConfig) -and
                ($headerVersion -ge $MinimumVulkanHeaderVersion)) {
                return @{ Root = $root; Bin = $bin; HeaderVersion = $headerVersion }
            }
        }
    }
    return $null
}

Require-Command cmake

$sourceReady = (Test-Path (Join-Path $SourceRoot "CMakeLists.txt")) -and
    (Test-Path (Join-Path $SourceRoot "ggml\CMakeLists.txt")) -and
    (Test-Path (Join-Path $SourceRoot "thirdparty\libwebm\CMakeLists.txt"))
if (-not $sourceReady) {
    Require-Command git
    Write-Host "Initializing stable-diffusion.cpp and pinned nested dependencies..."
    & git -C $RepoRoot submodule update --init --recursive framework/stable-diffusion.cpp
    if ($LASTEXITCODE -ne 0) { throw "Failed to initialize framework/stable-diffusion.cpp." }
}
if (-not (Test-Path (Join-Path $SourceRoot "ggml\CMakeLists.txt"))) {
    throw "stable-diffusion.cpp nested dependencies are incomplete at $SourceRoot."
}

$configureArgs = @(
    "-S", $SourceRoot,
    "-B", $BuildRoot,
    "-DCMAKE_BUILD_TYPE=$BuildType",
    "-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=$(if ($Lto) { 'ON' } else { 'OFF' })",
    "-DSD_VULKAN=ON",
    "-DSD_BUILD_EXAMPLES=ON",
    "-DSD_SERVER_BUILD_FRONTEND=OFF",
    "-DSD_BUILD_SHARED_LIBS=OFF",
    "-DSD_BUILD_SHARED_GGML_LIB=OFF",
    "-DSD_WEBM=ON",
    "-DSD_WEBP=ON",
    "-DGGML_NATIVE=$(if ($Native) { 'ON' } else { 'OFF' })"
)

$runtimeDllSource = $null
$toolchain = "msvc"
$vulkanSdk = Find-VulkanSdk
if ($vulkanSdk) {
    $env:PATH = "$($vulkanSdk.Bin);$env:PATH"
    $env:VULKAN_SDK = $vulkanSdk.Root
    $configureArgs += "-DCMAKE_PREFIX_PATH=$($vulkanSdk.Root.Replace('\', '/'))"
}

if ((Get-Command cl.exe -ErrorAction SilentlyContinue) -and (Get-Command ninja.exe -ErrorAction SilentlyContinue)) {
    $configureArgs += "-G", "Ninja"
} elseif ((Get-Command cl.exe -ErrorAction SilentlyContinue) -and (Get-Command nmake.exe -ErrorAction SilentlyContinue)) {
    $configureArgs += "-G", "NMake Makefiles"
} else {
    $msys2 = Find-Msys2Ucrt64Toolchain
    if (-not $msys2) {
        throw "No supported Vulkan C++ toolchain found. Install Visual Studio Build Tools plus Vulkan SDK, or MSYS2 UCRT64 gcc/ninja/shaderc."
    }
    $env:PATH = "$($msys2.Bin);$env:PATH"
    $runtimeDllSource = $msys2.Bin
    $toolchain = "msys2-ucrt64"
    $configureArgs += @(
        "-G", "Ninja",
        "-DCMAKE_PREFIX_PATH=$($msys2.Prefix.Replace('\', '/'))",
        "-DCMAKE_C_COMPILER=$($msys2.Gcc.Replace('\', '/'))",
        "-DCMAKE_CXX_COMPILER=$($msys2.Gpp.Replace('\', '/'))",
        "-DCMAKE_MAKE_PROGRAM=$($msys2.Ninja.Replace('\', '/'))"
    )
}

if (-not (Get-Command glslc.exe -ErrorAction SilentlyContinue)) {
    throw "A current Vulkan SDK was not found. Install Vulkan SDK 1.4.301 or newer, including SPIRV-Headers and glslc, or matching MSYS2 UCRT64 packages."
}

if ($Clean -and (Test-Path -LiteralPath $BuildRoot)) {
    Remove-Item -LiteralPath $BuildRoot -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $BuildRoot, $BinRoot, $LogRoot | Out-Null

Write-Host "Configuring $BackendId ($toolchain)..."
& cmake @configureArgs
if ($LASTEXITCODE -ne 0) { throw "CMake configure failed for $BackendId." }

Write-Host "Building sd-server.exe and sd-cli.exe..."
& cmake --build $BuildRoot --target sd-server sd-cli --config $BuildType --parallel
if ($LASTEXITCODE -ne 0) { throw "CMake build failed for $BackendId." }

Get-ChildItem -LiteralPath $BinRoot -File -ErrorAction SilentlyContinue | Remove-Item -Force
$builtBin = Join-Path $BuildRoot "bin"
Get-ChildItem -LiteralPath $builtBin -Recurse -File | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $BinRoot $_.Name) -Force
}
if ($runtimeDllSource) {
    foreach ($name in @("libstdc++-6.dll", "libgcc_s_seh-1.dll", "libwinpthread-1.dll", "libgomp-1.dll")) {
        $source = Join-Path $runtimeDllSource $name
        if (Test-Path -LiteralPath $source) { Copy-Item -LiteralPath $source -Destination (Join-Path $BinRoot $name) -Force }
    }
}
foreach ($name in @("sd-server.exe", "sd-cli.exe")) {
    if (-not (Test-Path -LiteralPath (Join-Path $BinRoot $name))) {
        throw "Build finished but $name was not copied into $BinRoot."
    }
}

if ($SmokeTest) {
    & (Join-Path $BinRoot "sd-server.exe") --help | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "sd-server.exe smoke test failed." }
    & (Join-Path $BinRoot "sd-cli.exe") --help | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "sd-cli.exe smoke test failed." }
}

Write-Host "$BackendId build complete: $BinRoot"
