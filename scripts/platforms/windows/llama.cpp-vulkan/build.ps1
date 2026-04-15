param(
    [string]$BuildType = "Release"
)

$ErrorActionPreference = "Stop"

$ScriptRoot = $PSScriptRoot
$RepoRoot = Resolve-Path (Join-Path $ScriptRoot "..\..\..\..")
$PackageRoot = Join-Path $RepoRoot ".local\runtime\windows\llama.cpp-vulkan"
$LlamaRoot = Resolve-Path (Join-Path $RepoRoot "framework\llama.cpp")
$BuildRoot = Join-Path $PackageRoot "build\llama.cpp-vulkan"
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
        $glslc = Join-Path $ucrt64Bin "glslc.exe"
        if ((Test-Path $gcc) -and (Test-Path $gpp) -and (Test-Path $ninja) -and (Test-Path $glslc)) {
            return @{ Root = $ucrt64Bin; Gcc = $gcc; Gpp = $gpp; Ninja = $ninja; Glslc = $glslc }
        }
    }
    return $null
}

function Find-VulkanSdk {
    $candidates = @()
    if ($env:VULKAN_SDK) {
        $candidates += $env:VULKAN_SDK
    }
    $candidates += @(
        "C:\VulkanSDK",
        "C:\Program Files\VulkanSDK"
    )

    foreach ($candidate in $candidates) {
        if (-not (Test-Path -LiteralPath $candidate)) {
            continue
        }
        $roots = @()
        if (Test-Path (Join-Path $candidate "Bin")) {
            $roots += $candidate
        } else {
            $roots += Get-ChildItem -Path $candidate -Directory -ErrorAction SilentlyContinue |
                Sort-Object Name -Descending |
                Select-Object -ExpandProperty FullName
        }

        foreach ($root in $roots) {
            $binDir = Join-Path $root "Bin"
            $glslc = Join-Path $binDir "glslc.exe"
            if (Test-Path $glslc) {
                return @{
                    Root = $root
                    Bin = $binDir
                    Glslc = $glslc
                }
            }
        }
    }

    return $null
}

function Get-GppThreadModel {
    param([string]$Compiler = "g++")
    $versionOutput = (cmd /c """$Compiler"" -v -x c++ -E NUL 2>&1" | Out-String)
    $match = [regex]::Match($versionOutput, "Thread model:\s*(\S+)")
    if ($match.Success) {
        return $match.Groups[1].Value
    }
    return ""
}

Require-Command cmake

$generator = $null
$buildArgs = @()
$toolchainKind = $null
$runtimeDllSource = $null
$configureArgs = @(
    "-S", $LlamaRoot,
    "-B", $BuildRoot,
    "-DCMAKE_BUILD_TYPE=$BuildType",
    "-DBUILD_SHARED_LIBS=OFF",
    "-DLLAMA_BUILD_TESTS=OFF",
    "-DLLAMA_BUILD_EXAMPLES=OFF",
    "-DLLAMA_BUILD_SERVER=ON",
    "-DLLAMA_OPENSSL=OFF",
    "-DGGML_VULKAN=ON",
    "-DGGML_NATIVE=OFF"
)

$vulkanSdk = Find-VulkanSdk
if ($vulkanSdk) {
    $env:PATH = "$($vulkanSdk.Bin);$env:PATH"
    $env:VULKAN_SDK = $vulkanSdk.Root
    $configureArgs += "-DCMAKE_PREFIX_PATH=$($vulkanSdk.Root.Replace('\','/'))"
}

if ((Get-Command cl -ErrorAction SilentlyContinue) -and (Get-Command nmake -ErrorAction SilentlyContinue)) {
    $generator = "NMake Makefiles"
    $buildArgs = @("--", "/NOLOGO")
    $toolchainKind = "msvc"
} else {
    $msys2 = Find-Msys2Ucrt64Toolchain
    if ($msys2) {
        $env:PATH = "$($msys2.Root);$env:PATH"
        $generator = "Ninja"
        $toolchainKind = "msys2-ucrt64-vulkan"
        $runtimeDllSource = $msys2.Root
        $configureArgs += @(
            "-DCMAKE_C_COMPILER=$($msys2.Gcc.Replace('\','/'))",
            "-DCMAKE_CXX_COMPILER=$($msys2.Gpp.Replace('\','/'))",
            "-DCMAKE_MAKE_PROGRAM=$($msys2.Ninja.Replace('\','/'))"
        )
    } else {
        Require-Command gcc
        Require-Command g++
        Require-Command mingw32-make

        $threadModel = Get-GppThreadModel -Compiler "g++"
        if ($threadModel -eq "win32") {
            throw "The detected MinGW toolchain uses the 'win32' thread model and cannot build llama-server reliably."
        }

        $generator = "MinGW Makefiles"
        $buildArgs = @("--", "-j", "4")
        $toolchainKind = "mingw-posix-vulkan"
        $runtimeDllSource = Split-Path (Get-Command g++.exe).Source -Parent
    }
}

if (-not (Get-Command glslc -ErrorAction SilentlyContinue)) {
    throw "glslc.exe was not found. Install the Vulkan SDK (or MSYS2 shaderc package), ensure glslc is in PATH, and retry."
}

New-Item -ItemType Directory -Force -Path $BuildRoot, $BinRoot | Out-Null
$configureArgs += @("-G", $generator)

Write-Host "Configuring llama.cpp Vulkan build..."
cmake @configureArgs

Write-Host "Building Vulkan llama-server.exe..."
cmake --build $BuildRoot --target llama-server --config $BuildType @buildArgs

Get-ChildItem (Join-Path $BuildRoot "bin") -File | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $BinRoot $_.Name) -Force
}

if ($runtimeDllSource) {
    @(
        (Join-Path $runtimeDllSource "libstdc++-6.dll"),
        (Join-Path $runtimeDllSource "libgcc_s_seh-1.dll"),
        (Join-Path $runtimeDllSource "libwinpthread-1.dll"),
        (Join-Path $runtimeDllSource "libgomp-1.dll")
    ) | ForEach-Object {
        if (Test-Path $_) {
            Copy-Item -LiteralPath $_ -Destination (Join-Path $BinRoot (Split-Path $_ -Leaf)) -Force
        }
    }
}

if (-not (Test-Path (Join-Path $BinRoot "llama-server.exe"))) {
    throw "Build finished but llama-server.exe was not copied into $BinRoot."
}

Write-Host ""
Write-Host "Vulkan build complete."
Write-Host "Toolchain: $toolchainKind"
if ($vulkanSdk) {
    Write-Host "Vulkan SDK: $($vulkanSdk.Root)"
}
Write-Host "Binary package location: $BinRoot"
