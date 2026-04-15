param(
    [string]$BuildType = "Release"
)

$ErrorActionPreference = "Stop"

$ScriptRoot = $PSScriptRoot
$RepoRoot = Resolve-Path (Join-Path $ScriptRoot "..\..\..\..")
$PackageRoot = Join-Path $RepoRoot ".local\runtime\windows\llama.cpp-cpu"
$LlamaRoot = Resolve-Path (Join-Path $RepoRoot "framework\llama.cpp")
$BuildRoot = Join-Path $PackageRoot "build\llama.cpp-cpu"
$BinRoot = Join-Path $PackageRoot "bin"

function Require-Command {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command '$Name' was not found in PATH."
    }
}

Require-Command cmake

function Find-Msys2Ucrt64Toolchain {
    $candidates = @()

    # 1. Environment variable (highest priority)
    if ($env:MSYS2_ROOT) { $candidates += $env:MSYS2_ROOT }

    # 2. Windows registry (MSYS2 installer writes InstallLocation here)
    foreach ($key in @(
        "HKLM:\SOFTWARE\MSYS2",
        "HKCU:\SOFTWARE\MSYS2",
        "HKLM:\SOFTWARE\WOW6432Node\MSYS2"
    )) {
        try {
            $loc = (Get-ItemProperty -Path $key -ErrorAction SilentlyContinue).InstallLocation
            if ($loc) { $candidates += $loc }
        } catch {}
    }

    # 3. PATH: if gcc.exe is already in PATH, derive MSYS2 root from it
    $gccInPath = Get-Command gcc.exe -ErrorAction SilentlyContinue
    if ($gccInPath) {
        $binDir = Split-Path $gccInPath.Source
        if ($binDir -match 'ucrt64\\bin$') {
            $candidates += (Split-Path (Split-Path $binDir))  # msys64 root
        }
    }

    # 4. Scan all drive roots for common MSYS2 install locations
    foreach ($drive in (Get-PSDrive -PSProvider FileSystem -ErrorAction SilentlyContinue)) {
        $root = $drive.Root  # e.g. "C:\"
        $candidates += Join-Path $root "msys64"
        $candidates += Join-Path $root "msys2"
    }
    # Also check well-known non-root paths (chocolatey, scoop, custom)
    if ($env:ChocolateyInstall) { $candidates += Join-Path $env:ChocolateyInstall "lib\msys2\msys64" }
    if ($env:SCOOP) { $candidates += Join-Path $env:SCOOP "apps\msys2\current" }

    foreach ($msys2Root in $candidates) {
        $ucrt64Bin = Join-Path $msys2Root "ucrt64\bin"
        if (-not (Test-Path $ucrt64Bin)) { continue }
        $gcc   = Join-Path $ucrt64Bin "gcc.exe"
        $gpp   = Join-Path $ucrt64Bin "g++.exe"
        $ninja = Join-Path $ucrt64Bin "ninja.exe"
        if ((Test-Path $gcc) -and (Test-Path $gpp) -and (Test-Path $ninja)) {
            return @{
                Root  = $ucrt64Bin
                Gcc   = $gcc
                Gpp   = $gpp
                Ninja = $ninja
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
    "-DGGML_NATIVE=OFF"
)

if ((Get-Command cl -ErrorAction SilentlyContinue) -and (Get-Command nmake -ErrorAction SilentlyContinue)) {
    $generator = "NMake Makefiles"
    $buildArgs = @("--", "/NOLOGO")
    $toolchainKind = "msvc"
} else {
    $msys2 = Find-Msys2Ucrt64Toolchain
    if ($msys2) {
        $env:PATH = "$($msys2.Root);$env:PATH"
        $generator = "Ninja"
        $toolchainKind = "msys2-ucrt64"
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
            $gccPath = (Get-Command g++.exe).Source
            throw @"
The MinGW g++ found at '$gccPath' uses the 'win32' thread model, which cannot build llama-server.

Fix (pick one):
  1. Install MSYS2 (https://www.msys2.org/) then run in MSYS2 terminal:
       pacman -S mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-ninja mingw-w64-ucrt-x86_64-cmake
     Then add C:\msys64\ucrt64\bin to PATH (or set `$env:MSYS2_ROOT`).

  2. Install Visual Studio Build Tools (https://visualstudio.microsoft.com/downloads/#build-tools)
     and run this script from a Developer PowerShell.
"@
        }

        $generator = "MinGW Makefiles"
        $buildArgs = @("--", "-j", "4")
        $toolchainKind = "mingw-posix"
        $runtimeDllSource = Split-Path (Get-Command g++.exe).Source -Parent
    }
}

New-Item -ItemType Directory -Force -Path $BuildRoot, $BinRoot | Out-Null
$configureArgs += @("-G", $generator)

Write-Host "Configuring llama.cpp CPU build..."
cmake @configureArgs

Write-Host "Building llama-server.exe..."
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
Write-Host "CPU build complete."
Write-Host "Toolchain: $toolchainKind"
Write-Host "Binary package location: $BinRoot"
