param(
    [string]$BuildType = "Release"
)

$ErrorActionPreference = "Stop"

$ScriptRoot = $PSScriptRoot
$RepoRoot = Resolve-Path (Join-Path $ScriptRoot "..\..\..\..")
$PackageRoot = Join-Path $RepoRoot ".local\runtime\windows\ik_llama.cpp-cpu"
$LlamaRoot = Resolve-Path (Join-Path $RepoRoot "framework\ik_llama.cpp")
$BuildRoot = Join-Path $PackageRoot "build\ik_llama.cpp-cpu"
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
            Write-Host ""
            Write-Host "  Problem: " -ForegroundColor Red -NoNewline
            Write-Host "The g++ at '$gccPath' uses the 'win32' thread model."
            Write-Host "           This cannot build llama-server (requires posix thread model)."
            Write-Host ""
            Write-Host "  Fix (pick one):" -ForegroundColor Yellow
            Write-Host ""
            Write-Host "    1. " -ForegroundColor Cyan -NoNewline
            Write-Host "Install MSYS2 ucrt64 toolchain"
            Write-Host "       Download:  https://www.msys2.org/"
            Write-Host "       Then run:  pacman -S mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-ninja mingw-w64-ucrt-x86_64-cmake"
            Write-Host '       Then set:  $env:MSYS2_ROOT = "C:\msys64"  (or your install path)'
            Write-Host ""
            Write-Host "    2. " -ForegroundColor Cyan -NoNewline
            Write-Host "Install Visual Studio Build Tools"
            Write-Host "       Download:  https://visualstudio.microsoft.com/downloads/#build-tools"
            Write-Host "       Then run this script from a Developer PowerShell."
            Write-Host ""
            exit 1
        }

        $generator = "MinGW Makefiles"
        $buildArgs = @("--", "-j", "4")
        $toolchainKind = "mingw-posix"
        $runtimeDllSource = Split-Path (Get-Command g++.exe).Source -Parent
    }
}

New-Item -ItemType Directory -Force -Path $BuildRoot, $BinRoot | Out-Null
$configureArgs += @("-G", $generator)

Write-Host "Configuring ik_llama.cpp CPU build..."
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
Write-Host "ik_llama.cpp CPU build complete."
Write-Host "Toolchain: $toolchainKind"
Write-Host "Binary package location: $BinRoot"
