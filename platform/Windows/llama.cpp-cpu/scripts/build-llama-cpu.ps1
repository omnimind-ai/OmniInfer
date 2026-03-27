param(
    [string]$BuildType = "Release"
)

$ErrorActionPreference = "Stop"

$PackageRoot = Split-Path -Parent $PSScriptRoot
$RepoRoot = Resolve-Path (Join-Path $PackageRoot "..\..\..")
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
    $roots = @(
        "E:\Coding\Tools\MSYS2\ucrt64\bin",
        "C:\msys64\ucrt64\bin"
    )

    foreach ($root in $roots) {
        $gcc = Join-Path $root "gcc.exe"
        $gpp = Join-Path $root "g++.exe"
        $ninja = Join-Path $root "ninja.exe"
        if ((Test-Path $gcc) -and (Test-Path $gpp) -and (Test-Path $ninja)) {
            return @{
                Root = $root
                Gcc = $gcc
                Gpp = $gpp
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
            throw "The detected MinGW toolchain uses the 'win32' thread model and cannot build llama-server reliably."
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
