param(
    [string]$PackageName = "OmniInfer"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$BuildRoot = Join-Path $PSScriptRoot "build"
$PortableRoot = Join-Path $PSScriptRoot "portable"
$DistRoot = Join-Path $PortableRoot $PackageName
$WorkRoot = Join-Path $BuildRoot "pyinstaller-work"
$SpecRoot = Join-Path $BuildRoot "pyinstaller-spec"
$ConfigTemplate = Join-Path $PSScriptRoot "config\omniinfer.json"
$UsageTemplate = Join-Path $RepoRoot "tmp\usage.md"

function Require-Command {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command '$Name' was not found in PATH."
    }
}

function Ensure-PyInstaller {
    $probe = python -c "import importlib.util; print(importlib.util.find_spec('PyInstaller') is not None)"
    if ($probe.Trim().ToLower() -ne "true") {
        Write-Host "Installing PyInstaller..."
        python -m pip install pyinstaller
    }
}

function Reset-Directory {
    param([string]$Path)
    if (Test-Path $Path) {
        Remove-Item -LiteralPath $Path -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $Path | Out-Null
}

function Copy-RuntimeBin {
    param(
        [string]$SourceRoot,
        [string]$TargetRoot
    )

    New-Item -ItemType Directory -Force -Path $TargetRoot | Out-Null
    Copy-Item -LiteralPath (Join-Path $SourceRoot "bin") -Destination $TargetRoot -Recurse -Force
    New-Item -ItemType Directory -Force -Path (Join-Path $TargetRoot "logs") | Out-Null
}

Require-Command python
Ensure-PyInstaller

Reset-Directory -Path $BuildRoot
Reset-Directory -Path $PortableRoot

Write-Host "Building OmniInfer.exe with PyInstaller..."
python -m PyInstaller `
    --noconfirm `
    --clean `
    --onedir `
    --windowed `
    --name $PackageName `
    --distpath $PortableRoot `
    --workpath $WorkRoot `
    --specpath $SpecRoot `
    (Join-Path $RepoRoot "omniinfer_gateway.py")

New-Item -ItemType Directory -Force -Path (Join-Path $DistRoot "config") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $DistRoot "runtime") | Out-Null

Copy-Item -LiteralPath $ConfigTemplate -Destination (Join-Path $DistRoot "config\omniinfer.json") -Force

Copy-RuntimeBin `
    -SourceRoot (Join-Path $RepoRoot "platform\Windows\llama.cpp-cpu") `
    -TargetRoot (Join-Path $DistRoot "runtime\llama.cpp-cpu")

Copy-RuntimeBin `
    -SourceRoot (Join-Path $RepoRoot "platform\Windows\llama.cpp-cuda") `
    -TargetRoot (Join-Path $DistRoot "runtime\llama.cpp-cuda")

if (Test-Path $UsageTemplate) {
    Copy-Item -LiteralPath $UsageTemplate -Destination (Join-Path $DistRoot "usage.md") -Force
}

Write-Host ""
Write-Host "Portable package ready:"
Write-Host "  $DistRoot"
Write-Host ""
Write-Host "Run with:"
Write-Host "  $DistRoot\\$PackageName.exe"
Write-Host "  $DistRoot\\$PackageName.exe --window-mode visible"
