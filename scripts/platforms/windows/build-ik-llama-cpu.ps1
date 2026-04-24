param(
    [string]$BuildType = "Release",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$PlatformRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$InnerScript = Join-Path $PlatformRoot "ik_llama.cpp-cpu\build.ps1"

if (-not (Test-Path -LiteralPath $InnerScript)) {
    throw "CPU build script not found: $InnerScript"
}

$argsList = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", $InnerScript,
    "-BuildType", $BuildType
)

Write-Host "Running ik_llama.cpp CPU backend build script:"
Write-Host "  powershell $($argsList -join ' ')"

if ($DryRun) {
    Write-Host "Dry run enabled. No build steps were executed."
    exit 0
}

powershell @argsList
exit $LASTEXITCODE
