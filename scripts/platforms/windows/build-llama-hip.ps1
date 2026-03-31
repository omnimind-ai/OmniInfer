param(
    [string]$BuildType = "Release",
    [string]$GpuTargets = "",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$PlatformRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$InnerScript = Join-Path $PlatformRoot "llama.cpp-hip\build.ps1"

if (-not (Test-Path -LiteralPath $InnerScript)) {
    throw "HIP build script not found: $InnerScript"
}

$argsList = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", $InnerScript,
    "-BuildType", $BuildType
)

if ($GpuTargets) {
    $argsList += @("-GpuTargets", $GpuTargets)
}

Write-Host "Running HIP backend build script:"
Write-Host "  powershell $($argsList -join ' ')"

if ($DryRun) {
    Write-Host "Dry run enabled. No build steps were executed."
    exit 0
}

powershell @argsList
exit $LASTEXITCODE
