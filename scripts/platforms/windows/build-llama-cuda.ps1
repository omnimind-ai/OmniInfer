param(
    [string]$BuildType = "Release",
    [string]$CudaArchitectures = "",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$PlatformRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$InnerScript = Join-Path $PlatformRoot "llama.cpp-cuda\build.ps1"

if (-not (Test-Path -LiteralPath $InnerScript)) {
    throw "CUDA build script not found: $InnerScript"
}

$argsList = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", $InnerScript,
    "-BuildType", $BuildType
)
if ($CudaArchitectures) {
    $argsList += @("-CudaArchitectures", $CudaArchitectures)
}

Write-Host "Running CUDA backend build script:"
Write-Host "  powershell $($argsList -join ' ')"

if ($DryRun) {
    Write-Host "Dry run enabled. No build steps were executed."
    exit 0
}

powershell @argsList
exit $LASTEXITCODE
