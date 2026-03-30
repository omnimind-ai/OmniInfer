param(
    [string]$BuildType = "Release",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$PlatformRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$InnerScript = Join-Path $PlatformRoot "llama.cpp-vulkan\build.ps1"

if (-not (Test-Path -LiteralPath $InnerScript)) {
    throw "Vulkan build script not found: $InnerScript"
}

$argsList = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", $InnerScript,
    "-BuildType", $BuildType
)

Write-Host "Running Vulkan backend build script:"
Write-Host "  powershell $($argsList -join ' ')"

if ($DryRun) {
    Write-Host "Dry run enabled. No build steps were executed."
    exit 0
}

powershell @argsList
exit $LASTEXITCODE
