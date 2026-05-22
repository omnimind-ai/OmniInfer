param(
    [Parameter(Mandatory = $true)]
    [string]$PortableRoot
)

$ErrorActionPreference = "Stop"

$portablePath = Resolve-Path -LiteralPath $PortableRoot
$cliPath = Join-Path $portablePath "omniinfer.exe"
if (-not (Test-Path -LiteralPath $cliPath)) {
    throw "Portable CLI binary not found: $cliPath"
}

Set-Content `
    -LiteralPath (Join-Path $portablePath "omniinfer.cmd") `
    -Encoding ASCII `
    -Value "@echo off`r`n`"%~dp0omniinfer.exe`" %*`r`nexit /b %ERRORLEVEL%`r`n"

$psWrapper = @'
$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$cli = Join-Path $scriptDir "omniinfer.exe"
& $cli @args
exit $LASTEXITCODE
'@

Set-Content `
    -LiteralPath (Join-Path $portablePath "omniinfer.ps1") `
    -Encoding ASCII `
    -Value $psWrapper

Write-Host "Portable launchers ready:"
Write-Host "  $portablePath\omniinfer.ps1  (PowerShell, recommended for TUI)"
Write-Host "  $portablePath\omniinfer.cmd  (cmd.exe compatibility)"
