$ErrorActionPreference = "Stop"

$scriptDir = if ($PSScriptRoot) {
    $PSScriptRoot
} else {
    Split-Path -Parent $MyInvocation.MyCommand.Path
}
$rustCli = Join-Path $scriptDir "target\debug\omniinfer-rs.exe"

if (Test-Path -LiteralPath $rustCli) {
    & $rustCli @args
    exit $LASTEXITCODE
}

Write-Error "Rust OmniInfer CLI was not found at $rustCli. Run: cargo build -p omniinfer-cli"
exit 1
