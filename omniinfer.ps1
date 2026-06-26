$ErrorActionPreference = "Stop"

$scriptDir = if ($PSScriptRoot) {
    $PSScriptRoot
} else {
    Split-Path -Parent $MyInvocation.MyCommand.Path
}
$rustCli = Join-Path $scriptDir "target\debug\omniinfer-rs.exe"
$entry = Join-Path $scriptDir "omniinfer.py"

function Test-ForcePython {
    if (-not $env:OMNIINFER_FORCE_PYTHON) {
        return $false
    }
    return @("1", "true", "yes", "on") -contains $env:OMNIINFER_FORCE_PYTHON.Trim().ToLowerInvariant()
}

if (-not (Test-ForcePython)) {
    if (Test-Path -LiteralPath $rustCli) {
        & $rustCli @args
        exit $LASTEXITCODE
    }
    Write-Error "Rust OmniInfer CLI was not found at $rustCli. Run: cargo build -p omniinfer-cli"
    Write-Error "To use the Python fallback now, set OMNIINFER_FORCE_PYTHON=1."
    exit 1
}

if ($env:OMNIINFER_PYTHON) {
    & $env:OMNIINFER_PYTHON $entry @args
    exit $LASTEXITCODE
}

if (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3 $entry @args
    exit $LASTEXITCODE
}

if (Get-Command python -ErrorAction SilentlyContinue) {
    & python $entry @args
    exit $LASTEXITCODE
}

Write-Error "Python 3 was not found in PATH."
exit 1
