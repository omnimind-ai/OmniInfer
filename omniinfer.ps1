$ErrorActionPreference = "Stop"

$scriptDir = if ($PSScriptRoot) {
    $PSScriptRoot
} else {
    Split-Path -Parent $MyInvocation.MyCommand.Path
}
$entry = Join-Path $scriptDir "omniinfer.py"

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
