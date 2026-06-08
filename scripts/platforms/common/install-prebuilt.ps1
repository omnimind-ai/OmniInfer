param(
    [Parameter(Mandatory = $true)][string]$Platform,
    [Parameter(Mandatory = $true)][string]$Backend,
    [Parameter(Mandatory = $true)][string]$RuntimeDir,
    [Parameter(Mandatory = $true)][string]$ModelsDir,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$ScriptRoot = $PSScriptRoot
$RepoRoot = Resolve-Path (Join-Path $ScriptRoot "..\..\..")
$Helper = Join-Path $RepoRoot "scripts\platforms\common\install-prebuilt.py"
$Catalog = Join-Path $RepoRoot "scripts\prebuilt_backends.json"

function Find-Python {
    foreach ($candidate in @("python", "python3", "py")) {
        $command = Get-Command $candidate -ErrorAction SilentlyContinue
        if ($command) {
            return $command.Source
        }
    }
    throw "Python was not found in PATH; it is required for prebuilt backend installation."
}

$python = Find-Python
$argsList = @(
    $Helper,
    "--catalog", $Catalog,
    "--platform", $Platform,
    "--backend", $Backend,
    "--runtime-dir", $RuntimeDir,
    "--models-dir", $ModelsDir
)
if ($DryRun) {
    $argsList += "--dry-run"
}

& $python @argsList
exit $LASTEXITCODE
