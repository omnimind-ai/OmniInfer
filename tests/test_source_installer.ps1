param(
    [string]$Installer = (Join-Path (Split-Path -Parent $PSScriptRoot) "scripts\install-from-source.ps1")
)

$ErrorActionPreference = "Stop"
$tokens = $null
$errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile(
    (Resolve-Path $Installer),
    [ref]$tokens,
    [ref]$errors
)
if ($errors.Count -gt 0) {
    throw "Source installer has PowerShell parse errors: $($errors -join '; ')"
}

$installerText = [IO.File]::ReadAllText((Resolve-Path $Installer))
if (-not $installerText.Contains('Write-Host "    .\omniinfer.ps1 serve --detach"')) {
    throw "Source installer completion guidance must start the gateway before model loading"
}
if ($installerText.Contains("The CLI auto-starts the service if needed.")) {
    throw "Source installer must not claim that model commands auto-start the gateway"
}
if (-not $installerText.Contains('Invoke-OmniInfer serve --detach --port $OmniPort --no-restore-model')) {
    throw "Source installer backend activation must not restore a previous model"
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$diffusionBuildScript = Join-Path $repoRoot "scripts\platforms\windows\stable-diffusion.cpp-vulkan\build.ps1"
$diffusionTokens = $null
$diffusionErrors = $null
[System.Management.Automation.Language.Parser]::ParseFile(
    (Resolve-Path $diffusionBuildScript),
    [ref]$diffusionTokens,
    [ref]$diffusionErrors
) | Out-Null
if ($diffusionErrors.Count -gt 0) {
    throw "Windows stable-diffusion.cpp build script has parse errors: $($diffusionErrors -join '; ')"
}
$diffusionBuildText = [IO.File]::ReadAllText((Resolve-Path $diffusionBuildScript))
if (-not $diffusionBuildText.Contains('$WindowsTargetVersion = "0x0A00"') -or
    -not $diffusionBuildText.Contains('-D_WIN32_WINNT=$WindowsTargetVersion -DWINVER=$WindowsTargetVersion')) {
    throw "Windows stable-diffusion.cpp must target Windows 10 for cpp-httplib"
}

$waitFunction = $ast.Find({
    param($node)
    $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
        $node.Name -eq "Wait-ForExitKey"
}, $true)
if (-not $waitFunction) {
    throw "Wait-ForExitKey function was not found"
}

$fatalFunction = $ast.Find({
    param($node)
    $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
        $node.Name -eq "Stop-Fatal"
}, $true)
if (-not $fatalFunction -or $fatalFunction.Extent.Text -notmatch '\bWait-ForExitKey\b') {
    throw "Stop-Fatal must use the shared non-interactive exit helper"
}

$loggedBuildFunction = $ast.Find({
    param($node)
    $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
        $node.Name -eq "Invoke-LoggedPowerShellBuild"
}, $true)
if (-not $loggedBuildFunction) {
    throw "Invoke-LoggedPowerShellBuild function was not found"
}

Invoke-Expression $waitFunction.Extent.Text
Invoke-Expression $loggedBuildFunction.Extent.Text
$NonInteractive = $true
$timer = [System.Diagnostics.Stopwatch]::StartNew()
$output = & { Wait-ForExitKey } 2>&1 | Out-String
$timer.Stop()

if ($output.Trim()) {
    throw "Non-interactive failure path produced an exit prompt: $output"
}
if ($timer.Elapsed.TotalSeconds -ge 1) {
    throw "Non-interactive failure path waited for $($timer.Elapsed.TotalSeconds) seconds"
}

$tempRoot = Join-Path ([IO.Path]::GetTempPath()) ("omniinfer-source-installer-" + [Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $tempRoot | Out-Null
try {
    $nativeScript = Join-Path $tempRoot "native-stderr.ps1"
    $buildLog = Join-Path $tempRoot "build.log"
    @'
[Console]::Error.WriteLine("expected native status")
exit 0
'@ | Set-Content -LiteralPath $nativeScript -Encoding ASCII

    $ErrorActionPreference = "Stop"
    $exitCode = Invoke-LoggedPowerShellBuild `
        -Arguments @("-NoLogo", "-NoProfile", "-NonInteractive", "-File", $nativeScript) `
        -LogPath $buildLog
    if ($exitCode -ne 0) {
        throw "Native stderr success path returned exit code $exitCode"
    }
    if (-not (Select-String -LiteralPath $buildLog -SimpleMatch "expected native status" -Quiet)) {
        throw "Native stderr was not preserved in the build log"
    }

    $failingScript = Join-Path $tempRoot "native-failure.ps1"
    $failingLog = Join-Path $tempRoot "failure.log"
    @'
[Console]::Error.WriteLine("expected native failure")
exit 7
'@ | Set-Content -LiteralPath $failingScript -Encoding ASCII

    $exitCode = Invoke-LoggedPowerShellBuild `
        -Arguments @("-NoLogo", "-NoProfile", "-NonInteractive", "-File", $failingScript) `
        -LogPath $failingLog
    if ($exitCode -ne 7) {
        throw "Native failure path returned exit code $exitCode instead of 7"
    }
} finally {
    Remove-Item -LiteralPath $tempRoot -Recurse -Force -ErrorAction SilentlyContinue
}

# The expected native failure above leaves PowerShell's process-wide status at
# 7 even though every assertion passed. Do not leak that fixture status to the
# caller or CI step.
$global:LASTEXITCODE = 0
Write-Host "source installer PowerShell tests passed"
