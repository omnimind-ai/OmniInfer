param(
    [string]$PackageName = "OmniInfer",
    [switch]$BuildCpuBackend,
    [switch]$BuildCudaBackend,
    [string]$BuildType = "Release",
    [string]$CudaArchitectures = "",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$PlatformRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $PlatformRoot "..\..")
$CpuScript = Join-Path $PlatformRoot "build-llama-cpu.ps1"
$CudaScript = Join-Path $PlatformRoot "build-llama-cuda.ps1"
$ReleaseScript = Join-Path $RepoRoot "release\build_portable.ps1"
$PortableRoot = Join-Path $RepoRoot "release\portable\OmniInfer"

if (-not (Test-Path -LiteralPath $ReleaseScript)) {
    throw "Release build script not found: $ReleaseScript"
}

function Stop-RunningPortableRelease {
    if (-not (Test-Path -LiteralPath $PortableRoot)) {
        return
    }

    $prefix = ([System.IO.Path]::GetFullPath($PortableRoot)).TrimEnd('\') + '\'
    $rows = Get-CimInstance Win32_Process |
        Where-Object {
            $_.ExecutablePath -and
            ([System.IO.Path]::GetFullPath($_.ExecutablePath)).StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)
        }

    foreach ($row in $rows) {
        Write-Host ("Stopping running portable release process: PID {0} ({1})" -f $row.ProcessId, $row.Name)
        if (-not $DryRun) {
            Stop-Process -Id $row.ProcessId -Force -ErrorAction SilentlyContinue
        }
    }
}

if ($BuildCpuBackend) {
    $cpuArgs = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $CpuScript,
        "-BuildType", $BuildType
    )
    if ($DryRun) {
        $cpuArgs += "-DryRun"
    }
    Write-Host "Preparing CPU backend before packaging..."
    powershell @cpuArgs
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

if ($BuildCudaBackend) {
    $cudaArgs = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $CudaScript,
        "-BuildType", $BuildType
    )
    if ($CudaArchitectures) {
        $cudaArgs += @("-CudaArchitectures", $CudaArchitectures)
    }
    if ($DryRun) {
        $cudaArgs += "-DryRun"
    }
    Write-Host "Preparing CUDA backend before packaging..."
    powershell @cudaArgs
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

Stop-RunningPortableRelease

$releaseArgs = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", $ReleaseScript,
    "-PackageName", $PackageName
)

Write-Host "Running Windows portable release build:"
Write-Host "  powershell $($releaseArgs -join ' ')"

if ($DryRun) {
    Write-Host "Dry run enabled. Release packaging was not executed."
    exit 0
}

powershell @releaseArgs
exit $LASTEXITCODE
