param(
    [string]$PackageName = "OmniInfer",
    [string[]]$Backends = @(),
    [switch]$BuildCpuBackend,
    [switch]$BuildCudaBackend,
    [switch]$BuildVulkanBackend,
    [switch]$BuildArm64Backend,
    [switch]$BuildSyclBackend,
    [switch]$BuildHipBackend,
    [string]$BuildType = "Release",
    [string]$CudaArchitectures = "",
    [string]$GpuTargets = "",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$PlatformRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $PlatformRoot "..\..\..")
$CpuScript = Join-Path $PlatformRoot "build-llama-cpu.ps1"
$CudaScript = Join-Path $PlatformRoot "build-llama-cuda.ps1"
$VulkanScript = Join-Path $PlatformRoot "build-llama-vulkan.ps1"
$Arm64Script = Join-Path $PlatformRoot "build-llama-arm64.ps1"
$SyclScript = Join-Path $PlatformRoot "build-llama-sycl.ps1"
$HipScript = Join-Path $PlatformRoot "build-llama-hip.ps1"
$LauncherScript = Join-Path $PlatformRoot "write-portable-launchers.ps1"
$PortableRoot = Join-Path $RepoRoot "release\portable\windows-x64\OmniInfer"
$BuildRoot = Join-Path $RepoRoot "release\build"
$CliWorkRoot = Join-Path $BuildRoot "pyinstaller-work-cli"
$CliSpecRoot = Join-Path $BuildRoot "pyinstaller-spec-cli"
$CliDistRoot = Join-Path $BuildRoot "cli-dist"
$LocalRuntimeRoot = Join-Path $RepoRoot ".local\runtime\windows"
$ModelCatalogRoot = Join-Path $RepoRoot "service_core\model_catalogs"
$UsageTemplate = Join-Path $RepoRoot "tmp\usage.md"

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

function Require-Command {
    param([string]$Name, [string]$Hint = "")
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        $message = "Required command '$Name' was not found in PATH."
        if ($Hint) {
            $message += " $Hint"
        }
        throw $message
    }
}

function Ensure-PyInstaller {
    $probe = python -c "import importlib.util; print(importlib.util.find_spec('PyInstaller') is not None)" 2>&1
    if ($probe.ToString().Trim().ToLower() -ne "true") {
        Write-Host "PyInstaller not found. Installing..."
        python -m pip install pyinstaller
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to install PyInstaller."
        }
    }
}

function Reset-Directory {
    param([string]$Path)
    if (Test-Path -LiteralPath $Path) {
        Remove-Item -LiteralPath $Path -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $Path | Out-Null
}

function Get-InstalledBackends {
    $items = @()
    if (Test-Path -LiteralPath $LocalRuntimeRoot) {
        foreach ($dir in Get-ChildItem -LiteralPath $LocalRuntimeRoot -Directory) {
            $launcher = Join-Path $dir.FullName "bin\llama-server.exe"
            if (Test-Path -LiteralPath $launcher) {
                $items += $dir.Name
            }
        }
    }
    return $items
}

function Assert-BackendRuntimeSelfContained {
    param([string]$Backend, [string]$BinRoot)

    $requiredPatterns = @("llama-server.exe")
    if ($Backend -like "*cuda*") {
        $requiredPatterns += @(
            "cudart64*.dll",
            "cublas64*.dll",
            "cublasLt64*.dll",
            "msvcp140.dll",
            "vcruntime140.dll",
            "vcruntime140_1.dll",
            "vcomp140.dll"
        )
    }

    foreach ($pattern in $requiredPatterns) {
        if (-not (Get-ChildItem -LiteralPath $BinRoot -Filter $pattern -File -ErrorAction SilentlyContinue)) {
            throw "Backend '$Backend' is not self-contained: missing $pattern in $BinRoot."
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

if ($BuildVulkanBackend) {
    $vulkanArgs = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $VulkanScript,
        "-BuildType", $BuildType
    )
    if ($DryRun) {
        $vulkanArgs += "-DryRun"
    }
    Write-Host "Preparing Vulkan backend before packaging..."
    powershell @vulkanArgs
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

if ($BuildArm64Backend) {
    $arm64Args = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $Arm64Script,
        "-BuildType", $BuildType
    )
    if ($DryRun) {
        $arm64Args += "-DryRun"
    }
    Write-Host "Preparing arm64 backend before packaging..."
    powershell @arm64Args
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

if ($BuildSyclBackend) {
    $syclArgs = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $SyclScript,
        "-BuildType", $BuildType
    )
    if ($DryRun) {
        $syclArgs += "-DryRun"
    }
    Write-Host "Preparing SYCL backend before packaging..."
    powershell @syclArgs
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

if ($BuildHipBackend) {
    $hipArgs = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $HipScript,
        "-BuildType", $BuildType
    )
    if ($GpuTargets) {
        $hipArgs += @("-GpuTargets", $GpuTargets)
    }
    if ($DryRun) {
        $hipArgs += "-DryRun"
    }
    Write-Host "Preparing HIP backend before packaging..."
    powershell @hipArgs
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

Stop-RunningPortableRelease

$installedBackends = @(Get-InstalledBackends)
if (-not $installedBackends) {
    throw "No built backends found under $LocalRuntimeRoot. Build at least one backend first."
}

$requestedBackends = @(
    $Backends |
    Where-Object { $_ } |
    ForEach-Object { $_ -split "," } |
    ForEach-Object { $_.Trim() } |
    Where-Object { $_ }
)
if ($requestedBackends.Count -gt 0) {
    $missingBackends = @($requestedBackends | Where-Object { $installedBackends -notcontains $_ })
    if ($missingBackends.Count -gt 0) {
        throw "Requested backend(s) not found under ${LocalRuntimeRoot}: $($missingBackends -join ', ')"
    }
    $backends = @($requestedBackends)
}
else {
    $backends = @($installedBackends)
}

$preferenceOrder = @(
    "llama.cpp-cpu",
    "llama.cpp-cuda",
    "llama.cpp-vulkan",
    "llama.cpp-windows-arm64",
    "llama.cpp-sycl",
    "llama.cpp-hip",
    "ik_llama.cpp-cpu",
    "ik_llama.cpp-cuda"
)
$defaultBackend = $backends[0]
foreach ($candidate in $preferenceOrder) {
    if ($backends -contains $candidate) {
        $defaultBackend = $candidate
        break
    }
}

Write-Host "Packaged $($backends.Count) backend(s): $($backends -join ', ')"
Write-Host "Default backend: $defaultBackend"
Write-Host "Package root: $PortableRoot"

if ($DryRun) {
    Write-Host "Dry run enabled. Release packaging was not executed."
    exit 0
}

Require-Command python "Install Python 3.10+ and ensure it is on PATH."
Ensure-PyInstaller

Reset-Directory -Path $BuildRoot
Reset-Directory -Path $PortableRoot

$cliEntry = Join-Path $RepoRoot "omniinfer.py"
if (-not (Test-Path -LiteralPath $cliEntry)) {
    throw "CLI entry point not found: $cliEntry"
}

$pyinstallerExcludes = @(
    "cv2",
    "matplotlib",
    "mkl",
    "numpy",
    "pandas",
    "PIL",
    "scipy",
    "sklearn",
    "sympy",
    "torch",
    "torchvision"
)
$pyinstallerArgs = @(
    "--noconfirm",
    "--clean",
    "--onedir",
    "--console",
    "--name", "omniinfer-cli",
    "--distpath", $CliDistRoot,
    "--workpath", $CliWorkRoot,
    "--specpath", $CliSpecRoot,
    "--add-data", "$ModelCatalogRoot;service_core\model_catalogs"
)
foreach ($exclude in $pyinstallerExcludes) {
    $pyinstallerArgs += @("--exclude-module", $exclude)
}
$pyinstallerArgs += $cliEntry

Write-Host ""
Write-Host "Building omniinfer-cli.exe (CLI) with PyInstaller..."
python -m PyInstaller @pyinstallerArgs
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed for CLI."
}

$cliDistDir = Join-Path $CliDistRoot "omniinfer-cli"
$cliExe = Join-Path $cliDistDir "omniinfer-cli.exe"
if (-not (Test-Path -LiteralPath $cliExe)) {
    throw "CLI build succeeded but omniinfer-cli.exe not found at $cliExe"
}
Copy-Item -Path (Join-Path $cliDistDir "*") -Destination $PortableRoot -Recurse -Force

if (Test-Path -LiteralPath $LauncherScript) {
    & $LauncherScript -PortableRoot $PortableRoot
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

New-Item -ItemType Directory -Force -Path (Join-Path $PortableRoot "config") | Out-Null
$configContent = @"
{
  "host": "127.0.0.1",
  "port": 9000,
  "default_backend": "$defaultBackend",
  "default_thinking": "off",
  "window_mode": "hidden",
  "startup_timeout": 60,
  "runtime_root": "runtime"
}
"@
$utf8NoBom = New-Object System.Text.UTF8Encoding $false
[System.IO.File]::WriteAllText((Join-Path $PortableRoot "config\omniinfer.json"), $configContent, $utf8NoBom)

$runtimeRoot = Join-Path $PortableRoot "runtime"
New-Item -ItemType Directory -Force -Path $runtimeRoot | Out-Null
$copiedRuntimes = @()
foreach ($backend in $backends) {
    $sourceRoot = Join-Path $LocalRuntimeRoot $backend
    $sourceBin = Join-Path $sourceRoot "bin"
    $targetRoot = Join-Path $runtimeRoot $backend
    $targetBin = Join-Path $targetRoot "bin"

    New-Item -ItemType Directory -Force -Path $targetBin | Out-Null
    New-Item -ItemType Directory -Force -Path (Join-Path $targetRoot "logs") | Out-Null
    Copy-Item -Path (Join-Path $sourceBin "*") -Destination $targetBin -Recurse -Force
    Assert-BackendRuntimeSelfContained -Backend $backend -BinRoot $targetBin
    $copiedRuntimes += $backend
}

if (Test-Path -LiteralPath $UsageTemplate) {
    Copy-Item -LiteralPath $UsageTemplate -Destination (Join-Path $PortableRoot "README.md") -Force
}

Write-Host ""
Write-Host "============================================"
Write-Host "Portable release ready."
Write-Host "  Location:  $PortableRoot"
Write-Host "  Backends:  $($copiedRuntimes -join ', ')"
Write-Host "  Default:   $defaultBackend"
Write-Host "============================================"
Write-Host ""
Write-Host "Run with:"
Write-Host "  $PortableRoot\omniinfer.ps1 backend list"
Write-Host "  $PortableRoot\omniinfer.ps1 chat `"Hello`""
Write-Host "  $PortableRoot\omniinfer.cmd backend list  (cmd.exe compatibility)"
