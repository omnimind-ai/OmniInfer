# ----------------------------------------------------------------
#  OmniInfer source installer for Windows (PowerShell)
#
#  Usage:
#    irm "https://raw.githubusercontent.com/omnimind-ai/OmniInfer/main/scripts/install-from-source.ps1?$(Get-Random)" | iex
# ----------------------------------------------------------------

param(
    [string]$InstallDir = "$(Join-Path ([Environment]::GetFolderPath('UserProfile')) 'OmniInfer')",
    [Alias("m")]
    [string]$Model = "",
    [switch]$NoModel,
    [switch]$SkipBuild,
    [switch]$Prebuilt,
    [string]$Backend = "",
    [switch]$NonInteractive
)

$ErrorActionPreference = "Stop"
$RepoSsh   = "git@github.com:omnimind-ai/OmniInfer.git"
$RepoHttps = "https://github.com/omnimind-ai/OmniInfer.git"
$script:BuildLogPath = ""
$script:BuildStatus = "not-run"
$script:SummaryPath = ""
$script:CudaEffectiveArch = ""

if ($NoModel -and $Model) {
    throw "Cannot use -Model and -NoModel together."
}

# Helpers

function Write-Info  { param([string]$Msg) Write-Host "[INFO] $Msg" -ForegroundColor Cyan }
function Write-Ok    { param([string]$Msg) Write-Host "[ OK ] $Msg" -ForegroundColor Green }
function Write-Warn  { param([string]$Msg) Write-Host "[WARN] $Msg" -ForegroundColor Yellow }
function Write-Err   { param([string]$Msg) Write-Host "[ERR ] $Msg" -ForegroundColor Red }
function Wait-ForExitKey {
    if ($NonInteractive) { return }
    try {
        if ([Console]::IsInputRedirected) { return }
    } catch {
        return
    }
    Write-Host "Press any key to exit ..." -ForegroundColor DarkGray
    try { [void][Console]::ReadKey($true) } catch {}
}
function Stop-Fatal  {
    param([string]$Msg)
    Write-Err $Msg
    Write-Host ""
    Wait-ForExitKey
    exit 1
}

function Invoke-LoggedPowerShellBuild {
    param(
        [string[]]$Arguments,
        [string]$LogPath
    )

    # Windows PowerShell 5.1 promotes a native child's stderr records when the
    # caller uses Stop. CMake writes normal configure output to stderr, so keep
    # that stream in the build log and decide success from the process exit code.
    $previousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & powershell.exe @Arguments 2>&1 |
            Tee-Object -FilePath $LogPath |
            Out-Host
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }

    return $exitCode
}

function Test-Command {
    param([string]$Name, [string]$Hint)
    if (Get-Command $Name -ErrorAction SilentlyContinue) {
        Write-Ok $Name
    } else {
        Stop-Fatal "'$Name' is required but not found. $Hint"
    }
}

function Get-CudaEffectiveArch {
    if ($env:CMAKE_CUDA_ARCHITECTURES) { return $env:CMAKE_CUDA_ARCHITECTURES }
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $nvidiaSmi) { return "" }
    try {
        $cap = (& $nvidiaSmi.Source --query-gpu=compute_cap --format=csv,noheader 2>$null | Select-Object -First 1)
        if ($cap) { return (($cap.ToString()).Trim() -replace '\.', '') }
    } catch {}
    return ""
}

function Write-InstallSummary {
    $script:SummaryPath = Join-Path $InstallDir ".local\install-summary.json"
    $summaryDir = Split-Path -Parent $script:SummaryPath
    if (-not (Test-Path $summaryDir)) { New-Item -ItemType Directory -Force -Path $summaryDir | Out-Null }
    $summary = [ordered]@{
        install_dir = $InstallDir
        backend = $SelectedBackend
        model_configured = [bool]$ModelConfigured
        model_path = if ($ModelPath) { $ModelPath } else { $null }
        port = $OmniPort
        skip_build = [bool]$SkipBuild
        build_status = $script:BuildStatus
        build_log = if ($script:BuildLogPath) { $script:BuildLogPath } else { $null }
        cuda_effective_arch = if ($script:CudaEffectiveArch) { $script:CudaEffectiveArch } else { $null }
    }
    $summary | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $script:SummaryPath -Encoding UTF8
    Write-Ok "Install summary written: $script:SummaryPath"
}

function Sync-ProcessPathFromRegistry {
    $paths = @()
    foreach ($rawPath in @(
        $env:PATH,
        [System.Environment]::GetEnvironmentVariable("PATH", "User"),
        [System.Environment]::GetEnvironmentVariable("PATH", "Machine")
    )) {
        if (-not $rawPath) { continue }
        foreach ($entry in ($rawPath -split ';')) {
            $expanded = [System.Environment]::ExpandEnvironmentVariables($entry.Trim())
            if ($expanded -and ($paths -notcontains $expanded)) {
                $paths += $expanded
            }
        }
    }
    if ($paths.Count -gt 0) {
        $env:PATH = ($paths -join ';')
    }
}

function Add-UniqueCandidate {
    param([System.Collections.ArrayList]$List, [string]$Path)
    if ($Path -and ($List -notcontains $Path)) {
        [void]$List.Add($Path)
    }
}

function Import-VsDevEnvironment {
    param([string]$VcVarsAll)
    $envLines = cmd /c "`"$VcVarsAll`" x64 >nul 2>&1 && set" 2>&1
    if ($LASTEXITCODE -ne 0) { return $false }
    foreach ($line in $envLines) {
        if ($line -match '^([^=]+)=(.*)$') {
            [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
        }
    }
    return [bool](Get-Command cl.exe -ErrorAction SilentlyContinue)
}

function Find-And-Load-Msvc {
    if (Get-Command cl.exe -ErrorAction SilentlyContinue) { return $true }

    $candidates = [System.Collections.ArrayList]::new()

    foreach ($base in @($env:VSINSTALLDIR, $env:VCINSTALLDIR)) {
        if (-not $base) { continue }
        $root = $base
        if ($root -match '\\VC\\?$') { $root = Split-Path $root -Parent }
        Add-UniqueCandidate $candidates (Join-Path $root "VC\Auxiliary\Build\vcvarsall.bat")
    }

    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $installPaths = @()
        try {
            $installPaths = & $vswhere -all -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
        } catch {}
        if (-not $installPaths) {
            try { $installPaths = & $vswhere -all -products * -property installationPath 2>$null } catch {}
        }
        foreach ($installPath in $installPaths) {
            Add-UniqueCandidate $candidates (Join-Path $installPath "VC\Auxiliary\Build\vcvarsall.bat")
        }
    }

    foreach ($drive in (Get-PSDrive -PSProvider FileSystem -ErrorAction SilentlyContinue)) {
        foreach ($year in @("18", "2022", "2019", "2017")) {
            foreach ($edition in @("BuildTools", "Community", "Professional", "Enterprise")) {
                Add-UniqueCandidate $candidates (Join-Path $drive.Root "Program Files\Microsoft Visual Studio\$year\$edition\VC\Auxiliary\Build\vcvarsall.bat")
                Add-UniqueCandidate $candidates (Join-Path $drive.Root "Program Files (x86)\Microsoft Visual Studio\$year\$edition\VC\Auxiliary\Build\vcvarsall.bat")
            }
        }
    }

    foreach ($vcvarsall in $candidates) {
        if (-not (Test-Path $vcvarsall)) { continue }
        Write-Info "Found Visual Studio toolchain at $vcvarsall"
        if (Import-VsDevEnvironment $vcvarsall) {
            Write-Info "Loaded Visual Studio C++ environment"
            return $true
        }
    }

    return $false
}

function Set-GitHubSubmodulesToHttps {
    param([string]$RepoDir)
    $gitmodules = Join-Path $RepoDir ".gitmodules"
    if (-not (Test-Path $gitmodules)) { return 0 }

    $converted = 0
    $moduleUrls = git -C $RepoDir config --file .gitmodules --get-regexp '^submodule\..*\.url$' 2>$null
    foreach ($line in $moduleUrls) {
        if ($line -notmatch '^(submodule\..+)\.url\s+(.+)$') { continue }
        $keyPrefix = $matches[1]
        $url = $matches[2]
        if ($url -notmatch '^git@github\.com:(.+)$') { continue }

        $httpsUrl = "https://github.com/$($matches[1])"
        git -C $RepoDir config --local "$keyPrefix.url" $httpsUrl
        if ($LASTEXITCODE -eq 0) { $converted++ }
    }

    return $converted
}

# Arrow-key menu selector. Returns 0-based index.
# Falls back to numbered list when console is not interactive (e.g. irm | iex).
function Select-Menu {
    param([int]$Default, [string[]]$Options)
    if ($NonInteractive) { return $Default }

    $count = $Options.Count

    # Detect if interactive console is available
    $hasConsole = $true
    try { [void][Console]::CursorVisible } catch { $hasConsole = $false }
    if (-not $hasConsole -or -not [Environment]::UserInteractive) {
        # Fallback: numbered list
        for ($i = 0; $i -lt $count; $i++) {
            $marker = if ($i -eq $Default) { "*" } else { " " }
            Write-Host "  $marker $($i + 1). $($Options[$i])"
        }
        Write-Host ""
        $choice = Read-Host "  Enter number (default: $($Default + 1))"
        if ($choice -match '^\d+$' -and [int]$choice -ge 1 -and [int]$choice -le $count) {
            return ([int]$choice - 1)
        }
        return $Default
    }

    # Interactive: arrow-key selector
    $cur = $Default
    [Console]::CursorVisible = $false

    function Draw-Menu {
        for ($i = 0; $i -lt $count; $i++) {
            if ($i -eq $cur) {
                Write-Host "  > $($Options[$i])" -ForegroundColor Cyan
            } else {
                Write-Host "    $($Options[$i])"
            }
        }
    }

    Draw-Menu

    while ($true) {
        $key = [Console]::ReadKey($true)
        if ($key.Key -eq "UpArrow" -and $cur -gt 0) { $cur-- }
        elseif ($key.Key -eq "DownArrow" -and $cur -lt ($count - 1)) { $cur++ }
        elseif ($key.Key -eq "Enter") { break }
        else { continue }

        [Console]::SetCursorPosition(0, [Console]::CursorTop - $count)
        Draw-Menu
    }

    [Console]::CursorVisible = $true
    return $cur
}

# Banner

Write-Host ""
Write-Host "============================================================"
Write-Host "           OmniInfer Interactive Installer"
Write-Host "     Local LLM/VLM inference on every device"
Write-Host "============================================================"
Write-Host ""

# Step 1: Check prerequisites

Write-Info "Step 1/6: Checking prerequisites ..."
Sync-ProcessPathFromRegistry
Test-Command "git"    "Install from https://git-scm.com/"
Test-Command "cargo"  "Install Rust from https://rustup.rs/"

# Python: try python, python3, then uv run python
$script:PythonCmd = $null
foreach ($candidate in @("python", "python3")) {
    if (Get-Command $candidate -ErrorAction SilentlyContinue) {
        $script:PythonCmd = $candidate
        break
    }
}
if (-not $script:PythonCmd -and (Get-Command "uv" -ErrorAction SilentlyContinue)) {
    # uv is available; use "uv run python" as the python command
    try {
        $uvCheck = & uv run python --version 2>$null
        if ($LASTEXITCODE -eq 0) { $script:PythonCmd = "__uv__" }
    } catch {}
}
if ($script:PythonCmd) {
    if ($script:PythonCmd -eq "__uv__") { Write-Ok "python (via uv)" } else { Write-Ok $script:PythonCmd }
} else {
    Stop-Fatal "'python' is required but not found. Install from https://python.org/ or use uv: https://docs.astral.sh/uv/"
}

# C/C++ toolchain (needed for building backends)
# Check PATH first, then installed Visual Studio, then MSYS2_ROOT, then registry, then scan drives
$hasMsvc = [bool](Get-Command cl.exe -ErrorAction SilentlyContinue)
if (-not $hasMsvc) {
    $hasMsvc = Find-And-Load-Msvc
}
$hasMsys2Gcc = [bool](Get-Command gcc.exe -ErrorAction SilentlyContinue)
$msys2Ucrt64Bin = $null
if (-not $hasMsvc -and -not $hasMsys2Gcc) {
    # Try to find MSYS2 ucrt64 even if not in PATH
    $msys2Candidates = @()
    # Read MSYS2_ROOT from process env, then directly from registry (conda/venv may not inherit new system vars)
    if ($env:MSYS2_ROOT) { $msys2Candidates += $env:MSYS2_ROOT }
    foreach ($scope in @("Machine", "User")) {
        $regVal = [System.Environment]::GetEnvironmentVariable("MSYS2_ROOT", $scope)
        if ($regVal -and ($regVal -notin $msys2Candidates)) { $msys2Candidates += $regVal }
    }
    foreach ($key in @("HKLM:\SOFTWARE\MSYS2","HKCU:\SOFTWARE\MSYS2","HKLM:\SOFTWARE\WOW6432Node\MSYS2")) {
        try { $loc = (Get-ItemProperty -Path $key -ErrorAction SilentlyContinue).InstallLocation; if ($loc) { $msys2Candidates += $loc } } catch {}
    }
    foreach ($drive in (Get-PSDrive -PSProvider FileSystem -ErrorAction SilentlyContinue)) {
        $msys2Candidates += Join-Path $drive.Root "msys64"
    }
    foreach ($root in $msys2Candidates) {
        $ucrt = Join-Path $root "ucrt64\bin"
        if ((Test-Path (Join-Path $ucrt "gcc.exe")) -and (Test-Path (Join-Path $ucrt "g++.exe"))) {
            $msys2Ucrt64Bin = $ucrt
            break
        }
    }
    if ($msys2Ucrt64Bin) {
        # Auto-add to PATH for this session so build scripts can find it
        Write-Info "Found MSYS2 ucrt64 at $msys2Ucrt64Bin, adding to PATH"
        $env:PATH = "$msys2Ucrt64Bin;$env:PATH"
        $hasMsys2Gcc = $true
    }
}
if (-not $hasMsvc -and -not $hasMsys2Gcc) {
    # Diagnostic: show what we searched
    Write-Host ""
    Write-Host "  Diagnosis:" -ForegroundColor Yellow
    Write-Host "    cl.exe (MSVC):  not found in PATH"
    Write-Host "    gcc.exe:        not found in PATH"
    $envVal = $env:MSYS2_ROOT
    $regVal = [System.Environment]::GetEnvironmentVariable("MSYS2_ROOT", "Machine")
    if ($envVal) {
        Write-Host "    `$env:MSYS2_ROOT = $envVal"
        $ucrtPath = Join-Path $envVal "ucrt64\bin\gcc.exe"
        if (Test-Path $ucrtPath) {
            Write-Host "    gcc.exe found at $ucrtPath" -ForegroundColor Green
            Write-Host "    But it was not detected; this is a bug, please report it." -ForegroundColor Red
        } else {
            Write-Host "    $ucrtPath does NOT exist" -ForegroundColor Red
            Write-Host "    -> MSYS2 ucrt64 toolchain is not installed."
        }
    } elseif ($regVal) {
        Write-Host "    MSYS2_ROOT (from registry) = $regVal"
        $ucrtPath = Join-Path $regVal "ucrt64\bin\gcc.exe"
        if (Test-Path $ucrtPath) {
            Write-Host "    gcc.exe found at $ucrtPath" -ForegroundColor Green
            Write-Host "    But it was not detected; this is a bug, please report it." -ForegroundColor Red
        } else {
            Write-Host "    $ucrtPath does NOT exist" -ForegroundColor Red
            Write-Host "    -> MSYS2 ucrt64 toolchain is not installed."
        }
    } else {
        Write-Host "    `$env:MSYS2_ROOT:  not set"
        Write-Host "    Registry MSYS2_ROOT:  not set"
        Write-Host "    Scanned drives for msys64/:  not found"
    }
    Write-Host ""
    Write-Err "No C/C++ compiler found."
    Write-Host ""
    Write-Host "  Fix (pick one):" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "    Option A: " -ForegroundColor Cyan -NoNewline
    Write-Host "Install MSYS2 + ucrt64 toolchain"
    Write-Host "      1. Download and install MSYS2: https://www.msys2.org/"
    Write-Host "      2. Open MSYS2 UCRT64 terminal and run:"
    Write-Host "           pacman -S mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-ninja mingw-w64-ucrt-x86_64-cmake" -ForegroundColor White
    Write-Host "      3. Set system environment variable:"
    Write-Host '           MSYS2_ROOT = C:\msys64  (or your install path)' -ForegroundColor White
    Write-Host "      4. Re-run this script."
    Write-Host ""
    Write-Host "    Option B: " -ForegroundColor Cyan -NoNewline
    Write-Host "Install Visual Studio Build Tools"
    Write-Host "      1. Download: https://visualstudio.microsoft.com/downloads/#build-tools"
    Write-Host "      2. Install the 'Desktop development with C++' workload."
    Write-Host "      3. Open 'Developer PowerShell for VS' and re-run this script."
    Write-Host ""
    Wait-ForExitKey
    exit 1
} elseif ($hasMsvc) {
    Write-Ok "C++ toolchain: MSVC (cl.exe)"
} else {
    Write-Ok "C++ toolchain: MSYS2 (gcc.exe)"
}
Write-Host ""

# Step 2: Clone or update repo

Write-Info "Step 2/6: Preparing repository ..."
if (Test-Path "$InstallDir\.git") {
    Write-Info "Found existing clone at $InstallDir, updating ..."
    try { git -C $InstallDir pull --ff-only 2>&1 | Out-Null } catch { Write-Warn "Pull failed, continuing with existing code" }
} else {
    Write-Info "Cloning OmniInfer to $InstallDir ..."
    $clonedViaHttps = $false
    Write-Info "Trying SSH (timeout 15s) ..."
    $prevEAP = $ErrorActionPreference; $ErrorActionPreference = "SilentlyContinue"
    $env:GIT_SSH_COMMAND = "ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no"
    git clone --depth 1 $RepoSsh $InstallDir *>&1 | Out-Null
    $sshExit = $LASTEXITCODE
    Remove-Item Env:GIT_SSH_COMMAND -ErrorAction SilentlyContinue
    $ErrorActionPreference = $prevEAP
    if ($sshExit -ne 0) {
        Write-Warn "SSH clone failed, falling back to HTTPS ..."
        if (Test-Path $InstallDir) { Remove-Item -Recurse -Force $InstallDir -ErrorAction SilentlyContinue }
        git clone --depth 1 $RepoHttps $InstallDir
        if ($LASTEXITCODE -ne 0) {
            Stop-Fatal "git clone failed via both SSH and HTTPS. Check your network connection and try again."
        }
        $clonedViaHttps = $true
    }
}
if (-not (Test-Path (Join-Path $InstallDir "omniinfer.ps1"))) {
    Stop-Fatal "Repository clone appears incomplete; omniinfer.ps1 not found in $InstallDir"
}
Write-Ok "Repository ready at $InstallDir"

# Hand off to the repo's own copy of this script so that fixes in the repo
# take effect immediately, even when the irm-downloaded script is stale.
$repoScript = Join-Path $InstallDir "scripts\install-from-source.ps1"
if (-not $env:OMNIINFER_INSTALL_HANDOFF -and (Test-Path $repoScript)) {
    $env:OMNIINFER_INSTALL_HANDOFF = "1"
    $handoffArgs = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $repoScript, "-InstallDir", $InstallDir)
    if ($Model)   { $handoffArgs += @("-m", $Model) }
    if ($NoModel) { $handoffArgs += "-NoModel" }
    if ($Backend) { $handoffArgs += @("-Backend", $Backend) }
    if ($Prebuilt)       { $handoffArgs += "-Prebuilt" }
    if ($SkipBuild)      { $handoffArgs += "-SkipBuild" }
    if ($NonInteractive) { $handoffArgs += "-NonInteractive" }
    & powershell.exe @handoffArgs
    exit $LASTEXITCODE
}

Write-Info "Building OmniInfer CLI ..."
Push-Location $InstallDir
try {
    & cargo build --locked -p omniinfer-cli
    if ($LASTEXITCODE -ne 0) {
        Stop-Fatal "Failed to build the OmniInfer CLI. Check the Cargo output above."
    }
} finally {
    Pop-Location
}
$builtCli = Join-Path $InstallDir "target\debug\omniinfer.exe"
if (-not (Test-Path -LiteralPath $builtCli -PathType Leaf)) {
    Stop-Fatal "Cargo did not produce $builtCli"
}
& (Join-Path $InstallDir "omniinfer.ps1") --version
if ($LASTEXITCODE -ne 0) {
    Stop-Fatal "The freshly built OmniInfer CLI failed its version check."
}
Write-Ok "OmniInfer CLI is ready"

# Ensure a usable port

$OmniPort = 9000

function Test-PortFree {
    param([int]$Port)
    try {
        $tcp = New-Object System.Net.Sockets.TcpClient
        $tcp.Connect("127.0.0.1", $Port)
        $tcp.Close()
        # Connection succeeded = something is listening = port in use
        return $false
    } catch {
        # Connection refused = nothing listening = port is free
        return $true
    }
}

if (-not (Test-PortFree $OmniPort)) {
    Write-Warn "Port $OmniPort is in use; leaving the existing service untouched and looking for a free port ..."
    $found = $false
    foreach ($tryPort in 9001, 9002, 9003, 9004, 9005, 9010, 9020, 9050, 9100, 8900, 8800, 19000) {
        if (Test-PortFree $tryPort) {
            $OmniPort = $tryPort
            $found = $true
            break
        }
    }
    if (-not $found) {
        Stop-Fatal "Could not find a free port"
    }
    Write-Info "Using port $OmniPort"
    $configDir = Join-Path $InstallDir "config"
    if (-not (Test-Path $configDir)) { New-Item -ItemType Directory -Path $configDir -Force | Out-Null }
    $configFile = Join-Path $configDir "omniinfer.json"
    $configPayload = [ordered]@{ host = "127.0.0.1"; port = $OmniPort } | ConvertTo-Json
    [IO.File]::WriteAllText($configFile, $configPayload, (New-Object Text.UTF8Encoding($false)))
    Write-Ok "Config written: $configFile (port $OmniPort)"
}
Write-Host ""

# Step 3: Detect platform and choose backend
# Get available backends from CLI

Write-Info "Step 3/6: Detecting platform and hardware ..."

$omniinferScript = Join-Path $InstallDir "omniinfer.ps1"

# Helper: invoke the Rust control-plane launcher. Service settings, including
# the selected port, are read from the repository config.
function Invoke-OmniInfer {
    & $omniinferScript @args
}

# Cleanup: shut down any gateway service started by the CLI on script exit
Register-EngineEvent PowerShell.Exiting -Action {
    try { Invoke-OmniInfer shutdown 2>$null } catch {}
} | Out-Null

$BackendIds   = @()
$BackendDescs = @()
# The stable text table keeps backend IDs in its first column. Querying the
# local CLI does not require starting or replacing a gateway.
$rawOutput = (Invoke-OmniInfer backend list --scope compatible 2>$null) -join "`n"
foreach ($line in $rawOutput -split "`n") {
    $columns = $line.Trim() -split '\s+'
    if ($columns.Count -eq 0) { continue }
    $backendId = $columns[0]
    if ($backendId -in @("", "Compatible", "Backend", "Install") -or $backendId -match '^-+$') {
        continue
    }
    if ($backendId -match '^[a-zA-Z0-9._-]+$') {
        $BackendIds += $backendId
        $BackendDescs += $backendId
    }
}

if ($BackendIds.Count -eq 0) {
    Stop-Fatal "No backends found. Check your platform support."
}

function Test-PrebuiltBackend {
    param([string]$BackendId)
    $catalogPath = Join-Path $InstallDir "scripts\prebuilt_backends.json"
    if (-not (Test-Path -LiteralPath $catalogPath)) {
        return $false
    }
    try {
        $catalogRaw = Get-Content -LiteralPath $catalogPath -Raw -Encoding UTF8
        if ($catalogRaw.Length -gt 0 -and $catalogRaw[0] -eq [char]0xFEFF) {
            $catalogRaw = $catalogRaw.Substring(1)
        }
        $catalog = $catalogRaw | ConvertFrom-Json
        $windowsEntries = $catalog.platforms.windows
        if (-not $windowsEntries) {
            return $false
        }
        return ($windowsEntries.PSObject.Properties.Name -contains $BackendId)
    } catch {
        return $false
    }
}

Write-Info "Platform: Windows"
$script:CudaEffectiveArch = Get-CudaEffectiveArch
if ($script:CudaEffectiveArch) {
    Write-Info "CUDA effective architecture: $script:CudaEffectiveArch"
}
Write-Host ""

if ($Backend) {
    $SelectedBackend = $Backend
} else {
    $Prebuilt = $false
    $prebuiltIds = @()
    $prebuiltDescs = @()
    for ($i = 0; $i -lt $BackendIds.Count; $i++) {
        if (Test-PrebuiltBackend $BackendIds[$i]) {
            $prebuiltIds += $BackendIds[$i]
            $prebuiltDescs += "$($BackendDescs[$i])  (prebuilt)"
        }
    }

    $menuDescs = @($BackendDescs)
    if ($prebuiltIds.Count -gt 0) {
        $menuDescs += $prebuiltDescs
    }
    $menuDescs += "Skip for now  -  install backend manually later"

    Write-Host "  Available backends (arrow keys to move, Enter to select):"
    Write-Host ""

    $idx = Select-Menu -Default 0 -Options $menuDescs

    if ($idx -lt $BackendIds.Count) {
        $SelectedBackend = $BackendIds[$idx]
    } elseif ($idx -lt ($BackendIds.Count + $prebuiltIds.Count)) {
        $Prebuilt = $true
        $SelectedBackend = $prebuiltIds[$idx - $BackendIds.Count]
    } else {
        Write-Info "Skipping backend selection. You can install a backend later with:"
        Write-Host "    cd $InstallDir"
        Write-Host "    .\omniinfer.ps1 backend list --scope compatible"
        Write-Host "    .\omniinfer.ps1 build <backend-id>"
        $SkipBuild = $true
        $SelectedBackend = $BackendIds[0]
    }
}

Write-Ok "Selected: $SelectedBackend"
if ($Prebuilt) {
    Write-Info "Install mode: prebuilt"
}
Write-Host ""

# Step 4: Build backend

if ($Prebuilt) {
    Write-Info "Step 4/6: Installing prebuilt backend ..."
} else {
    Write-Info "Step 4/6: Building backend ..."
}

if (-not $Prebuilt) {
    # Windows build scripts do NOT auto-bootstrap submodules.
    $sourceSubmodule = if ($SelectedBackend.StartsWith("stable-diffusion.cpp")) {
        "framework/stable-diffusion.cpp"
    } else {
        "framework/llama.cpp"
    }
    $sourceLabel = Split-Path $sourceSubmodule -Leaf
    $sourceDir = Join-Path $InstallDir ($sourceSubmodule.Replace('/', '\'))
    if (-not (Test-Path (Join-Path $sourceDir "CMakeLists.txt"))) {
        Write-Info "Initializing $sourceLabel submodule ..."
        git -C $InstallDir submodule update --init --recursive --depth 1 --progress $sourceSubmodule
        if ($LASTEXITCODE -ne 0) {
            Write-Warn "SSH submodule clone failed, retrying with HTTPS ..."
            $converted = Set-GitHubSubmodulesToHttps $InstallDir
            if ($converted -eq 0) {
                Stop-Fatal "Failed to initialize $sourceLabel submodule and no GitHub SSH submodule URLs could be converted to HTTPS."
            }
            git -C $InstallDir submodule update --init --recursive --depth 1 --progress $sourceSubmodule
            if ($LASTEXITCODE -ne 0) {
                Stop-Fatal "Failed to initialize $sourceLabel submodule via SSH and HTTPS. Check network access to GitHub and retry."
            }
        }
        if (-not (Test-Path (Join-Path $sourceDir "CMakeLists.txt"))) {
            Stop-Fatal "$sourceLabel submodule initialized but CMakeLists.txt is missing in $sourceDir"
        }
        Write-Ok "Submodule ready"
        Write-Host ""
    }
}

# Discover build script by convention: scripts/platforms/windows/<backend_id>/build.ps1
$fullScript = Join-Path $InstallDir "scripts\platforms\windows\$SelectedBackend\build.ps1"
if (-not (Test-Path $fullScript)) {
    Stop-Fatal "Build script not found: $fullScript"
}

if ($SkipBuild) {
    Write-Info "Skipping build (-SkipBuild)"
    $script:BuildStatus = "skipped"
} else {
    $runtimeCheck = Invoke-OmniInfer backend list --scope installed 2>$null
    $isBuilt = ($runtimeCheck -join "`n") -match "(?m)^\s*$([regex]::Escape($SelectedBackend))\s+"
    if ($isBuilt) {
        Write-Ok "Backend $SelectedBackend already installed, skipping"
        $script:BuildStatus = if ($Prebuilt) { "prebuilt" } else { "already-built" }
    } else {
        if ($Prebuilt) {
            Write-Info "Installing prebuilt $SelectedBackend ..."
        } else {
            Write-Info "Building $SelectedBackend (this may take a few minutes) ..."
        }
        $buildLogDir = Join-Path $InstallDir "tmp\test_results\install"
        if (-not (Test-Path $buildLogDir)) { New-Item -ItemType Directory -Force -Path $buildLogDir | Out-Null }
        $logKind = if ($Prebuilt) { "prebuilt" } else { "build" }
        $script:BuildLogPath = Join-Path $buildLogDir ("{0}-{1}-{2}.log" -f $SelectedBackend, $logKind, (Get-Date -Format "yyyyMMdd-HHmmss"))
        Write-Info "Build log: $script:BuildLogPath"
        $scriptArgs = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $fullScript)
        if ($Prebuilt) {
            $scriptArgs += "-Prebuilt"
        }
        $buildExitCode = Invoke-LoggedPowerShellBuild `
            -Arguments $scriptArgs `
            -LogPath $script:BuildLogPath
        if ($buildExitCode -ne 0) {
            $script:BuildStatus = "failed"
            Write-Host ""
            Write-InstallSummary
            Write-Err "Backend install failed (exit code $buildExitCode). See $script:BuildLogPath for details."
            exit 1
        }
        # Verify build produced the expected binary
        $binDir = Join-Path $InstallDir ".local\runtime\windows\$SelectedBackend\bin"
        if (-not (Test-Path $binDir) -or (Get-ChildItem $binDir -File -ErrorAction SilentlyContinue).Count -eq 0) {
            $script:BuildStatus = "failed"
            Write-InstallSummary
            Write-Err "Build completed but no binaries found in $binDir"
            exit 1
        }
        $script:BuildStatus = if ($Prebuilt) { "prebuilt" } else { "built" }
        Write-Ok "Backend install complete"
    }
}

if (-not $SkipBuild) {
    Write-Info "Starting the local gateway to activate $SelectedBackend ..."
    Invoke-OmniInfer serve --detach --port $OmniPort --no-restore-model
    if ($LASTEXITCODE -ne 0) { Stop-Fatal "Failed to start the local OmniInfer gateway." }
    Invoke-OmniInfer backend select $SelectedBackend
    if ($LASTEXITCODE -ne 0) { Stop-Fatal "Failed to activate backend $SelectedBackend." }
}
Write-Host ""

# Step 5: Model configuration

Write-Info "Step 5/6: Model configuration"
Write-Host ""
Write-Host "  How would you like to set up a model?"
Write-Host ""

$ModelConfigured = $false
$ModelPath = ""

if ($Model) {
    $Model = $Model.Trim('"', "'", ' ')
    Write-Info "Using provided model: $Model"
    $ModelPath = $Model
    $ModelConfigured = $true
} elseif ($NoModel) {
    Write-Info "Skipping model configuration (-NoModel)"
} else {
    $modelChoice = Select-Menu -Default 0 -Options @(
        "Download a recommended model",
        "Use a local model file",
        "Skip (configure later)"
    )

    switch ($modelChoice) {
        0 {
            Write-Info "Reading bundled model catalog ..."
            try {
                $catalogPath = Join-Path $InstallDir "crates\omniinfer-core\model_catalogs\windows.json"
                if (-not (Test-Path -LiteralPath $catalogPath)) {
                    throw "Bundled model catalog not found: $catalogPath"
                }
                $catalogRaw = Get-Content -LiteralPath $catalogPath -Raw -Encoding UTF8
                if ($catalogRaw.Length -gt 0 -and $catalogRaw[0] -eq [char]0xFEFF) {
                    $catalogRaw = $catalogRaw.Substring(1)
                }
                $catalog = $catalogRaw | ConvertFrom-Json

                $modelList = @()
                $seen = @{}
                foreach ($backendName in $catalog.PSObject.Properties.Name) {
                    $families = $catalog.$backendName
                    foreach ($famName in $families.PSObject.Properties.Name) {
                        $famModels = $families.$famName
                        foreach ($modelName in $famModels.PSObject.Properties.Name) {
                            $modelInfo = $famModels.$modelName
                            $quants = $modelInfo.quantization
                            if (-not $quants) { continue }
                            foreach ($qName in @("Q4_K_M", "Q6_K", "Q8_0")) {
                                $q = $quants.$qName
                                if (-not $q) { continue }
                                $dl = $q.download
                                $sizeStr = $q.size
                                if (-not $dl -or -not $sizeStr) { continue }
                                $dedup = "$modelName|$qName"
                                if ($seen.ContainsKey($dedup)) { continue }
                                $seen[$dedup] = $true
                                try { $sizeGib = [double]$sizeStr } catch { continue }
                                if ($sizeGib -gt 6.0 -or $sizeGib -lt 0.1) { continue }
                                $modelList += [PSCustomObject]@{
                                    Name  = $modelName
                                    Quant = $qName
                                    Size  = $sizeGib
                                    Url   = $dl
                                }
                                break
                            }
                        }
                    }
                }

                $modelList = $modelList | Sort-Object Size | Select-Object -First 6

                if ($modelList.Count -eq 0) {
                    Write-Warn "No suitable models found in catalog."
                } else {
                    Write-Host ""
                    Write-Host "  Recommended models:"
                    Write-Host ""

                    $dlLabels = @()
                    foreach ($m in $modelList) {
                        $dlLabels += ("{0,-32} {1,-10} {2:F2} GiB" -f $m.Name, $m.Quant, $m.Size)
                    }

                    $dlIdx = Select-Menu -Default 0 -Options $dlLabels
                    $selected = $modelList[$dlIdx]

                    $modelsDir = Join-Path $InstallDir ".local\models"
                    if (-not (Test-Path $modelsDir)) { New-Item -ItemType Directory -Force -Path $modelsDir | Out-Null }
                    $dlFilename = Split-Path $selected.Url -Leaf
                    $ModelPath = Join-Path $modelsDir $dlFilename

                    if (Test-Path $ModelPath) {
                        Write-Ok "Model already downloaded: $ModelPath"
                    } else {
                        Write-Info "Downloading $($selected.Name) ($($selected.Quant), $($selected.Size.ToString('F2')) GiB) ..."
                        Write-Info "Saving to: $ModelPath"
                        Invoke-WebRequest -Uri $selected.Url -OutFile $ModelPath -UseBasicParsing
                        Write-Ok "Download complete: $ModelPath"
                    }
                    $ModelConfigured = $true
                }
            } catch {
                Write-Warn "Could not read bundled model catalog: $_"
                Write-Warn "You can configure a model manually later."
            }
        }
        1 {
            Write-Host ""
            $localPath = (Read-Host "  Enter model path").Trim('"', "'", ' ')
            if ($localPath -and (Test-Path $localPath)) {
                $ModelPath = $localPath
                $ModelConfigured = $true
                Write-Ok "Model: $ModelPath"
            } else {
                Write-Warn "Path not found: $localPath"
                Write-Warn "Skipping model configuration."
            }
        }
        default {
            Write-Info "Skipping model configuration."
        }
    }
}

Write-Host ""

# Step 6: Load model and finish

Write-Info "Step 6/6: Finishing up ..."
Write-Host ""

# Finish message (reused by both paths)
function Print-Finish {
    Write-Host ""
    if (-not $SkipBuild) { Invoke-OmniInfer shutdown 2>$null }
    Write-InstallSummary

    Write-Host ""
    Write-Host "============================================================"
    if ($ModelConfigured) {
        Write-Host "                   Setup Complete!"
    } else {
        Write-Host "                   Install Complete!"
    }
    Write-Host "============================================================"
    Write-Host ""
    Write-Host "  Install:  $InstallDir"
    Write-Host "  Backend:  $SelectedBackend"
    if ($ModelPath) {
        Write-Host "  Model:    $(Split-Path $ModelPath -Leaf)"
        Write-Host ""
        Write-Host "  Your backend selection is saved. Next time just run:"
        Write-Host ""
        Write-Host "    cd $InstallDir"
        Write-Host "    .\omniinfer.ps1 serve --detach"
        Write-Host "    .\omniinfer.ps1 model load -m $ModelPath"
        Write-Host "    .\omniinfer.ps1 chat --message `"Hello`""
    } else {
        Write-Host ""
        Write-Host "  To start chatting, load a model first:"
        Write-Host ""
        Write-Host "    cd $InstallDir"
        Write-Host "    .\omniinfer.ps1 serve --detach"
        Write-Host "    .\omniinfer.ps1 model load -m C:\path\to\model.gguf"
        Write-Host "    .\omniinfer.ps1 chat --message `"Hello`""
    }
    Write-Host ""
    Write-Host "  The model needs to be loaded each time after a restart."
    Write-Host ""
    Write-Host "  Other useful commands:"
    Write-Host "    .\omniinfer.ps1 backend list          # list available backends"
    Write-Host "    .\omniinfer.ps1 backend select <backend>  # switch backend"
    Write-Host "    .\omniinfer.ps1 model list            # browse supported models"
    Write-Host "    .\omniinfer.ps1 status                # check current state"
    Write-Host "    .\omniinfer.ps1 serve                 # start API server (http://127.0.0.1:9000)"
    Write-Host "    .\omniinfer.ps1 shutdown              # stop the service"
    Write-Host ""
    Write-Host "  Full documentation:"
    Write-Host "    CLI guide:   $InstallDir\docs\CLI.md"
    Write-Host "    API guide:   $InstallDir\docs\API.md"
    Write-Host "    Build guide: $InstallDir\docs\build.md"
    Write-Host ""
}

if ($ModelConfigured -and $ModelPath) {
    Write-Info "Loading model ..."
    Invoke-OmniInfer model load -m $ModelPath
    if ($LASTEXITCODE -ne 0) {
        Write-Err "Failed to load model. Make sure the backend is built and the model path is correct."
        Write-InstallSummary
        Write-Host ""
        Write-Host "  Try building the backend first, then re-run:"
        Write-Host "    cd $InstallDir"
        Write-Host "    .\omniinfer.ps1 serve --detach"
        Write-Host "    .\omniinfer.ps1 model load -m $ModelPath"
        Write-Host ""
        exit 1
    }
    Write-Ok "Model loaded"
    Write-Host ""

    # Interactive chat loop
    Write-Ok "Setup complete! Try chatting with the model (type 'exit' to quit, Ctrl+C to stop)."
    Write-Host ""

    # Capture Ctrl+C ourselves so we can run cleanup
    [Console]::TreatControlCAsInput = $true
    $chatDone = $false

    while (-not $chatDone) {
        Write-Host "You: " -ForegroundColor Cyan -NoNewline
        $inputBuf = ""

        # Read char-by-char to detect Ctrl+C
        while ($true) {
            $keyInfo = [Console]::ReadKey($true)
            if ($keyInfo.Key -eq "Enter") {
                Write-Host ""
                break
            }
            if ($keyInfo.Modifiers -band [ConsoleModifiers]::Control -and $keyInfo.Key -eq "C") {
                Write-Host ""
                $chatDone = $true
                break
            }
            if ($keyInfo.Key -eq "Backspace") {
                if ($inputBuf.Length -gt 0) {
                    $inputBuf = $inputBuf.Substring(0, $inputBuf.Length - 1)
                    Write-Host "`b `b" -NoNewline
                }
                continue
            }
            $inputBuf += $keyInfo.KeyChar
            Write-Host $keyInfo.KeyChar -NoNewline
        }

        if ($chatDone) { break }
        if ([string]::IsNullOrWhiteSpace($inputBuf)) { continue }
        if ($inputBuf -eq "exit" -or $inputBuf -eq "quit") { break }

        Write-Host "AI: " -ForegroundColor Green -NoNewline
        Invoke-OmniInfer chat --message $inputBuf
        Write-Host ""
    }

    [Console]::TreatControlCAsInput = $false
    Print-Finish
} else {
    Print-Finish
}
