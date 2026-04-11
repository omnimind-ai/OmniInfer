# ──────────────────────────────────────────────────────────────
#  OmniInfer interactive installer for Windows (PowerShell)
#
#  Usage:
#    irm https://raw.githubusercontent.com/omnimind-ai/OmniInfer/main/scripts/install.ps1 | iex
# ──────────────────────────────────────────────────────────────

param(
    [string]$InstallDir = "$(Get-Location)\OmniInfer",
    [Alias("m")]
    [string]$Model = "",
    [switch]$SkipBuild,
    [string]$Backend = "",
    [switch]$NonInteractive
)

$ErrorActionPreference = "Stop"
$RepoUrl = "https://github.com/omnimind-ai/OmniInfer.git"
$CatalogUrl = "https://omnimind-model.oss-cn-beijing.aliyuncs.com/backend/windows/model_list.json"

# ── Helpers ─────────────────────────────────────────────────

function Write-Info  { param([string]$Msg) Write-Host "[INFO] $Msg" -ForegroundColor Cyan }
function Write-Ok    { param([string]$Msg) Write-Host "[ OK ] $Msg" -ForegroundColor Green }
function Write-Warn  { param([string]$Msg) Write-Host "[WARN] $Msg" -ForegroundColor Yellow }
function Write-Err   { param([string]$Msg) Write-Host "[ERR ] $Msg" -ForegroundColor Red }
function Stop-Fatal  { param([string]$Msg) Write-Err $Msg; exit 1 }

function Test-Command {
    param([string]$Name, [string]$Hint)
    if (Get-Command $Name -ErrorAction SilentlyContinue) {
        Write-Ok $Name
    } else {
        Stop-Fatal "'$Name' is required but not found. $Hint"
    }
}

# Arrow-key menu selector. Returns 0-based index.
function Select-Menu {
    param([int]$Default, [string[]]$Options)
    if ($NonInteractive) { return $Default }

    $cur = $Default
    $count = $Options.Count

    # Hide cursor
    [Console]::CursorVisible = $false

    # Draw
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

        # Move cursor up and redraw
        [Console]::SetCursorPosition(0, [Console]::CursorTop - $count)
        Draw-Menu
    }

    [Console]::CursorVisible = $true
    return $cur
}

# ── Banner ──────────────────────────────────────────────────

Write-Host ""
Write-Host "============================================================"
Write-Host "           OmniInfer Interactive Installer"
Write-Host "     Local LLM/VLM inference on every device"
Write-Host "============================================================"
Write-Host ""

# ── Step 1: Check prerequisites ─────────────────────────────

Write-Info "Step 1/6: Checking prerequisites ..."
Test-Command "git"    "Install from https://git-scm.com/"
Test-Command "cmake"  "Install from https://cmake.org/download/"
Test-Command "python" "Install from https://python.org/"
Write-Host ""

# ── Step 2: Clone or update repo ────────────────────────────

Write-Info "Step 2/6: Preparing repository ..."
if (Test-Path "$InstallDir\.git") {
    Write-Info "Found existing clone at $InstallDir, updating ..."
    git -C $InstallDir pull --ff-only 2>$null
} else {
    Write-Info "Cloning OmniInfer to $InstallDir ..."
    git clone --depth 1 $RepoUrl $InstallDir
}
Write-Ok "Repository ready at $InstallDir"

Write-Info "Initializing llama.cpp submodule ..."
git -C $InstallDir submodule update --init --recursive --depth 1 --progress framework/llama.cpp
Write-Ok "Submodule ready"
Write-Host ""

# ── Step 3: Detect platform & choose backend ────────────────

Write-Info "Step 3/6: Detecting platform and hardware ..."

$Arch = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString().ToLower()

$BackendIds     = @()
$BackendLabels  = @()
$BackendScripts = @()
$Recommended    = 0

if ($Arch -eq "arm64") {
    $BackendIds     += "llama.cpp-windows-arm64"
    $BackendLabels  += "llama.cpp CPU (ARM64)"
    $BackendScripts += "windows\build-llama-arm64.ps1"
} else {
    $BackendIds     += "llama.cpp-cpu"
    $BackendLabels  += "llama.cpp CPU (x64)"
    $BackendScripts += "windows\build-llama-cpu.ps1"

    if (Get-Command "nvidia-smi" -ErrorAction SilentlyContinue) {
        try {
            nvidia-smi 2>$null | Out-Null
            Write-Ok "NVIDIA GPU detected"
            $BackendIds     += "llama.cpp-cuda"
            $BackendLabels  += "llama.cpp CUDA (NVIDIA)"
            $BackendScripts += "windows\build-llama-cuda.ps1"
            $Recommended = $BackendIds.Count - 1

            $BackendIds     += "llama.cpp-vulkan"
            $BackendLabels  += "llama.cpp Vulkan"
            $BackendScripts += "windows\build-llama-vulkan.ps1"
        } catch {}
    }

    if (Get-Command "hipconfig" -ErrorAction SilentlyContinue) {
        Write-Ok "AMD HIP detected"
        $BackendIds     += "llama.cpp-hip"
        $BackendLabels  += "llama.cpp HIP (AMD)"
        $BackendScripts += "windows\build-llama-hip.ps1"
        $Recommended = $BackendIds.Count - 1
    }

    if (Get-Command "vulkaninfo" -ErrorAction SilentlyContinue) {
        if (-not ($BackendIds -contains "llama.cpp-vulkan")) {
            $BackendIds     += "llama.cpp-vulkan"
            $BackendLabels  += "llama.cpp Vulkan"
            $BackendScripts += "windows\build-llama-vulkan.ps1"
        }
    }
}

Write-Info "Platform: Windows ($Arch)"
Write-Host ""

if ($Backend) {
    $SelectedBackend = $Backend
    $idx = [array]::IndexOf($BackendIds, $Backend)
    if ($idx -lt 0) { Stop-Fatal "Unknown backend: $Backend" }
    $SelectedScript = $BackendScripts[$idx]
} else {
    Write-Host "  Available backends (arrow keys to move, Enter to select):"
    Write-Host ""

    $menuLabels = @()
    for ($i = 0; $i -lt $BackendIds.Count; $i++) {
        $label = $BackendLabels[$i]
        if ($i -eq $Recommended) { $label += " (recommended)" }
        $menuLabels += $label
    }

    $idx = Select-Menu -Default $Recommended -Options $menuLabels

    $SelectedBackend = $BackendIds[$idx]
    $SelectedScript  = $BackendScripts[$idx]
}

Write-Ok "Selected: $SelectedBackend"
Write-Host ""

# ── Step 4: Build backend ───────────────────────────────────

Write-Info "Step 4/6: Building backend ..."

$cmdPath = Join-Path $InstallDir "omniinfer.cmd"
$fullScript = Join-Path $InstallDir "scripts\platforms\$SelectedScript"

$runtimeDir = Join-Path $InstallDir ".local\runtime\windows\$SelectedBackend\bin"
$alreadyBuilt = Test-Path $runtimeDir

if ($SkipBuild) {
    Write-Info "Skipping build (-SkipBuild)"
} elseif ($alreadyBuilt) {
    Write-Ok "Backend $SelectedBackend already built, skipping"
} else {
    Write-Info "Building $SelectedBackend (this may take a few minutes) ..."
    powershell -NoProfile -ExecutionPolicy Bypass -File $fullScript
    Write-Ok "Build complete"
}

& cmd /c $cmdPath select $SelectedBackend 2>$null
Write-Ok "Backend $SelectedBackend selected"
Write-Host ""

# ── Step 5: Model configuration ─────────────────────────────

Write-Info "Step 5/6: Model configuration"
Write-Host ""
Write-Host "  How would you like to set up a model?"
Write-Host ""

$ModelConfigured = $false
$ModelPath = ""

if ($Model) {
    Write-Info "Using provided model: $Model"
    $ModelPath = $Model
    $ModelConfigured = $true
} else {
    $modelChoice = Select-Menu -Default 0 -Options @(
        "Download a recommended model",
        "Use a local model file",
        "Skip (configure later)"
    )

    switch ($modelChoice) {
        0 {
            Write-Info "Fetching model catalog ..."
            try {
                $catalogRaw = (Invoke-WebRequest -Uri $CatalogUrl -UseBasicParsing).Content
                if ($catalogRaw.Length -gt 0 -and $catalogRaw[0] -eq [char]0xFEFF) {
                    $catalogRaw = $catalogRaw.Substring(1)
                }
                $catalog = $catalogRaw | ConvertFrom-Json

                $modelList = @()
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

                    $modelsDir = Join-Path $InstallDir "models"
                    if (-not (Test-Path $modelsDir)) { New-Item -ItemType Directory -Path $modelsDir | Out-Null }
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
                Write-Warn "Could not fetch model catalog: $_"
                Write-Warn "You can configure a model manually later."
            }
        }
        1 {
            Write-Host ""
            $localPath = Read-Host "  Enter model path"
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

# ── Step 6: Load model & finish ──────────────────────────────

Write-Info "Step 6/6: Finishing up ..."
Write-Host ""

if ($ModelConfigured -and $ModelPath) {
    Write-Info "Loading model ..."
    & cmd /c $cmdPath model load -m $ModelPath
    Write-Ok "Model loaded"
    Write-Host ""

    # ── Cleanup function (runs on exit or Ctrl+C) ──────────
    function Print-Finish {
        Write-Host ""
        & cmd /c $cmdPath shutdown 2>$null

        Write-Host ""
        Write-Host "============================================================"
        Write-Host "                   Setup Complete!"
        Write-Host "============================================================"
        Write-Host ""
        Write-Host "  Install:  $InstallDir"
        Write-Host "  Backend:  $SelectedBackend"
        Write-Host "  Model:    $(Split-Path $ModelPath -Leaf)"
        Write-Host ""
        Write-Host "  Your backend selection is saved. Next time just run:"
        Write-Host ""
        Write-Host "    cd $InstallDir"
        Write-Host "    .\omniinfer.cmd model load -m $ModelPath"
        Write-Host "    .\omniinfer.cmd chat --message `"Hello`""
        Write-Host ""
        Write-Host "  The model needs to be loaded each time after a restart."
        Write-Host "  The CLI auto-starts the service if needed."
        Write-Host ""
        Write-Host "  Full documentation:"
        Write-Host "    CLI guide:   $InstallDir\docs\CLI.md"
        Write-Host "    API guide:   $InstallDir\docs\API.md"
        Write-Host "    Build guide: $InstallDir\docs\build.md"
        Write-Host ""
    }

    # ── Interactive chat loop ──────────────────────────────
    Write-Ok "Setup complete! Try chatting with the model (type 'exit' to quit)."
    Write-Host ""
    try {
        while ($true) {
            Write-Host "You: " -ForegroundColor Cyan -NoNewline
            $userMsg = Read-Host
            if ([string]::IsNullOrWhiteSpace($userMsg)) { continue }
            if ($userMsg -eq "exit" -or $userMsg -eq "quit") { break }
            Write-Host "AI: " -ForegroundColor Green -NoNewline
            & cmd /c $cmdPath chat --message $userMsg
            Write-Host ""
        }
    } finally {
        Print-Finish
    }
} else {
    Write-Host ""
    Write-Host "============================================================"
    Write-Host "                   Install Complete!"
    Write-Host "============================================================"
    Write-Host ""
    Write-Host "  Install:  $InstallDir"
    Write-Host "  Backend:  $SelectedBackend"
    Write-Host ""
    Write-Host "  To start chatting, load a model first:"
    Write-Host ""
    Write-Host "    cd $InstallDir"
    Write-Host "    .\omniinfer.cmd model load -m C:\path\to\model.gguf"
    Write-Host "    .\omniinfer.cmd chat --message `"Hello`""
    Write-Host ""
    Write-Host "  The model needs to be loaded each time after a restart."
    Write-Host ""
    Write-Host "  Full documentation:"
    Write-Host "    CLI guide:   $InstallDir\docs\CLI.md"
    Write-Host "    API guide:   $InstallDir\docs\API.md"
    Write-Host "    Build guide: $InstallDir\docs\build.md"
    Write-Host ""
}
