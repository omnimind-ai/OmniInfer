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

function Read-Choice {
    param([string]$Prompt, [string]$Default)
    if ($NonInteractive) { return $Default }
    $input = Read-Host $Prompt
    if ([string]::IsNullOrWhiteSpace($input)) { return $Default }
    return $input.Trim()
}

# ── Banner ──────────────────────────────────────────────────

Write-Host ""
Write-Host ([char]0x2554 + ("=" * 58) + [char]0x2557)
Write-Host ([char]0x2551 + "             OmniInfer Interactive Installer              " + [char]0x2551)
Write-Host ([char]0x2551 + "       Local LLM/VLM inference on every device            " + [char]0x2551)
Write-Host ([char]0x255A + ("=" * 58) + [char]0x255D)
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
git -C $InstallDir submodule update --init --recursive --depth 1 framework/llama.cpp 2>$null
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
    $Recommended = 0
} else {
    $BackendIds     += "llama.cpp-cpu"
    $BackendLabels  += "llama.cpp CPU (x64)"
    $BackendScripts += "windows\build-llama-cpu.ps1"
    $Recommended = 0

    # Detect NVIDIA GPU
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

    # Detect AMD GPU (HIP)
    if (Get-Command "hipconfig" -ErrorAction SilentlyContinue) {
        Write-Ok "AMD HIP detected"
        $BackendIds     += "llama.cpp-hip"
        $BackendLabels  += "llama.cpp HIP (AMD)"
        $BackendScripts += "windows\build-llama-hip.ps1"
        $Recommended = $BackendIds.Count - 1
    }

    # Vulkan fallback
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
    Write-Host "  Available backends:"
    Write-Host ""
    for ($i = 0; $i -lt $BackendIds.Count; $i++) {
        $marker = ""
        if ($i -eq $Recommended) { $marker = " (recommended)" }
        Write-Host ("    [{0}] {1}{2}" -f ($i + 1), $BackendLabels[$i], $marker)
    }
    Write-Host ""

    $choice = Read-Choice "  Select backend [$($Recommended + 1)]" "$($Recommended + 1)"
    $idx = [int]$choice - 1

    if ($idx -lt 0 -or $idx -ge $BackendIds.Count) {
        Write-Warn "Invalid choice, using recommended"
        $idx = $Recommended
    }

    $SelectedBackend = $BackendIds[$idx]
    $SelectedScript  = $BackendScripts[$idx]
}

Write-Ok "Selected: $SelectedBackend"
Write-Host ""

# ── Step 4: Build backend ───────────────────────────────────

Write-Info "Step 4/6: Building backend ..."

$cmdPath = Join-Path $InstallDir "omniinfer.cmd"
$fullScript = Join-Path $InstallDir "scripts\platforms\$SelectedScript"

# Check if already built
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
Write-Host "    [1] Download a recommended model"
Write-Host "    [2] Use a local model file"
Write-Host "    [3] Skip (configure later)"
Write-Host ""

$ModelConfigured = $false

if ($Model) {
    Write-Info "Using provided model: $Model"
    $ModelPath = $Model
    $ModelConfigured = $true
} else {
    $modelChoice = Read-Choice "  Choose [1]" "1"

    switch ($modelChoice) {
        "1" {
            Write-Info "Fetching model catalog ..."
            try {
                $catalogRaw = (Invoke-WebRequest -Uri $CatalogUrl -UseBasicParsing).Content
                # Remove BOM if present
                if ($catalogRaw.Length -gt 0 -and $catalogRaw[0] -eq [char]0xFEFF) {
                    $catalogRaw = $catalogRaw.Substring(1)
                }
                $catalog = $catalogRaw | ConvertFrom-Json

                # Parse catalog to find small models
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
                                break  # one quant per model
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
                    Write-Host ("    {0,-4} {1,-35} {2,-10} {3}" -f "#", "Model", "Quant", "Size")
                    Write-Host ("    {0,-4} {1,-35} {2,-10} {3}" -f "---", "---", "---", "---")
                    for ($i = 0; $i -lt $modelList.Count; $i++) {
                        $m = $modelList[$i]
                        Write-Host ("    [{0}]  {1,-35} {2,-10} {3:F2} GiB" -f ($i+1), $m.Name, $m.Quant, $m.Size)
                    }
                    Write-Host ""

                    $dlChoice = Read-Choice "  Select a model [1]" "1"
                    $dlIdx = [int]$dlChoice - 1
                    if ($dlIdx -lt 0 -or $dlIdx -ge $modelList.Count) { $dlIdx = 0 }
                    $selected = $modelList[$dlIdx]

                    $modelsDir = Join-Path $InstallDir "models"
                    if (-not (Test-Path $modelsDir)) { New-Item -ItemType Directory -Path $modelsDir | Out-Null }
                    $dlFilename = Split-Path $selected.Url -Leaf
                    $ModelPath = Join-Path $modelsDir $dlFilename

                    if (Test-Path $ModelPath) {
                        Write-Ok "Model already downloaded: $dlFilename"
                    } else {
                        Write-Info "Downloading $($selected.Name) ($($selected.Quant), $($selected.Size.ToString('F2')) GiB) ..."
                        Invoke-WebRequest -Uri $selected.Url -OutFile $ModelPath -UseBasicParsing
                        Write-Ok "Download complete: $dlFilename"
                    }
                    $ModelConfigured = $true
                }
            } catch {
                Write-Warn "Could not fetch model catalog: $_"
                Write-Warn "You can configure a model manually later."
            }
        }
        "2" {
            Write-Host ""
            $localPath = Read-Choice "  Enter model path" ""
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

    $modelBasename = Split-Path $ModelPath -Leaf
    Write-Host ""
    Write-Host "============================================================"
    Write-Host "                   Setup Complete!"
    Write-Host "============================================================"
    Write-Host ""
    Write-Host "  Install location:  $InstallDir"
    Write-Host "  Backend:           $SelectedBackend"
    Write-Host "  Model:             $modelBasename"
    Write-Host ""
    Write-Host "  -- Quick Reference --"
    Write-Host ""
    Write-Host "  Start the API server:"
    Write-Host "    cd $InstallDir"
    Write-Host "    .\omniinfer.cmd serve"
    Write-Host ""
    Write-Host "  API endpoint:  http://127.0.0.1:9000"
    Write-Host ""
    Write-Host "  Chat completion (curl):"
    Write-Host '    curl -X POST http://127.0.0.1:9000/v1/chat/completions \'
    Write-Host '      -H "Content-Type: application/json" \'
    Write-Host '      -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}"'
    Write-Host ""
    Write-Host "  CLI chat:"
    Write-Host "    .\omniinfer.cmd chat --message `"Hello`""
    Write-Host ""
    Write-Host "  Stop the service:"
    Write-Host "    .\omniinfer.cmd shutdown"
    Write-Host ""
    Write-Host "============================================================"
    Write-Host ""

    Write-Info "Let's try a quick chat to make sure everything works ..."
    Write-Host ""
    & cmd /c $cmdPath chat --message "Hello! Introduce yourself in one sentence."
    Write-Host ""
    Write-Host ""
    Write-Ok "Everything is working! Enjoy OmniInfer."
    Write-Host ""
    Write-Host "  To start the API server anytime:"
    Write-Host "    cd $InstallDir"
    Write-Host "    .\omniinfer.cmd serve"
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "============================================================"
    Write-Host "                   Install Complete!"
    Write-Host "============================================================"
    Write-Host ""
    Write-Host "  Install location:  $InstallDir"
    Write-Host "  Backend:           $SelectedBackend"
    Write-Host ""
    Write-Host "  -- Next Steps --"
    Write-Host ""
    Write-Host "  1. Load a model:"
    Write-Host "     cd $InstallDir"
    Write-Host "     .\omniinfer.cmd model load -m C:\path\to\model.gguf"
    Write-Host ""
    Write-Host "  2. Start the API server:"
    Write-Host "     .\omniinfer.cmd serve"
    Write-Host ""
    Write-Host "  3. Use the API (http://127.0.0.1:9000):"
    Write-Host ""
    Write-Host "     Health check:"
    Write-Host "       curl http://127.0.0.1:9000/health"
    Write-Host ""
    Write-Host "     Chat completion:"
    Write-Host '     curl -X POST http://127.0.0.1:9000/v1/chat/completions \'
    Write-Host '       -H "Content-Type: application/json" \'
    Write-Host '       -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}"'
    Write-Host ""
    Write-Host "     CLI chat:"
    Write-Host "       .\omniinfer.cmd chat --message `"Hello`""
    Write-Host ""
    Write-Host "  4. Stop the service:"
    Write-Host "     .\omniinfer.cmd shutdown"
    Write-Host ""
    Write-Host "  -- Other Useful Commands --"
    Write-Host ""
    Write-Host "  List backends:   .\omniinfer.cmd backend list"
    Write-Host "  Switch backend:  .\omniinfer.cmd select <backend>"
    Write-Host "  Check status:    .\omniinfer.cmd status"
    Write-Host "  Browse models:   .\omniinfer.cmd model list"
    Write-Host ""
    Write-Host "============================================================"
    Write-Host ""
}
