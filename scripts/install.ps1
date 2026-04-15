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
    try { git -C $InstallDir pull --ff-only 2>&1 | Out-Null } catch { Write-Warn "Pull failed, continuing with existing code" }
} else {
    Write-Info "Cloning OmniInfer to $InstallDir ..."
    git clone --depth 1 $RepoUrl $InstallDir
    if ($LASTEXITCODE -ne 0) {
        Stop-Fatal "git clone failed (exit code $LASTEXITCODE). Check your network connection and try again."
    }
}
if (-not (Test-Path (Join-Path $InstallDir "omniinfer.py"))) {
    Stop-Fatal "Repository clone appears incomplete — omniinfer.py not found in $InstallDir"
}
Write-Ok "Repository ready at $InstallDir"

# ── Ensure a usable port ────────────────────────────────────

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
    Write-Warn "Port $OmniPort is in use, looking for a free port ..."
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
    & python -c "import json; json.dump({'host':'127.0.0.1','port':$OmniPort}, open(r'$configFile','w',encoding='utf-8'), indent=2)"
    Write-Ok "Config written: $configFile (port $OmniPort)"
}
Write-Host ""

# ── Step 3: Detect platform & choose backend ────────────────
# Get available backends from CLI

Write-Info "Step 3/6: Detecting platform and hardware ..."

$cmdPath = Join-Path $InstallDir "omniinfer.cmd"
$pyScript = Join-Path $InstallDir "omniinfer.py"

# Helper: invoke omniinfer CLI via python directly (avoids cmd /c path issues)
function Invoke-OmniInfer {
    & python $pyScript @args
}

# Cleanup: shut down any gateway service started by the CLI on script exit
$_gatewayPort = 9000
try { $_gatewayPort = [int](Get-Content (Join-Path $InstallDir "release\config\omniinfer.json") | ConvertFrom-Json).port } catch {}
Register-EngineEvent PowerShell.Exiting -Action {
    try { Invoke-RestMethod -Uri "http://127.0.0.1:$($_gatewayPort)/omni/shutdown" -Method POST -TimeoutSec 3 2>$null } catch {}
} | Out-Null

$BackendIds   = @()
$BackendDescs = @()
# Query the gateway API for compatible backends (hardware-matched)
$_backendsJson = $null
try {
    Invoke-OmniInfer status 2>$null | Out-Null  # ensure gateway is running
    $_backendsJson = Invoke-RestMethod -Uri "http://127.0.0.1:$_gatewayPort/omni/backends?scope=compatible" -TimeoutSec 15 -ErrorAction Stop
} catch {}

if ($_backendsJson -and $_backendsJson.data) {
    foreach ($b in $_backendsJson.data) {
        $BackendIds   += $b.id
        $desc = $b.description
        if ($desc) { $BackendDescs += "$($b.id)  -  $desc" } else { $BackendDescs += $b.id }
    }
    if ($_backendsJson.recommended) {
        # Move recommended backend to the top of the list
        $recIdx = [Array]::IndexOf($BackendIds, $_backendsJson.recommended)
        if ($recIdx -gt 0) {
            $recId   = $BackendIds[$recIdx];   $recDesc = $BackendDescs[$recIdx]
            $BackendIds   = @($recId)   + ($BackendIds   | Where-Object { $_ -ne $recId })
            $BackendDescs = @("$recDesc  (recommended)") + ($BackendDescs | Where-Object { $_ -ne $recDesc })
        } elseif ($recIdx -eq 0) {
            $BackendDescs[0] = "$($BackendDescs[0])  (recommended)"
        }
    }
} else {
    # Fallback: parse CLI text output
    $rawOutput = Invoke-OmniInfer backend list 2>$null
    $currentId = ""
    foreach ($line in $rawOutput -split "`n") {
        $line = $line.Trim()
        if ($line -match '^[*\s]*([a-zA-Z0-9._-]+)$') {
            $currentId = $Matches[1]
            $BackendIds += $currentId
            $BackendDescs += $currentId
        }
        if ($line -match '^Description:\s*(.+)$' -and $currentId) {
            $BackendDescs[$BackendDescs.Count - 1] = "$currentId  -  $($Matches[1])"
        }
    }
}

if ($BackendIds.Count -eq 0) {
    Stop-Fatal "No backends found. Check your platform support."
}

Write-Info "Platform: Windows"
Write-Host ""

if ($Backend) {
    $SelectedBackend = $Backend
} else {
    Write-Host "  Available backends (arrow keys to move, Enter to select):"
    Write-Host ""

    $idx = Select-Menu -Default 0 -Options $BackendDescs
    $SelectedBackend = $BackendIds[$idx]
}

Write-Ok "Selected: $SelectedBackend"
Write-Host ""

# Select backend via CLI
Invoke-OmniInfer select $SelectedBackend 2>$null

# ── Step 4: Build backend ───────────────────────────────────
# Windows build scripts do NOT auto-bootstrap submodules.

Write-Info "Step 4/6: Building backend ..."

$llamaCppDir = Join-Path $InstallDir "framework\llama.cpp"
if (-not (Test-Path (Join-Path $llamaCppDir "CMakeLists.txt"))) {
    Write-Info "Initializing llama.cpp submodule ..."
    git -C $InstallDir submodule update --init --recursive --depth 1 --progress framework/llama.cpp
    Write-Ok "Submodule ready"
    Write-Host ""
}

# Check for a C/C++ toolchain before attempting to build
$hasMsvc = [bool](Get-Command cl.exe -ErrorAction SilentlyContinue)
$hasMsys2Gcc = [bool](Get-Command gcc.exe -ErrorAction SilentlyContinue)
if (-not $hasMsvc -and -not $hasMsys2Gcc) {
    # Try registry detection for MSYS2 (not in PATH but installed)
    $msys2Found = $false
    foreach ($key in @("HKLM:\SOFTWARE\MSYS2","HKCU:\SOFTWARE\MSYS2","HKLM:\SOFTWARE\WOW6432Node\MSYS2")) {
        try { if ((Get-ItemProperty -Path $key -ErrorAction SilentlyContinue).InstallLocation) { $msys2Found = $true; break } } catch {}
    }
    if ($msys2Found) {
        Write-Warn "MSYS2 is installed but its ucrt64\bin is not in PATH."
        Write-Host "  Add it to PATH and retry, e.g.:"
        Write-Host '    $env:PATH = "C:\msys64\ucrt64\bin;$env:PATH"'
        Write-Host ""
    }
    Stop-Fatal "No C/C++ compiler found. Install one of:`n  - Visual Studio Build Tools (cl.exe): https://visualstudio.microsoft.com/downloads/#build-tools`n  - MSYS2 with ucrt64 toolchain (gcc.exe): https://www.msys2.org/`n    Then: pacman -S mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-ninja mingw-w64-ucrt-x86_64-cmake"
} elseif ($hasMsvc) {
    Write-Ok "C++ toolchain: MSVC (cl.exe)"
} else {
    Write-Ok "C++ toolchain: MSYS2 (gcc.exe)"
}

# Discover build script by convention: scripts/platforms/windows/<backend_id>/build.ps1
$fullScript = Join-Path $InstallDir "scripts\platforms\windows\$SelectedBackend\build.ps1"
if (-not (Test-Path $fullScript)) {
    Stop-Fatal "Build script not found: $fullScript"
}

if ($SkipBuild) {
    Write-Info "Skipping build (-SkipBuild)"
} else {
    $runtimeCheck = Invoke-OmniInfer backend list 2>$null
    $isBuilt = ($runtimeCheck -join "`n") -match "$([regex]::Escape($SelectedBackend))[\s\S]*?Runtime available: yes"
    if ($isBuilt) {
        Write-Ok "Backend $SelectedBackend already built, skipping"
    } else {
        Write-Info "Building $SelectedBackend (this may take a few minutes) ..."
        powershell -NoProfile -ExecutionPolicy Bypass -File $fullScript
        # Verify build produced the expected binary
        $binDir = Join-Path $InstallDir ".local\runtime\windows\$SelectedBackend\bin"
        if (-not (Test-Path $binDir) -or (Get-ChildItem $binDir -File -ErrorAction SilentlyContinue).Count -eq 0) {
            Write-Err "Build failed — no binaries produced."
            Write-Host ""
            Write-Host "  Common fix: run from a Visual Studio Developer PowerShell"
            Write-Host "  so that cl.exe, link.exe, and rc.exe are in PATH."
            Write-Host ""
            exit 1
        }
        Write-Ok "Build complete"
    }
}
Write-Host ""

# ── Step 5: Model configuration ─────────────────────────────

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

# ── Step 6: Load model & finish ──────────────────────────────

Write-Info "Step 6/6: Finishing up ..."
Write-Host ""

# ── Finish message (reused by both paths) ──────────────────
function Print-Finish {
    Write-Host ""
    Invoke-OmniInfer shutdown 2>$null

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
        Write-Host "    .\omniinfer.cmd model load -m $ModelPath"
        Write-Host "    .\omniinfer.cmd chat --message `"Hello`""
    } else {
        Write-Host ""
        Write-Host "  To start chatting, load a model first:"
        Write-Host ""
        Write-Host "    cd $InstallDir"
        Write-Host "    .\omniinfer.cmd model load -m C:\path\to\model.gguf"
        Write-Host "    .\omniinfer.cmd chat --message `"Hello`""
    }
    Write-Host ""
    Write-Host "  The model needs to be loaded each time after a restart."
    Write-Host "  The CLI auto-starts the service if needed."
    Write-Host ""
    Write-Host "  Other useful commands:"
    Write-Host "    .\omniinfer.cmd backend list          # list available backends"
    Write-Host "    .\omniinfer.cmd select <backend>      # switch backend"
    Write-Host "    .\omniinfer.cmd model list            # browse supported models"
    Write-Host "    .\omniinfer.cmd status                # check current state"
    Write-Host "    .\omniinfer.cmd serve                 # start API server (http://127.0.0.1:9000)"
    Write-Host "    .\omniinfer.cmd shutdown              # stop the service"
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
        Write-Host ""
        Write-Host "  Try building the backend first, then re-run:"
        Write-Host "    cd $InstallDir"
        Write-Host "    .\omniinfer.cmd model load -m $ModelPath"
        Write-Host ""
        exit 1
    }
    Write-Ok "Model loaded"
    Write-Host ""

    # ── Interactive chat loop ──────────────────────────────
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
