# Installing OmniInfer

Backend installation accepts the names printed by `omniinfer backend list`, such as `llama.cpp-cpu`. See [backend names](backend-names.md) for legacy compatibility and runtime IDs used by packaging scripts.

OmniInfer separates the CLI installation from backend runtimes and models. The default installers download the official CLI-only GitHub Release, verify its SHA-256 checksum, and install it without elevated privileges.

## Install the Release CLI

Linux x64 and macOS arm64:

```bash
curl -fsSL https://raw.githubusercontent.com/omnimind-ai/OmniInfer/main/scripts/install.sh | bash
```

Windows x64 PowerShell:

```powershell
irm https://raw.githubusercontent.com/omnimind-ai/OmniInfer/main/scripts/install.ps1 | iex
```

The Unix installer writes to `~/.local/bin` by default. The Windows installer writes to `%LOCALAPPDATA%\Programs\OmniInfer\bin` and adds that directory to the user PATH. Neither installer clones the repository, installs a backend, downloads a model, or requests sudo/administrator access.

After installation:

```bash
omniinfer --version
omniinfer backend list
omniinfer backend install <backend>
```

## Pin a Release or Change the Install Directory

Linux or macOS:

```bash
curl -fsSL https://raw.githubusercontent.com/omnimind-ai/OmniInfer/main/scripts/install.sh | \
  bash -s -- --version v0.3.24 --install-dir "$HOME/.local/bin"
```

Windows PowerShell:

```powershell
$installer = Join-Path $env:TEMP "install-omniinfer.ps1"
irm https://raw.githubusercontent.com/omnimind-ai/OmniInfer/main/scripts/install.ps1 -OutFile $installer
& $installer -Version v0.3.24 -InstallDir "$env:LOCALAPPDATA\Programs\OmniInfer\bin"
```

Use `-NoPathUpdate` when a Windows deployment tool manages PATH separately.

## Complete Source Setup

The source installers clone or update the full repository, build the Rust CLI, select a backend, install a configured prebuilt runtime or build it from source, and optionally configure a model.

Linux and macOS:

```bash
curl -fsSL https://raw.githubusercontent.com/omnimind-ai/OmniInfer/main/scripts/install-from-source.sh | bash
```

Windows PowerShell:

```powershell
irm "https://raw.githubusercontent.com/omnimind-ai/OmniInfer/main/scripts/install-from-source.ps1?$(Get-Random)" | iex
```

Source setup requires Git, Rust/Cargo, Python, a C/C++ toolchain, and any dependencies required by the selected source-built backend. Prebuilt backend installation does not require CMake unless the selected backend build path needs it.

If the repository is already cloned, use the platform and backend commands in the [Build Guide](build.md) instead of running the complete source installer again.

## Manual Installation

Official archives and `checksums.txt` are published on [GitHub Releases](https://github.com/omnimind-ai/OmniInfer/releases). Verify the archive against `checksums.txt`, extract it, and place its executable launchers in a directory on PATH.

## Remove the Release CLI

Linux and macOS:

```bash
rm "$HOME/.local/bin/omniinfer"
```

Windows PowerShell:

```powershell
$bin = "$env:LOCALAPPDATA\Programs\OmniInfer\bin"
Remove-Item "$bin\omniinfer.exe", "$bin\omniinfer.cmd", "$bin\omniinfer.ps1"
```

Removing the CLI does not remove separately installed backend runtimes, models, or state.
