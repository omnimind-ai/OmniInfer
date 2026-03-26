# OmniInfer Windows Portable Release

This directory contains the files needed to build a portable Windows release:

- `OmniInfer.exe`
- `config/omniinfer.json`
- bundled backend files
- no bundled model weights

Build the portable package from the repository root with:

```powershell
powershell -ExecutionPolicy Bypass -File .\release\build_portable.ps1
```

The built package will be placed under:

```text
release\portable\OmniInfer
```

Run the packaged service with:

```powershell
.\release\portable\OmniInfer\OmniInfer.exe
```

Model discovery is driven by the caller-provided `models_path` query parameter on `GET /omni/models`.
Portable builds do not ship with a default `runtime\models` directory.
