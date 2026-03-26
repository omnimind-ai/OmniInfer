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

Portable builds do not ship with a default `runtime\models` directory.
`GET /v1/models` is currently not maintained in OmniInfer.
Use `GET /omni/supported-models?system=windows` or `GET /omni/supported-models?system=mac`
to fetch the latest supported-model catalog from the remote source.
