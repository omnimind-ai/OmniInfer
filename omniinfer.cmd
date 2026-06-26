@echo off
setlocal

set "RUST_CLI=%~dp0target\debug\omniinfer-rs.exe"
if /I "%OMNIINFER_FORCE_PYTHON%"=="1" goto python_fallback
if /I "%OMNIINFER_FORCE_PYTHON%"=="true" goto python_fallback
if /I "%OMNIINFER_FORCE_PYTHON%"=="yes" goto python_fallback
if /I "%OMNIINFER_FORCE_PYTHON%"=="on" goto python_fallback

if exist "%RUST_CLI%" (
    "%RUST_CLI%" %*
    exit /b %errorlevel%
)

echo Rust OmniInfer CLI was not found at %RUST_CLI%. Run: cargo build -p omniinfer-cli
echo To use the Python fallback now, set OMNIINFER_FORCE_PYTHON=1.
exit /b 1

:python_fallback
where py >nul 2>nul
if %errorlevel%==0 (
    py -3 "%~dp0omniinfer.py" %*
    exit /b %errorlevel%
)

where python >nul 2>nul
if %errorlevel%==0 (
    python "%~dp0omniinfer.py" %*
    exit /b %errorlevel%
)

echo Python 3 was not found in PATH.
exit /b 1
