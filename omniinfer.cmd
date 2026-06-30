@echo off
setlocal

set "RUST_CLI=%~dp0target\debug\omniinfer-rs.exe"

if not exist "%RUST_CLI%" goto missing_rust
"%RUST_CLI%" %*
exit /b %errorlevel%

:missing_rust
echo Rust OmniInfer CLI was not found at %RUST_CLI%. Run: cargo build -p omniinfer-cli
exit /b 1
