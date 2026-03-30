@echo off
setlocal

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
