@echo off
rem Baige cross-cut analyzer - Windows launcher (CRLF, ASCII only - do not re-save as UTF-8/LF)
rem Requires Python 3.9+ ; when installing tick "Add python.exe to PATH".
setlocal
cd /d "%~dp0"
title Baige Cross-Cut Analyzer

set "BAIGE_PY="

py -3 -c "import sys" >nul 2>nul
if not errorlevel 1 set "BAIGE_PY=py -3"
if defined BAIGE_PY goto found

python -c "import sys" >nul 2>nul
if not errorlevel 1 set "BAIGE_PY=python"
if defined BAIGE_PY goto found

python3 -c "import sys" >nul 2>nul
if not errorlevel 1 set "BAIGE_PY=python3"
if defined BAIGE_PY goto found

echo.
echo [ERROR] Python 3.9+ was not found on this computer.
echo         Install it from https://www.python.org/downloads/
echo         Tick "Add python.exe to PATH" during setup, then double-click start.bat again.
echo         (ZH) Qing xian an zhuang Python 3.9+, an zhuang shi gou xuan Add python.exe to PATH.
echo.
pause
exit /b 1

:found
chcp 65001 >nul 2>nul
%BAIGE_PY% "%~dp0launcher.py" %*
if errorlevel 1 pause
