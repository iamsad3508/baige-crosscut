@echo off
rem 百格附着力图像分析 — Windows 双击启动
rem 需要 Python 3.9+（安装时勾选 Add python.exe to PATH）
chcp 65001 >nul
cd /d "%~dp0"
title 百格附着力图像分析

set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY (
  where python >nul 2>nul && set "PY=python"
)
if not defined PY (
  echo.
  echo [错误] 没找到 Python。
  echo        请到 https://www.python.org/downloads/ 安装 3.9 或更高版本，
  echo        安装时务必勾选 "Add python.exe to PATH"，然后重新双击本文件。
  echo.
  pause
  exit /b 1
)

%PY% launcher.py %*
if errorlevel 1 (
  echo.
  echo 启动失败，请把上面的信息截图反馈。
  pause
)
