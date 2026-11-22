#!/bin/sh
# 百格附着力图像分析 — Linux / macOS 双击启动
# 文件管理器里双击若只打开编辑器：右键 → 属性 → 允许作为程序执行；
# 或命令行：./start.sh
cd "$(dirname "$0")" || exit 1
if command -v python3 >/dev/null 2>&1; then
  exec python3 launcher.py "$@"
fi
if command -v python >/dev/null 2>&1; then
  exec python launcher.py "$@"
fi
echo "[错误] 没找到 python3。Debian/Ubuntu: sudo apt install python3 python3-venv"
printf '按回车退出…'
read _
exit 1
