#!/bin/sh
# 百格附着力图像分析 — macOS 双击启动（Finder 里双击本文件）
cd "$(dirname "$0")" || exit 1
exec ./start.sh "$@"
