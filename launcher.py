#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""百格分析 · 一键启动器（双击 start.bat / start.sh 就会跑这个文件）

做的事，按顺序：
  1. 检查 Python 版本；
  2. 在程序目录建虚拟环境 .venv（已存在就复用，依赖清单变过会补装）；
  3. 装 requirements.txt 里的依赖（先走默认源，失败自动换国内镜像再试一次）；
  4. 起本地服务（默认 127.0.0.1:8765，端口被占就往后找一个空的）；
  5. 等服务真的起来了，自动打开浏览器；
  6. Ctrl+C 或关窗口 = 停服务。

只依赖 Python 标准库，所以「刚下载下来、还没装任何东西」也能跑起来。
"""

import argparse
import hashlib
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
REQ = ROOT / "requirements.txt"
MARK = VENV / ".baige_deps"          # 记「当前 .venv 是按哪份 requirements 装的」
MIN_PY = (3, 9)
MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"


def say(msg=""):
    print(msg, flush=True)


def die(msg):
    say("\n[错误] " + msg)
    say("\n按回车退出…")
    try:
        input()
    except Exception:
        pass
    sys.exit(1)


def venv_python():
    """虚拟环境里的解释器路径（Windows 在 Scripts，其它在 bin）。"""
    if os.name == "nt":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def req_hash():
    try:
        data = REQ.read_bytes()
    except OSError:
        return "no-requirements"
    return hashlib.sha256(data).hexdigest()[:16]


def check_python():
    if sys.version_info < MIN_PY:
        die("需要 Python %d.%d 或更高版本，当前是 %d.%d。\n"
            "      Windows：到 python.org 下载安装（安装时勾选 Add python.exe to PATH）\n"
            "      Linux：sudo apt install python3 python3-venv"
            % (MIN_PY[0], MIN_PY[1], sys.version_info[0], sys.version_info[1]))


def ensure_venv(reset=False):
    """建/复用虚拟环境，依赖不全就装。返回虚拟环境里的 python 路径。"""
    py = venv_python()
    if reset and VENV.exists():
        say("· 按要求重建虚拟环境…")
        shutil.rmtree(VENV, ignore_errors=True)
    if not py.exists():
        say("· 第一次运行：创建虚拟环境 .venv（约 10 秒）…")
        cmd = [sys.executable, "-m", "venv", str(VENV)]
        r = subprocess.run(cmd)
        if r.returncode != 0 or not py.exists():
            die("创建虚拟环境失败。\n"
                "      Linux 若提示 ensurepip 缺失，先装：sudo apt install python3-venv")
    want = req_hash()
    have = ""
    try:
        have = MARK.read_text(encoding="utf-8").strip()
    except OSError:
        pass
    if have == want:
        say("· 依赖已就绪（%s）" % want)
        return py
    say("· 安装依赖（第一次约 1~3 分钟，看网速）…")
    base = [str(py), "-m", "pip", "install", "--disable-pip-version-check",
            "-r", str(REQ)]
    r = subprocess.run(base)
    if r.returncode != 0:
        say("· 默认源装不上，换国内镜像重试…")
        r = subprocess.run(base + ["-i", MIRROR])
    if r.returncode != 0:
        die("依赖安装失败。\n"
            "      1) 确认能上网；2) 或在命令行里手动跑：\n"
            "         %s -m pip install -r requirements.txt -i %s" % (py, MIRROR))
    try:
        MARK.write_text(want, encoding="utf-8")
    except OSError:
        pass
    return py


def free_port(preferred):
    """从 preferred 开始找一个没被占的端口。"""
    for p in range(preferred, preferred + 30):
        with socket.socket() as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", p))
            except OSError:
                continue
            return p
    die("从 %d 往后 30 个端口都被占了，请用 --port 指定一个空闲端口。" % preferred)


def wait_health(url, timeout=180):
    """等服务真正能应答（首次 import numpy 要几秒）。"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url + "api/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, OSError, ValueError):
            pass
        time.sleep(0.5)
    return False


def main():
    ap = argparse.ArgumentParser(description="百格附着力图像分析 · 本地启动器")
    ap.add_argument("--port", type=int, default=8765, help="监听端口（默认 8765）")
    ap.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    ap.add_argument("--reset", action="store_true", help="重建虚拟环境（依赖坏了用）")
    args = ap.parse_args()

    say("=" * 58)
    say(" 百格附着力图像分析 — 本地版（用 DeepSeek API 开发）")
    say("=" * 58)
    check_python()
    py = ensure_venv(reset=args.reset)
    port = free_port(args.port)
    url = "http://127.0.0.1:%d/" % port
    if port != args.port:
        say("· 端口 %d 被占用，改用 %d" % (args.port, port))

    say("· 启动服务…")
    env = dict(os.environ)
    env["PYTHONPATH"] = ""                      # 别让外部 site-packages 漏进来
    env.setdefault("PYTHONUTF8", "1")
    cmd = [str(py), "-m", "uvicorn", "server:app",
           "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"]
    proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env)
    try:
        if not wait_health(url):
            say("[错误] 服务 180 秒内没起来，下面是可能的输出，按回车退出…")
            try:
                proc.wait(timeout=1)
            except Exception:
                pass
            try:
                input()
            except Exception:
                pass
            return 1
        say("· 服务已就绪：%s" % url)
        if not args.no_browser:
            try:
                webbrowser.open(url)
            except Exception:
                say("  （浏览器没能自动打开，请手动访问上面的地址）")
        say("")
        say("  用法：打开图片 → 吸两个颜色（漆膜/露底）→ 框出划格区 → 计算")
        say("  关掉这个窗口或按 Ctrl+C 即停止服务。数据都在本目录 data/ 里。")
        say("")
        proc.wait()
    except KeyboardInterrupt:
        say("\n· 正在停止…")
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
