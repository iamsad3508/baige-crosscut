#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""百格分析 · 一键启动器 / One-click launcher（双击 start.bat / start.sh 就会跑这个文件）

做的事 / what it does:
  1. 检查 Python 版本 / check Python version
  2. 建/复用虚拟环境 .venv / create or reuse a local venv
  3. 装 requirements.txt 的依赖（失败换镜像）/ install deps (mirror fallback)
  4. 起本地服务（默认 127.0.0.1:8765，被占就往后找）/ start the local server
  5. 等服务就绪后打开浏览器 / open the browser once it answers
  6. Ctrl+C 或关窗口 = 停服务 / closing the window stops it

语言 / language: --lang zh|en，或环境变量 BAIGE_LANG，否则跟随系统。
只依赖标准库 / standard library only.
"""

import argparse
import hashlib
import locale
import os
import shutil
import signal
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

MSG = {
    "zh": {
        "title": "百格附着力图像分析 — 本地版（用 DeepSeek API 开发）",
        "py_old": "需要 Python %d.%d 或更高版本，当前是 %d.%d。\n"
                  "      Windows：到 python.org 下载安装（安装时勾选 Add python.exe to PATH）\n"
                  "      Linux：sudo apt install python3 python3-venv",
        "recreate": "按要求重建虚拟环境…",
        "creating": "第一次运行：创建虚拟环境 .venv（约 10 秒）…",
        "venv_fail": "创建虚拟环境失败。\n      Linux 若提示 ensurepip 缺失，先装：sudo apt install python3-venv",
        "deps_ok": "依赖已就绪（%s）",
        "installing": "安装依赖（第一次约 1~3 分钟，看网速）…",
        "mirror": "默认源装不上，换国内镜像重试…",
        "deps_fail": "依赖安装失败。\n      1) 确认能上网；2) 或手动跑：\n         %s -m pip install -r requirements.txt -i %s",
        "ports_taken": "从 %d 往后 30 个端口都被占了，请用 --port 指定一个空闲端口。",
        "port_switch": "端口 %d 被占用，改用 %d",
        "starting": "启动服务…",
        "ready": "服务已就绪：%s",
        "no_browser": "（浏览器没能自动打开，请手动访问上面的地址）",
        "usage": "用法：打开图片 → 吸两个颜色（漆膜/露底）→ 框出划格区 → 计算",
        "stop_hint": "关掉这个窗口或按 Ctrl+C 即停止服务。数据都在本目录 data/ 里。",
        "timeout": "服务 180 秒内没起来，下面是可能的输出，按回车退出…",
        "stopping": "正在停止…",
        "press_enter": "按回车退出…",
        "err_prefix": "[错误] ",
    },
    "en": {
        "title": "Baige Cross-Cut Analyzer — local build (developed with the DeepSeek API)",
        "py_old": "Python %d.%d or newer is required, found %d.%d.\n"
                  "      Windows: install from python.org and tick 'Add python.exe to PATH'\n"
                  "      Linux:   sudo apt install python3 python3-venv",
        "recreate": "recreating the virtual environment as requested…",
        "creating": "first run: creating virtual environment .venv (about 10 s)…",
        "venv_fail": "could not create the virtual environment.\n"
                     "      On Linux, if ensurepip is missing: sudo apt install python3-venv",
        "deps_ok": "dependencies already in place (%s)",
        "installing": "installing dependencies (first run: 1-3 minutes)…",
        "mirror": "default index failed, retrying with a mirror…",
        "deps_fail": "dependency installation failed.\n"
                     "      1) check your network; 2) or run manually:\n"
                     "         %s -m pip install -r requirements.txt -i %s",
        "ports_taken": "ports %d and the next 29 are all busy; pick one with --port.",
        "port_switch": "port %d is busy, using %d instead",
        "starting": "starting the server…",
        "ready": "server ready: %s",
        "no_browser": "(could not open a browser automatically — open the URL above)",
        "usage": "usage: open image -> pick two colours (coating / substrate) -> frame the cross-hatch area -> calculate",
        "stop_hint": "close this window or press Ctrl+C to stop. Data stays in data/ next to the code.",
        "timeout": "the server did not answer within 180 s; see output above, press Enter to exit…",
        "stopping": "stopping…",
        "press_enter": "press Enter to exit…",
        "err_prefix": "[error] ",
    },
}

LANG = "zh"


def T(key, *args):
    table = MSG.get(LANG) or MSG["zh"]
    s = table.get(key, MSG["zh"].get(key, key))
    return s % args if args else s


def say(msg=""):
    print(msg, flush=True)


def detect_lang():
    """--lang > BAIGE_LANG > 系统语言；认不出就中文（原作者环境）。"""
    for k in ("BAIGE_LANG", "LANGUAGE", "LC_ALL", "LC_MESSAGES", "LANG"):
        v = (os.environ.get(k) or "").lower()
        if v.startswith("zh"):
            return "zh"
        if v.startswith("en"):
            return "en"
    if os.name == "nt":
        try:
            import ctypes
            lid = ctypes.windll.kernel32.GetUserDefaultUILanguage()
            return "zh" if (lid & 0x3FF) == 0x04 else "en"
        except Exception:
            pass
    try:
        name = (locale.getlocale()[0] or "").lower()
        if name.startswith("zh"):
            return "zh"
        if name.startswith("en"):
            return "en"
    except Exception:
        pass
    return "zh"


def die(msg):
    say("\n" + T("err_prefix") + msg)
    say("\n" + T("press_enter"))
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
        die(T("py_old", MIN_PY[0], MIN_PY[1], sys.version_info[0], sys.version_info[1]))


def ensure_venv(reset=False):
    """建/复用虚拟环境，依赖不全就装。返回虚拟环境里的 python 路径。"""
    py = venv_python()
    if reset and VENV.exists():
        say("· " + T("recreate"))
        shutil.rmtree(VENV, ignore_errors=True)
    if not py.exists():
        say("· " + T("creating"))
        r = subprocess.run([sys.executable, "-m", "venv", str(VENV)])
        if r.returncode != 0 or not py.exists():
            die(T("venv_fail"))
    want = req_hash()
    have = ""
    try:
        have = MARK.read_text(encoding="utf-8").strip()
    except OSError:
        pass
    if have == want:
        say("· " + T("deps_ok", want))
        return py
    say("· " + T("installing"))
    base = [str(py), "-m", "pip", "install", "--disable-pip-version-check",
            "-r", str(REQ)]
    r = subprocess.run(base)
    if r.returncode != 0:
        say("· " + T("mirror"))
        r = subprocess.run(base + ["-i", MIRROR])
    if r.returncode != 0:
        die(T("deps_fail", py, MIRROR))
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
    die(T("ports_taken", preferred))


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


def _install_signal_handlers():
    """SIGTERM/SIGHUP/SIGBREAK 一律转成 KeyboardInterrupt，这样 finally 能关掉子进程。"""
    def _raise(signum, frame):
        raise KeyboardInterrupt
    for name in ("SIGTERM", "SIGHUP", "SIGBREAK"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, _raise)
        except Exception:
            pass


def main():
    global LANG
    ap = argparse.ArgumentParser(description="百格附着力图像分析 · 本地启动器 / local launcher")
    ap.add_argument("--port", type=int, default=8765, help="监听端口 / port (default 8765)")
    ap.add_argument("--no-browser", action="store_true", help="不自动开浏览器 / no browser")
    ap.add_argument("--reset", action="store_true", help="重建虚拟环境 / recreate the venv")
    ap.add_argument("--lang", choices=("zh", "en"), help="界面语言 / interface language")
    args = ap.parse_args()

    LANG = args.lang or detect_lang()
    try:                                  # Windows 管道/重定向下中文不炸
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    _install_signal_handlers()            # 被 kill/关窗口时也要收掉 uvicorn 子进程

    say("=" * 62)
    say(" " + T("title"))
    say("=" * 62)
    check_python()
    py = ensure_venv(reset=args.reset)
    port = free_port(args.port)
    url = "http://127.0.0.1:%d/" % port
    if port != args.port:
        say("· " + T("port_switch", args.port, port))

    say("· " + T("starting"))
    env = dict(os.environ)
    env["PYTHONPATH"] = ""                      # 别让外部 site-packages 漏进来
    env.setdefault("PYTHONUTF8", "1")
    env["BAIGE_LANG"] = LANG
    cmd = [str(py), "-m", "uvicorn", "server:app",
           "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"]
    proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env)
    try:
        if not wait_health(url):
            say(T("timeout"))
            try:
                proc.wait(timeout=1)
            except Exception:
                pass
            try:
                input()
            except Exception:
                pass
            return 1
        say("· " + T("ready", url))
        if not args.no_browser:
            try:
                webbrowser.open(url)
            except Exception:
                say("  " + T("no_browser"))
        say("")
        say("  " + T("usage"))
        say("  " + T("stop_hint"))
        say("")
        proc.wait()
    except KeyboardInterrupt:
        say("\n· " + T("stopping"))
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
