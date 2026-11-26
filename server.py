#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2026 iamsad3508
# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of Baige (百格附着力分析). Commercial licence: see COMMERCIAL.md
"""百格附着力分析 —— Web 版服务端

算法部分完全复用 crosscut.py 里的纯函数（classify_parts / labels_from_parts /
roi_stats / split_lines / measure_line_width / astm_class / iso_class / rot_rect_mask），
一行没改；本文件只负责：上传、按分辨率缓存分析图、把 GUI 里的 compute()/auto_detect()
/measure_groove() 那条链复刻成 HTTP 接口，以及导出 CSV / 标注图。

启动：uvicorn server:app --host 127.0.0.1 --port 8765 --workers 1
"""
import ctypes
import hashlib
import io
import json
import math
import os
import secrets
import shutil
import sys
import threading
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
from starlette.concurrency import run_in_threadpool

import crosscut as cc
import gridfit as gf
import srvi18n
from srvi18n import L, is_en


def clamp(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
UPLOADS = os.path.join(DATA, "uploads")
OVERLAYS = os.path.join(DATA, "overlays")
os.makedirs(UPLOADS, exist_ok=True)
os.makedirs(OVERLAYS, exist_ok=True)

MAX_UPLOAD_MB = 60
MAX_PIXELS = 40_000_000
MAX_ANALYSIS_PIXELS = 6_000_000   # 分析图最多这么多像素（≈3000×2000）。实测 3.5 MP → 6 MP 时
                                  # “厚于 7px 的成片脱落”会被少算 15~20%，6 MP 之后基本不变
WORK_MAX_DEFAULT = 1800
DEFAULT_RES = os.environ.get("BAIGE_RES", "full")   # 默认分析分辨率：full = 原图像素（受 3.5 MP 上限）
MAX_SESSIONS = 3        # 最多同时留 3 张图的分析缓存（够 2-3 人共用）
KEEP_HOURS = 24
# 「选区原生像素」模式的像素预算：只裁选区，所以正常只有 0.2~1 MP；
# 框到整张 12 MP 图时才会触发降采样，避免一张图吃光内存。
MAX_ROI_PIXELS = 8_000_000

# 算法版本：每次改动会影响判级数值的代码（吸色、容差、刀缝、判级阈值）就 +1。
# 记录里存的是当时的版本，重开时能看出"数字为什么和存档不一样"。
APP_VERSION = "2.50"      # 软件版本（发版时改）
ALGO_VERSION = "2.15"      # 算法版本（改判级数值就 +1）

app = FastAPI(title="百格分析", docs_url=None, redoc_url=None, openapi_url=None)


@app.middleware("http")
async def _lang_middleware(request, call_next):
    """按请求头定服务端文案语言（X-Baige-Lang: zh|en）。

    注意：call_next 绝不能放进 try —— 那会把业务异常吞成别的状态码（v2.02 踩过）。
    """
    srvi18n.set_lang(request.headers.get("x-baige-lang")
                     or request.headers.get("accept-language"))
    return await call_next(request)


# ---------------------------------------------------------------- 基础工具


def _jpeg(im: Image.Image, q: int = 92) -> bytes:
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=q, subsampling=0)
    return buf.getvalue()


def _png(im: Image.Image) -> bytes:
    buf = io.BytesIO()
    im.save(buf, "PNG", compress_level=3)   # 6 MP 时比默认 6 快 ~4 倍，体积只大 ~10%
    return buf.getvalue()


def _res_to_side_sz(res, size) -> int:
    s = str(res or "").strip()
    if s in ("full", "全分辨率", "0"):
        side = max(size)
    else:
        try:
            v = int(float(s))
        except Exception:
            v = WORK_MAX_DEFAULT
        side = max(200, v)
    w, h = size
    if w * h > MAX_ANALYSIS_PIXELS:      # 按像素预算夹一下，别把内存吃穿
        side = min(side, int(math.sqrt(MAX_ANALYSIS_PIXELS * max(w, h) / float(min(w, h)))))
    return max(200, min(side, max(w, h)))


def _res_to_side(res, full: Image.Image) -> int:
    return _res_to_side_sz(res, full.size)


def _iso_class(pct):
    """ISO 2409 / GB/T 9286 分级（Gt0~Gt5）。

    crosscut.py 里的 iso_class 把 "≤5%" 归成 Gt0，与标准差一档：标准的 Gt1 才是
    "脱落 ≤5%"，Gt0 是 "完全不脱落"。crosscut.py 按约定不动，只在服务端
    显示/导出这一层用标准口径。
    """
    if pct <= cc.NOISE_PCT:
        return "Gt0"
    if pct < 5.0:
        return "Gt1"
    if pct < 15.0:
        return "Gt2"
    if pct < 35.0:
        return "Gt3"
    if pct < 65.0:
        return "Gt4"
    return "Gt5"


class Session:
    def __init__(self, sid: str, name: str, path: str, size, orig_size=None):
        self.id = sid
        self.name = name
        self.path = path
        self.full: Image.Image | None = None
        self.size = size          # 原图尺寸
        self.orig_size = orig_size or size   # 上传前的尺寸（超 MAX_PIXELS 会被缩，见 _meta）
        self.rot = 0.0
        self._works: dict = {}    # key -> dict(rgb, lab, size, work)
        self._rois: dict = {}     # (x0,y0,x1,y1) -> dict(rgb, lab, size, work) 选区原生裁剪
        self._cls = None          # (key, best, mind, t)
        self.lock = threading.RLock()
        self.touch = time.time()

    def full_img(self) -> Image.Image:
        """解码好的原图，进程内只留一份（别的会话一解码，就把上一张放掉）。

        12.5 MP 的 RGB 是 37 MB，50 MP 要 150 MB，还带着 glibc arena；三个会话都挂着
        就白吃内存。重新解码 12.5 MP 约 0.3 s、50 MP 约 1.5 s，比一直占着划算。
        """
        if self.full is None:
            with FULL_LOCK:
                if self.full is None:
                    for sid, s in list(SESSIONS.items()):
                        if sid != self.id and s.full is not None:
                            s.full = None
                    self.full = cc.load_image(self.path)
        return self.full

    def work_img(self, side: int):
        """只要图（+rgb），不转 Lab。出图给浏览器看用它：6 MP 的 Lab 要 1 s 多，
        用户开记录时并不需要立刻算——真分析时再补。"""
        key = (int(side), round(float(self.rot), 4))
        got = self._works.get(key)
        if got is not None:
            return got
        rgb_im = self.full_img()
        w0 = cc.fit_work(rgb_im, int(side))
        if abs(self.rot) > 1e-6:
            w0 = w0.rotate(self.rot, resample=Image.BICUBIC, fillcolor=(248, 248, 248))
        rgb = np.asarray(w0).astype(np.uint8)
        got = {"work": w0, "rgb": rgb, "lab": None,
               "size": w0.size, "side": int(side)}
        self._works.clear()          # 只留当前分辨率，内存别叠
        self._works[key] = got
        self._cls = None
        return got

    def work(self, side: int):
        """分析图 + rgb + lab，按 (side, rot) 缓存（Lab 用到时才补算）"""
        got = self.work_img(side)
        if got.get("lab") is None:
            got["lab"] = lab_chunked(got["rgb"])
        return got

    def roi_work(self, x0: int, y0: int, x1: int, y1: int):
        """选区原生裁剪图（原图像素）+ rgb + lab，只留最近一份。

        这就是「只分析框选的位置」：框外一个像素都不碰，所以 0.2~1 MP 的选区
        能按原生像素分析，比整图压到 6 MP 还省内存（实测 0.24 s / 200 MB vs 3.18 s / 306 MB）。
        """
        key = (int(x0), int(y0), int(x1), int(y1))
        got = self._rois.get(key)
        if got is not None:
            return got
        im = self.full_img().crop(key).convert("RGB")
        w, h = im.size
        if w * h > MAX_ROI_PIXELS:            # 框到整张大图时兜底，别吃光内存
            f = math.sqrt(MAX_ROI_PIXELS / float(w * h))
            im = im.resize((max(1, int(round(w * f))), max(1, int(round(h * f)))), Image.LANCZOS)
        rgb = np.asarray(im).astype(np.uint8)
        got = {"work": im, "rgb": rgb, "lab": lab_chunked(rgb),
               "size": im.size, "side": max(im.size)}
        self._rois.clear()                 # 内存别叠：只留当前这一块
        self._rois[key] = got
        self._cls = None
        return got


SESSIONS: dict[str, Session] = {}
SESS_LOCK = threading.Lock()
HEAVY = threading.Semaphore(1)      # 单 worker 串行跑重活，避免内存被打爆
FULL_LOCK = threading.Lock()        # 解码原图的全局锁（同一时刻只留一张解码好的原图）
NO_TRIM = os.environ.get("BAIGE_NO_TRIM") == "1"


def trim_mem():
    """把已 free 的堆还给系统。

    glibc 会把 6 MP 的 Lab(float32)、开运算临时数组这类大块留在 arena 里不归还，
    Python 侧看着已经释放，RSS 却一直挂着（实测单个会话能积到 650 MB 匿名页）。
    每跑完一次重活调一次，RSS 会掉回几百 MB。
    """
    if NO_TRIM:                 # BAIGE_NO_TRIM=1 只用于内存 A/B 对比，生产别开
        return
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


class _HeavyBlock:
    """串行跑重活 + 跑完把堆还给系统（顺带把 Lab/开运算的临时数组真正交回去）。"""

    def __enter__(self):
        HEAVY.acquire()
        return self

    def __exit__(self, *exc):
        HEAVY.release()
        trim_mem()
        return False


HEAVY_BLOCK = _HeavyBlock()



def get_session(sid: str) -> Session:
    with SESS_LOCK:
        s = SESSIONS.get(sid)
        if s is None:
            raise HTTPException(404, L("图片已过期，请重新上传",
                                   "The image expired — please upload it again"))
        s.touch = time.time()
        return s


def lab_chunked(rgb: np.ndarray, step: int = 256) -> np.ndarray:
    """rgb -> Lab，按行分块算：结果与 cc.rgb_to_lab(rgb) 逐像素相同，峰值内存小很多"""
    h = rgb.shape[0]
    out = np.empty(rgb.shape[:2] + (3,), np.float32)
    for y0 in range(0, h, step):
        y1 = min(h, y0 + step)
        out[y0:y1] = cc.rgb_to_lab(rgb[y0:y1])
    return out


def classify_parts_chunked(lab: np.ndarray, sample_labs, roles):
    """rows 分块调用原版 classify_parts —— 结果与整图一次算完全一致，只是省内存"""
    h, w = lab.shape[:2]
    best = np.empty((h, w), np.int16)
    mind = np.empty((h, w), np.float32)
    t = np.empty((h, w), np.float32)
    t_none = False
    step = 192
    for y0 in range(0, h, step):
        y1 = min(h, y0 + step)
        b, m, tt = cc.classify_parts(lab[y0:y1], sample_labs, roles)
        best[y0:y1] = b
        mind[y0:y1] = m
        if tt is None:
            t_none = True
        else:
            t[y0:y1] = tt
    return best, mind, (None if t_none else t)


def _role_name(role):
    """角色名（按请求语言）——/api/pick 等接口直接回给前端显示，别用 cc.ROLE_NAME 的中文"""
    return {"coat": L("漆膜", "coating"), "sub": L("基材", "substrate"),
            "ignore": L("忽略", "ignored")}.get(role, str(role))


def _sample_labs(samples):
    return [cc.rgb_to_lab(np.array(s["rgb"], np.float32)) for s in samples]


def _roles(samples):
    return [s["role"] for s in samples]


def _classify_pixels(lab_flat: np.ndarray, sample_labs, roles, chunk: int = 400_000):
    """对一串像素（N×3）做分类，分块省内存；与整图算的逐像素结果完全一致"""
    n = lab_flat.shape[0]
    best = np.empty(n, np.int16)
    mind = np.empty(n, np.float32)
    t = np.empty(n, np.float32)
    t_none = False
    for i in range(0, n, chunk):
        j = min(n, i + chunk)
        # crosscut 内部按 (h, w) 分配数组，像素列表摊成 (k, 1) 再压回去即可
        b, m, tt = cc.classify_parts(lab_flat[i:j].reshape(-1, 1, 3), sample_labs, roles)
        best[i:j] = np.asarray(b).reshape(-1)
        mind[i:j] = np.asarray(m).reshape(-1)
        if tt is None:
            t_none = True
        else:
            t[i:j] = np.asarray(tt).reshape(-1)
    return best, mind, (None if t_none else t)


def _msig(rois):
    """选区签名（进分类缓存 key）"""
    if not rois:
        return None
    return tuple((tuple(r["box"]), round(float(r.get("ang") or 0.0), 3)) for r in rois)


def _roi_bbox(shape2, masks):
    """所有选区的并集外接矩形 (y0, y1, x0, x1)；没有选区返回 None"""
    if not masks:
        return None
    m = np.zeros(shape2, bool)
    for x in masks:
        m |= x
    if not m.any():
        return None
    ys = np.flatnonzero(m.any(1))
    xs = np.flatnonzero(m.any(0))
    return (int(ys[0]), int(ys[-1]) + 1, int(xs[0]), int(xs[-1]) + 1)


def _labels_for(sess: Session, work, tol: float, purity: float, samples, masks=None, sig=None,
                vkey=None):
    """带缓存：样本不变时只重算 labels（容差/纯度变了也不用重跑最近色）

    masks 给了就**只算这些选区里的像素**（按像素索引，斜对角的框也不会把中间的空白算进去），
    结果贴回整幅（框外标 -1 = 未分类），下游 stats/诊断/覆盖层不用改。
    vkey 是「分析视图」的身份（缩放/原点），换视图必须让缓存失效。"""
    sample_labs = _sample_labs(samples)
    roles = _roles(samples)
    key = (tuple((tuple(s["rgb"]), s["role"]) for s in samples), sig, vkey)
    shape2 = work["lab"].shape[:2]
    with sess.lock:
        if sess._cls is None or sess._cls[0] != key:
            if masks:
                m = np.zeros(shape2, bool)
                for x in masks:
                    m |= x
                idx = np.flatnonzero(m)
                flat = work["lab"].reshape(-1, 3)[idx]
                b0, m0, t0 = _classify_pixels(flat, sample_labs, roles)
                best = np.full(shape2[0] * shape2[1], -1, np.int16)
                mind = np.full(shape2[0] * shape2[1], np.float32(1e6), np.float32)
                t = np.zeros(shape2[0] * shape2[1], np.float32)
                best[idx] = b0
                mind[idx] = m0
                if t0 is not None:
                    t[idx] = t0
                best = best.reshape(shape2)
                mind = mind.reshape(shape2)
                t = t.reshape(shape2)
            else:
                best, mind, t = classify_parts_chunked(work["lab"], sample_labs, roles)
            sess._cls = (key, best, mind, t)
        _, best, mind, t = sess._cls
    return cc.labels_from_parts(best, mind, t, float(tol), float(purity)), best, mind, t


def _span_of(samples):
    """漆膜与基材样本在 Lab 空间里的最小间距（自动容差 = 间距 × 45%）"""
    labs = _sample_labs(samples)
    roles = _roles(samples)
    coat = [l for l, r in zip(labs, roles) if r == cc.ROLE_COAT]
    sub = [l for l, r in zip(labs, roles) if r == cc.ROLE_SUB]
    if not coat or not sub:
        return None
    return min(float(np.linalg.norm(c - s)) for c in coat for s in sub)


def _auto_tol_of(span):
    """自动容差：整数，6~120（和桌面版同一套换算，只是不再带小数）"""
    if span is None:
        return None
    return int(round(max(6.0, min(120.0, span * 0.45))))


def _masks(shape, rois):
    return [cc.rot_rect_mask(shape, tuple(float(v) for v in r["box"]), float(r.get("ang") or 0.0))
            for r in rois]


class View:
    """这一次分析真正用的那张图，以及它和「画布坐标」之间的换算。

    两种视图：

    * 默认（roi_native=False）：视图 = 整图压到 ≤6 MP。scale = s6，原点 (0,0)。
    * 选区原生（roi_native=True）：视图 = 选区并集外接矩形按**原图像素**裁出来的一块。
      scale = 1（原生 px），原点是裁剪框左上角。

    对外一律用 wire 坐标——今天画布上那张 6 MP 分析图的坐标，所以前端坐标系不用改；
    k / 阈值也留在 wire 口径里（`len_in` 换算成视图像素再去跑开运算），
    这样同一个 k 在两种模式下代表同一段物理宽度，历史数字仍然可比。
    """

    def __init__(self, work, s6, scale=None, ox=0.0, oy=0.0, native=False):
        self.work = work
        self.s6 = float(s6)                                   # wire px / 原生 px
        self.scale = float(s6 if scale is None else scale)    # 视图 px / 原生 px
        self.ox = float(ox)
        self.oy = float(oy)
        self.native = bool(native)

    def len_in(self, v_wire):          # wire px → 视图 px
        return float(v_wire) * self.scale / self.s6

    def len_out(self, v_view):         # 视图 px → wire px
        return float(v_view) * self.s6 / self.scale

    def to_wire(self, x, y):           # 视图点 → wire
        return ((float(x) / self.scale + self.ox) * self.s6,
                (float(y) / self.scale + self.oy) * self.s6)

    def to_view(self, x, y):           # wire 点 → 视图
        return ((float(x) / self.s6 - self.ox) * self.scale,
                (float(y) / self.s6 - self.oy) * self.scale)

    def box_in(self, box):
        x0, y0 = self.to_view(box[0], box[1])
        x1, y1 = self.to_view(box[2], box[3])
        return [x0, y0, x1, y1]

    def rois_in(self, rois):
        out = []
        for r in rois:
            r2 = dict(r)
            r2["box"] = self.box_in(r["box"])
            out.append(r2)
        return out

    @property
    def key(self):
        w, h = self.work["size"]
        return ("roi" if self.native else "full", round(self.scale, 6),
                round(self.ox, 3), round(self.oy, 3), int(w), int(h))


ROI_MIN_WIRE = 4.0        # 选区每边至少这么宽（wire px）；前端新建框的下限是 6


def _view_for(sess: Session, req):
    """按请求决定分析视图。返回 (work, view)。

    选区原生模式只在「有选区、且整图没旋转过」时生效；其它情况自动退回默认视图
    （整图旋转后裁剪的几何关系会变，宁可不省那点像素也不出错）。

    顺带把选区框规整成「有序 + 落在画面内」：反框（x0>x1）、越界框以前会一路算到
    分母为 0 / 负面积，静默给出 0B、5B 这种假结论，选区原生路径还会在回退分支上
    抛 UnboundLocalError 变 500。现在就地修正，修不动（细成一条线）就 400。
    """
    side = _res_to_side(getattr(req, "res", "full"), sess.full_img())
    wi = sess.work_img(side)                  # 只构图，不转 Lab（6 MP 的 Lab 要 1 s 多）
    s6 = wi["size"][0] / float(sess.size[0]) if sess.size[0] else 1.0
    W, H = sess.size
    wmax, hmax = W * s6, H * s6               # wire 口径的画面尺寸
    rois = []
    for r in (getattr(req, "rois", None) or []):
        b = [float(v) for v in r.box]
        x0, x1 = (b[0], b[2]) if b[0] <= b[2] else (b[2], b[0])
        y0, y1 = (b[1], b[3]) if b[1] <= b[3] else (b[3], b[1])
        x0, x1 = min(max(x0, 0.0), wmax), min(max(x1, 0.0), wmax)
        y0, y1 = min(max(y0, 0.0), hmax), min(max(y1, 0.0), hmax)
        bw, bh = x1 - x0, y1 - y0
        if bw < ROI_MIN_WIRE or bh < ROI_MIN_WIRE:
            raise HTTPException(400, L("选区太小或不在画面上（宽 %.0f × 高 %.0f px，"
                                       "画面 %.0f×%.0f）：请重新框选",
                                       "Region too small or off-canvas (%.0f × %.0f px, canvas "
                                       "%.0f×%.0f) — please frame it again",
                                       bw, bh, wmax, hmax))
        nb = [x0, y0, x1, y1]
        if nb != b:
            r.box = nb                        # 回写模型：下游统计用的是同一份框
        rois.append({"box": nb, "ang": float(r.ang or 0.0)})
    if not getattr(req, "roi_native", False) or not rois or abs(sess.rot) > 1e-6:
        work6 = sess.work(side)               # 默认口径：整图 + Lab
        return work6, View(work6, s6)

    xs0 = ys0 = float("inf")
    xs1 = ys1 = float("-inf")
    for r in rois:
        box = [float(v) for v in r["box"]]
        ang = float(r.get("ang") or 0.0)
        if abs(ang) > 1e-6:
            pts = cc.rect_corners(tuple(box), ang)
            bx0, by0 = min(p[0] for p in pts), min(p[1] for p in pts)
            bx1, by1 = max(p[0] for p in pts), max(p[1] for p in pts)
        else:
            bx0, by0, bx1, by1 = box
        xs0, ys0 = min(xs0, bx0), min(ys0, by0)
        xs1, ys1 = max(xs1, bx1), max(ys1, by1)
    pad = 2                                    # 外扩 2 px：斜切缝的边缘不被裁断
    X0 = max(0, int(math.floor(xs0 / s6)) - pad)
    Y0 = max(0, int(math.floor(ys0 / s6)) - pad)
    X1 = min(W, int(math.ceil(xs1 / s6)) + pad)
    Y1 = min(H, int(math.ceil(ys1 / s6)) + pad)
    if X1 - X0 < 4 or Y1 - Y0 < 4:             # 裁剪小到没意义 → 退回默认口径
        work6 = sess.work(side)
        return work6, View(work6, s6)
    crop = sess.roi_work(X0, Y0, X1, Y1)
    scale = crop["size"][0] / float(X1 - X0)   # 只有当裁剪超过 MAX_ROI_PIXELS 时才 <1
    return crop, View(crop, s6, scale=scale, ox=X0, oy=Y0, native=True)


def _paste_codes(codes, labels, view, mask, a, bx, by, roles, sub_code=1):
    """把视图坐标里的分类结果贴进一张目标画布（codes: 0 空 / 1 露底 / 2 漆膜 / 3 刀缝）。

    sub_code 用来区分“算作脱落的基材”(1) 与“被开运算排除的刀缝”(3)。

    a = 视图 px → 目标 px，(bx, by) = 视图原点在目标画布上的位置。
    """
    h, w = labels.shape[:2]
    bb = _roi_bbox((h, w), [mask])
    if not bb:
        return
    y0, y1, x0, x1 = bb
    sub_lab = labels[y0:y1, x0:x1]
    tw = max(1, int(round((x1 - x0) * a)))
    th = max(1, int(round((y1 - y0) * a)))
    if (tw, th) != (x1 - x0, y1 - y0):
        lab_s = np.asarray(Image.fromarray(
            np.clip(sub_lab.astype(np.int16) + 2, 0, 255).astype(np.uint8)
        ).resize((tw, th), Image.NEAREST)).astype(np.int16) - 2
    else:
        lab_s = sub_lab
    code = np.zeros(lab_s.shape, np.uint8)
    for i, role in enumerate(roles):
        if role == cc.ROLE_SUB:
            code[lab_s == i] = sub_code
        elif role == cc.ROLE_COAT:
            code[lab_s == i] = 2
    oy, ox = int(round(y0 * a + by)), int(round(x0 * a + bx))
    oh, ow = codes.shape[:2]
    hh = min(th, oh - oy)
    ww = min(tw, ow - ox)
    if hh <= 0 or ww <= 0 or oy < 0 or ox < 0:
        return
    part = codes[oy:oy + hh, ox:ox + ww]
    src = code[:hh, :ww]
    part[src > 0] = src[src > 0]
    codes[oy:oy + hh, ox:ox + ww] = part


def _cheb_dist(mask: np.ndarray, kmax: int = 24) -> np.ndarray:
    """棋盘距离变换（= 逐层 3×3 腐蚀的次数），与 cc.thickness_map 的结果逐像素相同，
    但快 15~20 倍：cc 那版每次腐蚀都要过一次 PIL 滤镜。"""
    INF = np.int16(kmax + 1)
    d = np.where(mask, INF, np.int16(0)).astype(np.int16)
    h = d.shape[0]
    for y in range(1, h):
        prev, cur = d[y - 1], d[y]
        np.minimum(cur, prev + 1, out=cur)
        np.minimum(cur[1:], prev[:-1] + 1, out=cur[1:])
        np.minimum(cur[:-1], prev[1:] + 1, out=cur[:-1])
        np.minimum(cur[1:], np.minimum(cur[:-1] + 1, cur[1:]), out=cur[1:])
    for y in range(h - 2, -1, -1):
        nxt, cur = d[y + 1], d[y]
        np.minimum(cur, nxt + 1, out=cur)
        np.minimum(cur[1:], nxt[:-1] + 1, out=cur[1:])
        np.minimum(cur[:-1], nxt[1:] + 1, out=cur[:-1])
        np.minimum(cur[:-1], np.minimum(cur[1:] + 1, cur[:-1]), out=cur[:-1])
    return d


HIST_BINS = ((1, 3), (4, 6), (7, 10), (11, 16), (17, 10 ** 6))


def _hist_for(labels, mask, roles, px=1.0):
    """复刻 cc.thickness_hist（对选区内的露底像素做厚度分布），只把厚度图算快了

    px = 视图 px → wire px 的系数：厚度分档永远按 wire（6 MP 分析图）口径给，
    这样「选区原生」模式下这张直方图仍然和历史数字直接可比。"""
    sub_idx = [i for i, r in enumerate(roles) if r == cc.ROLE_SUB]
    if not sub_idx or not mask.any():
        return None
    rows = np.flatnonzero(mask.any(1))
    cols = np.flatnonzero(mask.any(0))
    if not rows.size or not cols.size:
        return None
    r0, r1 = rows[0], rows[-1] + 1
    c0, c1 = cols[0], cols[-1] + 1
    subc = np.zeros((r1 - r0, c1 - c0), bool)
    for i in sub_idx:
        subc |= (labels[r0:r1, c0:c1] == i)
    subc &= mask[r0:r1, c0:c1]
    if not subc.any():
        return [(lo, hi, 0, 0.0) for lo, hi in HIST_BINS]
    vals = (2 * np.minimum(_cheb_dist(subc), 24))[subc] * float(px)
    total = max(1, vals.size)
    return [(lo, hi, int(((vals >= lo) & (vals <= hi)).sum()),
             100.0 * int(((vals >= lo) & (vals <= hi)).sum()) / total) for lo, hi in HIST_BINS]


def _stats(labels, masks, roles, line_px, mode, want_hist=False, px=1.0, loc_tile=64, loc_factor=1.6):
    rows = []
    tot = {"coat_px": 0, "sub_raw": 0, "sub_open": 0, "area": 0, "unassigned": 0,
           "n_line": 0, "line_area": 0, "trans": 0, "thist": None,
           "loc_med": None, "loc_alt_open": 0}
    for m in masks:
        st = cc.roi_stats(labels, m, roles, int(line_px), mode, False,
                          loc_tile=int(loc_tile), loc_factor=float(loc_factor))
        if want_hist:
            st["thist"] = _hist_for(labels, m, roles, px)
        rows.append(st)
        for k in ("coat_px", "sub_raw", "sub_open", "area", "unassigned",
                  "n_line", "line_area", "trans"):
            tot[k] += st[k]
        if st.get("loc_alt_open") is not None:
            tot["loc_alt_open"] += st["loc_alt_open"]
            if st.get("loc_med") is not None and tot["loc_med"] is None:
                tot["loc_med"] = st["loc_med"]
        if want_hist and st.get("thist"):
            if tot["thist"] is None:
                tot["thist"] = [list(x) for x in st["thist"]]
            else:
                for a, b in zip(tot["thist"], st["thist"]):
                    a[2] += b[2]
    if tot["thist"]:
        s = sum(a[2] for a in tot["thist"]) or 1
        for a in tot["thist"]:
            a[3] = 100.0 * a[2] / s
    denom = tot["coat_px"] + tot["sub_raw"]
    pct = 100.0 * tot["sub_open"] / denom if denom else 0.0
    return rows, tot, pct


def _row_out(st):
    return {"pct": st["pct"], "pct_raw": st["pct_raw"], "astm": cc.astm_class(st["pct"]),
            "iso": _iso_class(st["pct"]), "area": st["area"], "coat_px": st["coat_px"],
            "sub_raw": st["sub_raw"], "sub_open": st["sub_open"],
            "unassigned": st["unassigned"], "trans": st["trans"],
            "n_line": st["n_line"], "line_area": st["line_area"]}


# ---------------------------------------------------------------- 请求模型


class ROI(BaseModel):
    box: list[float]
    ang: float = 0.0


class Sample(BaseModel):
    rgb: list[int]
    role: str = "coat"


class AnalyzeReq(BaseModel):
    id: str
    res: str = "1800"
    rois: list[ROI] = []
    samples: list[Sample] = []
    tol: float = 35.0
    purity: float = 0.75
    line_px: int = 1
    mode: str = "width"        # width / shape / local（实验 C：局部宽度）
    loc_tile: int = 64         # local 模式：分块边长（px，分析图口径）
    loc_factor: float = 1.6    # local 模式：排除「薄于 本块基线×factor」的露底
    want_hist: bool = True
    detail: bool = False
    overlay: bool = True
    ruler_hist: list[float] = []
    ruler_full: list[float] = []
    soft: bool = False          # True 时额外返回软权重/区间（实验，不影响上面任何字段）
    roi_native: bool = False     # v1.36 起默认「选区原生」：只裁选区、按原图像素分析（界面已无开关）
    want_suggest: bool = True     # False = 别算「按露底厚度猜刀缝」的建议值（k 由判法定的时候用不着）


class AutoReq(BaseModel):
    id: str
    res: str = "1800"
    rois: list[ROI] = []
    soft: bool = False
    roi_native: bool = False
    purity: float = 0.75               # 与界面上的纯度一致，否则半自动和手动算出的数对不上
    line_px: Optional[float] = None    # 由“刀缝判法”定好的 k：给了就直接用，不再自己猜
    detect_only: bool = False          # 只认色，不分析（前端拿到样色后先定 k，再回来算）


class Stroke(BaseModel):
    t: int                       # 1=确认脱落 2=确认刀缝 3=无法判断
    r: float = 12.0              # 笔刷半径（分析图像素）
    pts: list[list[float]] = []


class BlocksReq(BaseModel):
    id: str
    res: str = "1800"
    rois: list[ROI] = []
    samples: list[Sample] = []
    tol: float = 35.0
    purity: float = 0.75
    line_px: int = 1
    mode: str = "width"          # 必须和 /api/analyze 用同一个判法，否则块算出来的数对不上结果栏
    loc_tile: int = 64
    loc_factor: float = 1.6
    min_area: int = 30
    max_blocks: int = 240
    roi_native: bool = False


class AnnotReq(BaseModel):
    id: str
    res: str = "1800"
    rois: list[ROI] = []
    strokes: list[Stroke] = []
    roi_native: bool = False


class PickReq(BaseModel):
    id: str
    res: str = "1800"
    x: float
    y: float
    samples: list[Sample] = []


class SamplesReq(BaseModel):
    samples: list[Sample] = []


class RulerReq(BaseModel):
    id: str
    res: str = "1800"
    p0: list[float]
    p1: list[float]
    on_full: bool = True


class RotateReq(BaseModel):
    id: str
    deg: float = 0.0
    res: str = "1800"


# ---------------------------------------------------------------- 图片


def _meta(sess: Session, side: int):
    w, h = sess.work_img(side)["size"]      # 只构图，不转 Lab
    ow, oh = sess.size                       # 原图尺寸，别为此解码整张图
    capped = (ow * oh > MAX_ANALYSIS_PIXELS and side < max(ow, oh)
              and side <= int(math.sqrt(MAX_ANALYSIS_PIXELS * max(ow, oh) / float(min(ow, oh)))))
    return {"id": sess.id, "name": sess.name, "w": w, "h": h, "ow": ow, "oh": oh,
            "side": side, "rot": sess.rot, "scale": ow / float(w) if w else 1.0,
            "orig_w": sess.orig_size[0], "orig_h": sess.orig_size[1],
            "shrunk": sess.orig_size != sess.size,
            "shrunk_note": (L("原图 %d×%d，上传时超过 %d MP 上限，存档已缩到 %d×%d（%.0f%%）",
                              "Original %d×%d exceeded the %d MP upload cap; stored at %d×%d (%.0f%%)",
                              sess.orig_size[0], sess.orig_size[1], MAX_PIXELS // 1_000_000,
                              ow, oh, 100.0 * ow / float(sess.orig_size[0]))
                            if sess.orig_size != sess.size else ""),
            "note": (L("已按内存上限（%.1f MP）降采样分析",
                       "analysed downscaled to the %.1f MP memory cap",
                       MAX_ANALYSIS_PIXELS / 1e6) if capped else "")}


def _raw_exposure_stats(rp):
    """从 RAW 的原始数据量曝光：过曝比例、用到量程的多少。JPEG 里没有这个信息。
    只在整数域算，不转 float（12 MP 转 float 要 50 MB）。"""
    try:
        raw = rp.raw_image_visible
        wl = float(rp.white_level)
        bl = float(np.mean(rp.black_level_per_channel))
        span = max(1.0, wl - bl)
        n = float(raw.size)
        clip = float((raw >= bl + 0.98 * span).sum()) / n * 100.0
        near = float((raw >= bl + 0.95 * span).sum()) / n * 100.0
        mx = (float(raw.max()) - bl) / span
        med = (float(np.median(raw[::4, ::4])) - bl) / span
        return {"clip_pct": clip, "near_pct": near, "max_frac": mx, "median_frac": med}
    except Exception:
        return {}


def _raw_exposure_msg(st):
    """把 RAW 曝光情况翻成一句人话（有毛病才说，正常就轻描淡写）"""
    if not st:
        return ""
    clip, mx = st["clip_pct"], st["max_frac"]
    if clip > 0.05:
        return L("⚠ RAW 有 %.2f%% 像素过曝（最亮处已到白电平），白色区域细节被削平，"
                 "结果可能偏低——建议降 1/3 档重拍",
                 "⚠ %.2f%% of the RAW pixels are clipped (brightest already at white level), so "
                 "highlight detail is flattened and the result may read low — reshoot 1/3 stop darker",
                 clip)
    if mx < 0.40:
        return L("RAW 曝光偏暗：最亮只用到量程的 %.0f%%，过曝 0；能拍就再进 1 档，噪点更低",
                 "RAW is underexposed: the brightest pixel uses only %.0f%% of full scale, no clipping; "
                 "shoot one stop brighter if you can — less noise",
                 mx * 100)
    return L("RAW 曝光正常：最亮用到量程 %.0f%%，过曝 %.3f%%",
             "RAW exposure normal: brightest uses %.0f%% of full scale, %.3f%% clipped",
             mx * 100, clip)


def _decode_dng(raw_bytes, note):
    """DNG → RGB。要点：
    1) 相机白平衡不可信（Pixel 写的是 [0,1,0,0]），auto_wb 是灰世界会把黄色场景灰化，
       所以自己用“点白平衡”：取最亮的 1% 像素当纸白，把三通道拉到相等。
    2) 曝光对齐到纸白 ≈ 230/255（不溢出）。
    3) 解码后立刻压到分析尺寸，避免 12 MP 的 float 中间数组（≈300 MB）。"""
    try:
        import rawpy
    except Exception:
        raise HTTPException(400, L("服务器没装 rawpy，不能读 DNG（可先导出 TIFF/JPEG 再上传）",
                                   "rawpy is not installed on the server, so DNG cannot be read "
                                   "(export TIFF/JPEG first and upload that)"))
    stats = {}
    try:
        with rawpy.imread(io.BytesIO(raw_bytes)) as rp:
            stats = _raw_exposure_stats(rp)          # 先量曝光（只有 RAW 才有这个信息）
            rgb = rp.postprocess(use_camera_wb=False, use_auto_wb=False, no_auto_bright=True,
                                 output_bps=8, gamma=(2.4, 12.92),
                                 output_color=rawpy.ColorSpace.sRGB,
                                 demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD)
    except Exception as exc:
        raise HTTPException(400, L("DNG 解码失败：%s", "DNG decoding failed: %s", exc))
    h, w = rgb.shape[:2]
    if w * h > 60_000_000:
        raise HTTPException(400, L("这张 DNG 太大（%d 万像素），请用 12 MP RAW",
                                   "This DNG is too large (%d ×10k pixels) — use a 12 MP RAW",
                                   w * h // 10000))
    s = rgb[::8, ::8].astype(np.float32)
    lum = s.sum(-1)
    thr = np.percentile(lum, 99)
    if thr <= 0:
        thr = lum.max()
    m = s[lum >= thr].mean(0)
    gains = (m.mean() / np.maximum(m, 1e-6)).astype(np.float32)
    out = Image.fromarray(np.clip(rgb.astype(np.float32) * gains, 0, 255).astype(np.uint8))
    del rgb
    # 曝光对齐：把亮部的 99.5 分位拉到 L=96（和相机 JPEG 的观感一致，且与离线对比实验同一口径）
    small = np.asarray(out.resize((max(16, out.width // 6), max(16, out.height // 6)),
                                  Image.BILINEAR)).astype(np.float32)
    lo, hi = 0.3, 4.0
    for _ in range(12):
        kk = (lo + hi) / 2
        p = float(np.percentile(cc.rgb_to_lab(np.clip(small * kk, 0, 255).astype(np.float32))[..., 0], 99.5))
        if p > 96:
            hi = kk
        else:
            lo = kk
    kk = (lo + hi) / 2
    if abs(kk - 1.0) > 0.02:
        out = Image.fromarray(np.clip(np.asarray(out).astype(np.float32) * kk, 0, 255).astype(np.uint8))
    note["dng"] = L("已解码 DNG：点白平衡 + 曝光对齐（×%.2f），未做锐化",
                    "DNG decoded: white balance + exposure aligned (×%.2f), not sharpened", kk)
    if stats:
        note["raw"] = {k: round(float(v), 4) for k, v in stats.items()}
        note["raw_msg"] = _raw_exposure_msg(stats)
    note["wb_gain"] = [round(float(v) * kk, 3) for v in gains]
    return out


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    raw = await file.read()
    return _new_session_from_bytes(raw, file.filename or "photo.jpg")


def _new_session_from_bytes(raw: bytes, filename: str):
    """把一份完整的图片字节装进新会话。分片上传（/api/upload/finish）也走这里，保证两条路完全一致。"""
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, L("文件超过 %d MB", "File exceeds %d MB", MAX_UPLOAD_MB))
    sid = secrets.token_urlsafe(9)
    d = os.path.join(UPLOADS, sid)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "orig.png")      # 分析源图：无损，避免 q95 再编码漂移
    note = {}
    fname = (filename or "").lower()
    is_dng = fname.endswith((".dng", ".raw", ".cr2", ".nef", ".arw"))
    if is_dng:
        im = _decode_dng(raw, note)
    else:
        try:
            im = Image.open(io.BytesIO(raw))
            im = ImageOps.exif_transpose(im).convert("RGB")
        except Exception:
            if raw[:4] == b"II*\x00" or raw[:4] == b"MM\x00*":
                im = _decode_dng(raw, note)
            else:
                raise HTTPException(400, L("图片读不出来（既不是常见图片，也不是 DNG）",
                                   "Cannot read this image (neither a common image format nor DNG)"))
    orig_size = im.size                     # 存档前的原始尺寸（用来告诉用户“缩过没有”）
    if im.size[0] * im.size[1] > MAX_PIXELS:
        im.thumbnail((6000, 6000), Image.LANCZOS)
        note["shrunk"] = {"from": [orig_size[0], orig_size[1]], "to": [im.size[0], im.size[1]]}
    im.save(path, "PNG", compress_level=3)   # level 3：比 6 快 3 倍，体积只大 ~10%
    name = (filename or "photo.jpg").strip()
    sess = Session(sid, name, path, im.size, orig_size=orig_size)
    with SESS_LOCK:
        while len(SESSIONS) >= MAX_SESSIONS:
            old = min(SESSIONS.values(), key=lambda s: s.touch)
            SESSIONS.pop(old.id, None)
            old._works.clear()
            old._cls = None
            old.full = None
            # 被挤出去的会话，上传目录就是死重量（记录里存的是独立副本），直接删
            try:
                od = os.path.dirname(old.path)
                if os.path.abspath(od).startswith(os.path.abspath(UPLOADS)):
                    shutil.rmtree(od, ignore_errors=True)
            except Exception:
                pass
            trim_mem()
        SESSIONS[sid] = sess
    side = _res_to_side_sz(DEFAULT_RES, sess.size)      # 只要尺寸，别为它解码整张原图
    meta = _meta(sess, side)
    meta["url"] = "/api/image/%s?res=%s" % (sid, DEFAULT_RES)
    meta["default_res"] = DEFAULT_RES
    if note:
        meta.update(note)
        if note.get("raw_msg"):
            meta["note"] = note["raw_msg"] + (("｜" + meta["note"]) if meta.get("note") else "")
    return meta


CHUNK_DIR = os.path.join(UPLOADS, "_chunks")
CHUNK_MAX_BYTES = 4 * 1024 * 1024      # 单片硬上限；前端用 400 KB，给反代留余量
CHUNK_KEEP_S = 3 * 3600


def _chunk_uid_ok(uid: str) -> bool:
    return bool(uid) and len(uid) <= 40 and all(c.isalnum() or c in "-_" for c in uid)


@app.post("/api/upload/chunk")
async def upload_chunk(request: Request, uid: str, idx: int, total: int, name: str = ""):
    """分片上传：前端把大图切成 400 KB 的小块逐个发。

    存在的理由：外层反代（nginx + oauth2）对超过约 1 MB 的请求体直接回 500，
    10 MB 的手机照根本传不上来。分片让每个请求都远小于这个门槛，拼回来的字节与整传完全一致。
    """
    if not _chunk_uid_ok(uid):
        raise HTTPException(400, L("上传号不合法", "Invalid upload id"))
    if idx < 0 or total <= 0 or idx >= total:
        raise HTTPException(400, L("分片序号不合法", "Invalid chunk index"))
    raw = await request.body()
    if not raw:
        raise HTTPException(400, L("空分片", "Empty chunk"))
    if len(raw) > CHUNK_MAX_BYTES:
        raise HTTPException(413, L("单片太大（%d 字节）", "Chunk too large (%d bytes)", len(raw)))
    d = os.path.join(CHUNK_DIR, uid)
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, "%05d.part" % idx)
    with open(tmp + ".tmp", "wb") as f:
        f.write(raw)
    os.replace(tmp + ".tmp", tmp)
    with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"name": name, "total": total, "ts": time.time()}, f)
    return {"ok": True, "idx": idx, "got": len(raw)}


@app.post("/api/upload/finish")
def upload_finish(body: dict):
    uid = str(body.get("uid") or "")
    if not _chunk_uid_ok(uid):
        raise HTTPException(400, L("上传号不合法", "Invalid upload id"))
    d = os.path.join(CHUNK_DIR, uid)
    if not os.path.isdir(d):
        raise HTTPException(404, L("分片不见了，重传一次", "Upload chunk disappeared — send it again"))
    parts = sorted(x for x in os.listdir(d) if x.endswith(".part"))
    if not parts:
        raise HTTPException(400, L("一个分片都没收到", "No chunk received at all"))
    buf = bytearray()
    for i, nm in enumerate(parts):
        if nm != "%05d.part" % i:
            raise HTTPException(400, L("分片缺了第 %d 片", "Chunk %d is missing", i))
        with open(os.path.join(d, nm), "rb") as f:
            buf += f.read()
    name = body.get("name") or ""
    if not name:
        try:
            name = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8")).get("name") or ""
        except Exception:
            name = ""
    try:
        return _new_session_from_bytes(bytes(buf), name or "photo.jpg")
    finally:
        shutil.rmtree(d, ignore_errors=True)


@app.post("/api/load")
def load(body: dict):
    sess = get_session(body.get("id"))
    side = _res_to_side(body.get("res", WORK_MAX_DEFAULT), sess.full_img())
    with sess.lock:
        return _meta(sess, side)


@app.get("/api/image/{sid}")
def image(sid: str, res: str = str(WORK_MAX_DEFAULT)):
    sess = get_session(sid)
    side = _res_to_side(res, sess.full_img())
    with sess.lock:
        w = sess.work_img(side)["work"]
        data = _jpeg(w, 92)
    return Response(data, media_type="image/jpeg",
                    headers={"Cache-Control": "private, max-age=3600"})


@app.post("/api/rotate")
def rotate(body: RotateReq):
    sess = get_session(body.id)
    with sess.lock:
        sess.rot = float(body.deg)
        sess._works.clear()
        sess._cls = None
    side = _res_to_side(body.res, sess.full_img())
    return _meta(sess, side)


# ---------------------------------------------------------------- 分析


def _overlay_key(sess, side, samples, tol, purity, sig=None, vkey=None, line_px=0, mode="width"):
    h = hashlib.sha1(json.dumps([sess.id, side, round(sess.rot, 3),
                                 [[s["rgb"], s["role"]] for s in samples],
                                 round(tol, 3), round(purity, 3),
                                 # 必须用"选区签名"而不是外接矩形：新框落在老外接矩形里时
                                 # bbox 不变，就会命中旧图，少涂一个框
                                 [list(x) for x in sig] if sig else None,
                                 list(vkey) if vkey else None,
                                 ],
                                separators=(",", ":")).encode())
    return h.hexdigest()[:20]



def _otsu_1d(vals, bins: int = 160):
    """一维 Otsu：在 t（混色比 0=纯漆膜, 1=纯基材）的分布上找两类分界。
    比"手调纯度"客观；但它假设双峰，若混色像素占主导，分界会落在混色带中间——所以只作建议。"""
    vals = np.asarray(vals, dtype=np.float32).ravel()
    if vals.size < 500:
        return None
    h, edges = np.histogram(np.clip(vals, 0.0, 1.0), bins=bins, range=(0.0, 1.0))
    c = (edges[:-1] + edges[1:]) / 2.0
    w = h.astype(np.float64)
    tot = w.sum()
    if tot <= 0:
        return None
    cw = np.cumsum(w)
    cm = np.cumsum(w * c)
    best_v, best_t = -1.0, None
    for i in range(1, bins - 1):
        w0 = cw[i]
        w1 = tot - w0
        if w0 <= 0 or w1 <= 0:
            continue
        m0 = cm[i] / w0
        m1 = (cm[-1] - cm[i]) / w1
        v = w0 * w1 * (m0 - m1) ** 2
        if v > best_v:
            best_v, best_t = v, float(c[i])
    return best_t


def _granulometry(curve, big: float = 0.05):
    """粒度谱判读：pct(k) 相对 k=1 的每一步跌幅。

    * 跌幅集中在**一处** → 缝宽统一，陡降之后就是真实脱落量（建议 k = 陡降点 + 1）。
    * 跌幅散在**多处** → 徒手缝宽不等（3px 到 8px 都有），一个全局 k 顾不过来，
      这时按宽度判出来的等级不可信，应该用「按形状」或线基几何法。
    """
    out = {"steps": [], "verdict": "empty", "main_drop_k": None, "n_big": 0,
           "plateau": None, "suggest_k": None}
    if len(curve) < 4 or curve[0]["pct"] <= 0:
        return out
    p0 = curve[0]["pct"]
    steps = []
    for a, b in zip(curve, curve[1:]):
        d = a["pct"] - b["pct"]
        steps.append({"k": a["k"], "pct": a["pct"], "drop": round(d, 3),
                      "drop_rel": round(100.0 * d / p0, 2)})
    out["steps"] = steps
    n_big = sum(1 for x in steps if x["drop_rel"] >= 100.0 * big)
    out["n_big"] = n_big
    main = max(steps, key=lambda x: x["drop"])
    out["main_drop_k"] = main["k"] if main["drop_rel"] >= 100.0 * big else None
    if out["main_drop_k"] is None:
        out["verdict"] = "flat"          # 曲线几乎不随 k 变：这张图基本没有细结构
        return out
    if n_big >= 3:
        out["verdict"] = "uneven"        # 缝宽不均，单一 k 不可信
        return out
    # 陡降之后是否基本躺平
    after = [x for x in steps if x["k"] > main["k"]]
    rest = max((abs(x["drop_rel"]) for x in after), default=0.0)
    if rest <= 100.0 * big:
        out["verdict"] = "clear"
        out["suggest_k"] = int(main["k"]) + 1
        out["plateau"] = [int(main["k"]) + 1, int(curve[-1]["k"])]
    else:
        out["verdict"] = "uneven"
    return out


# v1.39：/api/kcurve（粒度谱）已下线，实现在 releases/v1.38_fix_kcurve.tgz 与
# 冻结备份 baige_线上备份_v1.37_20261030 里；_otsu_1d / _granulometry 两个纯函数保留。

def _build_overlay(sess: Session, work, samples, tol, purity, side, masks=None, sig=None,
                   view=None, full_size=None, line_px=0, mode="width") -> str:
    """把分类结果画在显示图上（红=露底、绿=完好），返回可访问路径

    masks 给了就只给选区内真正被分类的像素上色，其余（框外、未分类）全透明——
    千万别用外接矩形整块上色：两个斜对角的框会在中间糊出一块浅色，看着像把空白也算进去了。

    view 给了（选区原生模式）就把它贴回**整图**的显示画布：覆盖层的坐标口径永远和画布一致，
    前端不用知道分析是在哪块裁图上做的。"""
    bbox = _roi_bbox(work["rgb"].shape[:2], masks) if masks else None
    vkey = view.key if view is not None else None
    key = _overlay_key(sess, side, samples, tol, purity, sig, vkey, line_px, mode)
    path = os.path.join(OVERLAYS, key + ".png")
    if os.path.exists(path):
        return "api/overlay/%s.png" % key
    labels, _, _, _ = _labels_for(sess, work, tol, purity, samples, masks, sig, vkey)
    rgb = work["rgb"]
    h, w = rgb.shape[:2]
    fw, fh = full_size if full_size else (w, h)
    disp_side = min(1200, max(fh, fw))       # 覆盖层只是提示层
    sc = min(1.0, disp_side / float(max(fh, fw)))
    ow, oh = max(1, int(round(fw * sc))), max(1, int(round(fh * sc)))
    roles = [s["role"] for s in samples]
    # 画布：0 = 透明，1..n = 各角色色号；只在选区内上色，框外/未分类保持 0
    canvas = np.zeros((oh, ow), np.uint8)
    if view is None:
        a, bx, by = sc, 0.0, 0.0
    else:                                     # 视图 px → 整图显示 px
        a = view.s6 * sc / view.scale
        bx = view.ox * view.s6 * sc
        by = view.oy * view.s6 * sc
    pieces = masks if masks else [np.ones((h, w), bool)]
    sub_idx = [i for i, r in enumerate(roles) if r == cc.ROLE_SUB]
    for msk in pieces:
        if sub_idx:                       # 提示层会被缩到 ≤1200 px，1~2 px 的露底要先外扩 1 px 才看得见（不影响数字）
            lab2 = labels.copy()
            sm = np.isin(labels, sub_idx) & msk
            d = sm.copy()
            d[1:, :] |= sm[:-1, :]; d[:-1, :] |= sm[1:, :]
            d[:, 1:] |= sm[:, :-1]; d[:, :-1] |= sm[:, 1:]
            lab2[d] = sub_idx[0]
            _paste_codes(canvas, lab2, view, msk, a, bx, by, roles)
        else:
            _paste_codes(canvas, labels, view, msk, a, bx, by, roles)
    # 纯色半透明：底图透得出来，PNG 体积也小（平坦色块压缩率极高）
    out = np.zeros((oh, ow, 4), np.uint8)
    for code, (r_, g_, b_, a_) in ((1, (255, 40, 40, 200)), (2, (50, 210, 90, 110))):
        m = canvas == code
        out[m] = (r_, g_, b_, a_)
    Image.fromarray(out, "RGBA").save(path, "PNG", optimize=True)
    return "api/overlay/%s.png" % key


@app.get("/api/overlay/{key}.png")
def overlay(key: str):
    p = os.path.join(OVERLAYS, key + ".png")
    if not os.path.exists(p):
        raise HTTPException(404)
    return FileResponse(p, media_type="image/png", headers={"Cache-Control": "private, max-age=3600"})


def _margin_hint_en(pct):
    """cc.margin_hint() 的英文版（数字与口径逐项对应，别只翻文字）。"""
    parts = ["Margin: %.2f%% is %s" % (pct, cc.astm_class(pct))]
    for lim in cc.LINES:
        if pct < lim - 1e-9:
            parts.append("+%.2f pp (to %.1f%%) becomes %s"
                         % (lim - pct, lim, cc.astm_class(lim + 1e-6)))
            break
    for lim in reversed(cc.LINES):
        if pct >= lim - 1e-9:
            if lim == 0.5:
                parts.append("below %.1f%% becomes 5B (-%.2f pp)" % (0.5, pct - 0.5))
            else:
                parts.append("-%.2f pp (to %.0f%%) becomes %s"
                             % (pct - lim, lim, cc.astm_class(lim - 1e-6)))
            break
    return " | ".join(parts) + "\n"


def _margin_hint(pct):
    return L(cc.margin_hint(pct), _margin_hint_en(pct))


UNREL_UNASSIGNED = 12.0      # 未分类超过这个比例（占选区面积）→ 判级不可信
UNREL_COAT_SHARE = 5.0       # 漆膜占「漆膜+露底」低于这个比例 → 判级不可信


def _credibility(tot):
    """输入是否可信。返回 (ok, 理由)。

    两类情况会让等级看着"有个数"但根本没意义：
    · 未分类太高：吸的漆膜色没覆盖住真实漆膜（亮/暗档没吸全），分母里只剩基材；
    · 漆膜占比太低：多半是基材色吸到了背景，整片背景都算成露底。
    这时宁可不给等级，也别在徽章上写一个 0B 让人当真。
    """
    area = max(1, int(tot.get("area") or 0))
    un = 100.0 * float(tot.get("unassigned") or 0) / area
    cnt = float(tot.get("coat_px") or 0) + float(tot.get("sub_raw") or 0)
    coat_share = 100.0 * float(tot.get("coat_px") or 0) / max(1.0, cnt)
    bad = []
    if un > UNREL_UNASSIGNED:
        bad.append(L("未分类 %.0f%%（上限 %.0f%%）", "unclassified %.0f%% (limit %.0f%%)", un, UNREL_UNASSIGNED))
    if coat_share < UNREL_COAT_SHARE:
        bad.append(L("漆膜只占 %.0f%%（下限 %.0f%%）", "coating is only %.0f%% (floor %.0f%%)", coat_share, UNREL_COAT_SHARE))
    if not bad:
        return True, ""
    return False, L("、".join(bad) + "——多半是吸色不准或容差太小，先重吸一个颜色再判级",
                    "; ".join(bad) + " — the colour samples or tolerance are probably wrong; "
                    "re-sample a colour before trusting the grade")


def _lines_for(rows, tot, pct, tol, lo, hi, line_px, detail, diag, auto_warn="", purity=0.75):
    """复刻 GUI compute() 的输出文字"""
    lines = []
    for i, st in enumerate(rows, 1):
        lines.append(L("R%d   %5.2f%%   %s", "R%d   %5.2f%%   %s",
                       i, st["pct"], cc.astm_class(st["pct"])))
    if len(rows) > 1:
        lines.append("─" * 22)
        lines.append(L("合计 %5.2f%%   →  ASTM %s / ISO %s",
                       "Total %5.2f%%   →  ASTM %s / ISO %s",
                       pct, cc.astm_class(pct), _iso_class(pct)))
    if int(tot.get("sub_raw") or 0) == 0:
        lines.append(L("· 没检出露底（0 px）：要么漆膜完好（5B），要么基材色吸错 / 容差太小。"
                       "先看下面「未分类」——偏高就重吸一个基材点。",
                       "· No bare area detected (0 px): either the coating is intact (5B), or the substrate "
                       "colour is wrong / the tolerance too small. Check the unclassified share below — "
                       "if it is high, re-sample the substrate."))
    _ok, _why = _credibility(tot)
    if not _ok:
        lines.insert(0, L("⚠ 参数不可信：%s", "⚠ Parameters unreliable: %s", _why))
        lines.insert(1, "")
    lines.append(_margin_hint(pct).strip())
    # 未分类占比：只有两个角色以后，亮处漆膜不该被悄悄挖掉，得让用户看见
    try:
        _area = sum(int(r.get("area") or 0) for r in rows)
        _un = sum(int(r.get("unassigned") or 0) for r in rows)
        if _area and _un / float(_area) > 0.02:
            lines.append(L("· 选区内有 %.1f%% 的像素既不像漆膜也不像基材（未分类，不计入分子分母）。"
                           "如果那片其实是亮处/暗处的漆膜，再吸一个「漆膜」点把它们收进来，否则脱落率会偏高。",
                           "· %.1f%% of the pixels in the region match neither coat nor substrate "
                           "(unclassified, excluded from both numerator and denominator). If that area is "
                           "really coating under different light, sample one more coat point to include it — "
                           "otherwise the removal percentage reads high.",
                           100.0 * _un / _area))
    except Exception:
        pass
    if tot.get("thist"):
        seg = " | ".join(("%d-%dpx %.0f%%" % (a[0], a[1], a[3])) if a[1] < 10 ** 5
                         else ("≥%dpx %.0f%%" % (a[0], a[3])) for a in tot["thist"])
        lines.append(L("露底厚度：%s", "Bare-area thickness: %s", seg))
        thick = sum(a[3] for a in tot["thist"] if a[0] >= 7)
        if thick < 5 and tot["sub_raw"] > 0:
            lines.append(L("⚠ 几乎全部露底都薄于 7px —— 这些多半是刀缝，不是掉漆。"
                           "把「刀缝宽度/阈值」调到 7 以上再看，或换个真正的露底点重新吸色。",
                           "⚠ Nearly all bare pixels are thinner than 7px — most of them are cuts, "
                           "not coating removal. Raise the cut-width threshold above 7 and look again, "
                           "or sample a real bare spot instead."))

    if detail:
        lines.append(L("—— 诊断明细 ——", "— diagnostics —"))
        for i, st in enumerate(rows, 1):
            a = max(1, st["area"])
            cut = max(0, st["sub_raw"] - st["sub_open"])
            lines.append(L("R%d 面积 %d px｜漆膜 %.1f%%  露底 %.1f%%  刀缝排除 %.1f%%  "
                           "过渡 %.1f%%  未分类 %.1f%%",
                           "R%d area %d px | coat %.1f%%  bare %.1f%%  cut-excluded %.1f%%  "
                           "blend %.1f%%  unclassified %.1f%%",
                           i, st["area"], 100.0 * st["coat_px"] / a,
                           100.0 * st["sub_open"] / a, 100.0 * cut / a,
                           100.0 * st["trans"] / a, 100.0 * st["unassigned"] / a))
        raw = (100.0 * tot["sub_raw"] / (tot["coat_px"] + tot["sub_raw"])
               if (tot["coat_px"] + tot["sub_raw"]) else 0)
        lines.append(L("合计 漆膜 %d px / 基材 %d px；不排除刀缝时 %.2f%%",
                       "Total: coat %d px / substrate %d px; %.2f%% without cut exclusion",
                       tot["coat_px"], tot["sub_raw"], raw))
        if tot["trans"]:
            lines.append(L("过渡带（漆/基材混色，已排除）%d px，占选区 %.1f%%",
                           "Blend band (coat/substrate mixing, excluded) %d px, %.1f%% of the region",
                           tot["trans"], 100.0 * tot["trans"] / max(1, tot["area"])))
        if tot["n_line"]:
            lines.append(L("按形状识别线状刀缝 %d 段，排除 %.0f%% 的露底面积",
                           "Shape-based cut detection: %d strands, %.0f%% of bare area excluded",
                           tot["n_line"], 100.0 * tot["line_area"] / max(1, tot["sub_raw"])))
        lines.append(L("容差敏感性：%.1f%% ~ %.1f%%（容差 %.0f~%.0f）",
                       "Tolerance sensitivity: %.1f%% ~ %.1f%% (tol %.0f~%.0f)",
                       lo, hi, tol * 0.75, tol * 1.25))
        if diag:
            lines.append(L("色差分布：中位 %.0f / 90分位 %.0f / 99分位 %.0f；%s",
                           "Colour distance: median %.0f / p90 %.0f / p99 %.0f; %s",
                           diag["q50"], diag["q90"], diag["q99"], diag["sweep"]))

    if auto_warn:
        lines.append(auto_warn.strip())
    if purity <= 0.505 and int(line_px) <= 1 and tot["sub_raw"] > 0:
        lines.append(L("⚠ 纯度 0.50（不剔除过渡带）＋未排除刀缝：刀缝本身的露底会算进脱落，结果偏高。"
                       "本工具合成样图实测 13.6% 会读成 15.3%，足以把 3B 顶成 2B —— "
                       "要么到「量尺」页量一条真刀缝填进阈值，要么把纯度提到 0.75 左右。",
                       "⚠ Purity 0.50 (blend band kept) with no cut exclusion: the bare pixels inside the "
                       "cuts themselves count as removal, so the result reads high. On the built-in synthetic "
                       "sample 13.6% reads as 15.3% — enough to turn 3B into 2B. Either measure a real cut "
                       "on the ruler tab and set the threshold, or raise purity to about 0.75."))
    if cc.astm_class(lo) != cc.astm_class(hi):
        lines.append(L("⚠ 这一档不稳：容差稍动就会在 %s ~ %s 之间跳。",
                       "⚠ This grade is not stable: a small tolerance change flips it between %s and %s.",
                       cc.astm_class(lo), cc.astm_class(hi)))
    unass = 100.0 * tot["unassigned"] / max(1, tot["area"])
    if unass > 12:
        lines.append(L("⚠ 未分类 %.0f%%：容差偏小或颜色取样不全（多吸一两个漆膜明暗档）",
                       "⚠ Unclassified %.0f%%: tolerance too small or colours under-sampled "
                       "(sample one or two more coat shades)",
                       unass))
    if tot["sub_raw"] > 0 and tot["sub_open"] < 0.2 * tot["sub_raw"]:
        thick_share = None
        if tot.get("thist"):
            thick_share = sum(a[3] for a in tot["thist"] if a[0] >= 7)
        if thick_share is None or thick_share >= 5:
            raw = 100.0 * tot["sub_raw"] / max(1, tot["coat_px"] + tot["sub_raw"])
            lines.append(L("⚠ 刀缝阈值 %d px 可能太大：把细小的脱落也抹掉了（原始 %.1f%% → 现在 %.1f%%）",
                           "⚠ Cut threshold %d px may be too large: it also erases small removal "
                           "(raw %.1f%% → now %.1f%%)",
                           line_px, raw, pct))
    return lines


def _profile_full_width(rgb, p0, p1):
    """同一条线段上，除了半高宽(FWHM) 再算一个"基线全宽"：偏离 >10% 峰值的范围。
    刀缝是有软边的白线，FWHM 只到半高（偏窄 1.5~2 倍），全宽才接近肉眼看到的缝宽。"""
    prof, length = cc.sample_line(rgb, tuple(p0), tuple(p1))
    if prof is None or prof.size < 12:
        return None
    n = prof.size
    base = np.linspace(prof[0], prof[-1], n)
    dev = prof - base
    k = int(np.argmax(np.abs(dev)))
    if k <= 1 or k >= n - 2:
        return None
    sgn = 1.0 if dev[k] > 0 else -1.0
    lim = abs(dev[k]) * 0.10
    l = k
    while l > 0 and dev[l] * sgn > lim:
        l -= 1
    r = k
    while r < n - 1 and dev[r] * sgn > lim:
        r += 1
    return float(max(0.0, r - l) * length / (n - 1))


def _suggest_line(ruler_hist, thist, ruler_full=None):
    """刀缝阈值建议。
    - 量尺全宽（每像素单位与分析图一致）优先：缝宽 +1 就能整条吃掉。
    - 厚度分布只在"薄结构占主导"时才可信（90 分位 ≤10px）；成片脱落的图上
      90 分位会落在最后一档，+2 之后会把真实脱落一起删掉，所以必须弃用。
    - 两路都有就取大者。"""
    k_r = None
    src_r = ""
    vals = [v for v in (ruler_full or []) if v and v > 0]
    if vals:
        k_r = int(round(sum(vals) / len(vals))) + 1
        src_r = "按量尺全宽"
    elif ruler_hist:
        k_r = int(round((sum(ruler_hist) / len(ruler_hist)) * 1.8)) + 1
        src_r = "按量尺半高宽×1.8"
    k_t = None
    if thist:
        # 只看"薄结构"（≤10px）那一部分：厚的是成片脱落，不能用它的厚度推刀缝阈值。
        thin = [(lo, hi, pct) for lo, hi, c, pct in thist if hi <= 10]
        thin_share = sum(p[2] for p in thin)
        if thin and thin_share >= 50.0:      # 薄结构占一半以上，才认为刀缝主导
            cum, need = 0.0, 0.75 * thin_share
            for lo, hi, pct in thin:
                cum += pct
                if cum >= need:
                    k_t = hi + 2
                    break
    if k_r is not None and k_t is not None:
        return int(min(60, max(2, max(k_r, k_t)))), "量尺+厚度分布"
    if k_r is not None:
        return int(min(60, max(2, k_r))), src_r
    if k_t is not None:
        return int(min(60, max(2, k_t))), "按露底厚度分布"
    return None, ""


def _analyze_core(sess: Session, req, ruler_hist=None, auto_warn="", ruler_full=None):
    side = _res_to_side(req.res, sess.full_img())
    work6 = sess.work_img(side)              # 整图（显示/覆盖层用）
    work, view = _view_for(sess, req)        # 真正参与计算的图（默认就是 work6）
    k_view = max(1, int(round(view.len_in(req.line_px))))
    px_out = view.len_out(1.0)               # 视图 px → wire px
    samples = [s.model_dump() for s in req.samples]
    roles = _roles(samples)
    rois = [r.model_dump() for r in req.rois]
    if rois:
        rois_v = view.rois_in(rois)
    else:
        rois_v = []
    if not samples:
        return {"ok": False, "reason": L("请先吸两个颜色：①完好漆膜 ②露底基材",
                                         "Sample two colours first: (1) intact coating (2) exposed substrate"),
                "meta": _meta(sess, side)}
    if cc.ROLE_COAT not in roles or cc.ROLE_SUB not in roles:
        return {"ok": False, "reason": L("颜色角色不完整：至少需要一个「漆膜」和一个「基材」。",
                                             "Incomplete colour roles: at least one coat and one substrate "
                                             "sample are required."),
                "meta": _meta(sess, side)}

    shape2 = work["lab"].shape[:2]
    if rois_v:
        masks = _masks(shape2, rois_v)
    else:
        masks = [np.ones(shape2, bool)]

    bbox = _roi_bbox(shape2, masks) if rois_v else None
    msig = tuple((tuple(r["box"]), round(float(r.get("ang") or 0.0), 3)) for r in rois_v) if rois_v else None
    vkey = view.key
    labels, best, mind, t = _labels_for(sess, work, req.tol, req.purity, samples, masks, msig, vkey)
    rows, tot, pct = _stats(labels, masks, roles, k_view, req.mode, req.want_hist, px_out,
                     req.loc_tile, req.loc_factor)

    lo_labels, _, _, _ = _labels_for(sess, work, req.tol * 0.75, req.purity, samples, masks, msig, vkey)
    _, _, lo = _stats(lo_labels, masks, roles, k_view, req.mode, False, px_out,
                     req.loc_tile, req.loc_factor)
    # 缓存被上面改写，重算高容差需要原始 (best,mind,t)
    hi_labels = cc.labels_from_parts(best, mind, t, req.tol * 1.25, req.purity)
    _, _, hi = _stats(hi_labels, masks, roles, k_view, req.mode, False, px_out,
                     req.loc_tile, req.loc_factor)
    # 注意：_labels_for 内部缓存的是 best/mind/t，tol 改动不会丢，这里补回主标签
    labels = cc.labels_from_parts(best, mind, t, req.tol, req.purity)

    # 容差到底有没有用：只有 mind 落在 tol 附近的像素才会因为调 tol 改变分类，
    # 其余像素要么紧贴漆膜、要么紧贴基材，tol 怎么动都不影响。这里给前端一个判据。
    tol_eff = None
    try:
        _m_all = np.zeros(mind.shape, bool)
        for _m in masks:
            _m_all |= _m
        _mv = mind[_m_all]
        if _mv.size:
            # 判据用“结果差”：tol×0.75 与 tol×1.25 各算一遍露底率，差不动就是容差无效。
            # （只看落在边界带上的像素占比会误判：那些像素往往本来就归漆膜或未分类，不影响露底面积。）
            tol_eff = {"lo": round(float(lo), 4), "hi": round(float(hi), 4),
                       "delta": round(float(hi) - float(lo), 4),
                       "median": round(float(np.median(_mv)), 2),
                       "p90": round(float(np.percentile(_mv, 90)), 2),
                       "auto": round(float(_auto_tol_of(_span_of(samples))), 1)}
    except Exception:
        tol_eff = None

    diag = None
    if req.detail:
        m_all = np.zeros(mind.shape, bool)
        for m in masks:
            m_all |= m
        mv = mind[m_all]
        if mv.size:
            q = np.percentile(mv, [50, 90, 99])
            parts = [L("容差%.0f→未分类%.0f%%", "tol%.0f→unclassified %.0f%%",
                       req.tol * k, 100.0 * (mv > req.tol * k).mean())
                     for k in (0.6, 1.0, 1.5, 2.0)]
            diag = {"q50": float(q[0]), "q90": float(q[1]), "q99": float(q[2]),
                    "sweep": L("，", ", ").join(parts)}

    lines = _lines_for(rows, tot, pct, req.tol, lo, hi, req.line_px, req.detail, diag,
                       auto_warn, req.purity)
    # 多选区时，ASTM 是按"最差区域"评级的，合并面积算出来的 total 只能当参考
    worst = None
    if len(rows) > 1:
        wi = max(range(len(rows)), key=lambda i: rows[i].get("pct") or 0.0)
        wp = float(rows[wi].get("pct") or 0.0)
        worst = {"roi": wi + 1, "pct": wp, "astm": cc.astm_class(wp), "iso": _iso_class(wp)}
        # ★ 行不再写进结果文本：结果卡上已经有「R2 5.32%（最差）」和大判级
        # （worst 仍返回给前端卡片用）
    if ruler_hist is None:
        ruler_hist = getattr(req, "ruler_hist", None) or None
    if ruler_full is None:
        ruler_full = getattr(req, "ruler_full", None) or None
    sug, sug_src = (_suggest_line(ruler_hist, tot.get("thist"), ruler_full)
                    if getattr(req, "want_suggest", True) else (None, ""))
    # 建议值必须先试一遍：拿这个 k 真跑一次统计，如果它几乎把露底全删光（这把真实脱落也当刀缝），
    # 就别给建议——宁可让用户去量尺，也不能给一个把 3B 变成 0% 的数。
    if sug and sug > 1:
        k_sug = max(1, int(round(view.len_in(sug))))
        _, _tot_k, pct_k = _stats(labels, masks, roles, k_sug, req.mode, False, px_out,
                     req.loc_tile, req.loc_factor)
        # 划线交叉点比缝身宽 2~3 px：k 刚好等于缝宽时，交叉点那块白会留下来，被算成脱落，
        # 于是"肉眼 5B"被读成 4B。这里顺手把 k+2 也算一遍，差得多就明说出来。
        try:
            _, _t2, pct_k2 = _stats(labels, masks, roles,
                                    max(1, int(round(view.len_in(min(60, sug + 2))))),
                                    req.mode, False, px_out)
            if (req.mode == "width" and 0.15 < pct_k < 5.0 and pct_k2 < 0.6 * pct_k
                    and cc.astm_class(pct_k) != cc.astm_class(pct_k2)):
                lines.append(L("· 按建议 k=%d 排完还剩 %.2f%%（%s），几乎都卡在划线交叉点上；"
                               "交叉点比缝身宽 2 px 左右，把 k 提到 %d 就变 %.2f%%（%s）。"
                               "肉眼判级更接近后者。",
                               "· With the suggested k=%d, %.2f%% (%s) is left, almost all of it on cut "
                               "intersections: an intersection is about 2 px wider than the cut itself, so "
                               "raising k to %d gives %.2f%% (%s). Eyeballing matches the latter.",
                               sug, pct_k, cc.astm_class(pct_k),
                               min(60, sug + 2), pct_k2, cc.astm_class(pct_k2)))
        except Exception:
            pass
        if pct_k <= 0.05 * max(1e-9, pct):
            if not ruler_full:
                sug, sug_src = None, ""
                lines.append(L("· 露底厚度分布推不出可信的刀缝阈值（按它的建议排完只剩 %.2f%%，等于把真实"
                               "脱落一起删了），请到「量尺」页量一条刀缝。",
                               "· The bare-thickness distribution gives no trustworthy cut threshold "
                               "(excluding by its suggestion leaves %.2f%%, i.e. real removal is deleted "
                               "too) — measure a cut on the ruler tab.",
                               pct_k))
            else:
                lines.append(L("· 注意：按量尺全宽排完只剩 %.2f%%——这张图的脱落比刀缝还细，刀缝阈值"
                               "别照建议用（或改用「逐块确认」一块块看）。",
                               "· Note: excluding by the measured full width leaves only %.2f%% — here the "
                               "removal is finer than the cuts, so do not follow the suggestion (or review "
                               "block by block instead).",
                               pct_k))

    # 局部宽度口径：厚度只能取偶数（2×3×3 腐蚀步数），刀缝在 6MP 下常常只有 2~4 px，
    # 于是"排除到 2"还是"排除到 4"就决定了几倍差别 —— 直接给区间，别假装有精确值。
    if str(getattr(req, "mode", "")) == "local" and tot.get("loc_alt_open") is not None:
        try:
            den_local = tot["coat_px"] + tot["sub_raw"]
            if den_local > 0:
                alt_pct = 100.0 * tot["loc_alt_open"] / den_local
                med = tot.get("loc_med")
                lines.append(L("· 局部宽度对参数敏感：因子 %.1f → %.2f%%（%s），因子 2.2 → %.2f%%（%s）。"
                               "这个区间就是这张图的分辨极限%s —— 真值多半在中间，别只读一个数。",
                               "· Local width is parameter-sensitive: factor %.1f → %.2f%% (%s), factor 2.2 "
                               "→ %.2f%% (%s). That span is the resolution limit of this image%s — the truth "
                               "is usually in between; do not read a single number.",
                               float(getattr(req, "loc_factor", 1.6)), pct, cc.astm_class(pct),
                               alt_pct, cc.astm_class(alt_pct),
                               L("（基线缝宽中位 %.1f px）", " (median baseline cut width %.1f px)") % med
                               if med else ""))
        except Exception:
            pass

    # 旧的「按形状」对照提示已随该判法一起撤下（它会真的再跑一遍 shape 统计，白算）

    span = None
    coat = [l for l, r in zip(_sample_labs(samples), roles) if r == cc.ROLE_COAT]
    sub = [l for l, r in zip(_sample_labs(samples), roles) if r == cc.ROLE_SUB]
    if coat and sub:
        span = min(float(np.linalg.norm(c - s)) for c in coat for s in sub)

    _ok, _why = _credibility(tot)
    out = {
        "ok": True,
        "meta": _meta(sess, side),
        "analysis": {
            "roi_native": bool(view.native),
            "w": int(view.work["size"][0]), "h": int(view.work["size"][1]),
            "scale": round(view.scale / view.s6, 4),   # 视图相对 6 MP 分析图的线性倍率
            "px_out": round(px_out, 4),                # 视图 px → wire px
            "k_px": int(k_view),                       # 实际用于开运算的视图像素
            "k_wire": int(req.line_px),
            "note": (L("已启用「选区原生像素」：只分析框内 %.2f MP，未降采样（k 仍按画布/6MP 口径）",
                       "native-region pixels enabled: only %.2f MP inside the box is analysed, no "
                       "downscaling (k still in canvas/6MP units)",
                       view.work["size"][0] * view.work["size"][1] / 1e6)) if view.native else "",
        },
        "credible": _ok, "unreliable": _why, "algo": ALGO_VERSION,
        "rows": [_row_out(st) | {"astm": cc.astm_class(st["pct"]),
                                 "iso": _iso_class(st["pct"]),
                                 "margin": _margin_info(st["pct"]),
                                 "groove_px": st["sub_raw"] - st["sub_open"],
                                 "groove_share": 100.0 * (st["sub_raw"] - st["sub_open"]) / max(1, st["denom"])}
                 for st in rows],
        "total": {
            "pct": pct, "astm": cc.astm_class(pct), "iso": _iso_class(pct),
            "coat_px": tot["coat_px"], "sub_raw": tot["sub_raw"], "sub_open": tot["sub_open"],
            "area": tot["area"], "unassigned": tot["unassigned"], "trans": tot["trans"],
            "n_line": tot["n_line"], "line_area": tot["line_area"],
            "unassigned_pct": 100.0 * tot["unassigned"] / max(1, tot["area"]),
            # —— 以下为诊断字段（只加不改，v1 的 pct/判级口径完全不变）——
            "groove_px": tot["sub_raw"] - tot["sub_open"],
            "groove_share": 100.0 * (tot["sub_raw"] - tot["sub_open"]) / max(1, tot["coat_px"] + tot["sub_raw"]),
            "margin": _margin_info(pct),
            "algo": ALGO_VERSION,
            "worst": worst,
            "credible": _ok, "unreliable": _why,
        },
        "thist": tot.get("thist"),
        "sensitivity": {"lo": lo, "hi": hi, "lo_cls": cc.astm_class(lo), "hi_cls": cc.astm_class(hi)},
        "margin": _margin_hint(pct),
        "lines": lines,
        "span": span,
        "auto_tol": _auto_tol_of(span),
        "suggest_px": sug,
        "suggest_src": sug_src,
        "tol": req.tol,
        "tol_eff": tol_eff,
    }
    if req.overlay:
        try:
            out["overlay_url"] = _build_overlay(sess, view.work, samples, req.tol, req.purity, side,
                                                masks if rois_v else None, msig,
                                                view=view if view.native else None,
                                                full_size=work6["size"],
                                                line_px=req.line_px,
                                                mode=getattr(req, "mode", "width"))
        except Exception:
            out["overlay_url"] = None
    return out



THRESH = (0.5, 5.0, 15.0, 35.0, 65.0)


def _margin_info(pct):
    """距判级分界还有几个百分点——用来看这个数字稳不稳"""
    up = min([b for b in THRESH if b > pct + 1e-9], default=None)
    down = max([b for b in THRESH if b <= pct + 1e-9], default=None)
    return {"up": up, "up_pp": None if up is None else round(up - pct, 2),
            "down": down, "down_pp": None if down is None else round(pct - down, 2),
            "nearest_pp": round(min(abs(b - pct) for b in THRESH), 2)}


# ------------------------------------------------------- 软权重 + 区间（实验口径）
def _soft_block(sess: Session, req) -> dict:
    """按混色比 t 做软计数的替代口径——只在请求带 soft=true 时算，v1 的字段一个都不改。

    · 主值：每个像素按 t 计入（t=0 完好漆膜、t=1 纯露底），
            pct = Σt(删刀缝后) ÷ [Σt(删刀缝后) + Σ(1−t)(同选区其余像素)]
            —— 没有"纯度"硬阈值，所以不会因为 p 动 0.01 就跳档。
    · 下限：硬口径的露底像素 ÷ 选区面积（把"看不清"全当"没掉漆"）
    · 上限：(露底像素 + 不确定像素) ÷ 选区面积（把"看不清"全当"掉漆"）
    · unc_pct：不确定像素（未分类 + 过渡带）占选区面积的比例
    上下界跨档时 cross=True，表示这张图只够给"临界"结论。
    """
    side = _res_to_side(req.res, sess.full_img())
    work, view = _view_for(sess, req)
    samples = [s.model_dump() for s in req.samples]
    roles = _roles(samples)
    rois = [r.model_dump() for r in req.rois] if getattr(req, "rois", None) else []
    rois_v = view.rois_in(rois) if rois else []
    shape2 = work["lab"].shape[:2]
    masks = _masks(shape2, rois_v) if rois_v else None
    vkey = view.key
    labels, best, mind, t = _labels_for(sess, work, req.tol, req.purity, samples, masks,
                                        _msig(rois_v), vkey)
    if t is None:
        return {"ok": False, "reason": L("样本里缺少「漆膜」或「基材」，软权重算不了",
                                             "A coat or substrate sample is missing — soft weighting "
                                             "cannot run")}
    k = max(1, int(round(view.len_in(req.line_px))))
    shape2 = labels.shape[:2]
    masks = _masks(shape2, rois_v) if rois_v else [np.ones(shape2, bool)]

    rows, agg = [], {"A": 0, "D": 0, "U": 0, "num": 0.0, "den": 0.0}
    for m in masks:
        rr = np.flatnonzero(m.any(1))
        cc_ = np.flatnonzero(m.any(0))
        if not rr.size or not cc_.size:
            continue
        r0, r1, c0, c1 = rr[0], rr[-1] + 1, cc_[0], cc_[-1] + 1
        mb = m[r0:r1, c0:c1]
        # 软权重不依赖「露底纯度」：掩膜只按 t>0.5 切，过渡带一律不剔除，
        # 所以拖动纯度滑块不会改变这里的任何数字（实测 p=0.5~0.9 完全不动）
        tt = np.clip(t[r0:r1, c0:c1].astype(np.float32), 0.0, 1.0)
        subc = mb & (tt > 0.5)
        kept = cc.morph_open(subc, k) if k > 1 else subc
        ok = mb & (mind[r0:r1, c0:c1] <= req.tol)
        num = float(tt[kept].sum())
        den = num + float((1.0 - tt)[ok & ~kept].sum())
        pct = 100.0 * num / den if den else 0.0
        A = int(m.sum())
        D = int(kept.sum())
        # 不确定 = 颜色超出容差 或 混色比落在 0.25~0.75（固定带，不跟纯度走）
        unc = ((mind[r0:r1, c0:c1] > req.tol) | ((tt > 0.25) & (tt < 0.75))) & mb
        U = int(unc.sum())
        rows.append({"pct": round(pct, 2), "astm": cc.astm_class(pct), "iso": _iso_class(pct),
                     "lo": round(100.0 * D / A, 2) if A else 0.0,
                     "hi": round(100.0 * (D + U) / A, 2) if A else 0.0,
                     "unc_pct": round(100.0 * U / A, 2) if A else 0.0})
        for key in ("A", "D", "U"):
            agg[key] += {"A": A, "D": D, "U": U}[key]
        agg["num"] += num
        agg["den"] += den

    pct = 100.0 * agg["num"] / agg["den"] if agg["den"] else 0.0
    A = max(1, agg["A"])
    lo = 100.0 * agg["D"] / A
    hi = 100.0 * (agg["D"] + agg["U"]) / A
    g_lo, g_hi = cc.astm_class(lo), cc.astm_class(hi)
    return {
        "ok": True,
        "pct": round(pct, 2), "astm": cc.astm_class(pct), "iso": _iso_class(pct),
        "lo": round(lo, 2), "hi": round(hi, 2),
        "lo_cls": g_lo, "hi_cls": g_hi,
        "cross": g_lo != g_hi,
        "unc_pct": round(100.0 * agg["U"] / A, 2),
        "rows": rows,
    }


# ------------------------------------------------- 露底块（逐块确认，v1.10）
def _components(mask: np.ndarray, min_area: int = 30, cap: int = 240):
    """8 邻域连通域：返回 [(面积, 质心x, 质心y, x0,y0,x1,y1)]，按面积降序，最多 cap 个。
    纯 BFS，不用 scipy（现役依赖只有 numpy + PIL）。"""
    from collections import deque
    h, w = mask.shape
    seen = np.zeros((h, w), bool)
    ys, xs = np.nonzero(mask)
    out = []
    for sy, sx in zip(ys.tolist(), xs.tolist()):
        if seen[sy, sx]:
            continue
        seen[sy, sx] = True
        q = deque([(sy, sx)])
        area = 0
        sy_sum = sx_sum = 0
        x0 = x1 = sx
        y0 = y1 = sy
        while q:
            y, x = q.popleft()
            area += 1
            sy_sum += y
            sx_sum += x
            if x < x0: x0 = x
            elif x > x1: x1 = x
            if y < y0: y0 = y
            elif y > y1: y1 = y
            for ny in (y - 1, y, y + 1):
                if ny < 0 or ny >= h:
                    continue
                row = mask[ny]
                rs = seen[ny]
                for nx in (x - 1, x, x + 1):
                    if 0 <= nx < w and row[nx] and not rs[nx]:
                        rs[nx] = True
                        q.append((ny, nx))
        if area >= min_area:
            out.append((area, sx_sum / area, sy_sum / area, x0, y0, x1, y1))
    out.sort(key=lambda t: -t[0])
    return out[:cap], sum(a for a, *_ in out[cap:])



def _components_fast(mask: np.ndarray, min_area: int = 30, cap: int = 240):
    """8 邻域连通域（行程 + 并查集），返回与 _components 同结构、同口径。

    原 BFS 版逐像素入队，在 6MP 的选区上要 2~3 秒（占 /api/blocks 总耗时 90%）；
    这版按行游程合并，快一个数量级。8 邻接靠「行间游程外扩 1 px」判断。
    """
    h, w = mask.shape
    runs, rows = [], []
    for y in range(h):
        row = mask[y]
        if not row.any():
            rows.append(())
            continue
        d = np.diff(np.concatenate(([0], row.view(np.int8), [0])))
        ss = np.flatnonzero(d == 1).tolist()
        ee = np.flatnonzero(d == -1).tolist()
        cur = []
        for s0, e0 in zip(ss, ee):
            cur.append((s0, e0, len(runs)))
            runs.append((y, s0, e0))
        rows.append(cur)
    n = len(runs)
    if not n:
        return [], 0
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for y in range(1, h):
        prev, cur = rows[y - 1], rows[y]
        if not prev or not cur:
            continue
        i = j = 0
        while i < len(prev) and j < len(cur):
            ps, pe, pi = prev[i]
            cs, ce, ci = cur[j]
            if pe < cs - 1:
                i += 1
            elif ce < ps - 1:
                j += 1
            else:
                ra, rb = find(pi), find(ci)
                if ra != rb:
                    parent[rb] = ra
                if pe < ce:
                    i += 1
                else:
                    j += 1
    stats = {}
    for i, (y, s0, e0) in enumerate(runs):
        r = find(i)
        cnt = e0 - s0
        st = stats.get(r)
        if st is None:
            st = stats[r] = [0, s0, e0 - 1, y, y, 0.0, 0.0]
        st[0] += cnt
        if s0 < st[1]:
            st[1] = s0
        if e0 - 1 > st[2]:
            st[2] = e0 - 1
        if y > st[4]:
            st[4] = y
        st[5] += cnt * ((s0 + e0 - 1) / 2.0)
        st[6] += cnt * y
    out = [(st[0], st[5] / st[0], st[6] / st[0], st[1], st[3], st[2], st[4])
           for st in stats.values() if st[0] >= min_area]
    out.sort(key=lambda t: -t[0])
    return out[:cap], sum(t[0] for t in out[cap:])


def _rdp(pts, eps=1.0):
    """Douglas-Peucker 简化折线（纯 python，点几十到几百个，够快）"""
    if len(pts) < 3:
        return pts
    import math as _m
    ax, ay = pts[0]; bx, by = pts[-1]
    dx, dy = bx - ax, by - ay
    n = _m.hypot(dx, dy) or 1e-9
    imax, dmax = 0, -1.0
    for i in range(1, len(pts) - 1):
        px, py = pts[i]
        d = abs(dy * px - dx * py + bx * ay - by * ax) / n
        if d > dmax:
            imax, dmax = i, d
    if dmax > eps:
        return _rdp(pts[:imax + 1], eps)[:-1] + _rdp(pts[imax:], eps)
    return [pts[0], pts[-1]]


_NB = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]   # 顺时针，0=北


def _crack_loops(mask):
    """沿像素边（crack）描出所有闭合环：外轮廓 + 每个孔。

    比 marching squares 好在：顶点落在像素角上，**多边形面积与像素计数严格相等**
    （实测方块 144/144、L 120/120、U 268/268、带孔环 484-100=384/384），
    蚂蚁线也正好压在块边界上，不会有半像素内缩。
    方向统一为「内部在左手侧」，环拼接时按 右转 > 直行 > 左转 选边，
    鞍点处不会把对角的两块连成一条线。
    """
    h, w = mask.shape
    if h < 1 or w < 1:
        return []
    m = np.zeros((h + 2, w + 2), bool)
    m[1:-1, 1:-1] = mask
    up = np.zeros_like(m); up[1:] = m[:-1]
    dn = np.zeros_like(m); dn[:-1] = m[1:]
    lf = np.zeros_like(m); lf[:, 1:] = m[:, :-1]
    rt = np.zeros_like(m); rt[:, :-1] = m[:, 1:]

    E = {}

    def add(k1, k2):
        E.setdefault(k1, []).append(k2)

    for y, x in zip(*[a.tolist() for a in np.nonzero(m & ~up)]):
        add((x + 1, y), (x, y))              # 上边 → 往西
    for y, x in zip(*[a.tolist() for a in np.nonzero(m & ~rt)]):
        add((x + 1, y + 1), (x + 1, y))      # 右边 → 往北
    for y, x in zip(*[a.tolist() for a in np.nonzero(m & ~dn)]):
        add((x, y + 1), (x + 1, y + 1))      # 下边 → 往东
    for y, x in zip(*[a.tolist() for a in np.nonzero(m & ~lf)]):
        add((x, y), (x, y + 1))              # 左边 → 往南

    used = set()
    loops = []
    for st in list(E.keys()):
        while any((st, n) not in used for n in E[st]):
            cur, prev, loop = st, None, [st]
            while True:
                cand = [n for n in E.get(cur, ()) if (cur, n) not in used]
                if not cand:
                    break
                if len(cand) > 1 and prev is not None:
                    d = (cur[0] - prev[0], cur[1] - prev[1])
                    order = [(d[1], -d[0]), d, (-d[1], d[0])]   # 右转 / 直行 / 左转
                    cand.sort(key=lambda n: order.index((n[0] - cur[0], n[1] - cur[1]))
                              if (n[0] - cur[0], n[1] - cur[1]) in order else 9)
                nxt = cand[0]
                used.add((cur, nxt))
                loop.append(nxt)
                prev, cur = cur, nxt
                if nxt == st:
                    break
            if len(loop) > 3:
                loops.append(loop)
    # padding 去掉 1，坐标回到 mask 局部
    return [[(x - 1, y - 1) for (x, y) in lp] for lp in loops]


def _poly_area(loop):
    s = 0.0
    n = len(loop)
    for i in range(n):
        x1, y1 = loop[i]
        x2, y2 = loop[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


HOLE_MIN_AREA = 40.0     # 视图口径像素²：比这更小的孔描出来全是噪点，画上去只会糊成一片
HOLE_MAX = int(os.environ.get("BAIGE_HOLE_MAX", "200"))  # 每块最多描这么多孔（按面积从大到小取；实测 40→200 只多 5 KB）


def _simplify_pts(pts, eps):
    """折线抽稀（RDP）。首尾不重复；抽完太小就返回空。"""
    pts = [(float(x), float(y)) for (x, y) in pts]
    if len(pts) > 3 and pts[0] == pts[-1]:
        pts = pts[:-1]
    if len(pts) < 3:
        return []
    k = min(len(pts) // 2, max(1, int(np.argmax([abs(q[0] - pts[0][0]) + abs(q[1] - pts[0][1])
                                                 for q in pts]))))
    out = _rdp(pts[:k + 1], eps)[:-1] + _rdp(pts[k:], eps)
    bx = [q[0] for q in out]; by = [q[1] for q in out]
    if len(out) < 3 or (max(bx) - min(bx)) < 1.5 or (max(by) - min(by)) < 1.5:
        return []
    return [(round(float(x), 1), round(float(y), 1)) for x, y in out]


def _trace_loops(mask, eps=None):
    """块的全套轮廓：(外轮廓, [孔轮廓...])，mask 局部坐标、浮点、首尾不重复。

    以前只回外轮廓——一块成片脱落里那几十上百个“没掉干净的岛”，蚂蚁线只在外圈绕一圈，
    中间的洞一个都不描（用户看着就是“覆盖不到那些洞”）。
    `_crack_loops` 本来就把所有环都算出来了，孔是白捡的：多花的只是抽稀 + 序列化。
    """
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return [], []
    if eps is None:      # 小块要准（屏幕上就几十像素），大块可以粗（点数少一半，画得快）
        eps = 1.0 if len(ys) < 5000 else 1.8
    loops = _crack_loops(mask)
    box = [(float(xs.min()), float(ys.min())), (float(xs.max() + 1), float(ys.min())),
           (float(xs.max() + 1), float(ys.max() + 1)), (float(xs.min()), float(ys.max() + 1))]
    if not loops:
        return [(round(x, 1), round(y, 1)) for x, y in box], []
    loops = sorted(loops, key=_poly_area, reverse=True)
    outer = _simplify_pts(loops[0], eps) or [(round(x, 1), round(y, 1)) for x, y in box]
    holes = []
    if os.environ.get("BAIGE_NO_HOLES"):     # 对比实测用：关掉孔洞，只留外轮廓
        return outer, []
    for lp in loops[1:HOLE_MAX + 1]:
        if _poly_area(lp) < HOLE_MIN_AREA:
            break                                  # 已按面积降序：后面的更小，直接停
        simple = _simplify_pts(lp, eps)
        if simple:
            holes.append(simple)
    if os.environ.get("BAIGE_TRACE_DEBUG"):
        print("TRACE loops=%d outer=%d holes=%d hole_pts=%d px=%d" %
              (len(loops), len(outer), len(holes), sum(len(h) for h in holes), int(mask.sum())), flush=True)
    return outer, holes


def _trace_outline(mask, eps=None):
    """只要外轮廓（旧接口，导出/兼容用）。"""
    return _trace_loops(mask, eps)[0]


def _contour_png(keep, samples, view, req):
    """把「留下来的露底」画成一张只有细轮廓的透明 PNG（≤1200 px），画布直接贴上去。

    和导出图同一套画法（内 1 px 强调色 + 外 1 px 暗边），所以画布和导出看到的是同一个东西。
    强调色按样品配色挑：红膜上不用红线。"""
    import hashlib
    if keep is None or not keep.any():
        return None
    ACC = _accent_rgb(samples)
    h, w = keep.shape[:2]
    sc = min(1.0, 1200.0 / float(max(h, w)))
    try:
        rsig = [(round(float(x), 1), float(r.get("ang") or 0.0)) for r in
                (y.model_dump() for y in req.rois) for x in [r["box"]]]
    except Exception:
        rsig = []
    sig = "c|%s|%d|%s|%s|%s|%s" % (ACC, int(keep.sum()), (h, w),
                                   getattr(view, "key", ""), req.line_px, req.mode)
    if rsig:
        sig += "|" + repr(rsig)
    key = "c" + hashlib.md5(sig.encode()).hexdigest()[:16]
    path = os.path.join(OVERLAYS, key + ".png")
    if not os.path.exists(path):
        if sc < 1.0:
            small = np.asarray(Image.fromarray(keep.astype(np.uint8) * 255).resize(
                (max(1, int(round(w * sc))), max(1, int(round(h * sc)))), Image.NEAREST)) > 127
        else:
            small = keep
        inner = small & (_cheb_dist(small, 3) <= 1)
        outer = (~small) & (_cheb_dist(~small, 3) <= 1)
        out = np.zeros(small.shape + (4,), np.uint8)
        out[inner] = tuple(ACC) + (235,)
        out[outer] = (10, 10, 14, 170)
        Image.fromarray(out, "RGBA").save(path, "PNG", optimize=True)
    return "api/overlay/%s.png" % key


@app.post("/api/blocks")
def blocks(req: BlocksReq):
    """把“删掉刀缝后剩下的露底”拆成连通块，交给用户逐块确认（排除误判的刀缝段）。
    只返回几何量，前端点选后自己算面积，不需要来回请求。"""
    sess = get_session(req.id)
    side = _res_to_side(req.res, sess.full_img())
    with HEAVY_BLOCK:
        with sess.lock:
            work, view = _view_for(sess, req)
            samples = [s.model_dump() for s in req.samples]
            roles = _roles(samples)
            rois_ = [r.model_dump() for r in req.rois]
            rois_v = view.rois_in(rois_) if rois_ else []
            masks_ = _masks(work["lab"].shape[:2], rois_v) if rois_v else None
            labels, best, mind, t = _labels_for(sess, work, req.tol, req.purity, samples, masks_,
                                                _msig(rois_v), view.key)
        shape2 = labels.shape[:2]
        masks = _masks(shape2, rois_v) if rois_v else [np.ones(shape2, bool)]

        k = max(1, int(round(view.len_in(req.line_px))))
        _prof = os.environ.get("BAIGE_PROF")
        _t0 = time.time()
        keep_all = np.zeros(shape2, bool)       # 视图口径的全部块像素（画布轮廓图用）
        out = []
        for m in masks:
            rr = np.flatnonzero(m.any(1))
            cc_ = np.flatnonzero(m.any(0))
            if not rr.size or not cc_.size:
                out.append({"A": 0, "kept": 0, "denom": 0, "blocks": [], "others": 0})
                continue
            r0, r1, c0, c1 = rr[0], rr[-1] + 1, cc_[0], cc_[-1] + 1
            mb = m[r0:r1, c0:c1]
            subc = np.zeros(mb.shape, bool)
            for j, role in enumerate(roles):
                if role == cc.ROLE_SUB:
                    subc |= (labels[r0:r1, c0:c1] == j)
            subc &= mb
            kept, _n_line, _line_area, _loc = cc.keep_mask(subc, k, req.mode,
                                                           int(req.loc_tile), float(req.loc_factor))
            _th = time.time()
            comps, rest = _components_fast(kept, int(req.min_area), int(req.max_blocks))
            if _prof:
                print("PROF comps %.0f ms" % ((time.time() - _th) * 1000), flush=True)
            if os.environ.get("BAIGE_PROF_CHK"):
                slow, rest2 = _components(kept, int(req.min_area), int(req.max_blocks))
                same_n = len(slow) == len(comps)
                same_a = same_n and all(abs(a[0] - b[0]) <= 2 for a, b in zip(slow, comps))
                print("CHK fast=%d slow=%d 块数一致=%s 面积一致=%s" % (len(comps), len(slow), same_n, same_a), flush=True)
            for (ar_, _cx_, _cy_, px0, py0, px1, py1) in comps:
                keep_all[r0 + py0:r0 + py1 + 1, c0 + px0:c0 + px1 + 1] |= \
                    kept[py0:py1 + 1, px0:px1 + 1]
            coat_idx = [j for j, role in enumerate(roles) if role == cc.ROLE_COAT]
            coat_px = 0
            for j in coat_idx:
                coat_px += int((labels[r0:r1, c0:c1] == j)[mb].sum())
            sub_raw = int(subc.sum())
            bl = []
            for i, (a, cx, cy, x0, y0, x1, y1) in enumerate(comps):
                wx, wy = view.to_wire(cx + c0, cy + r0)       # 块坐标一律报 wire（画布口径）
                bx0, by0 = view.to_wire(x0 + c0, y0 + r0)
                bx1, by1 = view.to_wire(x1 + 1 + c0, y1 + 1 + r0)
                sub_b = kept[y0:y1 + 1, x0:x1 + 1]
                _outer, _holes = _trace_loops(sub_b)
                ol = [view.to_wire(px + x0 + c0, py + y0 + r0) for (px, py) in _outer]
                holes_w = [[[round(x, 1), round(y, 1)]
                            for (x, y) in [view.to_wire(px + x0 + c0, py + y0 + r0)
                                           for (px, py) in h]] for h in _holes]
                bl.append({"i": i, "area": a, "cx": round(wx, 1), "cy": round(wy, 1),
                           "box": [int(round(bx0)), int(round(by0)),
                                   int(round(bx1)), int(round(by1))],
                           "outline": [[round(x, 1), round(y, 1)] for (x, y) in ol],
                           "holes": holes_w})
            if _prof: print("PROF roi %.0f ms (含 trace/outline)" % ((time.time() - _th) * 1000), flush=True)
            out.append({"A": int(m.sum()), "kept": int(kept.sum()), "denom": coat_px + sub_raw,
                        "coat": coat_px, "sub_raw": sub_raw,
                        "blocks": bl, "others": int(rest)})
    return {"ok": True, "k": int(req.line_px), "k_px": k, "mode": req.mode, "rois": out,
            "contour_url": _contour_png(keep_all, samples, view, req)}


# --------------------------------------------------- 自动定刀缝（v1.74 实验）
class KneeReq(BlocksReq):
    k_max: int = 16
    rule: str = "area"          # area=按露底突增；count=按块数回落


@app.post("/api/knee")
def knee(req: KneeReq):
    """扫 k=1..k_max，找出“再小 1 px 块就大量冒出”的那一档。

    用户的手工判据：往下调 1 px 时块突然大批出现 → 说明那一档正好把刀缝全放出来了，
    所以往回调 1 px 就是刀缝宽度。这里把同样的动作算一遍：
    相邻两档之间块数成倍跳（且露底占比也跳）→ 取跳变的上侧那一档。
    """
    sess = get_session(req.id)
    with HEAVY_BLOCK:
        with sess.lock:
            work, view = _view_for(sess, req)
            samples = [s.model_dump() for s in req.samples]
            roles = _roles(samples)
            rois_ = [r.model_dump() for r in req.rois]
            rois_v = view.rois_in(rois_) if rois_ else []
            masks_ = _masks(work["lab"].shape[:2], rois_v) if rois_v else None
            labels, best, mind, t = _labels_for(sess, work, req.tol, req.purity, samples, masks_,
                                                _msig(rois_v), view.key)
        shape2 = labels.shape[:2]
        masks = _masks(shape2, rois_v) if rois_v else [np.ones(shape2, bool)]
        pieces = []
        for m in masks:
            rr = np.flatnonzero(m.any(1))
            cc_ = np.flatnonzero(m.any(0))
            if not rr.size or not cc_.size:
                pieces.append(None)
                continue
            r0, r1, c0, c1 = rr[0], rr[-1] + 1, cc_[0], cc_[-1] + 1
            mb = m[r0:r1, c0:c1]
            subc = np.zeros(mb.shape, bool)
            coat = 0
            for j, role in enumerate(roles):
                blk = (labels[r0:r1, c0:c1] == j)
                if role == cc.ROLE_SUB:
                    subc |= blk
                elif role == cc.ROLE_COAT:
                    coat += int(blk[mb].sum())
            subc &= mb
            pieces.append((subc, coat + int(subc.sum())))
        k_max = max(3, min(int(req.k_max), 24))
        # 直接按界面口径（wire px = 滑块上的整数）扫，前端拿到就能用；
        # 视图 px 由 len_in 换算，相邻整数 wire 值有时落在同一视图档，复用上一个结果。
        rows = []
        cache = {}
        for kw in range(1, k_max + 1):
            kv = max(1, int(round(view.len_in(kw))))
            if kv in cache:
                r0 = cache[kv]
                rows.append({"k": kw, "k_view": kv, "pct": r0["pct"], "n": r0["n"]})
                continue
            num = den = 0
            nb = 0
            for p in pieces:
                if p is None:
                    continue
                subc, d = p
                kept, _a, _b, _c = cc.keep_mask(subc, kv, "width")
                # 必须和 /api/blocks 用同一个连通域实现（8 邻域），否则这里的块数
                # 会和「块确认」页的数字对不上（4 邻域会把斜向相接的块拆成两个）
                cps = _components_fast(kept, int(req.min_area), int(req.max_blocks))[0]
                nb += len(cps)
                num += int(kept.sum())
                den += d
            r0 = {"k": kw, "k_view": kv,
                  "pct": round(100.0 * num / den, 3) if den else 0.0, "n": nb}
            cache[kv] = r0
            rows.append(r0)

    # 判据一（area）：把「k 降 1 → 露底涨多少」排成台阶，取最后一个显著台阶所在那一档。
    best_a, why_a, steps = None, "", []
    for i in range(1, len(rows)):
        prev, cur = rows[i - 1], rows[i]
        steps.append((cur, prev, prev["pct"] - cur["pct"]))
    if steps:
        mx = max(t[2] for t in steps)
        thr = max(0.5, 0.25 * mx)
        sig = [t for t in steps if t[2] >= thr]
        if sig:
            cur, prev, d = sig[-1]
            best_a = cur
            why_a = L("k=%d：再降到 %d，露底会从 %.2f%% 涨到 %.2f%%（+%.2f 个百分点，块数 %d→%d）；"
                      "再往上调就只是慢慢收缩",
                      "k=%d: one step down to %d makes the bare area jump from %.2f%% to %.2f%% "
                      "(+%.2f pp, blocks %d→%d); going up only shrinks slowly",
                      cur["k"], prev["k"], cur["pct"], prev["pct"], d, cur["n"], prev["n"])

    # 判据二（count）：块数 低→峰值→第二高→回落。
    #   ① 找块数峰值（它标志刀缝被拆开的那一刻）；
    #   ② 过峰值后第一档 n ≤ 25% 峰值（块数确实回落到低位）；
    #   ③ 只要「进这一档时露底还掉了 ≥2 个百分点」，说明还没走平，继续往下一档；
    #   ④ 走平时停下的那一档就是刀缝宽度。
    # 相邻 wire k 常落在同一视图档（_sq_min 把偶数当奇数+1），先按"值真的变了"去重。
    # 标定过的 5 条（都是 6MP 口径）：C9 YS→8、C1 NOX→10、C1 YS→8、test→8、PXL…101439799→8。
    levels = []
    for r in rows:
        if not levels or r["pct"] != levels[-1]["pct"]:
            levels.append(r)
    best_c, why_c = None, ""
    if levels:
        pk = max(range(len(levels)), key=lambda i: levels[i]["n"])
        peak = levels[pk]["n"]
        if peak > 0:
            k0 = next((i for i in range(pk + 1, len(levels)) if levels[i]["n"] <= 0.25 * peak), None)
            if k0 is None:
                why_c = L("块数一直没从峰值 %d 回落到 25%% 以内",
                          "the block count never falls back within 25%% of the peak %d", peak)
            else:
                i = k0
                # 「进这一档」的露底跌幅 = 上一档 - 本档；还在大跌说明没走平，继续往下一档
                while i + 1 < len(levels) and (levels[i - 1]["pct"] - levels[i]["pct"]) >= 2.0:
                    i += 1
                best_c = levels[i]
                why_c = L("块数峰值 %d（k=%d）→ 第一档回落到 25%%（≤%.0f 块）是 k=%d（%d 块）；"
                          "再看露底：进这一档掉了 %.2f 个百分点，%s → 取 k=%d（%d 块，露底 %.2f%%）",
                          "peak %d blocks (k=%d) → first step within 25%% (≤%.0f blocks) is k=%d (%d blocks); "
                          "bare area drops %.2f pp into it, %s → using k=%d (%d blocks, bare %.2f%%)",
                          peak, levels[pk]["k"], 0.25 * peak, levels[k0]["k"], levels[k0]["n"],
                          (levels[k0 - 1]["pct"] - levels[k0]["pct"]) if k0 else 0.0,
                          L("继续降", "still falling") if i != k0 else L("已经走平", "levelled off"),
                          levels[i]["k"], levels[i]["n"], levels[i]["pct"])
    # 块数回落没找到台阶 → 回落到「露底厚度分布」（和按宽度模式取初值是同一个来源）。
    # 只在 count 判据失败时触发，那 5 条标定记录走的还是原路径，不受影响。
    k_from = "rule"
    if req.rule == "count" and best_c is None:
        sug, sug_src, why_sug = None, "", ""
        try:
            px_k = view.len_out(1.0)
            _rs, _tot_h, _ = _stats(labels, masks, roles, 1, "width", True, px_k)
            sug, sug_src = _suggest_line(None, _tot_h.get("thist"), getattr(req, "ruler_full", None) or None)
        except Exception:
            sug, sug_src = None, ""
        if sug and sug > 1:
            pct1 = rows[0]["pct"] if rows else 0.0
            cand = next((x for x in rows if x["k"] >= int(sug)), rows[-1] if rows else None)
            if cand is not None and cand["pct"] <= 0.05 * max(1e-9, pct1):
                why_sug = L("；但按建议 k=%d 排完只剩 %.2f%%（会把真实脱落一起删掉），不给",
                            "; but excluding at the suggested k=%d leaves only %.2f%% (real removal deleted), "
                            "so it is not offered", cand["k"], cand["pct"])
                sug = None
        if sug and sug > 1:
            cand = next((x for x in rows if x["k"] >= int(sug)), rows[-1])
            best_c = cand
            k_from = "suggest"
            why_c = L("%s；改用露底厚度分布给的建议 k=%d（%s，露底 %.2f%%）。"
                      "这只是个兜底值，建议在曲线上点一档确认。",
                      "%s; fell back to the bare-thickness suggestion k=%d (%s, bare %.2f%%). "
                      "That is only a fallback — confirm it by clicking a row on the curve.",
                      why_c or L("块数没回落", "block count did not step down"),
                      cand["k"], L(sug_src, ({"按露底厚度": "from bare thickness",
                                              "量尺全宽": "from the measured full width"}.get(sug_src, sug_src))) or L("按露底厚度", "from bare thickness"),
                      cand["pct"])
        else:
            why_c = (why_c or L("块数没回落", "block count did not step down")) + \
                    (why_sug or L("；露底厚度分布也给不出可信阈值",
                                  "; the bare-thickness distribution gives no trustworthy threshold"))

    if best_a is None and not why_a:
        why_a = L("曲线全程露底几乎没变化（%.2f%%~%.2f%%）：选不出台阶，通常是漆膜完好，或吸色/容差不对",
                  "the bare area hardly changes across the whole curve (%.2f%%–%.2f%%): no step to pick, "
                  "usually an intact coating or a colour/tolerance problem",
                  rows[0]["pct"] if rows else 0.0, rows[-1]["pct"] if rows else 0.0)
    pick = best_c if req.rule == "count" else best_a
    # 按露底分布取到扫描上限（k=k_max）= 曲线一路只升不降，这个边界值不算「定出」：
    # 否则界面上会出现一个看着像结果的 16，用户没法分辨是算出来的还是兜底的。
    area_boundary = (req.rule != "count" and pick is not None and int(pick["k"]) >= int(req.k_max))
    if area_boundary:
        pick = None
    # 两种判据都没定出 k 时，给一条与判法无关的常驻提示（前端顶栏黄标 + 量尺页黄条都用它）。
    # 以前只有 rule=count 分支会写警告，而判法=按露底分布时 rule=area → 一句提示都没有。
    no_bare = (not rows) or all((x.get("pct") or 0.0) <= 1e-9 for x in rows)
    _order = {"5B": 5, "4B": 4, "3B": 3, "2B": 2, "1B": 1, "0B": 0}
    class_span = sorted({cc.astm_class(x["pct"]) for x in rows}, key=lambda c: -_order.get(c, 0))
    same_band = len(class_span) <= 1
    _rule_name = (L("按块数回落", "by block count") if req.rule == "count"
                  else L("按露底分布", "by bare area"))
    _scan = L("已扫 k=1…%d（%s）", "scanned k=1…%d (%s)", int(req.k_max), _rule_name)
    warn = ""
    if pick is None:
        _pk, _last = None, None
        _rows_n = [x for x in rows if isinstance(x.get("n"), (int, float))]
        if _rows_n:
            _pk = max(_rows_n, key=lambda x: x["n"])
            _last = _rows_n[-1]
        if no_bare:
            warn = L("⚠ %s：没检出露底（0 px）——要么漆膜完好（5B），要么基材色吸错 / 容差太小。"
                     "先看「未分类」占比，偏高就重吸一个基材点。", 
                     "⚠ %s: no bare area detected (0 px) — either the coating is intact (5B), or the substrate "
                     "colour is wrong / the tolerance too small. Check the unclassified share; if it is high, "
                     "re-sample the substrate.", _scan)
        elif area_boundary:
            warn = L("⚠ %s：露底曲线一路只升不降，最大档 k=%d 只是扫描上限、不算台阶 → 没定出 k。"
                     "现在沿用 k=%s px；这类图（大面积成片脱落）到量尺页量一条刀缝更稳。",
                     "⚠ %s: the bare-area curve only rises, so the largest k=%d is just the scan limit, not a "
                     "step — no k was derived. Keeping k=%s px; for images like this (large sheet removal) "
                     "measuring a cut on the ruler tab is safer.",
                     _scan, int(req.k_max), int(req.line_px))
        else:
            _span = (L("k=1…%d 全程都是 %s，k 取哪个都不改判级", "every k from 1 to %d stays %s, so the exact k "
                                                              "does not change the grade",
                       rows[-1]["k"] if rows else 0, class_span[0]) if same_band
                     else L("这条曲线跨 %s~%s（k=1…%d），k 必须用量尺定，别凭猜",
                            "the curve spans %s–%s (k=1…%d), so k has to be measured",
                            class_span[0], class_span[-1], rows[-1]["k"] if rows else 0))
            _cnt = (L("块数峰值 %d 块（k=%d），到 k=%d 还有 %d 块，要 ≤%d 才算回落；露底分布也没给出建议",
                      "block count peaks at %d (k=%d) and is still %d at k=%d (needs ≤%d to count as a fall); "
                      "the bare distribution gives no suggestion",
                      _pk["n"], _pk["k"], _last["k"], _last["n"], max(1, int(round(_pk["n"] * 0.25)))
                      if isinstance(_pk, dict) and _last else 0)
                    if (_pk and _last) else
                    L("块数一直没回落，露底分布也没给出建议", "no fall in block count, no bare suggestion"))
            warn = L("⚠ %s：%s → 没定出 k。现在沿用 k=%s px；请到量尺页量一条刀缝，或在 k 曲线里手点一档——%s。",
                     "⚠ %s: %s — no k was derived. Keeping k=%s px; measure a cut on the ruler tab, or click a "
                     "row in the k curve — %s.",
                     _scan, _cnt, int(req.line_px), _span)
    return {"ok": True, "rows": rows, "k_from": k_from,
            "k": pick["k"] if pick else None,
            "k_px": pick["k_view"] if pick else None,
            "k_wire": pick["k"] if pick else None,
            "why": (why_c if req.rule == "count" else why_a),
            "k_area": best_a["k"] if best_a else None,
            "why_area": why_a,
            "k_count": (best_c["k"] if (best_c and k_from == "rule") else None),
            "why_count": why_c, "warn": warn, "no_bare": bool(no_bare),
            "class_span": class_span, "same_band": bool(same_band)}


# --------------------------------------------------- 手绘标注口径（v1.9）
@app.post("/api/annot")
def annot(req: AnnotReq):
    """把用户的三色笔迹栅格化，按选区统计——完全不碰自动分类，纯面积。

    t=1 确认脱落 / t=2 确认刀缝 / t=3 无法判断；没涂到的地方算“未涂”，不算任何一边。
    · 主值 = D / (A − G)     扣掉你亲手确认的刀缝，分母是有效漆膜区
    · 下限 = D / A           刀缝都不扣
    · 上限 = (D+U) / A       把“无法判断”全算成脱落
    上下界跨档 → cross=True，只够给“临界”结论。
    """
    sess = get_session(req.id)
    side = _res_to_side(req.res, sess.full_img())
    with HEAVY_BLOCK:
        with sess.lock:
            work, view = _view_for(sess, req)
        shape2 = work["lab"].shape[:2]
        rois = [r.model_dump() for r in req.rois]
        rois_v = view.rois_in(rois) if rois else []
        masks = _masks(shape2, rois_v) if rois_v else [np.ones(shape2, bool)]

    h, w = shape2
    img = Image.new("L", (w, h), 0)
    dr = ImageDraw.Draw(img)
    n_used = 0
    for s in req.strokes:
        pts = [view.to_view(a, b) for a, b in s.pts]      # 笔迹是画布坐标 → 视图坐标
        if not pts:
            continue
        t = {1: 1, 2: 2}.get(int(s.t), 3)
        r = max(1.0, view.len_in(float(s.r)))
        if len(pts) == 1:
            dr.ellipse([pts[0][0] - r, pts[0][1] - r, pts[0][0] + r, pts[0][1] + r], fill=t)
        else:
            dr.line(pts, fill=t, width=max(1, int(round(2 * r))), joint="curve")
            for x, y in (pts[0], pts[-1]):
                dr.ellipse([x - r, y - r, x + r, y + r], fill=t)
        n_used += 1
    arr = np.asarray(img)          # 0 未涂 / 1 脱落 / 2 刀缝 / 3 不确定

    rows, agg = [], {"A": 0, "D": 0, "G": 0, "U": 0}
    for m in masks:
        A = int(m.sum())
        D = int(((arr == 1) & m).sum())
        G = int(((arr == 2) & m).sum())
        U = int(((arr == 3) & m).sum())
        BG = A - D - G - U
        pct = 100.0 * D / max(1, A - G)
        lo = 100.0 * D / max(1, A)
        hi = 100.0 * (D + U) / max(1, A)
        rows.append({"pct": round(pct, 2), "lo": round(lo, 2), "hi": round(hi, 2),
                     "astm": cc.astm_class(pct), "iso": _iso_class(pct),
                     "lo_cls": cc.astm_class(lo), "hi_cls": cc.astm_class(hi),
                     "A": A, "D": D, "G": G, "U": U, "BG": BG,
                     "d_pct": round(100.0 * D / max(1, A), 2),
                     "g_pct": round(100.0 * G / max(1, A), 2),
                     "u_pct": round(100.0 * U / max(1, A), 2),
                     "bg_pct": round(100.0 * BG / max(1, A), 2)})
        for k in ("A", "D", "G", "U"):
            agg[k] += {"A": A, "D": D, "G": G, "U": U}[k]

    A = max(1, agg["A"])
    D, G, U = agg["D"], agg["G"], agg["U"]
    pct = 100.0 * D / max(1, A - G)
    lo = 100.0 * D / A
    hi = 100.0 * (D + U) / A
    return {
        "ok": True, "n_stroke": n_used,
        "pct": round(pct, 2), "lo": round(lo, 2), "hi": round(hi, 2),
        "astm": cc.astm_class(pct), "iso": _iso_class(pct),
        "lo_cls": cc.astm_class(lo), "hi_cls": cc.astm_class(hi),
        "cross": cc.astm_class(lo) != cc.astm_class(hi),
        "d_pct": round(100.0 * D / A, 2), "g_pct": round(100.0 * G / A, 2),
        "u_pct": round(100.0 * U / A, 2), "bg_pct": round(100.0 * (A - D - G - U) / A, 2),
        "rows": rows,
    }


@app.post("/api/analyze")
async def analyze(req: AnalyzeReq):
    sess = get_session(req.id)

    def run():
        t0 = time.time()
        with HEAVY_BLOCK:
            with sess.lock:
                out = _analyze_core(sess, req)
            if getattr(req, "soft", False) and out.get("ok"):
                with sess.lock:
                    out["soft"] = _soft_block(sess, req)
        out["ms"] = round((time.time() - t0) * 1000, 1)
        return out

    return await run_in_threadpool(run)


# ---------------------------------------------------------------- 半自动


def _auto_detect(sess: Session, work, rois):
    """复刻 GUI._auto_detect：KMeans(4) 猜漆膜/基材"""
    lab_all = work["lab"]
    rgb_all = work["rgb"]
    shape2 = lab_all.shape[:2]
    if rois:
        m = np.zeros(shape2, bool)
        for r in rois:
            m |= cc.rot_rect_mask(shape2, tuple(float(v) for v in r["box"]),
                                  float(r.get("ang") or 0.0))
    else:
        m = np.ones(shape2, bool)
    idx_all = np.flatnonzero(m)
    if idx_all.size < 500:
        return None, L("选区太小", "region too small")
    step = max(1, idx_all.size // 20000)
    idx = idx_all[::step]
    lab = lab_all.reshape(-1, 3)[idx]
    rgb = rgb_all.reshape(-1, 3)[idx].astype(np.float32)

    K = 4
    rng = np.random.default_rng(0)
    cent = lab[rng.choice(len(lab), K, replace=False)].copy()
    lbl = np.zeros(len(lab), np.int32)
    for _ in range(18):
        d = ((lab[:, None, :] - cent[None, :, :]) ** 2).sum(-1)
        lbl = d.argmin(1)
        for k in range(K):
            mm = lbl == k
            if mm.any():
                cent[k] = lab[mm].mean(0)
    counts = np.bincount(lbl, minlength=K)
    usable = [k for k in range(K) if counts[k] >= 0.03 * len(lbl)] or list(range(K))

    # 一个簇一组。**不要**按 ab（色彩）把"同色不同光"的簇合并：灰漆膜 + 灰白基材本来就只差明度，
    # 合并会把两类并成一类（实测 demo 图直接报"只有一类颜色"）。代表色用**中位数**——均值会被
    # 亮/暗两侧拉偏，正是 C19 吸成 (99,131,165) 的一半原因。
    groups = []
    for k in usable:
        sel = lbl == k
        groups.append({"keys": [k], "w": int(counts[k]),
                       "lab": np.median(lab[sel], axis=0),
                       "rgb": np.median(rgb[sel], axis=0)})

    def _chroma(i):
        v = groups[i]["lab"]
        return float(np.hypot(float(v[1]), float(v[2])))

    coat_i = max(range(len(groups)), key=lambda i: groups[i]["w"])
    # 角色别只按面积定：成片脱落（或框里带进背景）时白底占多数，按面积会把基材当漆膜，
    # 脱落率直接被算成"反数"（实测 C19：白 53% 被当漆膜 → 12.14%/3B，实际 ~60%/1B）。
    # 规则：最大的那块若近乎无彩（灰/白，彩度<8），而另有一块明显有彩（≥12）→ 后者才是有色漆膜。
    if _chroma(coat_i) < 8.0:
        # 有“占比下限”：一小块彩色（标签、背景杂物、反光）不能把角色抢走；
        # 面积过半优先是默认，只有在过半的那类近乎无彩（灰/白底材）时，才让占比 ≥15% 的有彩色当漆膜。
        _n_all = max(1, int(len(lbl)))
        cand = [i for i in range(len(groups))
                if i != coat_i and _chroma(i) >= 12.0 and groups[i]["w"] >= 0.15 * _n_all]
        if cand:
            coat_i = max(cand, key=lambda i: groups[i]["w"])
    others = [i for i in range(len(groups)) if i != coat_i]
    if not others:
        return None, L("只有一类颜色", "only one colour")

    edge_frac = {}
    try:
        ring = m & ~cc._sq_min(m, 9)
        if ring.sum() > 50:
            lab_sel = lab_all.reshape(-1, 3)[idx_all]
            lbl_sel = ((lab_sel[:, None, :] - cent[None, :, :]) ** 2).sum(-1).argmin(1)
            in_ring = ring.reshape(-1)[idx_all]
            tot_ring = max(1, int(in_ring.sum()))
            for k in range(K):
                edge_frac[k] = float((in_ring & (lbl_sel == k)).sum()) / tot_ring
    except Exception:
        edge_frac = {}

    eg = {gi: sum(edge_frac.get(k, 0.0) for k in g["keys"]) for gi, g in enumerate(groups)}
    dist = {i: float(np.linalg.norm(groups[i]["lab"] - groups[coat_i]["lab"])) for i in others}
    ranked = sorted(others, key=lambda i: -dist[i])
    inner = [i for i in ranked if eg.get(i, 0.0) <= 0.5 and dist[i] >= 10]
    sub_i = inner[0] if inner else (ranked[0] if dist[ranked[0]] >= 10 else None)
    if sub_i is None:
        return None, L("两类颜色差得太少（<10）", "the two colours are too close (<10)")
    warn = ""
    if eg.get(sub_i, 0.0) > 0.6:
        warn = L("⚠ 选中的露底颜色占了选区边缘的大半，很可能是背景。"
                 "把框框紧一点，或取消半自动手动改色。",
                 "⚠ The colour picked as substrate covers most of the region border — it is probably the "
                 "background. Frame tighter, or turn off semi-auto and sample colours by hand.")

    def to_rgb(i):
        v = np.clip(groups[i]["rgb"], 0, 255)
        return [int(round(x)) for x in v]

    # 容差只按"两类颜色的间距×0.45"定太窄：亮处/暗处的漆膜离中心本来就远，会掉进「未分类」，
    # 从分母里消失 → 脱落率虚高。改成**自校准**：取选区里每个像素到最近一色的距离的 p99，
    # 让 99% 的像素都能被认下来，再用两类颜色的间距夹住（0.30~0.80×gap）防止两边串色。
    gap = float(np.linalg.norm(groups[coat_i]["lab"] - groups[sub_i]["lab"]))
    _dmin = np.minimum(np.linalg.norm(lab - groups[coat_i]["lab"], axis=1),
                       np.linalg.norm(lab - groups[sub_i]["lab"], axis=1))
    tol_auto = min(0.80 * gap, max(0.30 * gap, float(np.percentile(_dmin, 99.0))))

    samples = [{"rgb": to_rgb(coat_i), "role": cc.ROLE_COAT},
               {"rgb": to_rgb(sub_i), "role": cc.ROLE_SUB}]
    # v1.42：**只出两个**（漆膜 + 基材）。以前这里会把第三类颜色加成「忽略」，
    # 结果亮处的漆膜常被当成"忽略"从分母里挖掉 → 脱落率虚高。现在多余的颜色一律不进 samples，
    # 它们自然落到「未分类」（既不进分子也不进分母，但明确显示在结果里）。
    return {"samples": samples, "warn": warn, "ignored": [],
            "tol": max(6.0, min(120.0, round(tol_auto)))}, ""


@app.post("/api/autorun")
async def autorun(req: AutoReq):
    sess = get_session(req.id)
    side = _res_to_side(req.res, sess.full_img())
    # 同一条会话上的半自动请求排队时，只有最新那次需要算；
    # 老请求在拿到锁/认完色后自查一次，发现被取代就直接返回（客户端也会 abort，双保险）
    my_seq = getattr(sess, "auto_seq", 0) + 1
    sess.auto_seq = my_seq          # 整数赋值在 GIL 下是原子的，不在这里抢 sess.lock（会被重活堵住）

    def run():
        with HEAVY_BLOCK:
            with sess.lock:
                if getattr(sess, "auto_seq", my_seq) != my_seq:
                    return {"ok": False, "reason": "superseded"}
                work, view = _view_for(sess, req)
                rois_v = view.rois_in([r.model_dump() for r in req.rois]) if req.rois else []
                got, reason = _auto_detect(sess, work, rois_v)
                if got is None:
                    return {"ok": False, "reason": reason}
                if getattr(sess, "auto_seq", my_seq) != my_seq:
                    return {"ok": False, "reason": "superseded"}
                base = AnalyzeReq(id=req.id, res=req.res, rois=req.rois,
                                  samples=[Sample(**s) for s in got["samples"]],
                                  tol=35.0, purity=float(clamp(req.purity, 0.05, 0.95)), line_px=1, mode="width",
                                  want_hist=True, detail=False, overlay=True,
                                  roi_native=req.roi_native)
                span = None
                labs = _sample_labs([{"rgb": s["rgb"], "role": s["role"]} for s in got["samples"]])
                roles = [s["role"] for s in got["samples"]]
                coat = [l for l, r in zip(labs, roles) if r == cc.ROLE_COAT]
                sub = [l for l, r in zip(labs, roles) if r == cc.ROLE_SUB]
                if coat and sub:
                    span = min(float(np.linalg.norm(c - s)) for c in coat for s in sub)
                    base.tol = float(got.get("tol") or _auto_tol_of(span))
                # 只认色：把样色/容差交回去，让前端先按“刀缝判法”定 k，再回来算一次
                if req.detect_only:
                    return {"ok": True, "detect_only": True, "samples": got["samples"],
                            "ignored": got.get("ignored") or [], "auto_warn": got["warn"],
                            "tol": base.tol, "span": span, "suggest_px": None, "line_px": None}
                # k 由判法定好了就直接用（不复算、不再套服务器自己的建议）
                if req.line_px is not None:
                    base.line_px = int(clamp(round(float(req.line_px)), 1, 60))
                    res = _analyze_core(sess, base, auto_warn=got["warn"])
                    res["samples"] = got["samples"]
                    res["ignored"] = got.get("ignored") or []
                    res["auto_warn"] = got["warn"]
                    res["note"] = ""
                    res["line_px"] = base.line_px
                    res["tol"] = base.tol
                    if getattr(req, "soft", False):
                        with sess.lock:
                            res["soft"] = _soft_block(sess, base)
                    return res
                res = _analyze_core(sess, base, auto_warn=got["warn"])
                note = ""
                v = res.get("suggest_px")
                if v and int(round(base.line_px)) != v:
                    base.line_px = int(min(v, 25))
                    res2 = _analyze_core(sess, base, auto_warn=got["warn"])
                    if res2["total"]["pct"] <= 0.05 or res2["total"]["pct"] >= 60:
                        bad = res2["total"]["pct"]
                        keep = res["total"]["pct"]
                        base.line_px = 1
                        res2 = _analyze_core(sess, base, auto_warn=got["warn"])
                        note = L("⚠ 按建议的 %d px 排除刀缝后只剩 %.2f%%（不可信：等于把真实脱落"
                                 "也当刀缝删掉了），已先关掉刀缝排除，当前 %.2f%%。"
                                 "这张图的脱落比刀缝还细，请到「量尺」页量一条真刀缝再手动填阈值；"
                                 "或者别用自动建议。",
                                 "⚠ Excluding cuts at the suggested %d px leaves only %.2f%% (not "
                                 "trustworthy: real removal is being deleted as cuts), so cut exclusion is "
                                 "switched off and the current result is %.2f%%. Here the removal is finer "
                                 "than the cuts — measure a real cut on the ruler tab and enter the "
                                 "threshold by hand, or do not use the automatic suggestion.",
                                 v, bad, keep)
                    res = res2
                res["samples"] = got["samples"]
                res["ignored"] = got.get("ignored") or []
                res["auto_warn"] = got["warn"]
                res["note"] = note
                res["line_px"] = base.line_px
                res["tol"] = base.tol
                if getattr(req, "soft", False):
                    with sess.lock:
                        res["soft"] = _soft_block(sess, base)
                return res

    return await run_in_threadpool(run)


# ---------------------------------------------------------------- 吸色 / 量尺


@app.post("/api/samples")
def samples_info(req: SamplesReq):
    """只算样本间距和建议容差——改角色/删样本后前端拿它刷新自动容差"""
    sp = _span_of([s.model_dump() for s in req.samples])
    return {"span": sp, "auto_tol": _auto_tol_of(sp)}


@app.post("/api/pick")
async def pick(req: PickReq):
    sess = get_session(req.id)
    side = _res_to_side(req.res, sess.full_img())

    def run():
        with sess.lock:
            work = sess.work(side)
            rgb = work["rgb"]
            h, w = rgb.shape[:2]
            x, y = int(round(req.x)), int(round(req.y))
            if not (0 <= x < w and 0 <= y < h):
                return {"ok": False, "reason": L("点在图像外", "That point is outside the image")}
            r = 5
            y0, y1 = max(0, y - r), min(h, y + r + 1)
            x0, x1 = max(0, x - r), min(w, x + r + 1)
            patch = rgb[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32)
            patch_lab = cc.rgb_to_lab(patch)
            samples = [s.model_dump() for s in req.samples]
            existing = _sample_labs(samples)
            if existing:
                d = np.min(np.stack([np.linalg.norm(patch_lab - e, axis=1) for e in existing], 1),
                           axis=1)
                k = int(np.argmax(d))
                c = [int(v) for v in patch[k]]
                note = L("（与本块其他色差 %.0f，取最纯的一点）",
                         " (colour distance %.0f from the others in this patch; took the purest pixel)", d[k])
            else:
                c = [int(v) for v in np.median(patch, axis=0)]
                note = L("（本块中位色）", " (median colour of this patch)")
            role = cc.ROLE_COAT if not samples else (
                cc.ROLE_SUB if len(samples) == 1 else cc.ROLE_COAT)
            for i, s in enumerate(samples):
                if max(abs(s["rgb"][0] - c[0]), abs(s["rgb"][1] - c[1]),
                       abs(s["rgb"][2] - c[2])) <= 3:
                    return {"ok": False, "reason": L("这个颜色已经取过了（第 %d 个）",
                                                     "This colour is already sampled (sample %d)", i + 1),
                            "samples": samples}
            samples.append({"rgb": c, "role": role})
            span = _span_of(samples)
            return {"ok": True, "samples": samples, "rgb": c,
                    "role_name": _role_name(role), "note": note, "span": span,
                    "auto_tol": _auto_tol_of(span)}

    return await run_in_threadpool(run)


@app.post("/api/ruler")
async def ruler(req: RulerReq):
    sess = get_session(req.id)
    side = _res_to_side(req.res, sess.full_img())

    def run():
        with sess.lock:
            work = sess.work(side)
            ow = sess.full_img().size[0]
            scale = ow / float(work["size"][0])
            fw = None
            if req.on_full:
                full = np.asarray(sess.full_img()).astype(np.uint8)
                p0 = (req.p0[0] * scale, req.p0[1] * scale)
                p1 = (req.p1[0] * scale, req.p1[1] * scale)
                res = cc.measure_line_width(full, p0, p1)
                fw = _profile_full_width(full, p0, p1)
            else:
                res = cc.measure_line_width(work["rgb"], tuple(req.p0), tuple(req.p1))
                fw = _profile_full_width(work["rgb"], tuple(req.p0), tuple(req.p1))
            if res is None:
                return {"ok": False, "reason": L("没测到明显的沟槽：让线段垂直于刀缝、长度 30px 以上",
                                             "No clear groove measured: keep the segment perpendicular to "
                                             "the cut and at least 30 px long")}
            res["width_full"] = res["width_px"] if req.on_full else res["width_px"] * scale
            res["width_analysis"] = (res["width_px"] / scale) if req.on_full else res["width_px"]
            # 全宽同样换算到两个尺度
            if fw is None:
                res["full_full"] = None
                res["full_analysis"] = None
            else:
                res["full_full"] = fw if req.on_full else fw * scale
                res["full_analysis"] = (fw / scale) if req.on_full else fw
            res["reliable"] = res["contrast"] >= 8
            res["ok"] = True
            return res

    return await run_in_threadpool(run)


# ---------------------------------------------------------------- 导出


class ExportReq(AnalyzeReq):
    name: str = ""
    fmt: str = "pdf"          # pdf（默认，正式交付）| html（网页预览/自己打印）


def _csv_bytes(sess, req):
    side = _res_to_side(req.res, sess.full_img())
    work, view = _view_for(sess, req)
    samples = [s.model_dump() for s in req.samples]
    roles = _roles(samples)
    rois = [r.model_dump() for r in req.rois]
    rois_v = view.rois_in(rois) if rois else []
    px_out = view.len_out(1.0)
    k_view = max(1, int(round(view.len_in(req.line_px))))
    shape2 = work["lab"].shape[:2]
    masks = _masks(shape2, rois_v) if rois_v else [np.ones(shape2, bool)]
    labels, _, _, _ = _labels_for(sess, work, req.tol, req.purity, samples, masks,
                                  _msig(rois_v), view.key)
    rows, tot, pct = _stats(labels, masks, roles, k_view, req.mode, False, px_out,
                     req.loc_tile, req.loc_factor)
    import csv
    out = io.StringIO()
    wr = csv.writer(out)
    units = L("原生", "native") if view.native else L("分析图", "analysis")
    hdr = [L("面积px(%s)", "area px (%s)") % units,
           L("漆膜px(%s)", "coat px (%s)") % units,
           L("基材px(%s)", "substrate px (%s)") % units,
           L("未分类px(%s)", "unclassified px (%s)") % units,
           L("脱落%(原始)", "removal %% (raw)"),
           L("脱落%(忽略细线)", "removal %% (cuts excluded)"),
           "ASTM", "ISO", L("分析源", "analysis source")]
    wr.writerow(["ROI", "x0", "y0", "x1", "y1"] + hdr)
    src = L("选区原生", "native region") if view.native else L("整图6MP", "full image 6MP")
    for i, st in enumerate(rows, 1):
        r = rois[i - 1]["box"] if i - 1 < len(rois) else [0, 0, 0, 0]
        wr.writerow([i, round(r[0]), round(r[1]), round(r[2]), round(r[3]), st["area"],
                     st["coat_px"], st["sub_raw"], st["unassigned"],
                     round(st["pct_raw"], 2), round(st["pct"], 2),
                     cc.astm_class(st["pct"]), _iso_class(st["pct"]), src])
    wr.writerow([L("合计", "Total"), "", "", "", "", tot["area"], tot["coat_px"], tot["sub_raw"],
                 "", "", round(pct, 2), cc.astm_class(pct), _iso_class(pct), src])
    _ok, _why = _credibility(tot)
    if not _ok:                                    # 输入不可信：CSV 末尾留一行注释（13 列结构不变）
        wr.writerow([L("# ⚠ 参数不可信：", "# WARNING unreliable parameters: ") + _why])
    return ("\ufeff" + out.getvalue()).encode("utf-8")


EXPORT_TINT = 0.65            # 导出图整片染色强度（v1.60 还原 v1.55 观感：露底 65% 红、漆膜 25% 绿）
EXPORT_OUTLINE = False        # 导出图是否再叠一层细轮廓（默认关；画布那边仍是蚂蚁线）
EXPORT_NUM_MIN_AREA = 150     # 块面积（wire px²）≥ 这个值才编号；更小的只描边
EXPORT_NUM_CAP = 30           # 最多编号多少个块（按面积从大到小）


ACCENT_CANDS = [((0, 200, 255), "青"), ((255, 0, 200), "品红"), ((150, 255, 60), "亮绿"),
                ((255, 215, 0), "黄"), ((0, 120, 255), "蓝"), ((255, 120, 0), "橙"),
                ((255, 255, 255), "白")]


def _accent_rgb(samples):
    """挑一个离「漆膜色/基材色」最远的描边颜色。红线画在红色漆膜上根本分不出来。"""
    labs = [np.asarray(cc.rgb_to_lab(np.asarray(s["rgb"], np.uint8)), np.float32) for s in samples]
    if not labs:
        return (0, 220, 255)
    best = None
    for rgb, _name in ACCENT_CANDS:
        l = np.asarray(cc.rgb_to_lab(np.asarray(rgb, np.uint8)), np.float32)
        d = min(float(np.sqrt(((l - x) ** 2).sum())) for x in labs)
        if best is None or d > best[0]:
            best = (d, rgb)
    return best[1] if best[0] >= 22 else (255, 255, 255)


def _annotated_png(sess, req, bar=True):
    side = _res_to_side(req.res, sess.full_img())
    work6 = sess.work(side)                        # 导出图永远是整图（和以前一样）
    work, view = _view_for(sess, req)
    samples = [s.model_dump() for s in req.samples]
    roles = _roles(samples)
    rois = [r.model_dump() for r in req.rois]
    rois_v = view.rois_in(rois) if rois else []
    px_out = view.len_out(1.0)
    k_view = max(1, int(round(view.len_in(req.line_px))))
    shape2 = work["lab"].shape[:2]
    masks = _masks(shape2, rois_v) if rois_v else [np.ones(shape2, bool)]
    labels, _, _, _ = _labels_for(sess, work, req.tol, req.purity, samples, masks,
                                  _msig(rois_v), view.key)
    rows, tot, pct = _stats(labels, masks, roles, k_view, req.mode, False, px_out,
                     req.loc_tile, req.loc_factor)
    cls = cc.astm_class(pct)

    base = work6["rgb"].copy()                     # uint8 就地调色，省一次 72 MB 的 float32 拷贝
    h6, w6 = base.shape[:2]
    if view.native:                                # 视图 → 整图 6MP 画布
        a = view.s6 / view.scale
        bx, by = view.ox * view.s6, view.oy * view.s6
    else:
        a, bx, by = 1.0, 0.0, 0.0

    # ---- 逐块几何：与 /api/blocks 完全同一套（同一 keep_mask + 同一连通域）----
    keep_c = np.zeros((h6, w6), bool)              # 画布口径的「留下来的露底」
    blocks = []                                    # dict(roi, i, area, box) —— i 就是界面的块号-1
    for ri, m in enumerate(masks):
        subc = np.zeros(labels.shape[:2], bool)
        for j, role in enumerate(roles):
            if role == cc.ROLE_SUB:
                subc |= (labels == j)
        subc &= m
        kept_v = cc.keep_mask(subc, k_view, req.mode,
                              int(req.loc_tile), float(req.loc_factor))[0]
        comps = _components_fast(kept_v, 30, 240)[0]   # 同一口径，比 BFS 版快一个数量级
        # 只给「会进块列表」的块描边：_components 已经把 <30 px 的碎点滤掉了，
        # 所以用「块 bbox ∩ kept_v」就能精确还原这些块的像素（连通域彼此不重叠）。
        sel_v = np.zeros_like(kept_v)
        for (ar, _cx, _cy, qx0, qy0, qx1, qy1) in comps:
            sel_v[qy0:qy1 + 1, qx0:qx1 + 1] |= kept_v[qy0:qy1 + 1, qx0:qx1 + 1]
        bb = _roi_bbox(labels.shape[:2], [m])
        if bb:
            y0, y1, x0, x1 = bb
            sv = sel_v[y0:y1, x0:x1]
            tw, th = max(1, int(round((x1 - x0) * a))), max(1, int(round((y1 - y0) * a)))
            if (tw, th) != (x1 - x0, y1 - y0):
                sv = np.asarray(Image.fromarray(sv.astype(np.uint8) * 255)
                                .resize((tw, th), Image.NEAREST)) > 127
            oy, ox = int(round(y0 * a + by)), int(round(x0 * a + bx))
            hh, ww = min(th, h6 - oy), min(tw, w6 - ox)
            if hh > 0 and ww > 0 and oy >= 0 and ox >= 0:
                keep_c[oy:oy + hh, ox:ox + ww] |= sv[:hh, :ww]
        for bi, (ar, _cx, _cy, qx0, qy0, qx1, qy1) in enumerate(comps):
            blocks.append(dict(roi=ri, i=bi, area=ar * a * a,
                               box=(qx0 * a + bx, qy0 * a + by,
                                    (qx1 + 1) * a + bx, (qy1 + 1) * a + by)))

    # ---- 诊断着色：默认关。整片染红会把「这块到底白不白」直接盖掉，那才是判级的原始证据 ----
    if EXPORT_TINT > 0:
        codes = np.zeros((h6, w6), np.uint8)
        for m in masks:
            _paste_codes(codes, labels, view, m, a, bx, by, roles)
        sub, coat = codes == 1, codes == 2
        base[sub] = (base[sub].astype(np.float32) * 0.35 + np.array([255, 40, 40], np.float32) * 0.65).astype(np.uint8)
        base[coat] = (base[coat].astype(np.float32) * 0.75 + np.array([60, 220, 90], np.float32) * 0.25).astype(np.uint8)

    # ---- 块轮廓：内 1 px 强调色 + 外 1 px 暗边。强调色按样品配色挑，
    # 红膜上不用红线、黄膜上不用黄线（实测红膜上红轮廓基本看不见）。
    ACC = _accent_rgb(samples)
    ROI_C = (230, 0, 200) if (ACC[0] > 180 and ACC[1] > 150 and ACC[2] > 200) else (0, 200, 255)
    if EXPORT_OUTLINE and keep_c.any():
        din = _cheb_dist(keep_c, 3)
        dout = _cheb_dist(~keep_c, 3)
        inner = keep_c & (din <= 1)
        outer = (~keep_c) & (dout <= 1)
        base[outer] = (12, 12, 16)
        base[inner] = ACC

    boxes = [rois[i - 1]["box"] if i - 1 < len(rois) else [0, 0, 0, 0]
             for i in range(1, len(rows) + 1)]

    out = Image.fromarray(base)
    dr = ImageDraw.Draw(out)
    W, H = out.size
    fs = max(14, int(H / 55))
    font = cc._pil_font(fs)

    # ---- 编号默认不画（用户明确要求）：只留细线框出露底，编号在界面里看 ----
    texts = ["R%d  %.1f%%  %s" % (i, st["pct"], cc.astm_class(st["pct"]))
             for i, st in enumerate(rows, 1)]
    # 第一行只放结论（用户要求：不要 ISO、不要块数）
    txt = "removed %.2f%%  ASTM %s" % (pct, cls)
    if not _credibility(tot)[0]:
        # 导出图的字体没有中文字形（写中文会变成方块豆腐），这条**只能英文**，别包 L()
        txt += "   (params unreliable)"
    # 第二行：复现参数（刀缝宽度 k / 判法 / 容差 / 露底纯度）+ 漆膜、基材色样
    f2 = cc._pil_font(max(11, int(fs * 0.74)))
    sw = max(9, int(fs * 0.52))
    pre = "k=%d(%s)  tol=%.1f  purity=%.2f" % (int(req.line_px), req.mode, req.tol, req.purity)
    wpre = dr.textlength(pre, font=f2)
    segs = []
    for smp in samples[:4]:
        rgb = smp["rgb"]
        tag = {"coat": "C", "sub": "S"}.get(smp["role"], "I")
        t2 = "%s %d,%d,%d" % (tag, rgb[0], rgb[1], rgb[2])
        segs.append((tuple(rgb), t2, dr.textlength(t2, font=f2)))
    w2 = wpre + int(fs * 0.9) + sum(sw + 5 + w + int(fs * 0.7) for _c, _t, w in segs)
    try:      # 汇总条先量真实宽度，再让标签躲开它（以前 R1 贴着左上角会被盖住）
        _bw = int(max(dr.textbbox((0, 0), txt, font=font)[2] + 12, w2 + 12))
    except Exception:
        _bw = int(max(12 + fs * 0.62 * len(txt), w2 + 12))
    _bh = int(fs * 1.45 + fs * 1.15)
    banner = (0, 0, _bw, _bh) if bar else None
    for (lx, ly, lw, lh, tx) in cc.place_labels(boxes, texts, (0, 0, W, H),
                                                char_w=fs * 0.62, line_h=int(fs * 1.55),
                                                avoid=[banner] if bar else []):
        dr.rectangle([lx, ly, lx + lw, ly + lh], fill=(15, 15, 15), outline=(140, 140, 140))
        dr.text((lx + 6, ly + lh * 0.5 - fs * 0.55), tx, fill=(255, 213, 74), font=font)
    lw_ = max(2, W // 500)
    for i, box in enumerate(boxes):
        ang = rois[i]["ang"] if i < len(rois) else 0.0
        pts = cc.rect_corners(tuple(box), ang)
        dr.line(list(pts) + [pts[0]], fill=ROI_C, width=lw_)
    if bar:      # 报告里这条黑底条没必要（表格里都有），传 bar=False 关掉
        dr.rectangle([0, 0, _bw, _bh], fill=(0, 0, 0))
        dr.text((6, 4), txt, fill=(255, 255, 255), font=font)
        yy = int(fs * 1.5)
        xx = 6
        dr.text((xx, yy), pre, fill=(200, 205, 212), font=f2)
        xx += int(wpre) + int(fs * 0.9)
        for rgb, t2, w2s in segs:
            dr.rectangle([xx, yy + 2, xx + sw, yy + 2 + sw], fill=rgb, outline=(210, 210, 210))
            dr.text((xx + sw + 5, yy), t2, fill=(225, 225, 230), font=f2)
            xx += sw + 5 + w2s + int(fs * 0.7)
    return _png(out)


def _hole_stats(mask):
    """一块脱落里的“孔洞”（被露底包围的漆膜小岛）：返回 (孔洞数, 孔洞面积 px)。

    做法：把外圈补成前景，背景就分成“外界（一大块） + 各孔洞”，最大那块当外界丢掉。
    比描轮廓快得多，且不需要额外的库。
    """
    if not mask.any():
        return 0, 0
    m = np.pad(mask, 1, constant_values=True)
    comps, _rest = _components_fast(~m, 30, 5000)
    if len(comps) < 2:
        return 0, 0
    comps = sorted(comps, key=lambda t: -t[0])
    holes = comps[1:]
    return len(holes), int(sum(int(c[0]) for c in holes))


def _block_stats(labels, masks, rois_v, roles, k, mode, req):
    """报告用的块统计（只算面积/占比，不描轮廓——描轮廓留给 /api/blocks）。

    口径与「块确认」页完全一致：同一套 keep_mask + 连通域 + min_area=30。
    """
    shape2 = labels.shape[:2]
    out = []
    for m in masks:
        rr = np.flatnonzero(m.any(1)); ccx = np.flatnonzero(m.any(0))
        if not rr.size or not ccx.size:
            out.append({"blocks": [], "rest": 0, "kept": 0})
            continue
        r0, r1, c0, c1 = rr[0], rr[-1] + 1, ccx[0], ccx[-1] + 1
        mb = m[r0:r1, c0:c1]
        subc = np.zeros(mb.shape, bool)
        for j, role in enumerate(roles):
            if role == cc.ROLE_SUB:
                subc |= (labels[r0:r1, c0:c1] == j)
        subc &= mb
        kept, *_ = cc.keep_mask(subc, k, mode, int(req.loc_tile), float(req.loc_factor))
        comps, rest = _components_fast(kept, 30, 240)
        tot = max(1, int(kept.sum()))
        bl = [{"i": i, "area": int(a), "share": 100.0 * float(a) / tot}
              for i, (a, _cx, _cy, _x0, _y0, _x1, _y1) in enumerate(comps)]
        nh, ha = _hole_stats(kept)
        out.append({"blocks": bl, "rest": int(rest), "kept": int(kept.sum()),
                    "holes": nh, "hole_area": ha})
    return out


def _plain_png(sess, req, max_w=1500, quality=92):
    """A 图：**不加覆盖层**的原图，只描选区细框 —— 给人看“这到底长什么样”。"""
    work, view = _view_for(sess, req)
    rois = [r.model_dump() for r in req.rois]
    rois_v = view.rois_in(rois) if rois else []
    im = Image.fromarray(np.asarray(work["rgb"])).convert("RGB")
    dr = ImageDraw.Draw(im)
    lw = max(2, im.width // 700)
    font = cc._pil_font(max(16, im.width // 60))
    for i, r in enumerate(rois_v):
        pts = cc.rect_corners(tuple(r["box"]), r.get("ang", 0.0))
        dr.line(list(pts) + [pts[0]], fill=(0, 170, 200), width=lw)
        x, y = min(p[0] for p in pts), min(p[1] for p in pts)
        dr.rectangle([x + 4, y + 4, x + 34 + font.size * 1.1, y + 12 + font.size * 1.25], fill=(0, 0, 0))
        dr.text((x + 9, y + 7), "R%d" % (i + 1), font=font, fill=(255, 255, 255))
    if im.width > max_w:
        im = im.resize((max_w, max(1, int(round(im.height * max_w / im.width)))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def _overlay_png(sess, req, max_w=1500, quality=92):
    """B 图：分析视图 —— 判定脱落的像素盖一层半透明红，其余保持原样。

    不描线、不写号（用户明确要求：只覆盖红色就行，蚂蚁线和块号都不要）。
    返回 (jpeg, 每选区块明细) 供表格/报告使用。
    """
    work, view = _view_for(sess, req)
    samples = [s.model_dump() for s in req.samples]
    roles = _roles(samples)
    rois = [r.model_dump() for r in req.rois]
    rois_v = view.rois_in(rois) if rois else []
    k_view = max(1, int(round(view.len_in(req.line_px))))
    shape2 = work["lab"].shape[:2]
    masks = _masks(shape2, rois_v) if rois_v else [np.ones(shape2, bool)]
    labels, _, _, _ = _labels_for(sess, work, req.tol, req.purity, samples, masks,
                                  _msig(rois_v), view.key)

    base = np.asarray(work["rgb"]).astype(np.float32)
    meta = []
    for ri, m in enumerate(masks):
        subc = np.zeros(labels.shape[:2], bool)
        for j, role in enumerate(roles):
            if role == cc.ROLE_SUB:
                subc |= (labels == j)
        subc &= m
        kept, *_ = cc.keep_mask(subc, k_view, req.mode,
                                int(req.loc_tile), float(req.loc_factor))
        base[kept] = base[kept] * 0.56 + np.array([235.0, 58.0, 58.0]) * 0.44   # 红覆盖
        comps, rest = _components_fast(kept, 30, 240)
        tot = max(1, int(kept.sum()))
        bl = [{"i": bi, "area": int(ar), "share": 100.0 * ar / tot,
               "cx": float(cx), "cy": float(cy)}
              for bi, (ar, cx, cy, _x0, _y0, _x1, _y1) in enumerate(comps)]
        nh, ha = _hole_stats(kept)
        meta.append({"blocks": bl, "rest": int(rest), "kept": int(kept.sum()),
                     "holes": nh, "hole_area": ha})

    im = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).convert("RGB")
    dr = ImageDraw.Draw(im)
    lw = max(2, im.width // 700)
    font = cc._pil_font(max(16, im.width // 60))
    for i, r in enumerate(rois_v):
        pts = cc.rect_corners(tuple(r["box"]), r.get("ang", 0.0))
        dr.line(list(pts) + [pts[0]], fill=(0, 170, 200), width=lw)
        x, y = min(p[0] for p in pts), min(p[1] for p in pts)
        dr.rectangle([x + 4, y + 4, x + 34 + font.size * 1.1, y + 12 + font.size * 1.25],
                     fill=(0, 0, 0))
        dr.text((x + 9, y + 7), "R%d" % (i + 1), font=font, fill=(255, 255, 255))
    if im.width > max_w:
        im = im.resize((max_w, max(1, int(round(im.height * max_w / im.width)))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality, optimize=True)
    return buf.getvalue(), meta


def _report_data(sess, req):
    """报告的全部数据（HTML 与 PDF 两个渲染器共用，避免算两遍）。"""
    work, view = _view_for(sess, req)
    samples = [s.model_dump() for s in req.samples]
    roles = _roles(samples)
    rois = [r.model_dump() for r in req.rois]
    rois_v = view.rois_in(rois) if rois else []
    px_out = view.len_out(1.0)
    k_view = max(1, int(round(view.len_in(req.line_px))))
    shape2 = work["lab"].shape[:2]
    masks = _masks(shape2, rois_v) if rois_v else [np.ones(shape2, bool)]
    labels, _, _, _ = _labels_for(sess, work, req.tol, req.purity, samples, masks,
                                  _msig(rois_v), view.key)
    rows, tot, pct = _stats(labels, masks, roles, k_view, req.mode, False, px_out,
                            req.loc_tile, req.loc_factor)
    img_a = _plain_png(sess, req)
    img_b, blk = _overlay_png(sess, req)
    ok, why = _credibility(tot)
    role_name = {"coat": L("漆膜", "coating"), "sub": L("基材", "substrate"),
                 "ignore": L("忽略", "ignored")}
    struct = []
    for ri, bs in enumerate(blk, 1):
        big = max((b["area"] for b in bs["blocks"]), default=0)
        struct.append({"roi": ri, "n": len(bs["blocks"]),
                       "largest": 100.0 * big / max(1, bs["kept"]),
                       "holes": bs["holes"],
                       "hole_share": 100.0 * bs["hole_area"] / max(1, bs["kept"])})
    name = (req.name or sess.name or L("未命名", "untitled")).strip()
    return {
        "name": name, "rid": time.strftime("%Y%m%d-%H%M"),
        "now": time.strftime("%Y-%m-%d %H:%M"),
        "size": (sess.size[0], sess.size[1]), "ana": (int(shape2[1]), int(shape2[0])),
        "mode": {"auto_blk": L("按块数回落", "by block count"),
                 "width": L("按露底分布", "by bare area")}.get(req.mode, req.mode),
        "k": int(req.line_px), "tol": float(req.tol), "purity": float(req.purity),
        "algo": ALGO_VERSION, "app": APP_VERSION,
        "ok": ok, "why": why,
        "units": L("原生", "native") if view.native else L("分析图", "analysis"),
        "samples": [{"rgb": s["rgb"], "role": role_name.get(s["role"], s["role"])} for s in samples[:6]],
        "rows": [{"i": i, "pct": st["pct"], "astm": cc.astm_class(st["pct"]),
                  "iso": _iso_class(st["pct"]), "area": st["area"],
                  "coat": st["coat_px"], "sub": st["sub_raw"],
                  "unassigned": st["unassigned"], "raw": st["pct_raw"],
                  "blocks": (struct[i - 1]["n"] if i - 1 < len(struct) else 0)}
                 for i, st in enumerate(rows, 1)],
        "tot": {"pct": pct, "astm": cc.astm_class(pct), "iso": _iso_class(pct),
                "area": tot["area"], "coat": tot["coat_px"], "sub": tot["sub_raw"],
                "unassigned": tot["unassigned"],
                "raw": 100.0 * tot["sub_raw"] / max(1, tot["coat_px"] + tot["sub_raw"])},
        "struct": struct,
        "img_a": img_a, "img_b": img_b,
    }


REPORT_CSS = """<style>
:root{--ink:#111827;--mut:#6b7280;--line:#e5e7eb;--band:#f3f4f6;--warn:#b45309;--warnbg:#fffbeb}
*{box-sizing:border-box}
body{margin:0;padding:22px 26px;font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans SC",
     "Microsoft YaHei",sans-serif;color:var(--ink);background:#fff}
h1{margin:0 0 2px;font-size:21px;letter-spacing:.3px}
.sub{color:var(--mut);font-size:12.5px;margin-bottom:16px}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px 14px;margin:0 0 16px;font-size:12.5px}
.grid b{display:block;color:var(--mut);font-weight:500;font-size:11.5px;letter-spacing:.02em}
.kv{padding:7px 9px;background:var(--band);border-radius:6px}
.figs{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:0 0 6px}
.fig{border:1px solid var(--line);border-radius:8px;overflow:hidden}
.fig img{display:block;width:100%;height:auto}
.fig .t{font-size:11.5px;color:var(--mut);padding:5px 8px;background:var(--band)}
.cap{color:var(--mut);font-size:11.5px;margin:0 0 18px}
table{width:100%;border-collapse:collapse;font-size:12.5px;margin:0 0 16px}
th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th{background:var(--band);font-weight:600;color:#374151}
th:first-child,td:first-child{text-align:left}
tr.tot td{font-weight:700;border-top:2px solid var(--line)}
.grade{display:inline-block;min-width:26px;text-align:center;padding:1px 6px;border-radius:4px;
       background:#111827;color:#fff;font-weight:700;font-size:12px}
.grade.bad{background:#b91c1c}.grade.mid{background:#b45309}.grade.ok{background:#15803d}
.warn{border-left:4px solid var(--warn);background:var(--warnbg);color:#78350f;padding:9px 12px;
      border-radius:0 6px 6px 0;font-size:12.5px;margin:0 0 16px}
.sw{display:inline-block;width:12px;height:12px;border-radius:3px;border:1px solid #cbd5e1;
    vertical-align:-2px;margin-right:5px}
h2{font-size:13px;margin:20px 0 7px;color:#374151;text-transform:uppercase;letter-spacing:.06em}
footer{margin-top:22px;padding-top:12px;border-top:1px solid var(--line);color:var(--mut);font-size:11.5px;line-height:1.7}
@media print{body{padding:0}@page{size:A4;margin:12mm}h2{page-break-after:avoid}
  .fig,table{page-break-inside:avoid}}
</style>"""


def _report_html(D):
    """一页式报告（自包含 HTML）。判据数字与 CSV / PDF 同一套。"""
    import html as _h

    def esc(x):
        return _h.escape(str(x))

    def grade_cls(g):
        return "bad" if g in ("0B", "1B") else ("mid" if g in ("2B", "3B") else "ok")

    def gtag(g):
        return "<span class=\"grade %s\">%s</span>" % (grade_cls(g), g)

    trs = []
    for r in D["rows"]:
        trs.append("<tr><td>R%d</td><td>%.2f%%</td><td>%s</td><td>%s</td><td>%d</td><td>%d</td>"
                   "<td>%d</td><td>%d</td><td>%d</td><td>%.2f%%</td></tr>"
                   % (r["i"], r["pct"], gtag(r["astm"]), r["iso"], r["blocks"], r["area"], r["coat"],
                      r["sub"], r["unassigned"], r["raw"]))
    t = D["tot"]
    trs.append("<tr class=\"tot\"><td>%s</td><td>%.2f%%</td><td>%s</td><td>%s</td><td>%d</td><td>%d</td>"
               "<td>%d</td><td>%d</td><td>%d</td><td>%.2f%%</td></tr>"
               % (L("合计", "Total"), t["pct"], gtag(t["astm"]), t["iso"],
                  sum(r["blocks"] for r in D["rows"]), t["area"],
                  t["coat"], t["sub"], t["unassigned"], t["raw"]))

    sw = "".join("<span style=\"margin-right:16px;white-space:nowrap\"><i class=\"sw\" "
                 "style=\"background:rgb(%d,%d,%d)\"></i>%s %d,%d,%d</span>"
                 % (s["rgb"][0], s["rgb"][1], s["rgb"][2], esc(s["role"]),
                    s["rgb"][0], s["rgb"][1], s["rgb"][2]) for s in D["samples"])

    warn = ""
    if not D["ok"]:
        warn = "<div class=\"warn\">%s</div>" % esc(
            L("参数不可信：%s。下面的判级仅供参考，请重吸色或调整容差后重新生成报告。",
              "Parameters are unreliable: %s. The grade below is for reference only — re-sample the "
              "colours or adjust the tolerance, then regenerate.", D["why"]))

    parts = [
        "<!doctype html><html lang=\"%s\"><head><meta charset=\"utf-8\">" % ("en" if is_en() else "zh"),
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">",
        "<title>%s</title>" % esc(L("百格附着力测试报告", "Cross-cut adhesion test report")),
        REPORT_CSS, "</head><body>",
        "<h1>%s</h1>" % esc(L("百格附着力测试报告", "Cross-cut adhesion test report")),
        "<div class=\"sub\">%s · ASTM D3359 %s · ISO 2409</div>" % (esc(D["name"]), L("方法 B", "Method B")),
        warn,
        "<div class=\"grid\">",
        "<div class=\"kv\"><b>%s</b>%s</div>" % (L("样品", "Sample"), esc(D["name"])),
        "<div class=\"kv\"><b>%s</b>BG-%s</div>" % (L("报告编号", "Report no."), D["rid"]),
        "<div class=\"kv\"><b>%s</b>%s</div>" % (L("报告时间", "Report time"), D["now"]),
        "<div class=\"kv\"><b>%s</b>%d×%d → %d×%d</div>"
        % (L("图片 / 分析尺寸", "Image / analysis size"), D["size"][0], D["size"][1], D["ana"][0], D["ana"][1]),
        "<div class=\"kv\"><b>%s</b>%s</div>" % (L("判法", "k method"), esc(D["mode"])),
        "<div class=\"kv\"><b>%s</b>%d px</div>" % (L("刀缝宽度 k", "Cut width k"), D["k"]),
        "<div class=\"kv\"><b>%s</b>%.1f</div>" % (L("颜色容差", "Colour tolerance"), D["tol"]),
        "<div class=\"kv\"><b>%s</b>%.2f</div>" % (L("露底纯度", "Bare purity"), D["purity"]),
        "<div class=\"kv\"><b>%s</b>v%s</div>" % (L("算法版本", "Algorithm"), D["algo"]),
        "</div>",
        "<div class=\"figs\">",
        "<div class=\"fig\"><img alt=\"A\" src=\"data:image/jpeg;base64,%s\">"
        "<div class=\"t\">%s</div></div>" % (__import__("base64").b64encode(D["img_a"]).decode("ascii"),
                                            esc(L("A 原图（不加覆盖）", "A · original, no overlay"))),
        "<div class=\"fig\"><img alt=\"B\" src=\"data:image/jpeg;base64,%s\">"
        "<div class=\"t\">%s</div></div>" % (__import__("base64").b64encode(D["img_b"]).decode("ascii"),
                                            esc(L("B 脱落判定（红=脱落）", "B · removal (red = removed)"))),
        "</div>",
        "<div class=\"cap\">%s</div>" % esc(
            L("A 是原图，不加任何覆盖；B 是把判定脱落的像素盖成半透明红（刻线刀缝 %d px 已剔除，"
              "所以刀缝不上色）。两图同尺度，可直接对照。",
              "A is the untouched photo; B tints the pixels judged as removed in translucent red (the %d px "
              "cut grooves are excluded first, so the grooves themselves are not tinted). Same scale, so the "
              "two can be compared directly.", D["k"])),
        "<h2>%s</h2>" % esc(L("各选区结果", "Results per region")),
        "<table><thead><tr><th>%s</th><th>%s</th><th>ASTM</th><th>ISO</th><th>%s</th><th>%s</th>"
        "<th>%s</th><th>%s</th><th>%s</th><th>%s</th></tr></thead><tbody>"
        % (L("选区", "Region"), L("脱落率（已剔除刀缝）", "Removal % (grooves excluded)"),
           L("块数", "Blocks"), L("选区面积 px", "Region area px"), L("漆膜 px", "Coating px"),
           L("基材 px", "Substrate px"), L("未分类 px", "Unclassified px"),
           L("原始脱落率", "Removal % (raw)")),
        "".join(trs), "</tbody></table>",
        "<h2>%s</h2>" % esc(L("色样与口径", "Colour samples and basis")),
        "<div style=\"font-size:12.5px;line-height:1.9\">%s</div>" % sw,
        "<div class=\"cap\" style=\"margin-top:6px\">%s</div>" % esc(
            L("面积单位：%s像素（分析图 = 长边 6 MP）。分母为「漆膜 + 露底」；未分类像素不计入。"
              "「原始脱落率」未剔除刻线刀缝，仅作对照。",
              "Area unit: %s pixels (analysis image = 6 MP on the long side). The denominator is "
              "coating + bare area; unclassified pixels are excluded. “Raw” does not exclude the cut "
              "grooves and is shown for comparison only.", D["units"])),
        "<footer>",
        "%s<br>" % esc(L("判级依据 ASTM D3359（方法 B）/ ISO 2409；数值由图像自动分析得出，边界样品建议人工复核。",
                         "Grading follows ASTM D3359 (Method B) / ISO 2409. Values are produced by automated "
                         "image analysis; borderline samples should be reviewed by hand.")),
        "%s<br>" % esc(L("软件：百格附着力分析 v%s（算法 %s）· AGPL-3.0-or-later · © 2026 iamsad3508",
                         "Software: Baige cross-cut analyser v%s (algorithm %s) · AGPL-3.0-or-later · "
                         "© 2026 iamsad3508", D["app"], D["algo"])),
        "</footer></body></html>",
    ]
    return "".join(parts).encode("utf-8")


REPORT_FONT = os.environ.get("BAIGE_FONT", "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")


def _report_pdf(D):
    """同一个报告的 PDF 版（A4，中文用 wqy 字体；浏览器里打印不如直接给 PDF 正规）。"""
    from fpdf import FPDF

    pdf = FPDF("P", "mm", "A4")
    pdf.set_auto_page_break(True, margin=12)
    pdf.add_font("wqy", "", REPORT_FONT)
    pdf.set_font("wqy", "", 9)
    pdf.set_margins(12, 12, 12)
    pdf.add_page()
    W = pdf.w - pdf.l_margin - pdf.r_margin                 # ≈186 mm
    ink = (17, 24, 39)
    mut = (107, 114, 128)

    pdf.set_text_color(*ink)
    pdf.set_font("wqy", "", 16)
    pdf.cell(W, 8, L("百格附着力测试报告", "Cross-cut adhesion test report"),
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("wqy", "", 8.5)
    pdf.set_text_color(*mut)
    pdf.cell(W, 5, "%s · ASTM D3359 %s · ISO 2409 · BG-%s"
             % (D["name"], L("方法 B", "Method B"), D["rid"]), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    if not D["ok"]:
        pdf.set_fill_color(255, 251, 235)
        pdf.set_text_color(120, 53, 15)
        pdf.set_font("wqy", "", 8.5)
        pdf.multi_cell(W, 4.6, L("参数不可信：%s。下面的判级仅供参考。",
                                 "Parameters are unreliable: %s. The grade below is for reference only.",
                                 D["why"]), fill=True)
        pdf.ln(2)

    # —— 信息卡：4 列 × 2 行
    cells = [(L("样品", "Sample"), D["name"]),
             (L("报告时间", "Report time"), D["now"]),
             (L("图片 / 分析尺寸", "Image / analysis"), "%d×%d → %d×%d"
              % (D["size"][0], D["size"][1], D["ana"][0], D["ana"][1])),
             (L("判法", "k method"), D["mode"]),
             (L("刀缝宽度 k", "Cut width k"), "%d px" % D["k"]),
             (L("颜色容差", "Colour tolerance"), "%.1f" % D["tol"]),
             (L("露底纯度", "Bare purity"), "%.2f" % D["purity"]),
             (L("算法版本", "Algorithm"), "v%s" % D["algo"])]
    cw = W / 4.0
    y0 = pdf.get_y()
    pdf.set_text_color(*mut)
    pdf.set_font("wqy", "", 6.8)
    for i, (k, _v) in enumerate(cells):
        pdf.set_xy(pdf.l_margin + (i % 4) * cw, y0 + (i // 4) * 9.6)
        pdf.cell(cw - 2, 3.4, k)
    pdf.set_text_color(*ink)
    pdf.set_font("wqy", "", 8.2)
    for i, (_k, v) in enumerate(cells):
        pdf.set_xy(pdf.l_margin + (i % 4) * cw, y0 + (i // 4) * 9.6 + 3.6)
        pdf.cell(cw - 2, 4, str(v))
    pdf.set_y(y0 + 21)

    # —— A/B 两张图并排
    iw = (W - 6) / 2.0
    ya = pdf.get_y()
    hs = []
    for i, (buf, cap) in enumerate((
            (D["img_a"], L("A 原图（不加覆盖）", "A · original, no overlay")),
            (D["img_b"], L("B 脱落判定（红=脱落）", "B · removal (red = removed)")))):
        im = Image.open(io.BytesIO(buf))
        h = iw * im.height / max(1, im.width)
        w_use = iw
        if h > 104:                      # 竖构图别把整页吃掉，否则表格被挤到第 2 页只剩两行
            w_use = iw * 104.0 / h
            h = 104.0
        x = pdf.l_margin + i * (iw + 6) + (iw - w_use) / 2.0
        pdf.image(io.BytesIO(buf), x=x, y=ya, w=w_use)
        pdf.set_draw_color(203, 213, 225)
        pdf.set_line_width(0.25)
        pdf.rect(x, ya, w_use, h)
        hs.append(h)
        pdf.set_xy(x, ya + h + 1)
        pdf.set_font("wqy", "", 6.8)
        pdf.set_text_color(*mut)
        pdf.cell(iw, 3.4, cap)
    pdf.set_y(ya + max(hs) + 5.5)
    pdf.set_text_color(*mut)
    pdf.set_font("wqy", "", 6.8)
    pdf.multi_cell(W, 3.6, L("A 是原图，不加覆盖；B 把判定脱落的像素盖成半透明红（刻线刀缝 %d px 已先剔除，"
                             "刀缝本身不上色）。两图同尺度，可直接对照。",
                             "A is the untouched photo; B tints the pixels judged as removed in translucent "
                             "red (the %d px cut grooves are excluded first, so the grooves are not tinted). "
                             "Same scale, so the two can be compared directly.", D["k"]))
    pdf.ln(1)

    def table(head, rows, widths, aligns, total_row=False):
        pdf.set_font("wqy", "", 7.4)
        pdf.set_fill_color(243, 244, 246)
        pdf.set_text_color(55, 65, 81)
        for i, hlabel in enumerate(head):
            pdf.cell(widths[i], 5, hlabel, border="B", align=aligns[i], fill=True)
        pdf.ln(5)
        pdf.set_text_color(*ink)
        for n, row in enumerate(rows):
            fill = total_row and n == len(rows) - 1
            if fill:
                pdf.set_fill_color(243, 244, 246)
            pdf.set_font("wqy", "", 7.4)
            for i, v in enumerate(row):
                pdf.cell(widths[i], 5, str(v), border="B", align=aligns[i], fill=fill)
            pdf.ln(5)

    pdf.set_font("wqy", "", 8.6)
    pdf.set_text_color(55, 65, 81)
    pdf.cell(W, 5.5, L("各选区结果", "RESULTS PER REGION"), new_x="LMARGIN", new_y="NEXT")
    rows = [[ "R%d" % r["i"], "%.2f%%" % r["pct"], r["astm"], r["iso"], r["blocks"], r["area"],
              r["coat"], r["sub"], r["unassigned"], "%.2f%%" % r["raw"]] for r in D["rows"]]
    rows.append([L("合计", "Total"), "%.2f%%" % D["tot"]["pct"], D["tot"]["astm"], D["tot"]["iso"],
                 sum(r["blocks"] for r in D["rows"]), D["tot"]["area"], D["tot"]["coat"], D["tot"]["sub"],
                 D["tot"]["unassigned"], "%.2f%%" % D["tot"]["raw"]])
    table([L("选区", "Region"), L("脱落率", "Removal"), "ASTM", "ISO", L("块数", "Blocks"),
           L("面积 px", "Area px"), L("漆膜 px", "Coat px"), L("基材 px", "Sub px"),
           L("未分类 px", "Uncl. px"), L("原始脱落率", "Raw %")],
          rows, [15, 21, 12, 12, 13, 25, 25, 25, 19, 19],
          ["L", "R", "C", "C", "R", "R", "R", "R", "R", "R"], total_row=True)
    pdf.ln(2)

    pdf.set_font("wqy", "", 7.4)
    pdf.set_text_color(*mut)
    sw = " / ".join("%s %d,%d,%d" % (s["role"], s["rgb"][0], s["rgb"][1], s["rgb"][2])
                    for s in D["samples"])
    pdf.multi_cell(W, 4.2, "%s: %s\n%s" % (
        L("色样", "Colour samples"), sw,
        L("判级依据 ASTM D3359（方法 B）/ ISO 2409；数值由图像自动分析得出，边界样品建议人工复核。"
          "面积单位：%s像素，分母为「漆膜 + 露底」。",
          "Grading follows ASTM D3359 (Method B) / ISO 2409. Values are automated; borderline samples "
          "should be reviewed by hand. Area unit: %s px, denominator = coating + bare area.", D["units"])))
    pdf.ln(1)
    pdf.set_font("wqy", "", 6.6)
    pdf.multi_cell(W, 3.6, L("软件：百格附着力分析 v%s（算法 %s）· AGPL-3.0-or-later · © 2026 iamsad3508",
                             "Software: Baige cross-cut analyser v%s (algorithm %s) · AGPL-3.0-or-later · "
                             "© 2026 iamsad3508", D["app"], D["algo"]))
    return bytes(pdf.output())


class GridReq(AnalyzeReq):
    inset: float = 0.0           # 格子内缩（px，分析图口径）
    line_xs: list[float] = []    # 手动给定线位（回放/调试用，坐标为分析图口径）
    line_ys: list[float] = []
    # ---- 模型格网（v2.26）：不猜周期、不猜相位，格数/角度/刀缝宽由人给 ----
    model: bool = False
    cols: int = 0
    rows: int = 0
    angle: float = 0.0           # 格网倾角（度，逆时针为正，与 PIL rotate 同向）
    k_band: float = 0.0          # 刀缝带宽（wire px）；0 = 用 line_px
    auto_angle: bool = False     # 让服务端先估一个角度回来（只给建议，不改请求）
    auto_counts: bool = False    # 让服务端自己数格数（认得出才用，认不出沿用 cols/rows）
    auto_cols: bool = False      # 只认列（行由人定）；auto_counts 同时打开两个方向
    auto_rows: bool = False
    roles_auto: bool = False     # 颜色是自动分的（人没点过）→ 允许按格内面积纠正漆膜/基材角色
    ox: float = 0.0              # 整格网平移（wire px，图像坐标）
    oy: float = 0.0
    line_dx: list[float] = []    # 每根竖线的单独微调（wire px，沿格网 u 方向）
    line_dy: list[float] = []
    snap: bool = False           # 逐根线各自吸附到最近的暗缝上（透视/刀走偏时用）
    trace: bool = True           # 先认缝再按缝铺线（等分只在认不全时兜底）


def _model_geom(view, bbox, req):
    """模型格网的线位（视图 px）。等分 → 整体平移 → 逐线微调 → 可选逐根吸附。

    单独成函数是为了让「铺格」和「刀缝分离」用同一套线位：两边各算一次很容易悄悄分叉，
    而分叉之后图上画的线和统计用的线不是一套，最难看。
    返回 dict：xb/yb（各 cols+1、rows+1 根线在旋转后坐标系里的位置）、per_x/per_y、k、
    ang、cx/cy、ct/st（反向变换用）、snap_err。
    """
    x0, y0, x1, y1 = [float(v) for v in bbox]
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    bw, bh = max(4.0, x1 - x0), max(4.0, y1 - y0)
    cols, rows = max(2, int(req.cols)), max(2, int(req.rows))
    ang = float(req.angle or 0.0)
    auto_ang = 0.0
    # 歪角没填（或填了 0）而图是歪的：认缝按 0 度走会把一条斜缝摊到很多列上，连续性直接崩掉，
    # 于是“认不全 → 回退等分 → 看着对不准”。先自己估一个角度去认，认全了才采用。
    if getattr(req, "trace", True) and abs(ang) < 0.35:
        try:
            _g8 = np.asarray(view.work["rgb"]).astype(np.float32)
            if _g8.ndim == 3:
                _g8 = _g8.mean(2)
            _bx0, _by0 = int(max(0, x0)), int(max(0, y0))
            _bx1, _by1 = int(min(_g8.shape[1], x1)), int(min(_g8.shape[0], y1))
            _crop = _g8[_by0:_by1, _bx0:_bx1].clip(0, 255).astype(np.uint8)
            if _crop.size:
                _est = float(gf.estimate_angle(_crop))
                if abs(_est) >= 0.5:
                    auto_ang = _est
        except Exception:
            auto_ang = 0.0
    # 先认缝：刀缝是长条、漆膜是孤岛 → 用「连续性」区分，认全了就直接按缝铺线
    # （线落在沟的正中间；等分只在认不全时兜底，那才需要下面的逐根吸附去救）
    trace = None
    T = None
    if getattr(req, "trace", True):
        def _try_trace(a):
            try:
                _g = np.asarray(view.work["rgb"]).astype(np.float32)
                if _g.ndim == 3:
                    _g = _g.mean(2)
                _T = gf.trace_grid(_g.clip(0, 255).astype(np.uint8), a, (cx, cy),
                                   (x0, y0, x1, y1), cols, rows)
                _tr = {kk: _T.get(kk) for kk in ("ok_x", "ok_y", "n_x", "n_y", "pitch_x", "pitch_y",
                                                 "sign_x", "sign_y", "qual_x", "qual_y",
                                                 "w_x", "w_y", "msg")}
                _tr["angle"] = a
                return _T, _tr
            except Exception as _e:
                return None, {"ok_x": False, "ok_y": False, "msg": repr(_e), "angle": a}

        T, trace = _try_trace(ang)
        if auto_ang and abs(auto_ang - ang) >= 0.4:
            # 0° 常常也能认全（尤其图案本身是正的），但线压在缝边上。估出来的角度只要
            # 认得不差就采用它——否则界面上永远显示 0°，用户以为“自动定角没执行”。
            T2, tr2 = _try_trace(auto_ang)
            ok1 = int(bool(T.get("ok_x"))) + int(bool(T.get("ok_y")))
            ok2 = int(bool(T2.get("ok_x"))) + int(bool(T2.get("ok_y")))
            q1 = float(T.get("qual_x") or -1) + float(T.get("qual_y") or -1)
            q2 = float(T2.get("qual_x") or -1) + float(T2.get("qual_y") or -1)
            if ok2 > ok1 or (ok2 == ok1 and (q2 > q1 + 0.02 or ok2 == 0)):
                # ok2==ok1==0：两边都认不全，但估出来的角度是投影量出来的，比硬用 0° 更接近真实
                # （实测用户那张 CrossScan 截图：0° 一条缝都认不出，−1.35° 才铺得正）
                T, trace = T2, tr2
                ang = auto_ang
                th = math.radians(ang)          # 只在“按这个角度认得更全”时才改角度
                ct, st = math.cos(th), math.sin(th)

    kw = float(req.k_band) if req.k_band else float(req.line_px or 1)
    k = max(1.0, float(view.len_in(kw)))          # 视图 px
    per_x, per_y = bw / cols, bh / rows
    u0, v0 = cx - bw / 2.0, cy - bh / 2.0
    # 坐标系约定（合成图实测钉死的，别改符号）：
    #   _rot 用 PIL rotate(ang) 把「原图」转成「格网坐标」；PIL 的 +角度把 (du,dv) 送到
    #   (du·cos+dv·sin, -du·sin+dv·cos)，所以「原图→格网」= u=du·ct+dv·st、v=-du·st+dv·ct，
    #   「格网→原图」= x=cx+du·ct-dv·st、y=cy+du·st+dv·ct。两者都用 (ct,st)=(cos ang, sin ang)。
    #   这里以前写的是 radians(-ang)，等于把回算用成了正向矩阵 → 分析是对的、画出来的线却往
    #   反方向歪（用户看到的就是「自动定角给 -1.35，实际 +1.35 才像」）。
    th = math.radians(ang)
    ct, st = math.cos(th), math.sin(th)
    _ox, _oy = float(view.len_in(req.ox or 0.0)), float(view.len_in(req.oy or 0.0))
    _ca, _sa = math.cos(math.radians(ang)), math.sin(math.radians(ang))
    du, dv = _ox * _ca - _oy * _sa, _ox * _sa + _oy * _ca
    _dx = [float(view.len_in(v)) for v in (req.line_dx or [])]
    _dy = [float(view.len_in(v)) for v in (req.line_dy or [])]
    xi = [u0 + du + i * per_x for i in range(cols + 1)]
    yj = [v0 + dv + j * per_y for j in range(rows + 1)]
    xb = [xi[i] + (_dx[i - 1] if 0 < i <= len(_dx) else 0.0) for i in range(cols + 1)]
    yb = [yj[j] + (_dy[j - 1] if 0 < j <= len(_dy) else 0.0) for j in range(rows + 1)]
    snap_err = ""
    # 认全了的方向直接用真缝位置（线落在沟中间），并据此校正平均间距
    if T:
        # 真缝位置 + 用户自己拖的那点偏移（拖拽是叠加在真缝上，不能被覆盖掉）
        if T.get("ok_x"):
            xb = [float(T["xb"][i]) + (_dx[i - 1] if 0 < i <= len(_dx) else 0.0) for i in range(cols + 1)]
            per_x = (T["xb"][-1] - T["xb"][0]) / max(1, cols)
            trace["used_x"] = True
        if T.get("ok_y"):
            yb = [float(T["yb"][j]) + (_dy[j - 1] if 0 < j <= len(_dy) else 0.0) for j in range(rows + 1)]
            per_y = (T["yb"][-1] - T["yb"][0]) / max(1, rows)
            trace["used_y"] = True
    traced = bool(trace and (trace.get("used_x") or trace.get("used_y")))
    # 两个方向都认全了才完全跳过吸附；只认全一个方向时，另一个方向仍允许吸附兜底
    if getattr(req, "snap", False) and not (trace and trace.get("used_x") and trace.get("used_y")):
        try:
            g8 = np.asarray(view.work["rgb"]).astype(np.float32)
            if g8.ndim == 3:
                g8 = g8.mean(2)
            sx, sy = gf.snap_lines(g8.clip(0, 255).astype(np.uint8), ang, xb, yb,
                                   per_x, per_y, (cx, cy))
            if trace and trace.get("used_x"):
                sx = [0.0] * len(sx)          # 这个方向已经按真缝铺好了，别再动
            if trace and trace.get("used_y"):
                sy = [0.0] * len(sy)
            xb = [xb[i] + sx[i] for i in range(cols + 1)]
            yb = [yb[j] + sy[j] for j in range(rows + 1)]
        except Exception as _e:
            snap_err = repr(_e)
    return {"xb": xb, "yb": yb, "xi": xi, "yj": yj, "per_x": per_x, "per_y": per_y, "k": k, "ang": ang,
            "cx": cx, "cy": cy, "ct": ct, "st": st, "snap_err": snap_err,
            "cols": cols, "rows": rows, "trace": trace, "drag_x": _dx, "drag_y": _dy}


def _grid_model(view, labels, roles, union, bbox, req):
    """模型格网：选区外接框按 cols×rows 等分，逐格统计「格内剩余漆膜面积」。

    与自动拟合的分工：自动拟合去猜周期/相位（规则阵列好用，旧翻拍图上常直接失败）；
    这里格数、角度、刀缝宽都从请求来，工具只做两件事——把线画出来、把每格里的
    漆膜像素数出来。面积口径与主分析一致：只数被分成「漆膜」的像素，刀缝带按
    k/2 从每边扣掉（扣掉的既不算脱落、也不进分母）。
    """
    shape2 = union.shape

    coat = np.zeros(shape2, bool)
    sub = np.zeros(shape2, bool)
    for j, role in enumerate(roles):
        if role == cc.ROLE_COAT:
            coat |= (labels == j)
        elif role == cc.ROLE_SUB:
            sub |= (labels == j)

    def _rot(a):
        im = Image.fromarray((a * 255).astype(np.uint8))
        if abs(ang0) > 0.01:
            im = im.rotate(ang0, resample=Image.NEAREST, center=(cx0, cy0), fillcolor=0)
        return np.asarray(im) > 127

    G = _model_geom(view, bbox, req)
    x0, y0, x1, y1 = [float(v) for v in bbox]
    ang0 = G["ang"]; cx0 = G["cx"]; cy0 = G["cy"]
    coat_r, sub_r = _rot(coat), _rot(sub)
    cols, rows = G["cols"], G["rows"]
    k, ang = G["k"], G["ang"]
    cx, cy, ct, st = G["cx"], G["cy"], G["ct"], G["st"]
    ang0, cx0, cy0 = ang, cx, cy
    xb, yb = G["xb"], G["yb"]
    xi, yj = G["xi"], G["yj"]
    per_x, per_y = G["per_x"], G["per_y"]
    _snap_err = G["snap_err"]
    u0, v0 = xb[0], yb[0]
    bw = xb[-1] - xb[0]
    bh = yb[-1] - yb[0]

    def to_orig(u, v):
        du, dv = u - cx, v - cy
        x, y = cx + du * ct - dv * st, cy + du * st + dv * ct
        return [round(v2, 1) for v2 in view.to_wire(x, y)]

    def quad(a0, b0, a1, b1):
        return [to_orig(a0, b0), to_orig(a1, b0), to_orig(a1, b1), to_orig(a0, b1)]

    H, W = shape2
    cells, bands = [], []
    tot_a = tot_k = 0.0
    # 没吸样就没有任何"漆膜/露底"像素：这时不能把格子全丢掉——几何跟分类无关，
    # 该照常回给前端画框，只是不给百分比（用户点了"铺格"却什么都没出现，就是这里丢的）
    has_cls = bool((coat_r | sub_r).any())

    # ── 「谁是漆膜」按**格子内部**的面积判，而不是全框面积 ─────────────────────
    # 刀缝露白面积一大（参考图上那条宽带能占 40% 格距），全框面积会站在基材那边，
    # 角色一翻，百分比就变成 100−真值（实测同一张图 65.2% 与 34.8% 互为补数）。
    # 漆膜物理上盖住格子内部，基材集中在刀缝和崩边处，所以按格内计数是稳的。
    # 只在"颜色是自动分的"时才敢替用户翻（roles_auto），人点过的颜色一律不动。
    roles_flipped = False
    cell_coat = cell_sub = 0.0
    try:
        inner = np.zeros(shape2, bool)
        for i in range(cols):
            ia0 = max(0, int(round(xb[i] + k / 2.0)))
            ia1 = min(W, int(round(xb[i + 1] - k / 2.0)))
            if ia1 - ia0 < 2:
                continue
            for j in range(rows):
                ib0 = max(0, int(round(yb[j] + k / 2.0)))
                ib1 = min(H, int(round(yb[j + 1] - k / 2.0)))
                if ib1 - ib0 < 2:
                    continue
                inner[ib0:ib1, ia0:ia1] = True
        cell_coat = float((coat_r & inner).sum())
        cell_sub = float((sub_r & inner).sum())
        tot_in = max(cell_coat + cell_sub, 1.0)
        if (getattr(req, "roles_auto", False) and has_cls
                and cell_sub > 1.5 * max(cell_coat, 1.0) and cell_sub > 0.55 * tot_in):
            coat_r, sub_r = sub_r, coat_r
            cell_coat, cell_sub = cell_sub, cell_coat
            roles_flipped = True
    except Exception:
        pass
    for i in range(cols):
        for j in range(rows):
            a0 = xb[i] + k / 2.0
            a1 = xb[i + 1] - k / 2.0
            b0 = yb[j] + k / 2.0
            b1 = yb[j + 1] - k / 2.0
            ia0, ia1 = max(0, int(round(a0))), min(W, int(round(a1)))
            ib0, ib1 = max(0, int(round(b0))), min(H, int(round(b1)))
            if ia1 - ia0 < 2 or ib1 - ib0 < 2:
                continue
            # 分母只算「漆膜 + 露底」：过渡带/未分类的像素两头都不算，
            # 否则调高露底纯度会让 pct 反而变大（实测 0.50→0.75 从 36.6% 跳到 62.8%）
            pp = (coat_r[ib0:ib1, ia0:ia1] | sub_r[ib0:ib1, ia0:ia1])
            area = float(pp.sum())
            if area < 1:
                if has_cls:
                    continue                      # 有分类但这格全落在刀带里 → 这格没有可用像素
                cells.append({"i": i, "j": j, "quad": quad(a0, b0, a1, b1),
                              "area": 0.0, "kept": 0.0, "pct": None})
                continue
            kept = float((coat_r[ib0:ib1, ia0:ia1] & pp).sum())
            cells.append({"i": i, "j": j, "quad": quad(a0, b0, a1, b1),
                          "area": area, "kept": kept,
                          "pct": 100.0 * (1.0 - kept / area)})
            tot_a += area
            tot_k += kept

    # 每根内线单独回传「轴线号 + 法线方向」：前端靠它知道拖的是哪根、往哪边算偏移
    n_x = [ct, st]        # 格网 u 方向在图像坐标里的单位向量
    n_y = [-st, ct]       # v 方向
    for i in range(1, cols):
        u = xb[i]
        bands.append({"axis": "x", "idx": i, "n": [round(n_x[0], 6), round(n_x[1], 6)],
                      "quad": quad(u - k / 2.0, v0, u + k / 2.0, v0 + bh)})
    for j in range(1, rows):
        v = yb[j]
        bands.append({"axis": "y", "idx": j, "n": [round(n_y[0], 6), round(n_y[1], 6)],
                      "quad": quad(u0, v - k / 2.0, u0 + bw, v + k / 2.0)})

    # 回传每根内线相对「理想等分」的总偏移（wire px）：前端直接存，下次拖线在它上面累加
    # 给前端的两套数：shift_* 是“最终位置比等分挪了多少”（只用来提示），
    # used_* 回的是**用户拖拽量本身**。以前 used_* = 最终−等分，前端存下来下次又当拖拽
    # 叠加一遍，于是每跑一次就漂一点——换颜色/改刀宽重跑几次就彻底歪掉。
    _dx = G.get("drag_x") or []          # 用户拖拽量（在 _model_geom 里算的）
    _dy = G.get("drag_y") or []
    shift_dx = [round(float(view.len_out(xb[i] - xi[i])), 2) for i in range(1, cols)]
    shift_dy = [round(float(view.len_out(yb[j] - yj[j])), 2) for j in range(1, rows)]
    used_dx = [round(float(view.len_out(_dx[i - 1])), 2) if 0 < i <= len(_dx) else 0.0
               for i in range(1, cols)]
    used_dy = [round(float(view.len_out(_dy[j - 1])), 2) if 0 < j <= len(_dy) else 0.0
               for j in range(1, rows)]

    cell_area = (tot_a / len(cells)) if cells else max(1.0, (per_x - k) * (per_y - k))
    sc = view.len_out(1.0)                       # wire px / 视图 px；面积要平方
    return {"ok": True, "warn": "", "model": True, "angle": ang,
            "cols": cols, "rows": rows, "k": float(view.len_out(k)),
            "pitch_x": float(view.len_out(per_x)), "pitch_y": float(view.len_out(per_y)),
            "box": [float(view.to_wire(x0, y0)[0]), float(view.to_wire(x0, y0)[1]),
                    float(view.to_wire(x1, y1)[0]), float(view.to_wire(x1, y1)[1])],
            "bands": bands,
            "cells": [dict(c, area=round(c["area"] * sc * sc, 1),
                           kept=round(c["kept"] * sc * sc, 1)) for c in cells],
            "line_dx": used_dx, "line_dy": used_dy, "snap_err": _snap_err,
            "shift_dx": shift_dx, "shift_dy": shift_dy,
            "trace": G.get("trace"),
            "roles_flipped": roles_flipped,
            "cell_coat_frac": (round(cell_coat / max(cell_coat + cell_sub, 1.0), 3) if has_cls else None),
            "need_samples": (not has_cls),
            "total": {"pct": (100.0 * (1.0 - tot_k / max(tot_a, 1.0)) if has_cls else None),
                      "area": round(tot_a * sc * sc, 1),
                      "kept": round(tot_k * sc * sc, 1),
                      "cell_area": round(cell_area * sc * sc, 1),
                      "equiv": (tot_k / cell_area if has_cls else None)},
            # 等级按框选区的百分比给（与主分析同一套 cc.astm_class）；判法=格网 时结果卡直接用它
            "grade": (cc.astm_class(100.0 * (1.0 - tot_k / max(tot_a, 1.0))) if has_cls else None),
            "units": L("像素", "px")}



class AutoSampleReq(BaseModel):
    id: str
    res: str = "full"
    rois: list[ROI] = []


# ---------------------------------------------------------------------------
# 刀缝分离（v2.35）：把「刀划出来的沟」从「真脱落」里扣出来
# ---------------------------------------------------------------------------
# 为什么不做第三套「找格子」：这张样实测——没有任何一列/一行的暗像素占比超过 25%
# （score 峰值 0.25），刀缝是断续的、被光照切碎的；自动投影一定会混进假线（7 根里 2 根是假）。
# 所以这里不猜几何，直接用**模型格网铺好的那套线位**（人给的列数/行数/角度/刀宽，
# 可逐线拖、可自动对线），只负责把它们之间的距离场分成三层：
#     d <= k/2            刀缝本体      —— 与附着力无关，扣掉
#     k/2 < d <= k/2+edge 刀口崩边      —— 缝外被带下来的漆，单列
#     d > k/2+edge        格子内部      —— 这里才是「整格掉了」的脱落
# 三个数分开报，是为了让用户看到「等级是被哪一层撑着」，而不是一个揉在一起的百分比。

class SepReq(GridReq):
    edge_w: float = 60.0     # 崩边带宽（wire px）
    k_mult: float = 1.5      # 崩边阈值 = k_mult × 格内暗场 p98
    overlay: bool = True


def _dark_field(gray):
    """暗场：大核模糊 − 原图。细沟在这里变亮，均匀漆膜接近 0。"""
    def bz(a, r):
        return np.asarray(Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
                          .filter(ImageFilter.BoxBlur(int(r))), np.float32)
    g = np.asarray(gray, np.float32)
    return np.clip(bz(bz(g, 18), 18) - g, 0, None)


def _sep_overlay(seam, flare, blk, view, bbox, key, fw, fh) -> str:
    """三层上色：蓝=刀缝、橙=崩边、红=格内成块露底。坐标口径与画布一致。"""
    path = os.path.join(OVERLAYS, key + ".png")
    disp_side = min(1200, max(fh, fw))
    sc = min(1.0, disp_side / float(max(fh, fw)))
    ow, oh = max(1, int(round(fw * sc))), max(1, int(round(fh * sc)))
    if view is None:
        a, bx, by = sc, 0.0, 0.0
    else:
        a = view.s6 * sc / view.scale
        bx = view.ox * view.s6 * sc
        by = view.oy * view.s6 * sc
    y0, x0 = int(bbox[1]), int(bbox[0])
    out = np.zeros((oh, ow, 4), np.uint8)
    for m, rgba in ((seam, (70, 165, 255, 165)), (flare, (255, 190, 60, 205)), (blk, (255, 45, 45, 215))):
        if not m.any():
            continue
        sub = m[y0:y0 + int(bbox[3] - bbox[1]), x0:x0 + int(bbox[2] - bbox[0])]
        hh, ww = sub.shape
        tw, th = max(1, int(round(ww * a))), max(1, int(round(hh * a)))
        sm = np.asarray(Image.fromarray((sub * 255).astype(np.uint8))
                        .resize((tw, th), Image.NEAREST)) > 127
        oy, ox = int(round(y0 * a + by)), int(round(x0 * a + bx))
        h2 = min(th, oh - oy)
        w2 = min(tw, ow - ox)
        if h2 <= 0 or w2 <= 0:
            continue
        tgt = out[oy:oy + h2, ox:ox + w2]
        tgt[sm[:h2, :w2]] = rgba
    Image.fromarray(out, "RGBA").save(path, "PNG", optimize=True)
    return "api/overlay/%s.png" % key


@app.post("/api/separate")
def api_separate(req: SepReq):
    """按模型格网的线位把「刀缝 / 刀口崩边 / 格内脱落」分成三层各给一个数。"""
    if req.cols < 2 or req.rows < 2:
        raise HTTPException(400, L("先在格网里填列数/行数，再算分离（分离用的是你铺的那套线）",
                                   "Set columns/rows first — separation uses the grid you laid out"))
    sess = get_session(req.id)
    with HEAVY_BLOCK:
        with sess.lock:
            work, view = _view_for(sess, req)
            samples = [s.model_dump() for s in req.samples]
            roles = _roles(samples)
            rois_ = [r.model_dump() for r in req.rois]
            rois_v = view.rois_in(rois_) if rois_ else []
            shape2 = work["lab"].shape[:2]
            masks = _masks(shape2, rois_v) if rois_v else [np.ones(shape2, bool)]
            labels, *_ = _labels_for(sess, work, req.tol, req.purity, samples, masks,
                                     _msig(rois_v), view.key)

    union = np.zeros(shape2, bool)
    for m in masks:
        union |= m
    ys_i, xs_i = np.where(union)
    if xs_i.size == 0:
        raise HTTPException(400, L("先画一个选区", "Draw a region first"))
    gx0, gx1 = int(xs_i.min()), int(xs_i.max()) + 1
    gy0, gy1 = int(ys_i.min()), int(ys_i.max()) + 1

    rgb = work["rgb"]
    gray = np.asarray(rgb).astype(np.float32)
    if gray.ndim == 3:
        gray = gray @ np.array([0.299, 0.587, 0.114], np.float32)
    D = _dark_field(gray)

    G = _model_geom(view, (gx0, gy0, gx1, gy1), req)
    xb, yb = G["xb"], G["yb"]
    k, cx, cy, ct, st = G["k"], G["cx"], G["cy"], G["ct"], G["st"]

    # 距离场：旋转坐标是等距变换，所以 |U - 线位| 就是到那条线的垂直距离
    H2, W2 = shape2
    YY, XX = np.mgrid[0:H2, 0:W2].astype(np.float32)
    dxp, dyp = XX - cx, YY - cy
    U = cx + dxp * ct + dyp * st
    V = cy - dxp * st + dyp * ct
    dU = np.full((H2, W2), 1e6, np.float32)
    for x in xb:
        dU = np.minimum(dU, np.abs(U - float(x)))
    dV = np.full((H2, W2), 1e6, np.float32)
    for y in yb:
        dV = np.minimum(dV, np.abs(V - float(y)))
    dist = np.minimum(dU, dV)

    ew = float(view.len_in(req.edge_w))
    seam = dist <= k / 2.0
    edge = (dist > k / 2.0) & (dist <= k / 2.0 + ew)
    inner = (dist > k / 2.0 + ew) & union
    if inner.sum() < 0.05 * max(1, int(union.sum())):
        raise HTTPException(400, L("刀宽/崩边带宽设得太大，选区里没剩多少格子了",
                                   "Blade width / edge band too wide — nothing left inside the cells"))
    base = float(np.percentile(D[inner], 98.0))
    thr = req.k_mult * base
    flare = edge & (D > thr) & union

    cand = (D > thr) & inner
    er = np.asarray(Image.fromarray((cand * 255).astype(np.uint8))
                    .filter(ImageFilter.MinFilter(7)), np.float32) > 127
    blk = np.asarray(Image.fromarray((er * 255).astype(np.uint8))
                     .filter(ImageFilter.MaxFilter(7)), np.float32) > 127
    blk &= inner

    n_inner = float(inner.sum())
    pct_flare = 100.0 * float(flare.sum()) / n_inner
    pct_blk = 100.0 * float(blk.sum()) / n_inner
    sweep = []
    for f in (1.35, 1.5, 1.7, 1.9, 2.2):
        n = int((edge & (D > f * base) & union).sum())
        p = 100.0 * n / n_inner
        sweep.append({"k": f, "pct": round(p, 2), "astm": cc.astm_class(p)})

    out = {"ok": True, "cols": G["cols"], "rows": G["rows"],
           "k_wire": round(float(view.len_out(k)), 1), "edge_w": req.edge_w,
           "base_dark": round(base, 1), "k_mult": req.k_mult,
           "seam_pct": round(100.0 * float((seam & union).sum()) / max(1.0, float(union.sum())), 1),
           "flare_pct": round(pct_flare, 2), "blk_pct": round(pct_blk, 2),
           "astm": cc.astm_class(pct_flare + pct_blk), "iso": cc.iso_class(pct_flare + pct_blk),
           "sweep": sweep, "snap_err": G["snap_err"],
           "cell_area": int(n_inner)}
    if req.overlay:
        key = "sep_%s_%s_%d_%d" % (
            sess.id[:8],
            hashlib.sha1(("%s|%s|%s|%s" % (round(k, 2), [round(v, 1) for v in xb],
                                           [round(v, 1) for v in yb], req.edge_w)).encode()).hexdigest()[:10],
            int(req.edge_w), int(req.k_mult * 100))
        fw = int(round(sess.size[0] * view.s6))
        fh = int(round(sess.size[1] * view.s6))
        out["overlay_url"] = _sep_overlay(seam, flare, blk, view,
                                          (gx0, gy0, gx1, gy1), key, fw, fh)
    return out


@app.post("/api/autosample")
def api_autosample(req: AutoSampleReq):
    """按亮度把选区里的像素分成两簇，给出一对色样（漆膜 + 露底）。

    角色**不能按亮度定**：亮=漆膜的假设在「深色漆膜 / 浅色基材」的样品上会正好反
    （实测 ERICHSEN 参考图黑漆+白底：亮线被当漆膜 → 65.2%，反过来才是 34.8%）。
    物理先验是「漆膜盖住大部分面积，基材只在刀缝和掉块处露出来」：
      1) 面积大的那簇 = 漆膜；
      2) 例外——它近乎无彩（灰/白）而另一簇明显有彩（≥12）且占 ≥15%，则有彩的是漆膜
         （与半自动认色 _group_colours 同一条规则，两条自动路径不能各说各的）。
    只是起点：分错了图上点一下就能改。
    """
    sess = get_session(req.id)
    with HEAVY_BLOCK:
        with sess.lock:
            work, view = _view_for(sess, req.model_dump())
            rgb = work["rgb"]
            rois_v = view.rois_in([r.model_dump() for r in req.rois]) if req.rois else []
            masks = _masks(rgb.shape[:2], rois_v) if rois_v else [np.ones(rgb.shape[:2], bool)]
            m = np.zeros(rgb.shape[:2], bool)
            for mm in masks:
                m |= mm
            if not m.any():
                raise HTTPException(400, L("先画一个选区", "Draw a region first"))
            px = rgb[m].astype(np.float32)
            lum = px @ np.array([0.299, 0.587, 0.114], np.float32)
            lo, hi = float(np.percentile(lum, 1)), float(np.percentile(lum, 99))
            thr = 0.5 * (lo + hi)                    # 先用中位幅度切一刀
            for _ in range(6):                       # 再做几轮两均值迭代（一维 k-means）
                d = lum >= thr
                if d.all() or (~d).all():
                    break
                m1, m2 = float(lum[d].mean()), float(lum[~d].mean())
                thr = 0.5 * (m1 + m2)
            d = lum >= thr
            if d.sum() < 50 or (~d).sum() < 50:
                raise HTTPException(400, L("选区里只有一种颜色，分不出漆膜和底材",
                                           "Only one colour in the region"))
            bright = np.median(px[d], axis=0).round().astype(int).tolist()
            dark = np.median(px[~d], axis=0).round().astype(int).tolist()
            n_b, n_d = int(d.sum()), int((~d).sum())
            total = n_b + n_d

            def _chroma(c):
                v = cc.rgb_to_lab(np.array(c, np.float32))
                return float(np.hypot(float(v[1]), float(v[2])))

            # 「谁盖住面积」不如「谁成块、谁成网」：漆膜被刀缝切成一个个格子（连通块很多、
            # 单块占比小），基材是连通的一整张线网（块很少、最大块几乎占满）。
            # 光靠面积会在两种常见情况下判反：刀缝很宽（露白过半）、掉块成片（基材占多数）。
            def _topo(mask):
                tot = float(mask.sum())
                if tot < 200:
                    return 0, 1.0
                step = max(1, int(round(max(mask.shape) / 320.0)))
                sm = mask[::step, ::step]
                lab, _runs, stats = cc._components(sm)
                areas = sorted((v[0] for v in stats.values()), reverse=True)
                if not areas:
                    return 0, 1.0
                n = sum(1 for a in areas if a >= 0.004 * float(sm.sum()))
                return n, (areas[0] / max(1.0, float(sum(areas))))

            # 腐蚀：细线先没、格子还在（项目只有 numpy + PIL，不引 scipy/opencv）
            def _survive(mask, rad):
                k = 2 * int(rad) + 1
                im = Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.MinFilter(k))
                return float((np.asarray(im) > 127).sum()) / max(1.0, float(mask.sum()))

            # d 是「像素列表」上的布尔（px = rgb[m]），拓扑要二维图：按同样的 C 序填回家
            d2 = np.zeros(m.shape, bool)
            d2[m] = d
            s2 = m & ~d2
            nb, bigb = _topo(d2)
            ndk, bigd = _topo(s2)
            thick = "area"
            # 连通块数量差 3 倍以上 → 多的那簇是漆膜（格子被刀缝切开，块数≈格数）
            if max(nb, ndk) >= 3 * max(1, min(nb, ndk)) and max(nb, ndk) >= 4:
                coat_is_bright = nb > ndk
                thick = "cells"
            else:
                rad = max(2, int(round(0.02 * min(m.shape))))
                s_b, s_d = _survive(d2, rad), _survive(s2, rad)
                if max(s_b, s_d) > 0 and max(s_b, s_d) >= 1.6 * max(1e-6, min(s_b, s_d)):
                    coat_is_bright = s_b > s_d
                    thick = "shape"
                else:
                    coat_is_bright = n_b >= n_d      # 都分不出来 → 退回面积（多数情况是漆膜）

            coat_rgb, sub_rgb = (bright, dark) if coat_is_bright else (dark, bright)
            n_coat = n_b if coat_is_bright else n_d
            if _chroma(coat_rgb) < 8.0:
                other, n_other = (dark, n_d) if n_coat == n_b else (bright, n_b)
                if _chroma(other) >= 12.0 and n_other >= 0.15 * total:
                    coat_rgb, sub_rgb, n_coat = other, coat_rgb, n_other
            samples = [{"rgb": coat_rgb, "role": cc.ROLE_COAT},
                       {"rgb": sub_rgb, "role": cc.ROLE_SUB}]
            sp = _span_of(samples)
            frac = n_coat / max(1, total)
            warn = ""
            if frac < 0.60:
                warn = L("两类面积差不多（漆膜只占 %.0f%%），自动分色不一定对——"
                         "图上点一下完好漆膜、再点一下露底就能改" % (100 * frac),
                         "The two colours cover similar areas (coating %.0f%%), so the automatic "
                         "pick may be wrong — click the intact coating, then the bare substrate" % (100 * frac))
            return {"ok": True, "samples": samples, "span": sp, "auto_tol": _auto_tol_of(sp),
                    "ratio_bright": round(float(d.mean()), 3),
                    "coat_frac": round(frac, 3), "role_by": thick,
                    "shape": [round(float(nb), 1), round(float(ndk), 1)], "warn": warn}


def _count_estimate(gray, box):
    """自己在选区里数格数：估角 → 转正 → 自相关定格距 → 格数 = 边长 / 格距。

    两道闸，不放心就不改用户的数：
      ① 自相关残差 ≤ 0.12×格距（周期不整齐说明不是规则阵列）；
      ② 拿这个格数真的认一遍缝，两个方向都认全（trace_grid ok_x & ok_y）才算数。
    认不出返回 None → 调用方沿用请求里的 cols/rows，并在界面上说明“没认出来”。
    """
    try:
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        sub = np.asarray(gray[y0:y1, x0:x1]).astype(np.uint8)
        if min(sub.shape) < 60:
            return None
        ang = float(gf.estimate_angle(sub))
        fill = int(np.median(sub))
        rot = np.asarray(Image.fromarray(sub, "L").rotate(ang, resample=Image.BILINEAR, fillcolor=fill))
        r = gf.fit(rot)
        if not r.get("ok"):
            return None
        px, py = float(r.get("period_x") or 0), float(r.get("period_y") or 0)
        if px < 6 or py < 6:
            return None
        if float(r.get("resid_x") or 0) > 0.12 * px or float(r.get("resid_y") or 0) > 0.12 * py:
            return None
        cols = int(round((x1 - x0) / px))
        rows = int(round((y1 - y0) / py))
        if not (2 <= cols <= 40 and 2 <= rows <= 40):
            return None
        t = gf.trace_grid(gray, ang, ((x0 + x1) / 2.0, (y0 + y1) / 2.0), (x0, y0, x1, y1), cols, rows)
        conf = "trace" if (t.get("ok_x") and t.get("ok_y")) else "period"
        # 认缝复核不过时不硬拒：格距（自相关）本身是稳的，逐根认缝在原生分辨率/边缘上会失手
        # （实测同一张图 roi_native=True 时一个框 11/11、旁边挪 10px 的框 1/11）。降级成 period，
        # 界面照实说“刀缝没认全，格数是按格距算的”，用户看一眼就能改。
        return {"cols": cols, "rows": rows,
                "pitch_x": round(px, 1), "pitch_y": round(py, 1),
                "angle": round(ang, 2), "conf": conf,
                "w_x": t.get("w_x"), "w_y": t.get("w_y")}
    except Exception:
        return None


@app.post("/api/grid")
def api_grid(req: GridReq):
    """实验：在选区范围内自动拟合格网，逐格算脱落率。

    只为「规则阵列」设计（激光划格 / 刀割百格）：先在大核背景差图上做投影，
    用自相关定周期、最小二乘拟合相位，再按实测槽宽切出格子。
    失败时返回 warn（格网.歪了 / 找不到线），前端提示用户先转正或手画选区。
    """
    sess = get_session(req.id)
    with HEAVY_BLOCK:
        with sess.lock:
            work, view = _view_for(sess, req)
            samples = [s.model_dump() for s in req.samples]
            roles = _roles(samples)
            rois_ = [r.model_dump() for r in req.rois]
            rois_v = view.rois_in(rois_) if rois_ else []
            shape2 = work["lab"].shape[:2]
            masks = _masks(shape2, rois_v) if rois_v else [np.ones(shape2, bool)]
            labels, *_ = _labels_for(sess, work, req.tol, req.purity, samples, masks,
                                     _msig(rois_v), view.key)

    union = np.zeros(shape2, bool)
    for m in masks:
        union |= m
    ys_i, xs_i = np.where(union)
    if xs_i.size == 0:
        raise HTTPException(400, L("先画一个选区再拟合格网", "Draw a region before fitting the grid"))
    gx0, gx1 = int(xs_i.min()), int(xs_i.max()) + 1
    gy0, gy1 = int(ys_i.min()), int(ys_i.max()) + 1
    rgb = work["rgb"]
    gray = np.asarray(rgb).astype(np.float32)
    if gray.ndim == 3:
        gray = gray.mean(2)
    sub = np.ascontiguousarray(gray[gy0:gy1, gx0:gx1]).clip(0, 255).astype(np.uint8)

    if req.model and (req.auto_counts or req.auto_cols or req.auto_rows or (req.cols >= 2 and req.rows >= 2)):
        cnt = None
        want_c = req.auto_counts or req.auto_cols
        want_r = req.auto_counts or req.auto_rows
        if want_c or want_r or req.cols < 2 or req.rows < 2:
            cnt = _count_estimate(gray, (gx0, gy0, gx1, gy1))
            # 格数采用策略：认缝复核过（trace）直接用；只量到格距（period）也先用上——
            # 用户要的是"框完就自动"，而格距是自相关量出来的、残差过闸的，比默认 9×7 靠谱得多。
            # 面板必须写明这是按格距算的、可以改。实测用户那张 CrossScan 截图：按格距 11×11 +
            # −1.35° 得 35.9%（厂家软件自己显示 33.4%），而 9×7 是 52.5%。
            if cnt:
                req = req.model_copy(update={
                    "cols": cnt["cols"] if (want_c or req.cols < 2) else req.cols,
                    "rows": cnt["rows"] if (want_r or req.rows < 2) else req.rows})
        if req.cols < 2 or req.rows < 2:
            return {"ok": False, "warn": "grid.nocount",
                    "reason": L("认不出格数：把框贴着整片百格画紧一点，或手填列/行数",
                                "Could not count the cells — frame the grid tightly or type columns/rows")}
        out = _grid_model(view, labels, roles, union, (gx0, gy0, gx1, gy1), req)
        out["counts_est"] = cnt
        if req.auto_angle:
            out["angle_est"] = gf.estimate_angle(sub)
        if req.overlay:
            # 判定染色（红=露底 / 绿=漆膜）与主分析走同一条 _build_overlay，
            # 免得「格网报 33%、覆盖层却是另一套参数」这种对不上的情况
            try:
                side = _res_to_side(req.res, sess.full_img())
                out["overlay_url"] = _build_overlay(
                    sess, work, samples, req.tol, req.purity, side,
                    masks if rois_v else None, _msig(rois_v),
                    view=view if view.native else None,
                    full_size=(int(round(sess.size[0] * view.s6)), int(round(sess.size[1] * view.s6))),
                    line_px=req.line_px, mode=getattr(req, "mode", "width"))
            except Exception:
                out["overlay_url"] = None
        return out

    if req.line_xs and req.line_ys:
        r = {"ok": True, "warn": "", "angle": 0.0, "k_x": float(req.line_px), "k_y": float(req.line_px),
             "period_x": float(np.median(np.diff(req.line_xs))) if len(req.line_xs) > 1 else 0.0,
             "period_y": float(np.median(np.diff(req.line_ys))) if len(req.line_ys) > 1 else 0.0,
             "resid_x": 0.0, "resid_y": 0.0, "strength_x": 0.0, "strength_y": 0.0,
             "xs": list(req.line_xs), "ys": list(req.line_ys), "scale": 1.0}
    else:
        r = gf.fit(sub)
        if not r.get("ok"):
            return {"ok": False, "warn": r.get("warn") or "grid.nolines",
                    "angle": r.get("angle", 0.0), "dark_frac": r.get("dark_frac", 0.0)}
        r["xs"] = [gx0 + v for v in r["xs"]]
        r["ys"] = [gy0 + v for v in r["ys"]]

    cells = gf.cells(r["xs"], r["ys"], r["k_x"], r["k_y"], inset=float(req.inset))

    sub_all = np.zeros(shape2, bool)
    for j, role in enumerate(roles):
        if role == cc.ROLE_SUB:
            sub_all |= (labels == j)
    sub_all &= union
    k_view = max(1, int(round(view.len_in(req.line_px))))
    kept, *_ = cc.keep_mask(sub_all, k_view, req.mode, int(req.loc_tile), float(req.loc_factor))

    out, tot_a, tot_r = [], 0.0, 0.0
    for c in cells:
        bx0, by0, bx1, by1 = [int(round(v)) for v in c["box"]]
        bx0, by0 = max(0, bx0), max(0, by0)
        bx1, by1 = min(shape2[1], bx1), min(shape2[0], by1)
        if bx1 <= bx0 or by1 <= by0:
            continue
        u = union[by0:by1, bx0:bx1]
        a = float(u.sum())
        if a <= 0:
            continue
        rem = float((kept[by0:by1, bx0:bx1] & u).sum())
        pct = 100.0 * rem / max(1.0, a)
        out.append({"i": int(c["i"]), "j": int(c["j"]),
                    "box": [round(float(v), 1) for v in c["box"]],
                    "area": a, "removed": rem, "pct": pct, "astm": cc.astm_class(pct)})
        tot_a += a
        tot_r += rem
    total = 100.0 * tot_r / max(1.0, tot_a)
    box = [r["xs"][0], r["ys"][0], r["xs"][-1], r["ys"][-1]]
    # 自动寻线只认到零星几格时，别让它当成正常结果交出去（旧图/翻拍图上很常见）：
    # 用户看到"6 格 + 9.5%"会以为在算他那张图，其实框里大半没被格网盖住
    _warn = r.get("warn") or ""
    if not _warn:
        # 判据用「格网跨度 / 选区尺寸」：正常图用户框得紧，跨度≈100%；
        # 认错周期时格网只盖住一半，跨度明显缩水（实测 904 那张 96%+，这张只有 70~80%）
        _rw = max(1.0, float(gx1 - gx0))
        _rh = max(1.0, float(gy1 - gy0))
        _sx = (r["xs"][-1] - r["xs"][0]) / _rw if len(r["xs"]) > 1 else 0.0
        _sy = (r["ys"][-1] - r["ys"][0]) / _rh if len(r["ys"]) > 1 else 0.0
        if len(out) < 4 or _sx < 0.85 or _sy < 0.85:
            _warn = L("只自动铺了 %d 格，格网盖住选区的 %.0f%%×%.0f%%（框画松了也会这样）。"
                      "要是线不对，别在这条路上耗——用下面的「模型格网」自己给列数和行数。",
                      "Auto-fit placed only %d cells; the lattice covers %.0f%%×%.0f%% of the region "
                      "(a loose frame does this too). If the lines look wrong, skip this path and use "
                      "the model grid below.",
                      len(out), 100.0 * min(1.0, _sx), 100.0 * min(1.0, _sy))
    return {"ok": True, "warn": _warn, "angle": r.get("angle", 0.0),
            "box": box, "scope": L("视场", "field"),
            "period_x": r.get("period_x", 0.0), "period_y": r.get("period_y", 0.0),
            "resid_x": r.get("resid_x", 0.0), "resid_y": r.get("resid_y", 0.0),
            "k_x": r.get("k_x", 0.0), "k_y": r.get("k_y", 0.0),
            "xs": r["xs"], "ys": r["ys"], "cells": out,
            "total": {"pct": total, "astm": cc.astm_class(total), "area": tot_a, "removed": tot_r}}


@app.post("/api/export/report")
async def export_report(req: ExportReq):
    """一页式报告（取代原来的 CSV 导出按钮）。CSV 接口保留给脚本用。"""
    sess = get_session(req.id)

    want_pdf = (req.fmt or "pdf").lower() != "html"

    def run():
        with HEAVY_BLOCK:
            with sess.lock:
                D = _report_data(sess, req)
                return _report_pdf(D) if want_pdf else _report_html(D)

    data = await run_in_threadpool(run)
    import urllib.parse
    base = (req.name or "结果").rsplit(".", 1)[0]
    if want_pdf:
        name = base + "_百格报告.pdf"
        mt = "application/pdf"
    else:
        name = base + "_百格报告.html"
        mt = "text/html; charset=utf-8"
    return Response(data, media_type=mt,
                    headers={"Content-Disposition": "attachment; filename*=UTF-8''%s"
                             % urllib.parse.quote(name)})


@app.post("/api/export/csv")
async def export_csv(req: ExportReq):
    sess = get_session(req.id)

    def run():
        with HEAVY_BLOCK:
            with sess.lock:
                return _csv_bytes(sess, req)

    data = await run_in_threadpool(run)
    name = (req.name or "结果").rsplit(".", 1)[0] + "_百格.csv"
    return Response(data, media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": "attachment; filename*=UTF-8''%s"
                             % __import__("urllib.parse").parse.quote(name)})


@app.post("/api/export/png")
async def export_png(req: ExportReq):
    sess = get_session(req.id)

    def run():
        with HEAVY_BLOCK:
            with sess.lock:
                return _annotated_png(sess, req)

    data = await run_in_threadpool(run)
    name = (req.name or "结果").rsplit(".", 1)[0] + "_标注.png"
    return Response(data, media_type="image/png",
                    headers={"Content-Disposition": "attachment; filename*=UTF-8''%s"
                             % __import__("urllib.parse").parse.quote(name)})


# ---------------------------------------------------------------- 记录（手工保存，永不自动删除）
# 与 uploads 的会话缓存分开：SESSIONS/uploads 会按时间被清理，records 只在用户点删除时删。

RECORDS = os.path.join(DATA, "records")
os.makedirs(RECORDS, exist_ok=True)
MAX_RECORDS = 400
RECORD_VIEW_SIDE = 3000       # 存档图长边上限（分析本来就是 ≤6 MP，存 3000 足够精确复现）
RECORD_MIN_FREE_MB = 800


class SaveReq(BaseModel):
    id: str
    name: str = ""
    note: str = ""
    res: str = "full"
    rois: list[ROI] = []
    samples: list[Sample] = []
    tol: float = 25.0
    purity: float = 0.5
    line_px: int = 1
    mode: str = "width"
    roi_native: bool = False
    # 手动排除块之后，前端手里的“已生效”结果（不传就存自动口径）
    pct: float | None = None
    astm: str = ""
    iso: str = ""
    excl: list[list[int]] | None = None
    knee_warn: str = ""          # 判法没定出 k 时的警告文本（存起来，重开记录还看得见）
    knee_k: int | None = None
    # 模型格网（实验块）的快照：不存的话「记录里没有那些格子」，重开也回不到当时的格网
    gm: dict | None = None


def _rec_dir(rid: str) -> str:
    if not rid or "/" in rid or ".." in rid:
        raise HTTPException(400, L("记录号不合法", "Invalid record id"))
    return os.path.join(RECORDS, rid)


def _rec_json(rid: str) -> dict:
    p = os.path.join(_rec_dir(rid), "record.json")
    if not os.path.exists(p):
        raise HTTPException(404, L("记录不存在", "Record not found"))
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _rec_brief(rid: str) -> dict | None:
    try:
        d = _rec_json(rid)
    except Exception:
        return None
    res = d.get("result") or {}
    return {
        "id": rid,
        "name": d.get("name") or rid,
        "created": d.get("created"),
        "w": d.get("w"), "h": d.get("h"),
        "n_roi": len(d.get("rois") or []),
        "pct": res.get("pct"),
        "astm": res.get("astm"),
        "iso": res.get("iso"),
        "worst_pct": (res.get("worst") or {}).get("pct"),
        "worst_roi": (res.get("worst") or {}).get("roi"),
        "manual": len([i for a in (res.get("excl") or []) for i in (a or [])]),
        "auto_pct": res.get("auto_pct"),
        "gm": ({"cols": (d.get("gm") or {}).get("cols"), "rows": (d.get("gm") or {}).get("rows"),
                "pct": ((d.get("gm") or {}).get("total") or {}).get("pct"),
                "angle": (d.get("gm") or {}).get("angle"),
                "n": len((d.get("gm") or {}).get("cells") or [])}
               if d.get("gm") else None),
        "tol": d.get("tol"), "line_px": d.get("line_px"),
        "bytes": sum(os.path.getsize(os.path.join(_rec_dir(rid), f))
                     for f in os.listdir(_rec_dir(rid))
                     if os.path.isfile(os.path.join(_rec_dir(rid), f))),
    }


@app.post("/api/records/save")
def records_save(req: SaveReq):
    sess = get_session(req.id)
    names = [n for n in os.listdir(RECORDS) if os.path.isdir(os.path.join(RECORDS, n))]
    if len(names) >= MAX_RECORDS:
        raise HTTPException(409, L("记录已满（%d 条），请先删掉一些",
                                       "Record limit reached (%d) — delete some first", MAX_RECORDS))
    try:
        free_mb = shutil.disk_usage(RECORDS).free / 1e6
        if free_mb < RECORD_MIN_FREE_MB:
            raise HTTPException(507, L("服务器剩余空间只有 %.0f MB，先别存了",
                                       "Only %.0f MB left on the server — stop saving for now", free_mb))
    except HTTPException:
        raise
    except Exception:
        pass

    # 同一张图 + 同一组选区 → 就地更新，不再堆出一堆同名记录
    img_sha = hashlib.sha1(open(sess.path, "rb").read()).hexdigest()
    roi_sig = json.dumps([[round(float(v), 1) for v in r.box] + [round(float(r.ang), 1)]
                          for r in req.rois], sort_keys=True)
    rid = None
    upd = False
    cand = []                       # 同图 + 同选区的候选；优先挑原始文件名也一样的
    for name in names:
        try:
            old = _rec_json(name)
        except Exception:
            continue
        if old.get("img_sha1") == img_sha and old.get("roi_sig") == roi_sig:
            cand.append(name)
            if old.get("sess_name", "") == (sess.name or ""):
                rid = name
                upd = True
                break
    if rid is None and cand:
        rid = cand[0]
        upd = True
    if rid is None:
        rid = time.strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(2)
    d = _rec_dir(rid)
    os.makedirs(d, exist_ok=True)
    # 存档图 = 上传时的原始文件，字节级拷贝，绝不重新编码/缩放：
    # 重开时分析的就是同一张图，结果逐位一致（缩到别的长边会换掉分析尺度，数字漂 0.2~0.3 pp）
    view_name = "view" + (".png" if sess.path.lower().endswith(".png") else ".jpg")
    view = os.path.join(d, view_name)
    shutil.copyfile(sess.path, view)
    th = Image.open(view)
    th.thumbnail((320, 320), Image.LANCZOS)
    th.save(os.path.join(d, "thumb.jpg"), "JPEG", quality=82)
    with Image.open(view) as _v:
        im_size = _v.size

    out = _analyze_core(sess, AnalyzeReq(
        id=req.id, res=req.res, rois=req.rois, samples=req.samples, tol=req.tol,
        purity=req.purity, line_px=req.line_px, mode=req.mode,
        want_hist=False, detail=False, overlay=False, roi_native=req.roi_native))
    total = (out.get("total") or {}) if isinstance(out, dict) else {}
    rec = {
        "id": rid,
        "name": (req.name or sess.name or "未命名").strip()[:80],
        "note": (req.note or "").strip()[:400],
        "created": (time.strftime("%Y-%m-%d %H:%M:%S") if not upd else
                    (_rec_json(rid).get("created") or time.strftime("%Y-%m-%d %H:%M:%S"))),
        "saved_at": time.time(),
        "updated": (time.strftime("%Y-%m-%d %H:%M:%S") if upd else None),
        "img_sha1": img_sha,
        "roi_sig": roi_sig,
        "view": view_name,
        "sess_name": sess.name or "",
        "w": im_size[0], "h": im_size[1],
        "ow": sess.size[0], "oh": sess.size[1],
        "res": req.res,
        "samples": [s.model_dump() for s in req.samples],
        "algo": ALGO_VERSION,
        "tol": req.tol, "purity": req.purity, "line_px": req.line_px, "mode": req.mode,
        "roi_native": bool(req.roi_native),
        "rois": [{"box": list(r.box), "ang": r.ang} for r in req.rois],
        "suggest_px": out.get("suggest_px") if isinstance(out, dict) else None,
        "knee_warn": req.knee_warn, "knee_k": req.knee_k,
        "result": {
            "pct": total.get("pct"), "astm": total.get("astm"), "iso": total.get("iso"),
            "worst": total.get("worst"),
            "rows": [{"pct": r.get("pct"), "astm": r.get("astm"), "iso": r.get("iso")}
                     for r in (out.get("rows") or [])],
        },
    }
    if req.gm:
        g = req.gm
        def _r(v, n=2):
            try:
                return round(float(v), n)
            except Exception:
                return None
        rec["gm"] = {
            "cols": int(g.get("cols") or 0), "rows": int(g.get("rows") or 0),
            "k_wire": _r(g.get("k_wire"), 1), "angle": _r(g.get("angle"), 2),
            "ox": _r(g.get("ox"), 2), "oy": _r(g.get("oy"), 2),
            "line_dx": [_r(v, 1) for v in (g.get("line_dx") or [])],
            "line_dy": [_r(v, 1) for v in (g.get("line_dy") or [])],
            "snap": bool(g.get("snap")),
            "total": {"pct": _r(g.get("total_pct"), 2),
                      "kept": _r(g.get("total_kept"), 0),
                      "area": _r(g.get("total_area"), 0)},
            "trace": g.get("trace") or None,
            "pitch": [g.get("pitch_x"), g.get("pitch_y")],
            "cells": [{"i": int(c.get("i") or 0), "j": int(c.get("j") or 0),
                       "pct": _r(c.get("pct"), 2), "kept": _r(c.get("kept"), 0)}
                      for c in (g.get("cells") or [])][:1600],
        }
    # 手动排除过块：存“已生效”的数字 + 自动值 + 排除了哪几块（重开时按编号复原）
    if req.pct is not None:
        rec["result"]["auto_pct"] = total.get("pct")
        rec["result"]["pct"] = req.pct
        rec["result"]["astm"] = req.astm or total.get("astm")
        rec["result"]["iso"] = req.iso or total.get("iso")
        rec["result"]["worst"] = None
        rec["result"]["excl"] = req.excl or []
    with open(os.path.join(d, "record.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    return {"ok": True, "updated": upd, "record": _rec_brief(rid)}


@app.get("/api/records/count")
def records_count():
    n = sum(1 for name in os.listdir(RECORDS) if os.path.isdir(os.path.join(RECORDS, name)))
    return {"ok": True, "total": n, "max": MAX_RECORDS}


@app.get("/api/records")
def records_list():
    items = []
    for n in os.listdir(RECORDS):
        if os.path.isdir(os.path.join(RECORDS, n)):
            b = _rec_brief(n)
            if b:
                items.append(b)
    items.sort(key=lambda x: x.get("created") or "", reverse=True)
    try:
        free_mb = shutil.disk_usage(RECORDS).free / 1e6
    except Exception:
        free_mb = None
    return {"ok": True, "items": items, "total": len(items), "max": MAX_RECORDS,
            "bytes": sum(i["bytes"] for i in items), "free_mb": free_mb}


@app.get("/api/records/{rid}/thumb.jpg")
def records_thumb(rid: str):
    p = os.path.join(_rec_dir(rid), "thumb.jpg")
    if not os.path.exists(p):
        raise HTTPException(404, L("没有缩略图", "No thumbnail"))
    return FileResponse(p, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})


@app.get("/api/records/{rid}/view.jpg")
def records_view(rid: str):
    p = os.path.join(_rec_dir(rid), "view.jpg")
    if not os.path.exists(p):
        raise HTTPException(404, L("没有存档图", "No archived image"))
    return FileResponse(p, media_type="image/jpeg")


@app.get("/api/records/{rid}/view.png")
def records_view_png(rid: str):
    p = os.path.join(_rec_dir(rid), "view.png")
    if not os.path.exists(p):
        raise HTTPException(404, L("没有存档图", "No archived image"))
    return FileResponse(p, media_type="image/png")


@app.post("/api/records/{rid}/open")
def records_open(rid: str):
    """把存档图重新装进一个会话，参数原样带回（前端恢复吸色/阈值/选区后自己重算）"""
    rec = _rec_json(rid)
    src = ""
    for nm in ("view.png", "view.jpg"):        # 新记录是 png，老记录只有 jpg
        p = os.path.join(_rec_dir(rid), nm)
        if os.path.exists(p):
            src = p
            break
    if not src:
        raise HTTPException(404, L("存档图丢了", "Archived image is gone"))
    sid = secrets.token_urlsafe(9)
    d = os.path.join(UPLOADS, sid)
    os.makedirs(d, exist_ok=True)
    dst = os.path.join(d, "orig" + os.path.splitext(src)[1])
    shutil.copyfile(src, dst)
    # 尺寸只用文件头（lazy）就够：以前这里 im.load()+convert 会把 12.5 MP 整张解码一遍，
    # 白等 0.7 s —— 真正解码发生在下面 _meta -> work_img，那一步省不掉。
    if rec.get("ow") and rec.get("oh"):
        sz = (int(rec["ow"]), int(rec["oh"]))
    else:
        with Image.open(dst) as _im:
            sz = _im.size
    # 会话名必须是“原始文件名”，不是记录名——记录名可以被重命名，
    # 拿它当会话名会让“同图同选区 = 就地更新”的判重失效，一存就多一条（v1.53 修）
    sess = Session(sid, rec.get("sess_name") or rec.get("name") or rid, dst, sz)
    with SESS_LOCK:
        while len(SESSIONS) >= MAX_SESSIONS:
            old = min(SESSIONS.values(), key=lambda s: s.touch)
            SESSIONS.pop(old.id, None)
            old._works.clear()
            old._cls = None
            old.full = None
        SESSIONS[sid] = sess
    side = _res_to_side_sz(DEFAULT_RES, sess.size)      # 只要尺寸，别为它解码整张原图
    meta = _meta(sess, side)
    meta["url"] = "/api/image/%s?res=%s" % (sid, DEFAULT_RES)
    meta["default_res"] = DEFAULT_RES
    return {"ok": True, "meta": meta, "record": rec}


class RenameReq(BaseModel):
    name: str = ""


@app.post("/api/records/{rid}/rename")
def records_rename(rid: str, req: RenameReq):
    """给记录改个名字（只动 record.json 的 name，图片/结果/选区一律不动）。"""
    d = _rec_dir(rid)
    if not os.path.isdir(d):
        raise HTTPException(404, L("记录不存在", "Record not found"))
    nm = " ".join((req.name or "").replace("\n", " ").split())
    nm = "".join(ch for ch in nm if ch.isprintable())[:60]
    if not nm:
        raise HTTPException(400, L("名字不能为空（删掉重输）", "Name cannot be blank (clear it and retype)"))
    p = os.path.join(d, "record.json")
    try:
        rec = _rec_json(rid)
    except Exception:
        raise HTTPException(500, L("读不到这条记录", "This record cannot be read"))
    rec["name"] = nm
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    os.replace(tmp, p)
    return {"ok": True, "item": _rec_brief(rid)}


@app.delete("/api/records/{rid}")
def records_delete(rid: str):
    d = _rec_dir(rid)
    if not os.path.isdir(d):
        raise HTTPException(404, L("记录不存在", "Record not found"))
    shutil.rmtree(d, ignore_errors=True)
    return {"ok": True, "deleted": rid,
            "total": sum(1 for n in os.listdir(RECORDS) if os.path.isdir(os.path.join(RECORDS, n)))}



@app.get("/api/health")
def health():
    return {"ok": True, "sessions": len(SESSIONS), "ts": time.time()}


# ---------------------------------------------------------------- 认证 / 清理 / 静态


# 应用层不再做登录：对外由反代/SSO 负责（内网直连只在本机或 WG 内网）。


def _cleanup():
    while True:
        time.sleep(300)
        try:
            now = time.time()
            for name in os.listdir(UPLOADS):
                p = os.path.join(UPLOADS, name)
                if now - os.path.getmtime(p) > KEEP_HOURS * 3600:
                    shutil.rmtree(p, ignore_errors=True)
            for name in os.listdir(OVERLAYS):
                p = os.path.join(OVERLAYS, name)
                if now - os.path.getmtime(p) > 6 * 3600:
                    os.remove(p)
            with SESS_LOCK:
                for sid, s in list(SESSIONS.items()):
                    idle = now - s.touch
                    if idle > 3600:
                        SESSIONS.pop(sid, None)
                    elif idle > 900:      # 15 分钟没动：把 rgb/lab/best 这些大数组放掉，图片还在
                        with s.lock:
                            s._works.clear()
                            s._rois.clear()
                            s._cls = None
                            s.full = None         # 解码好的原图也放掉（要再用时 0.3 s 重解）
                        trim_mem()
        except Exception:
            pass


threading.Thread(target=_cleanup, daemon=True).start()

app.mount("/", StaticFiles(directory=os.path.join(BASE, "static"), html=True), name="static")


@app.middleware("http")
async def _no_stale_static(request, call_next):
    """HTML/JS/CSS 一律要求回源校验：否则浏览器会拿几天前的 app.js 继续跑，
    改了界面用户却"看不见"（v1.46 加，就是被这个坑过）。"""
    resp = await call_next(request)
    p = request.url.path
    if p == "/" or p.endswith((".html", ".js", ".css")):
        resp.headers["Cache-Control"] = "no-cache, must-revalidate, max-age=0"
    return resp
