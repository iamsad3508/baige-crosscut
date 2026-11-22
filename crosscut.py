# -*- coding: utf-8 -*-
"""
百格附着力（Cross-cut / ASTM D3359 Method B / ISO 2409）图像定量分析工具

功能：
  1. 打开测试照片，在图上框出任意多个矩形选区（ROI）；
  2. 在图上吸取几种颜色（完好漆膜 / 露底基材 / 忽略），按 Lab 色差自动分类每个像素；
  3. 实时显示每个选区的颜色面积百分比、总面积加权的「脱落百分比」；
  4. 按 ASTM D3359 自动换算 0B~5B（并给出 ISO 2409 的 Gt 等级）；
  5. 可「忽略细线」：用形态学开运算抹掉划痕沟槽（比漆块细的线），避免把刀缝算成脱落；
  6. 导出 CSV 明细 + 带标注的结果图。

用法：
  图形界面：  run.bat          或  python crosscut.py
  命令行：    python crosscut.py --cli 图片.jpg --roi 0.3,0.3,0.5,0.5 --roi ... \
                                 --coating 236,232,226 --substrate 37,37,40 --out 结果.png
  自检：      python crosscut.py --self-test

依赖：Pillow、numpy（pip install pillow numpy）
"""

import argparse
import csv
import math
import os
import sys

try:
    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter, ImageOps
except Exception as exc:  # pragma: no cover
    sys.stderr.write(
        "缺少依赖：%s\n请先安装：pip install pillow numpy\n" % exc)
    raise

# ImageTk 只有桌面版（Tk 图形界面）要用。Linux 上没装 python3-tk 时
# `from PIL import ImageTk` 会抛 ImportError，把整个模块带死——网页版根本不需要 Tk，
# 所以这里降级成「有就用、没有就 None」，只有真去开图形界面时才报错。
try:
    from PIL import ImageTk
except Exception:  # pragma: no cover
    ImageTk = None

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
    from tkinter import ttk
except Exception:  # pragma: no cover
    tk = None
    filedialog = messagebox = None

WORK_MAX = 1800          # 分析用的工作分辨率（长边像素）
NOISE_PCT = 0.5          # 小于该百分比视为 0%（判 5B）
ROLE_COAT = "coat"
ROLE_SUB = "sub"
ROLE_IGN = "ignore"
ROLE_NAME = {ROLE_COAT: "漆膜", ROLE_SUB: "基材", ROLE_IGN: "忽略"}


# --------------------------------------------------------------------------
# 颜色与分级
# --------------------------------------------------------------------------
def rgb_to_lab(rgb):
    """rgb: (...,3) uint8/float -> Lab float32"""
    c = np.asarray(rgb, dtype=np.float32) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]], np.float32)
    xyz = c @ m.T
    wp = np.array([0.95047, 1.0, 1.08883], np.float32)
    t = xyz / wp
    d = 6.0 / 29.0
    f = np.where(t > d ** 3, np.cbrt(np.maximum(t, 1e-12)), t / (3 * d * d) + 4.0 / 29.0)
    return np.stack([116 * f[..., 1] - 16,
                     500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], axis=-1).astype(np.float32)


def astm_class(pct):
    """ASTM D3359 Method B：按脱落面积百分比给出 0B~5B"""
    if pct <= NOISE_PCT:
        return "5B"
    if pct < 5:
        return "4B"
    if pct < 15:
        return "3B"
    if pct < 35:
        return "2B"
    if pct < 65:
        return "1B"
    return "0B"


def iso_class(pct):
    """ISO 2409：Gt0~Gt5"""
    if pct <= 5:
        return "Gt0"
    if pct <= 15:
        return "Gt1"
    if pct <= 35:
        return "Gt2"
    if pct <= 65:
        return "Gt3"
    return "Gt4"


LINES = [0.5, 5.0, 15.0, 35.0, 65.0]


def margin_hint(pct):
    """离上/下一条档线各有多远——临界样品最该看的是这个"""
    parts = ["判定余量：%.2f%% 属 %s" % (pct, astm_class(pct))]
    for lim in LINES:
        if pct < lim - 1e-9:
            parts.append("再多 %.2f pp（到 %.1f%%）即 %s"
                         % (lim - pct, lim, astm_class(lim + 1e-6)))
            break
    for lim in reversed(LINES):
        if pct >= lim - 1e-9:
            if lim == 0.5:
                parts.append("降到 %.1f%% 以下即 5B（少 %.2f pp）" % (0.5, pct - 0.5))
            else:
                parts.append("少 %.2f pp（到 %.0f%%）即 %s"
                             % (pct - lim, lim, astm_class(lim - 1e-6)))
            break
    return "；".join(parts) + "\n"


# --------------------------------------------------------------------------
# 图像处理
# --------------------------------------------------------------------------
def load_image(path):
    im = Image.open(path)
    im = ImageOps.exif_transpose(im).convert("RGB")
    return im


def fit_work(im, max_side=WORK_MAX):
    w, h = im.size
    s = min(1.0, max_side / float(max(w, h)))
    if s < 1.0:
        im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
    return im


TRANS_LABEL = -2       # 过渡带（漆膜与基材的混合色：刀缝、掉漆块边缘）


def classify_parts(lab, sample_labs, roles):
    """算出每个像素的 (最近样本, 最近距离, 混合比 t)

    t = 到基材的距离 / (到漆膜的距离 + 到基材的距离)
      t≈0 纯漆膜   t≈0.5 一半一半（刀缝/边缘）   t≈1 纯露底
    """
    n = len(sample_labs)
    if n == 0:
        return (np.full(lab.shape[:2], -1, np.int16),
                np.full(lab.shape[:2], 1e9, np.float32), None)
    d = np.empty(lab.shape[:2] + (n,), np.float32)
    for i, s in enumerate(sample_labs):
        d[..., i] = np.sqrt(((lab - s) ** 2).sum(-1))
    best = d.argmin(-1).astype(np.int16)
    mind = d.min(-1)
    coat = [i for i, r in enumerate(roles) if r == ROLE_COAT]
    sub = [i for i, r in enumerate(roles) if r == ROLE_SUB]
    ign = [i for i, r in enumerate(roles) if r == ROLE_IGN]
    t = None
    if coat and sub:
        dc = d[..., coat].min(-1)
        ds = d[..., sub].min(-1)
        t = (dc / (dc + ds + 1e-6)).astype(np.float32)
        if ign:                      # 明确标成"忽略"的样本（背景、笔迹）不受过渡带影响
            t = np.where(np.isin(best, ign), -1.0, t)
    return best, mind, t


def labels_from_parts(best, mind, t, tol, purity):
    """按容差和"露底纯度"生成标签：-1 未分类，-2 过渡带"""
    labels = best.copy()
    labels[mind > tol] = -1
    if t is not None and purity < 0.999:
        band = (t > 1.0 - purity) & (t < purity) & (mind <= tol) & (t >= 0)
        labels[band] = TRANS_LABEL
    return labels


def classify(lab, sample_labs, tol):
    """纯最近色分类；超出容差记为 -1"""
    n = len(sample_labs)
    if n == 0:
        return np.full(lab.shape[:2], -1, np.int16), np.full(lab.shape[:2], 1e9, np.float32)
    d = np.empty(lab.shape[:2] + (n,), np.float32)
    for i, s in enumerate(sample_labs):
        d[..., i] = np.sqrt(((lab - s) ** 2).sum(-1))
    best = d.argmin(-1).astype(np.int16)
    mind = d.min(-1)
    best[mind > tol] = -1
    return best, mind


def _sq_min(mask, k):
    """k × k 方形核最小值（= 腐蚀），与 PIL MinFilter 逐像素相同。
    可分离：先横向 k 个 AND，再纵向 k 个 AND；边界按复制处理（PIL 同）。
    比 PIL rankfilter 快 14~24 倍（k=8: 0.46s -> 0.03s / 0.73MP）。"""
    from numpy.lib.stride_tricks import sliding_window_view as _swv
    k = int(k)
    if k <= 1:
        return np.asarray(mask, bool).copy()
    if k % 2 == 0:
        k += 1
    m = np.asarray(mask, bool)
    r = k // 2
    # 大图分条处理，避免 (h, w, k) 中间数组吃内存
    if m.size > 4_000_000:
        out = np.empty_like(m)
        step = max(1, 2_000_000 // max(1, m.shape[1]))
        for y0 in range(0, m.shape[0], step):
            y1 = min(m.shape[0], y0 + step)
            a0 = max(0, y0 - r)
            a1 = min(m.shape[0], y1 + r)
            out[y0:y1] = _sq_min(m[a0:a1], k)[y0 - a0:y1 - a0]
        return out
    p = np.pad(m, ((r, r), (r, r)), mode="edge")
    a = _swv(p, k, axis=1).all(-1)
    return _swv(a, k, axis=0).all(-1)


def _sq_max(mask, k):
    """k × k 方形核最大值（= 膨胀），= ~腐蚀(~mask)"""
    return ~_sq_min(~np.asarray(mask, bool), k)


def morph_open(mask, k):
    """k x k 开运算：抹掉比 k 细的线（刀缝），保留块状脱落"""
    k = int(k)
    if k <= 1:
        return mask
    m = np.asarray(mask, bool)
    return _sq_max(_sq_min(m, k), k)


def _components(mask):
    """4 连通行程标注（纯 numpy + 并查集，不用 scipy）

    返回 (lab, runs, stats)
      runs : [(y, x0, x1, root)]
      stats: root -> [面积, x0, x1, y0, y1, Σx, Σy, Σx², Σy², Σxy]
    """
    h, w = mask.shape
    runs = []
    rows = []
    for y in range(h):
        row = mask[y]
        if not row.any():
            rows.append(())
            continue
        d = np.diff(np.concatenate(([0], row.view(np.int8), [0])))
        ss = np.flatnonzero(d == 1)
        ee = np.flatnonzero(d == -1)
        cur = []
        for s, e in zip(ss, ee):
            cur.append((int(s), int(e), len(runs)))
            runs.append((y, int(s), int(e)))
        rows.append(tuple(cur))

    parent = list(range(len(runs)))

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
            if pe <= cs:
                i += 1
            elif ce <= ps:
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
    for i, (y, s, e) in enumerate(runs):
        r = find(i)
        st = stats.get(r)
        if st is None:
            st = stats[r] = [0, s, e - 1, y, y, 0.0, 0.0, 0.0, 0.0, 0.0]
        n = e - s
        st[0] += n
        if s < st[1]:
            st[1] = s
        if e - 1 > st[2]:
            st[2] = e - 1
        if y < st[3]:
            st[3] = y
        if y > st[4]:
            st[4] = y
        xm = (s + e - 1) / 2.0
        st[5] += n * xm
        st[6] += n * y
        # Σx² 必须带上**行程内部**的方差 (n²-1)/12；只取中点会把整条行程压成一个点，
        # x 方向方差恒为 0，"长宽比"退化成只看 y 向散布（横竖结论不一致的老 bug）。
        st[7] += n * (xm * xm + (n * n - 1) / 12.0)
        st[8] += n * (y * y)
        st[9] += n * xm * y

    lab = np.zeros((h, w), np.int32)
    for i, (y, s, e) in enumerate(runs):
        lab[y, s:e] = find(i) + 1
    return lab, [(y, s, e, find(i)) for i, (y, s, e) in enumerate(runs)], stats


ELONG_MIN = 4.0      # 长宽比 √(λ1/λ2) 超过它就算"线"（4:1；之前错把 λ 比当长宽比，等于只要求 2.8:1）
LINE_MIN_AREA = 24   # 太小的碎点不做线的判断


def split_lines(sub_mask, max_line_px):
    """把露底区域分成「块状脱落」和「线状刀缝」

    思路：比 max_line_px 粗的部分一定是块；剩下的薄片按形状判——
    细长的（长宽比 ≥ ELONG_MIN）判为刀缝，圆/团状（碎屑、脱漆环）保留。
    这样刀缝粗细不均也能一起判掉，不用卡一个固定宽度。

    返回 (保留的 mask, 线状结构条数, 判为刀缝的像素数)
    """
    thick = morph_open(sub_mask, max_line_px)
    thin = sub_mask & ~thick
    if not thin.any() or int(thin.sum()) < LINE_MIN_AREA:
        return sub_mask, 0, 0

    lab, runs, stats = _components(thin)
    is_line = {}
    for root, st in stats.items():
        area = st[0]
        if area < LINE_MIN_AREA:
            is_line[root] = False
            continue
        n = float(area)
        mx_, my_ = st[5] / n, st[6] / n
        a = st[7] / n - mx_ * mx_                   # 行程内方差已在 _components 里算进 Σx²
        c = st[8] / n - my_ * my_
        b = st[9] / n - mx_ * my_
        tr, det = a + c, a * c - b * b
        disc = max(0.0, tr * tr / 4.0 - det) ** 0.5
        l1 = tr / 2.0 + disc
        l2 = max(1e-6, tr / 2.0 - disc)
        is_line[root] = (l1 / l2) ** 0.5 >= ELONG_MIN   # 用长宽比，不是特征值比

    keep = thick.copy()
    line_mask = np.zeros_like(sub_mask)
    for (y, s, e, root) in runs:
        if is_line.get(root):
            line_mask[y, s:e] = True
        else:
            keep[y, s:e] = True
    n_line = sum(1 for v in is_line.values() if v)
    return keep, n_line, int(line_mask.sum())


def _rot2(x, y, cx, cy, ang):
    """局部坐标 → 屏幕坐标"""
    t = math.radians(ang)
    dx, dy = x - cx, y - cy
    return (cx + dx * math.cos(t) - dy * math.sin(t),
            cy + dx * math.sin(t) + dy * math.cos(t))


def _unrot2(x, y, cx, cy, ang):
    """屏幕坐标 → 局部坐标"""
    t = math.radians(ang)
    dx, dy = x - cx, y - cy
    return (cx + dx * math.cos(t) + dy * math.sin(t),
            cy - dx * math.sin(t) + dy * math.cos(t))


def rect_corners(box, ang=0.0):
    """四个角（顺时针，屏幕坐标）"""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    if abs(ang) < 1e-9:
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return [_rot2(px, py, cx, cy, ang) for (px, py) in
            ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]


def rot_rect_mask(shape, box, ang=0.0):
    """带角度的矩形掩膜（在局部坐标系里判 |dx|<=w/2 且 |dy|<=h/2）"""
    h, w = shape[:2]
    x0, y0, x1, y1 = box
    if abs(ang) < 1e-9:
        return rect_mask(shape, box)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    hw, hh = abs(x1 - x0) / 2.0, abs(y1 - y0) / 2.0
    t = math.radians(ang)
    ca, sa = math.cos(t), math.sin(t)
    ex = abs(hw * ca) + abs(hh * sa)
    ey = abs(hw * sa) + abs(hh * ca)
    ix0 = max(0, int(math.floor(cx - ex)))
    ix1 = min(w, int(math.ceil(cx + ex)))
    iy0 = max(0, int(math.floor(cy - ey)))
    iy1 = min(h, int(math.ceil(cy + ey)))
    m = np.zeros((h, w), bool)
    if ix1 <= ix0 or iy1 <= iy0:
        return m
    ys, xs = np.mgrid[iy0:iy1, ix0:ix1]
    dx = xs - cx
    dy = ys - cy
    lx = dx * ca + dy * sa
    ly = -dx * sa + dy * ca
    m[iy0:iy1, ix0:ix1] = (np.abs(lx) <= hw) & (np.abs(ly) <= hh)
    return m

def rect_mask(shape, rect):
    h, w = shape[:2]
    x0, y0, x1, y1 = rect
    x0, x1 = sorted((int(round(x0)), int(round(x1))))
    y0, y1 = sorted((int(round(y0)), int(round(y1))))
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    m = np.zeros((h, w), bool)
    if x1 > x0 and y1 > y0:
        m[y0:y1, x0:x1] = True
    return m


def _erode3(mask):
    return _sq_min(np.asarray(mask, bool), 3)


def thickness_map(mask, k_max=24):
    """每个露底像素的"厚度"（≈2×内切半径，用逐层腐蚀測）

    刀缝是薄的，掉漆块是厚的——这张图就能看出该把阈值放在哪。
    """
    m = mask.copy()
    d = np.zeros(mask.shape, np.int16)
    k = 0
    while k < k_max and m.any():
        k += 1
        d[m] = k
        m = _erode3(m)
        if not m.any():
            break
    return d * 2


def thickness_hist(subc, k_max=24, bins=((1, 3), (4, 6), (7, 10), (11, 16), (17, 10 ** 6))):
    d = thickness_map(subc, k_max)
    vals = d[subc]
    total = max(1, vals.size)
    out = []
    for lo, hi in bins:
        c = int(((vals >= lo) & (vals <= hi)).sum())
        out.append((lo, hi, c, 100.0 * c / total))
    return out



def local_width_keep(sub_mask, tile=64, factor=1.6, floor=3, alt_factor=2.2):
    """按**局部缝宽**排除刀缝（实验口径 C）。

    不拟合直线、也不设全局 k：把选区切成 tile×tile 的块，每块量一次"这儿的缝有多宽"
    （block 内露底像素厚度的中位数），只排除**薄于本块基线**的那些像素。
    徒手划线粗细不匀、网格有弧度时，比全局 k 稳，也不会把成块崩边删掉
    （块状脱落比局部基线厚）。

    返回 (保留的 mask, 被当刀缝排除的像素数, 基线中位数, 用 alt_factor 时的保留像素数)

    为什么要 alt_factor：这张量化的“厚度”只能是偶数（2×3×3腐蚀步数），刀缝在 6MP 下常常只有
    2~4 px，于是“排除到 2”还是“排除到 4”就决定了几倍的差别（本例 6.8% ↔ 1.6%）。
    所以同时算一个更宽的因子，给用户一个**区间**，而不是假装有个精确值。
    """
    m = np.asarray(sub_mask, bool)
    if not m.any():
        return m, 0, 0.0
    th = thickness_map(m, k_max=max(24, int(3 * tile // 8)))
    keep = np.zeros_like(m)
    keep_alt = np.zeros_like(m)
    meds = []
    for y0 in range(0, m.shape[0], tile):
        for x0 in range(0, m.shape[1], tile):
            s = m[y0:y0 + tile, x0:x0 + tile]
            if not s.any():
                continue
            t = th[y0:y0 + tile, x0:x0 + tile]
            med = float(np.median(t[s]))
            meds.append(med)
            blk = keep[y0:y0 + tile, x0:x0 + tile]
            blk[:] = s & (t > max(floor, factor * med))
            alb = keep_alt[y0:y0 + tile, x0:x0 + tile]
            alb[:] = s & (t > max(floor, alt_factor * med))
    med_all = float(np.median(meds)) if meds else 0.0
    return keep, int(m.sum() - keep.sum()), med_all, int(keep_alt.sum())


def keep_mask(subc, line_px, mode="width", loc_tile=64, loc_factor=1.6):
    """按“刀缝判法”决定哪些露底像素留下（不算刀缝）。

    统计（roi_stats）和逐块确认（/api/blocks）必须共用这一份逻辑，
    否则切了判法以后，块列表算出来的数会和结果栏对不上。
    返回 (保留的 mask, 判成刀缝的条数, 刀缝面积, local 的 (基线中位, 备用因子下的保留量))
    """
    if mode == "local":
        kept, _cut, med, alt = local_width_keep(subc, loc_tile, loc_factor)
        return kept, 0, 0, (med, alt)
    if line_px > 1 and mode == "shape":
        kept, n_line, line_area = split_lines(subc, line_px)
        return kept, n_line, line_area, None
    if line_px > 1:
        return morph_open(subc, line_px), 0, 0, None
    return subc.copy(), 0, 0, None


def roi_stats(labels, mask, roles, line_px, mode="width", want_hist=False, loc_tile=64, loc_factor=1.6):
    """统计一个选区内的各类面积与脱落百分比（只在选区外接矩形内做运算，快）

    mode="width"：按固定宽度开运算抹掉刀缝
    mode="shape"：按形状判——细长的算刀缝，团状的算脱落（网格连通时不适用，实验）
    mode="local"：按**局部宽度**判——每块自己定基线（实验 C，旋转不变、最快）
    """
    sel = labels[mask]
    n = len(roles)
    counts = np.bincount(sel[sel >= 0], minlength=n) if sel.size else np.zeros(n, np.int64)
    unassigned = int((sel == -1).sum())
    trans = int((sel == TRANS_LABEL).sum())

    coat_idx = [i for i, r in enumerate(roles) if r == ROLE_COAT]
    sub_idx = [i for i, r in enumerate(roles) if r == ROLE_SUB]

    coat_px = int(sum(counts[i] for i in coat_idx))
    sub_raw = int(sum(counts[i] for i in sub_idx))

    sub_open = 0
    n_line = 0
    line_area = 0
    loc_info = None
    hist = None
    if sub_idx and sub_raw:
        rows = np.flatnonzero(mask.any(1))
        cols = np.flatnonzero(mask.any(0))
        if rows.size and cols.size:
            r0, r1 = rows[0], rows[-1] + 1
            c0, c1 = cols[0], cols[-1] + 1
            subc = np.zeros((r1 - r0, c1 - c0), bool)
            for i in sub_idx:
                subc |= (labels[r0:r1, c0:c1] == i)
            subc &= mask[r0:r1, c0:c1]
            kept, n_line, line_area, loc_info = keep_mask(subc, line_px, mode, loc_tile, loc_factor)
            sub_open = int(kept.sum())
            if want_hist:
                hist = thickness_hist(subc)

    denom = coat_px + sub_raw
    pct_raw = 100.0 * sub_raw / denom if denom else 0.0
    pct_open = 100.0 * sub_open / denom if denom else 0.0
    return {
        "area": int(mask.sum()),
        "counts": [int(c) for c in counts],
        "coat_px": coat_px,
        "sub_raw": sub_raw,
        "sub_open": sub_open,
        "unassigned": unassigned,
        "trans": trans,
        "denom": denom,
        "pct_raw": pct_raw,
        "pct": pct_open,
        "n_line": n_line,
        "line_area": line_area,
        "loc_med": None if loc_info is None else loc_info[0],
        "loc_alt_open": None if loc_info is None else loc_info[1],
        "thist": hist,
    }


def sample_line(rgb, p0, p1, n=None):
    """沿 p0->p1 双线性采样灰度，返回 (profile, 段长px)"""
    x0, y0 = p0
    x1, y1 = p1
    length = math.hypot(x1 - x0, y1 - y0)
    if length < 1:
        return None, 0.0
    n = n or max(24, int(length * 2))
    t = np.linspace(0.0, 1.0, n)
    xs = x0 + (x1 - x0) * t
    ys = y0 + (y1 - y0) * t
    h, w = rgb.shape[:2]
    xs = np.clip(xs, 0, w - 1.001)
    ys = np.clip(ys, 0, h - 1.001)
    x0i = np.floor(xs).astype(np.int32)
    y0i = np.floor(ys).astype(np.int32)
    fx = xs - x0i
    fy = ys - y0i
    gray = rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
    g00 = gray[y0i, x0i]
    g10 = gray[y0i, np.minimum(x0i + 1, w - 1)]
    g01 = gray[np.minimum(y0i + 1, h - 1), x0i]
    g11 = gray[np.minimum(y0i + 1, h - 1), np.minimum(x0i + 1, w - 1)]
    prof = (g00 * (1 - fx) * (1 - fy) + g10 * fx * (1 - fy) +
            g01 * (1 - fx) * fy + g11 * fx * fy)
    return prof.astype(np.float32), float(length)


def measure_line_width(rgb, p0, p1):
    """横跨刀缝拖一条线，用剖面半高宽（FWHM）估计刀缝宽度

    返回 dict(width_px, contrast, length_px, polarity) 或 None
    """
    prof, length = sample_line(rgb, p0, p1)
    if prof is None or length < 10 or prof.size < 12:
        return None
    n = prof.size
    # 用两端连线做基线，去掉光照梯度
    base = np.linspace(prof[0], prof[-1], n)
    dev = prof - base
    k = int(np.argmax(np.abs(dev)))
    if k <= 1 or k >= n - 2:
        return None
    peak = float(dev[k])
    sgn = 1.0 if peak > 0 else -1.0
    half = abs(peak) * 0.5
    left = k
    while left > 0 and dev[left] * sgn > half:
        left -= 1
    right = k
    while right < n - 1 and dev[right] * sgn > half:
        right += 1
    if right - left < 1:
        return None

    def cross(i_in, i_out):
        """在 i_in（幅值仍高）与 i_out（已低于半高）之间线性插值出半高位置"""
        a, b = dev[i_in] * sgn, dev[i_out] * sgn
        if a == b:
            return float(i_in)
        return i_in + (half - a) / (b - a) * (i_out - i_in)

    xl = cross(left + 1, left) if left + 1 <= k else float(left)
    xr = cross(right - 1, right) if right - 1 >= k else float(right)
    px_per_sample = length / (n - 1)
    width = max(0.0, (xr - xl) * px_per_sample)
    return {
        "width_px": float(width),
        "contrast": abs(peak),
        "length_px": length,
        "polarity": "暗" if peak < 0 else "亮",
    }


def place_labels(boxes, texts, bounds, char_w=7.2, line_h=17, pad=4, avoid=()):
    """把标签摆在框外侧，尽量不互相遮挡、不出界。返回 [(x,y,w,h,text)]

    avoid：已经占用的矩形（比如左上角的汇总条），标签不能压上去。
    """
    xmin, ymin, xmax, ymax = bounds
    placed = [tuple(a) for a in avoid]
    out = []
    for (x0, y0, x1, y1), text in zip(boxes, texts):
        w = len(text) * char_w + 10
        h = line_h
        cands = [
            (x0, y0 - h - pad),          # 上-左
            (x1 - w, y0 - h - pad),      # 上-右
            (x0, y1 + pad),              # 下-左
            (x1 - w, y1 + pad),          # 下-右
            (x1 + pad, y0),              # 右
            (x0 - w - pad, y0),          # 左
            (x1 + pad, y1 - h),          # 右下
            (x0 - w - pad, y1 - h),      # 左下
            (x0 + pad, y0 + pad),        # 框内-左上（实在没地方才用）
            (x1 - w - pad, y1 - h - pad),# 框内-右下
            (x0 + pad, y1 - h - pad),    # 框内-左下
            (x1 - w - pad, y0 + pad),    # 框内-右上
        ]
        best = None
        for px, py in cands:
            px = max(xmin + 1, min(xmax - w - 1, px))
            py = max(ymin + 1, min(ymax - h - 1, py))
            r = (px, py, px + w, py + h)
            if not any(not (r[2] < q[0] or r[0] > q[2] or r[3] < q[1] or r[1] > q[3])
                       for q in placed):
                best = r
                break
        if best is None:       # 全部候选都被占：往下/往右挪，直到不压已占用区域
            px = max(xmin + 1, min(xmax - w - 1, cands[0][0]))
            py = max(ymin + 1, min(ymax - h - 1, cands[0][1]))
            for _ in range(24):
                r = (px, py, px + w, py + h)
                hit = [q for q in placed
                       if not (r[2] < q[0] or r[0] > q[2] or r[3] < q[1] or r[1] > q[3])]
                if not hit:
                    break
                py = min(ymax - h - 1, hit[0][3] + pad)
                if py >= ymax - h - 1:
                    px = min(xmax - w - 1, px + w + pad)
                    py = max(ymin + 1, cands[0][1])
            best = (px, py, px + w, py + h)
        placed.append(best)
        out.append((best[0], best[1], w, h, text))
    return out


# --------------------------------------------------------------------------
# 命令行
# --------------------------------------------------------------------------
def run_cli(args):
    im = load_image(args.cli)
    work = fit_work(im)
    sx = im.size[0] / float(work.size[0])
    sy = im.size[1] / float(work.size[1])
    rgb = np.asarray(work).astype(np.uint8)
    lab = rgb_to_lab(rgb)

    samples = []
    roles = []
    for spec in (args.coating or []):
        samples.append(parse_rgb(spec)); roles.append(ROLE_COAT)
    for spec in (args.substrate or []):
        samples.append(parse_rgb(spec)); roles.append(ROLE_SUB)
    for spec in (args.ignore or []):
        samples.append(parse_rgb(spec)); roles.append(ROLE_IGN)
    if not samples:
        sys.exit("至少需要 --coating 和 --substrate")
    sample_labs = [rgb_to_lab(np.array(s, np.float32)) for s in samples]

    best, mind, t = classify_parts(lab, sample_labs, roles)
    labels = labels_from_parts(best, mind, t, args.tol, args.purity)

    rois = []
    if args.roi:
        for r in args.roi:
            v = [float(t) for t in r.split(",")]
            if all(0 <= t <= 1 for t in v):      # 归一化坐标
                x0, y0, x1, y1 = v
                rois.append((x0 * work.size[0], y0 * work.size[1],
                             x1 * work.size[0], y1 * work.size[1]))
            else:
                rois.append((v[0] / sx, v[1] / sy, v[2] / sx, v[3] / sy))
    else:
        rois = [(0, 0, work.size[0], work.size[1])]

    rows = []
    tot = {"coat_px": 0, "sub_raw": 0, "sub_open": 0, "area": 0}
    for i, r in enumerate(rois, 1):
        m = rect_mask(labels.shape, r)
        st = roi_stats(labels, m, roles, args.line_px)
        rows.append((i, r, st))
        for k in tot:
            tot[k] += st[k]
    denom = tot["coat_px"] + tot["sub_raw"]
    pct = 100.0 * tot["sub_open"] / denom if denom else 0.0
    cls = astm_class(pct)

    print("图像: %s  原始 %dx%d  分析 %dx%d" % (args.cli, im.size[0], im.size[1], work.size[0], work.size[1]))
    print("样本: " + ", ".join("%s=%s" % (ROLE_NAME[r], s) for s, r in zip(samples, roles)))
    print("容差: %.0f   露底纯度: %.2f   刀缝阈值: %d px"
          % (args.tol, args.purity, args.line_px))
    for i, r, st in rows:
        print("  ROI%d  %.0f,%.0f,%.0f,%.0f  面积%8d  漆膜%8d  基材%8d  未分类%8d  脱落 %5.1f%% -> %s"
              % (i, r[0], r[1], r[2], r[3], st["area"], st["coat_px"], st["sub_raw"],
                 st["unassigned"], st["pct"], astm_class(st["pct"])))
    print("合计: 漆膜 %d px, 基材 %d px, 脱落 %.2f%%  =>  ASTM %s / ISO %s"
          % (tot["coat_px"], tot["sub_raw"], pct, cls, iso_class(pct)))

    out_png = args.out or os.path.splitext(args.cli)[0] + "_标注.png"
    save_annotated(work, labels, rois, rows, pct, cls, samples, roles, out_png)
    print("标注图: " + out_png)

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8-sig") as f:
            wr = csv.writer(f)
            wr.writerow(["ROI", "x0", "y0", "x1", "y1", "面积px", "漆膜px", "基材px",
                         "未分类px", "脱落%(原始)", "脱落%(忽略细线)", "ASTM", "ISO"])
            for i, r, st in rows:
                wr.writerow([i, round(r[0]), round(r[1]), round(r[2]), round(r[3]), st["area"],
                             st["coat_px"], st["sub_raw"], st["unassigned"],
                             round(st["pct_raw"], 2), round(st["pct"], 2),
                             astm_class(st["pct"]), iso_class(st["pct"])])
            wr.writerow(["合计", "", "", "", "", tot["area"], tot["coat_px"], tot["sub_raw"],
                         "", "", round(pct, 2), cls, iso_class(pct)])
        print("CSV: " + args.csv)
    return 0


def parse_rgb(s):
    v = [int(float(t)) for t in s.replace(" ", "").split(",")]
    if len(v) != 3:
        raise ValueError("颜色格式应为 R,G,B：" + s)
    return tuple(v)


def _pil_font(size):
    from PIL import ImageFont
    for name in ("msyh.ttc", "arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size)
    except Exception:
        return ImageFont.load_default()


def save_annotated(work, labels, rois, rows, pct, cls, samples, roles, path):
    base = np.asarray(work).astype(np.float32).copy()
    for i, r, st in rows:
        m = rect_mask(labels.shape, r)
        sub = np.zeros_like(m)
        coat = np.zeros_like(m)
        for idx, role in enumerate(roles):
            if role == ROLE_SUB:
                sub |= (labels == idx) & m
            elif role == ROLE_COAT:
                coat |= (labels == idx) & m
        base[sub] = base[sub] * 0.35 + np.array([255, 40, 40], np.float32) * 0.65
        base[coat] = base[coat] * 0.75 + np.array([60, 220, 90], np.float32) * 0.25
    out = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    dr = ImageDraw.Draw(out)
    W, H = out.size
    fs = max(14, int(H / 55))
    font = _pil_font(fs)
    texts = ["R%d  %.1f%%  %s" % (i, st["pct"], astm_class(st["pct"])) for i, r, st in rows]
    for (lx, ly, lw, lh, tx) in place_labels([r for i, r, st in rows], texts, (0, 0, W, H),
                                             char_w=fs * 0.62, line_h=int(fs * 1.55)):
        dr.rectangle([lx, ly, lx + lw, ly + lh], fill=(15, 15, 15), outline=(140, 140, 140))
        dr.text((lx + 6, ly + lh * 0.5 - fs * 0.55), tx, fill=(255, 213, 74), font=font)
    for i, r, st in rows:
        dr.rectangle([r[0], r[1], r[2], r[3]], outline=(0, 200, 255), width=max(2, W // 500))
    txt = "removed %.2f%%  ASTM %s  ISO %s" % (pct, cls, iso_class(pct))
    dr.rectangle([0, 0, int(12 + fs * 0.62 * len(txt)), int(fs * 1.7)], fill=(0, 0, 0))
    dr.text((6, 4), txt, fill=(255, 255, 255), font=font)
    out.save(path)


class ColorList(ttk.Frame):
    """带色块的样本列表：一眼看出是什么颜色"""

    def __init__(self, master, height=5, on_select=None):
        ttk.Frame.__init__(self, master)
        self._on_select = on_select
        self._rows = []
        self._sel = -1
        self._box = tk.Frame(self, bg="white", highlightthickness=1,
                             highlightbackground="#dcdfe4")
        self._box.pack(fill="both", expand=True)
        self._empty = tk.Label(self._box, text="（还没有颜色样本）", bg="white",
                               fg="#9aa0a6", anchor="w", padx=8, pady=8)
        self._empty.pack(fill="x")

    def clear(self):
        for w in self._rows:
            w[0].destroy()
        self._rows = []
        self._sel = -1

    def set_items(self, items):
        """items: [(rgb tuple, 文本, 备注)]"""
        self.clear()
        if not items:
            self._empty.pack(fill="x")
            return
        self._empty.pack_forget()
        for i, (rgb, text, note) in enumerate(items):
            row = tk.Frame(self._box, bg="white", padx=4, pady=3,
                           highlightthickness=0)
            row.pack(fill="x")
            chip = tk.Label(row, bg="#%02X%02X%02X" % rgb, width=3, height=1,
                            relief="flat", bd=1)
            chip.pack(side="left", padx=(2, 8))
            lbl = tk.Label(row, text=text, bg="white", anchor="w",
                           font=("Microsoft YaHei UI", 9))
            lbl.pack(side="left")
            if note:
                nt = tk.Label(row, text=note, bg="white", fg="#9aa0a6", anchor="w")
                nt.pack(side="left", padx=(6, 0))
            for w in (row, chip, lbl) if not note else (row, chip, lbl, nt):
                w.bind("<Button-1>", lambda e, idx=i: self.select(idx))
            self._rows.append((row, chip, lbl, note))
        self.select(0 if items else -1)

    def select(self, i):
        self._sel = i
        for k, (row, chip, lbl, _n) in enumerate(self._rows):
            on = (k == i)
            row.config(bg="#e8f0fe" if on else "white",
                       highlightthickness=1 if on else 0,
                       highlightbackground="#2563eb",
                       highlightcolor="#2563eb")
            lbl.config(bg="#e8f0fe" if on else "white")
        if self._on_select:
            self._on_select(i)

    def get_sel(self):
        return self._sel


class WinSlider(tk.Canvas):
    """Win11 风格的细滑条：圆头 + 已填充轨道"""

    def __init__(self, master, variable, lo, hi, step=1, command=None,
                 width=160, height=24, **kw):
        tk.Canvas.__init__(self, master, width=width, height=height,
                           highlightthickness=0, bd=0, bg=kw.pop("bg", "#ffffff"), **kw)
        self._var = variable
        self._lo, self._hi, self._step = float(lo), float(hi), float(step)
        self._cmd = command
        self._pad = 9
        self._drag = False
        self.bind("<Button-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Configure>", lambda e: self._redraw())
        self._var.trace_add("write", lambda *a: self._redraw())
        self._redraw()

    # --- 几何 ---
    def _x_of(self, v):
        w = max(1, self.winfo_width())
        f = (v - self._lo) / (self._hi - self._lo + 1e-12)
        return self._pad + f * (w - 2 * self._pad)

    def _value_of(self, x):
        w = max(1, self.winfo_width())
        f = (x - self._pad) / max(1e-6, (w - 2 * self._pad))
        v = self._lo + max(0.0, min(1.0, f)) * (self._hi - self._lo)
        if self._step >= 1:
            v = round(v / self._step) * self._step
        else:
            v = round(round(v / self._step) * self._step, 4)
        return max(self._lo, min(self._hi, v))

    # --- 绘制 ---
    def _redraw(self):
        self.delete("all")
        w = max(1, self.winfo_width())
        h = max(1, self.winfo_height())
        cy = h / 2.0
        try:
            v = float(self._var.get())
        except Exception:
            v = self._lo
        x = self._x_of(max(self._lo, min(self._hi, v)))
        # 轨道
        self.create_line(self._pad, cy, w - self._pad, cy,
                         fill="#d7dbe0", width=4, capstyle="round")
        # 已填充
        if x > self._pad + 0.5:
            self.create_line(self._pad, cy, x, cy,
                             fill="#2563eb", width=4, capstyle="round")
        # 圆头
        self.create_oval(x - 7, cy - 7, x + 7, cy + 7,
                         fill="#2563eb", outline="#ffffff", width=2)

    # --- 交互 ---
    def _set_from_x(self, x):
        v = self._value_of(x)
        if abs(v - float(self._var.get())) > 1e-9:
            self._var.set(v)
            if self._cmd:
                self._cmd()

    def _on_press(self, e):
        self._drag = True
        self._set_from_x(e.x)

    def _on_drag(self, e):
        if self._drag:
            self._set_from_x(e.x)

    def _on_release(self, e):
        self._drag = False


# --------------------------------------------------------------------------
# 图形界面
# --------------------------------------------------------------------------
class App:
    def __init__(self, root):
        self.root = root
        root.title("百格附着力图像分析  —  ASTM D3359 / ISO 2409")
        root.geometry("1400x900")
        root.minsize(1080, 680)

        self.path = None
        self.full = None          # 原始 PIL
        self.work = None          # 工作分辨率 PIL
        self.rgb = None
        self.lab = None
        self.samples = []         # [(rgb tuple, role)]
        self.rois = []            # [(x0,y0,x1,y1)] 工作坐标
        self.zoom = 1.0
        self.mode = "rect"
        self.drag = None
        self._imgtk = None
        self.labels = None
        self.last_pct = 0.0
        self.last_stats = []
        # 视图：panx/pany 是图像左上角在画布中的位置
        self.panx = 0.0
        self.pany = 0.0
        self._pan_anchor = None
        # 刀缝量尺
        self.ruler = None          # (x0,y0,x1,y1) 工作坐标，正在拖
        self.ruler_done = None     # 上一次量到的线段
        self.ruler_hist = []       # [(宽度px, 对比度, 段长px)]
        self.ruler_result = None   # dict
        # 分类缓存
        self._best = None          # 每像素最近的样本序号
        self._mind = None          # 每像素到最近样本的 Lab 距离
        self.roi_sel = -1          # 当前选中的选区
        self._roi_drag = None      # 拖动/缩放选区时的状态
        self._t = None             # 每像素的混合比
        self._key = None           # 缓存对应的样本指纹
        self._auto_setting = False
        self.rot = 0.0
        self.angles = []
        self._rotate_drag = None
        self.var_auto_mode = tk.BooleanVar(value=False)
        self._auto_reason = ""
        self.work0 = None
        self._full_rot = None
        self._last_thist = None
        # 渲染缓存（高分辨率不卡的关键）
        self._ver = 0
        self._base_small = None
        self._base_small_src = None
        self.lab_small = None
        self._lab_small_ver = -1
        self._disp_img = None
        self._disp_key = None
        self._img_id = None
        self._zooming = False
        self._zoom_job = None
        self.var_purity = tk.DoubleVar(value=0.75)
        self.var_detail = tk.BooleanVar(value=False)
        self.last_boxes = []       # 上次计算实际用到的框
        self.var_auto = tk.BooleanVar(value=True)
        self._job = None
        # 分辨率与全分辨率量尺
        self.work_max = WORK_MAX
        self.full_rgb = None
        self.scale2full = 1.0
        self.var_res = tk.StringVar(value=str(WORK_MAX))

        self._build()
        self._bind()

    # ---------------- 界面 ----------------
    def _style_setup(self, st):
        bg = "#f4f5f7"
        st.configure(".", background=bg, foreground="#202124",
                     font=("Microsoft YaHei UI", 9))
        st.configure("TFrame", background=bg)
        st.configure("TLabel", background=bg, foreground="#202124")
        st.configure("Hint.TLabel", foreground="#5f6368")
        st.configure("Title.TLabel", foreground="#1f2328",
                     font=("Microsoft YaHei UI", 10, "bold"))
        st.configure("Sum.TLabel", foreground="#0a5a2a",
                     font=("Microsoft YaHei UI", 11, "bold"))
        st.configure("Warn.TLabel", foreground="#b3261e",
                     font=("Microsoft YaHei UI", 11, "bold"))
        st.configure("TCheckbutton", background=bg)
        st.configure("TRadiobutton", background=bg)
        st.configure("TLabelframe", background=bg, bordercolor="#dcdfe4",
                     relief="solid", borderwidth=1)
        st.configure("TLabelframe.Label", background=bg, foreground="#5f6368",
                     font=("Microsoft YaHei UI", 9, "bold"))
        st.configure("TButton", padding=(9, 4), relief="flat",
                     background="#e6e8eb", bordercolor="#d3d7dc")
        st.map("TButton", background=[("active", "#d8dbe0"), ("pressed", "#cfd3d8")])
        st.configure("Accent.TButton", background="#2563eb", foreground="white",
                     bordercolor="#2563eb", font=("Microsoft YaHei UI", 9, "bold"))
        st.map("Accent.TButton",
               background=[("active", "#1d4ed8"), ("pressed", "#1a45c0")],
               foreground=[("disabled", "#c7d2fe")])
        st.configure("Mini.TButton", padding=(2, 0), width=2)
        st.configure("Mode.TRadiobutton", padding=(12, 4), background="#e6e8eb",
                     indicatoron=True)
        st.map("Mode.TRadiobutton",
               background=[("selected", "#2563eb"), ("active", "#d8dbe0")],
               foreground=[("selected", "white")])
        st.configure("TScrollbar", background="#e6e8eb", troughcolor="#f0f1f3",
                     bordercolor="#dcdfe4", arrowcolor="#5f6368")
        self.root.configure(bg=bg)

    def _build(self):
        st = ttk.Style()
        try:
            st.theme_use("clam")
        except Exception:
            pass
        self._style_setup(st)

        # ---- 顶部工具条 ----
        bar = ttk.Frame(self.root, padding=(10, 8, 10, 4))
        bar.pack(side="top", fill="x")
        ttk.Button(bar, text="打开图片", command=self.open_file).pack(side="left")
        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=10)
        ttk.Button(bar, text="▶ 计算", style="Accent.TButton",
                   command=self.compute).pack(side="left")
        ttk.Button(bar, text="导出 CSV", command=self.export_csv).pack(side="left", padx=(8, 0))
        ttk.Button(bar, text="保存标注图", command=self.export_png).pack(side="left", padx=(4, 0))
        self.lbl_sum = ttk.Label(bar, text="", style="Sum.TLabel")
        self.lbl_sum.pack(side="left", padx=(14, 0))
        ttk.Checkbutton(bar, text="半自动", variable=self.var_auto_mode,
                        command=self._on_auto_mode).pack(side="right")

        # ---- 主体 ----
        main = ttk.Frame(self.root)
        main.pack(side="top", fill="both", expand=True, padx=8, pady=(0, 8))

        self.canvas = tk.Canvas(main, bg="#1f2226", highlightthickness=0, takefocus=1)
        self.canvas.pack(side="left", fill="both", expand=True)

        wrap = ttk.Frame(main, width=430)
        wrap.pack(side="right", fill="y", padx=(8, 0))
        wrap.pack_propagate(False)
        self.pw = ttk.Panedwindow(wrap, orient="vertical")
        self.pw.pack(fill="both", expand=True)

        self.nb = ttk.Notebook(self.pw)
        self.pw.add(self.nb, weight=1)

        # ===== 页1：框选（选区几何 + 图像旋转）=====
        tab1 = ttk.Frame(self.nb, padding=(10, 10))
        self.nb.add(tab1, text="  框选  ")
        ttk.Label(tab1, text="图像旋转（图纸歪了就转正，选区跟着转）",
                  style="Hint.TLabel").pack(anchor="w")
        rrow = ttk.Frame(tab1)
        rrow.pack(fill="x", pady=(6, 0))
        ttk.Button(rrow, text="−0.2°", width=6,
                   command=lambda: self.apply_rotation(self.rot - 0.2)).pack(side="left")
        self.var_rot = tk.DoubleVar(value=0.0)
        ent_rot = ttk.Entry(rrow, width=7, justify="right", textvariable=self.var_rot)
        ent_rot.pack(side="left", padx=4)
        ent_rot.bind("<Return>", lambda e: self.apply_rotation())
        ent_rot.bind("<FocusOut>", lambda e: self.apply_rotation())
        ttk.Button(rrow, text="+0.2°", width=6,
                   command=lambda: self.apply_rotation(self.rot + 0.2)).pack(side="left")
        ttk.Button(rrow, text="归零", width=5,
                   command=lambda: self.apply_rotation(0)).pack(side="left", padx=(6, 0))

        sp = ttk.Separator(tab1, orient="horizontal")
        sp.pack(fill="x", pady=(14, 10))
        ttk.Label(tab1, text="选区", style="Hint.TLabel").pack(anchor="w")
        self.lbl_roi = ttk.Label(tab1, text="", style="Hint.TLabel")
        self.lbl_roi.pack(anchor="w", pady=(4, 0))
        rrow2 = ttk.Frame(tab1)
        rrow2.pack(fill="x", pady=(8, 0))
        ttk.Button(rrow2, text="清空全部选区", command=self.clear_roi).pack(side="left")
        nodge = ttk.Frame(tab1)
        nodge.pack(fill="x", pady=(8, 0))
        ttk.Label(nodge, text="微移", style="Hint.TLabel").pack(side="left", padx=(0, 6))
        for txt, dx, dy in (("←", -1, 0), ("→", 1, 0), ("↑", 0, -1), ("↓", 0, 1)):
            ttk.Button(nodge, text=txt, width=2,
                       command=lambda dx=dx, dy=dy: self.nudge_roi(dx, dy)
                       ).pack(side="left", padx=2)

        # ===== 页2：吸色（颜色样本 + 颜色参数）=====
        tab2 = ttk.Frame(self.nb, padding=(10, 10))
        self.nb.add(tab2, text="  吸色  ")
        ttk.Label(tab2, text="颜色样本（同色自动去重）", style="Hint.TLabel").pack(anchor="w")
        self.lst = ColorList(tab2, on_select=lambda i: None)
        self.lst.pack(fill="both", expand=True, pady=(4, 0))
        row = ttk.Frame(tab2)
        row.pack(fill="x", pady=(6, 0))
        ttk.Button(row, text="设为漆膜", command=lambda: self.set_role(ROLE_COAT)).pack(side="left")
        ttk.Button(row, text="设为基材", command=lambda: self.set_role(ROLE_SUB)
                   ).pack(side="left", padx=4)
        ttk.Button(row, text="忽略", command=lambda: self.set_role(ROLE_IGN)).pack(side="left")
        ttk.Button(row, text="删除", command=self.del_sample).pack(side="right")
        arow = ttk.Frame(tab2)
        arow.pack(fill="x", pady=(6, 0))
        ttk.Button(arow, text="重新识别颜色", width=12,
                   command=self.auto_run).pack(side="left")
        self.lbl_span = ttk.Label(tab2, text="样本间距：—（先吸漆膜和基材）",
                                  style="Hint.TLabel")
        self.lbl_span.pack(anchor="w", pady=(10, 0))
        self.var_auto = tk.BooleanVar(value=True)
        ttk.Checkbutton(tab2, text="自动容差（样本间距的 45%）", variable=self.var_auto,
                        command=self._on_auto_tol).pack(anchor="w")
        self.var_tol = tk.DoubleVar(value=25.0)
        self._param(tab2, "颜色容差", self.var_tol, 3, 120, 1, "%.0f", self._on_tol_change)
        self.var_purity = tk.DoubleVar(value=0.75)
        self._param(tab2, "露底纯度（越高越只认纯基材）", self.var_purity,
                    0.50, 0.99, 0.01, "%.2f", self.refresh_overlay)

        # ===== 页3：量尺（刀缝）=====
        tab3 = ttk.Frame(self.nb, padding=(10, 10))
        self.nb.add(tab3, text="  量尺  ")
        rowr = ttk.Frame(tab3)
        rowr.pack(fill="x")
        ttk.Button(rowr, text="用测量值", command=self.apply_ruler).pack(side="left")
        ttk.Button(rowr, text="用平均值", command=self.apply_ruler_mean).pack(side="left", padx=4)
        ttk.Button(rowr, text="清空", command=self.clear_ruler).pack(side="left")
        self.lbl_ruler = ttk.Label(tab3, text="还没有测量", style="Hint.TLabel",
                                   justify="left", wraplength=350)
        self.lbl_ruler.pack(anchor="w", pady=(8, 0))
        self.var_line = tk.DoubleVar(value=1.0)
        self._param(tab3, "刀缝宽度 / 线状阈值（px）", self.var_line, 1, 60, 1, "%.0f",
                    self.refresh_overlay)
        srow = ttk.Frame(tab3)
        srow.pack(fill="x", pady=(8, 0))
        self.lbl_suggest = ttk.Label(srow, text="建议刀缝阈值：先吸色算一次，或量一条刀缝",
                                     style="Hint.TLabel", wraplength=240, justify="left")
        self.lbl_suggest.pack(side="left")
        self.btn_suggest = ttk.Button(srow, text="采用建议", state="disabled",
                                      command=self.take_suggestion)
        self.btn_suggest.pack(side="right")
        exrow = ttk.Frame(tab3)
        exrow.pack(fill="x", pady=(12, 0))
        ttk.Label(exrow, text="排除方式", style="Hint.TLabel").pack(side="left", padx=(0, 6))
        self.var_exmode = tk.StringVar(value="width")
        for txt, val in (("按宽度", "width"), ("按形状", "shape")):
            ttk.Radiobutton(exrow, text=txt, value=val, variable=self.var_exmode,
                            style="Mode.TRadiobutton",
                            command=self.refresh_overlay).pack(side="left", padx=(0, 4))

        # ---- 结果 ----
        box5 = ttk.LabelFrame(self.pw, text=" 结果 ")
        self.pw.add(box5, weight=2)
        inner = ttk.Frame(box5, padding=(8, 6))
        inner.pack(fill="both", expand=True)
        hrow = ttk.Frame(inner)
        hrow.pack(fill="x")
        ttk.Checkbutton(hrow, text="诊断明细", variable=self.var_detail,
                        command=self.compute).pack(side="left")
        self.var_overlay = tk.BooleanVar(value=True)
        ttk.Checkbutton(hrow, text="覆盖层", variable=self.var_overlay,
                        command=self.refresh_overlay).pack(side="left", padx=(10, 0))
        ttk.Label(hrow, text="分辨率", style="Hint.TLabel").pack(side="right", padx=(0, 4))
        om = ttk.OptionMenu(hrow, self.var_res, self.var_res.get(),
                            "1500", "1800", "2400", "3000", "全分辨率",
                            command=self._on_resolution)
        om.pack(side="right", padx=(0, 10))
        trow = ttk.Frame(inner)
        trow.pack(fill="both", expand=True, pady=(6, 0))
        self.txt = tk.Text(trow, height=8, width=42, wrap="word", state="disabled",
                           relief="flat", bg="white", fg="#202124",
                           highlightthickness=1, highlightbackground="#dcdfe4",
                           font=("Microsoft YaHei UI", 9), padx=8, pady=6)
        self.txt.pack(side="left", fill="both", expand=True)
        sb3 = ttk.Scrollbar(trow, orient="vertical", command=self.txt.yview)
        sb3.pack(side="right", fill="y")
        self.txt.config(yscrollcommand=sb3.set)

        self.var_mode = tk.StringVar(value="rect")
        self.nb.bind("<<NotebookTabChanged>>", self._on_tab)
        for wdg in (self.txt,):
            wdg.bind("<MouseWheel>", self._on_widget_wheel)

    # ---------------- 旋转 ----------------
    def _rotated(self, im):
        if abs(self.rot) < 1e-6:
            return im
        return im.rotate(self.rot, resample=Image.BICUBIC, fillcolor=(248, 248, 248))

    @staticmethod
    def _rot_pt(x, y, deg, w, h):
        cx, cy = w / 2.0, h / 2.0
        t = math.radians(deg)
        dx, dy = x - cx, y - cy
        return (cx + dx * math.cos(t) + dy * math.sin(t),
                cy - dx * math.sin(t) + dy * math.cos(t))

    def apply_rotation(self, deg=None):
        """旋转分析图；已有选区跟着转（取旋转后四角的外接框）"""
        if self.work0 is None:
            return
        old_rot = self.rot
        if deg is not None:
            new_rot = float(deg)
        elif hasattr(self, "var_rot"):
            try:
                new_rot = float(self.var_rot.get())
            except Exception:
                return
        else:
            return
        if abs(new_rot - old_rot) < 1e-9 and self.work is not None:
            return                      # 角度没变（比如输入框失焦），不要重复转
        delta = new_rot - old_rot
        w, h = self.work0.size
        if abs(delta) > 1e-9 and self.rois:
            moved = []
            for (x0, y0, x1, y1) in self.rois:
                bw, bh = x1 - x0, y1 - y0
                ncx, ncy = self._rot_pt((x0 + x1) / 2.0, (y0 + y1) / 2.0, delta, w, h)
                nx0 = min(max(0.0, ncx - bw / 2.0), w - bw)
                ny0 = min(max(0.0, ncy - bh / 2.0), h - bh)
                moved.append((nx0, ny0, nx0 + bw, ny0 + bh))
            self.rois = moved      # 只转中心、尺寸不变：转回去能精确还原，面积也不虚增
        self.rot = new_rot
        self.work = self._rotated(self.work0)
        self.rgb = np.asarray(self.work).astype(np.uint8)
        self.lab = rgb_to_lab(self.rgb)
        self._best = self._mind = self._t = None
        self._key = None
        self.labels = None
        self._ver += 1
        self._base_small = None
        self._base_small_src = None
        self.lab_small = None
        self._lab_small_ver = -1
        self._disp_img = None
        self._disp_key = None
        self._img_id = None
        self._full_rot = None
        if hasattr(self, "var_rot"):
            self.var_rot.set(round(self.rot, 2))
        self.refresh_roi_list()
        self.fit_view()
        if self.samples:
            self._schedule_compute()

    # ---------------- 刀缝阈值建议 ----------------
    def _suggest_line(self):
        if self.ruler_hist:
            return int(min(60, max(2, round(sum(self.ruler_hist) / len(self.ruler_hist)) + 1)))
        th = getattr(self, "_last_thist", None)
        if not th:
            return None
        cum = 0.0
        for lo, hi, c, pct in th:
            cum += pct
            if cum >= 90:
                upper = hi if hi < 10 ** 5 else lo
                return int(min(60, max(2, upper + 2)))
        return None

    def update_suggestion(self):
        if not hasattr(self, "lbl_suggest"):
            return
        v = self._suggest_line()
        if v is None:
            self.lbl_suggest.config(text="建议刀缝阈值：先吸色算一次，或量一条刀缝")
            self.btn_suggest.config(text="采用建议", state="disabled")
        else:
            src = "按实测刀缝" if self.ruler_hist else "按露底厚度分布"
            self.lbl_suggest.config(text="建议刀缝阈值 ≈ %d px（%s）" % (v, src))
            self.btn_suggest.config(text="采用 %d px" % v, state="normal")

    def take_suggestion(self):
        v = self._suggest_line()
        if v is None:
            return
        self.var_line.set(float(v))
        self.refresh_overlay()

    def _on_tab(self, e=None):
        idx = self.nb.index(self.nb.select())
        self.var_mode.set(("rect", "pick", "ruler")[idx])
        self._mode_changed()

    # ---- Win11 风参数行：滑条 + 可手改的数字 ----
    def _param(self, parent, title, var, lo, hi, step, fmt, on_change):
        ttk.Label(parent, text=title, style="Hint.TLabel").pack(anchor="w", pady=(10, 2))
        row = ttk.Frame(parent)
        row.pack(fill="x")
        sl = WinSlider(row, var, lo, hi, step, command=on_change,
                       bg="#f4f5f7", width=150)
        sl.pack(side="left", fill="x", expand=True)
        ent = ttk.Entry(row, width=6, justify="right")
        ent.pack(side="left", padx=(8, 0))

        def sync(*a):
            try:
                cur = ent.get()
                new = fmt % float(var.get())
                if cur != new:
                    ent.delete(0, "end")
                    ent.insert(0, new)
            except Exception:
                pass

        def commit(e=None):
            try:
                v = float(ent.get())
            except Exception:
                sync()
                return
            v = max(lo, min(hi, v))
            if step >= 1:
                v = float(round(v))
            var.set(round(v, 4))
            on_change()
            sync()

        var.trace_add("write", sync)
        ent.bind("<Return>", commit)
        ent.bind("<FocusOut>", commit)
        ent.bind("<KP_Enter>", commit)
        sync()

    def _bind_panel_wheel(self, w):
        if isinstance(w, (tk.Text, tk.Listbox)):
            w.bind("<MouseWheel>", self._on_widget_wheel)
            return
        w.bind("<MouseWheel>", self._on_panel_wheel)
        for ch in w.winfo_children():
            self._bind_panel_wheel(ch)

    def _on_widget_wheel(self, e):
        w = e.widget
        try:
            w.yview_scroll(-1 if e.delta > 0 else 1, "units")
        except Exception:
            pass
        return "break"

    # ---------------- 半自动识别 ----------------
    def auto_detect(self):
        """把选区里占比最大的颜色当作漆膜，离它最远且不是沿边背景的一类当作露底"""
        try:
            return self._auto_detect()
        except Exception as exc:
            self._auto_reason = "内部错误 %s" % exc
            return False

    def _auto_detect(self):
        if self.lab is None:
            return False
        shape2 = self.lab.shape[:2]
        if self.rois:
            m = np.zeros(shape2, bool)
            for i, r in enumerate(self.rois):
                ang = self.angles[i] if i < len(self.angles) else 0.0
                m |= rot_rect_mask(shape2, r, ang)
        else:
            m = np.ones(shape2, bool)
        idx_all = np.flatnonzero(m)
        self._auto_reason = "选区太小"
        if idx_all.size < 500:
            return False
        step = max(1, idx_all.size // 20000)
        idx = idx_all[::step]
        lab = self.lab.reshape(-1, 3)[idx]
        rgb = self.rgb.reshape(-1, 3)[idx].astype(np.float32)

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
        rgb_means = {k: rgb[lbl == k].mean(0) for k in range(K)}

        usable = [k for k in range(K) if counts[k] >= 0.03 * len(lbl)] or list(range(K))
        coat = max(usable, key=lambda k: counts[k])
        others = [k for k in usable if k != coat]
        if not others:
            self._auto_reason = "只有一类颜色"
            return False

        # 看哪一类占满了选区边缘——那多半是背景
        edge_frac = {}
        try:
            ring = m & ~_sq_min(m, 9)
            if ring.sum() > 50:
                lab_all = self.lab.reshape(-1, 3)[idx_all]
                lbl_all = ((lab_all[:, None, :] - cent[None, :, :]) ** 2
                           ).sum(-1).argmin(1)
                in_ring = ring.reshape(-1)[idx_all]
                tot = max(1, int(in_ring.sum()))
                for k in range(K):
                    edge_frac[k] = float((in_ring & (lbl_all == k)).sum()) / tot
        except Exception:
            edge_frac = {}

        dist = {k: float(np.linalg.norm(cent[k] - cent[coat])) for k in others}
        ranked = sorted(others, key=lambda k: -dist[k])
        inner = [k for k in ranked if edge_frac.get(k, 0.0) <= 0.5 and dist[k] >= 10]
        sub = inner[0] if inner else (ranked[0] if dist[ranked[0]] >= 10 else None)
        if sub is None:
            self._auto_reason = "两类颜色差得太少（<10）"
            return False
        self._auto_warn = ""
        if edge_frac.get(sub, 0.0) > 0.6:
            self._auto_warn = ("⚠ 选中的露底颜色占了选区边缘的大半，很可能是背景。"
                               "把框框紧一点，或取消半自动手动改色。\n")

        def to_rgb(k):
            v = np.clip(rgb_means[k], 0, 255)
            return tuple(int(round(x)) for x in v)

        new_samples = [(to_rgb(coat), ROLE_COAT), (to_rgb(sub), ROLE_SUB)]
        for k in others:
            if k != sub and counts[k] > 0.02 * len(lbl):
                new_samples.append((to_rgb(k), ROLE_IGN))
        self.samples = new_samples
        self._after_sample_change()
        return True

    def auto_run(self):
        """一键：选框 → 自动取色 → 自动定刀缝阈值 → 出结果"""
        try:
            return self._auto_run()
        except Exception as exc:
            self._auto_reason = "内部错误 %s" % exc
            return False

    def _auto_run(self):
        if not self.rois:
            self._auto_reason = "还没有选区"
            return False
        if not self.auto_detect():
            return False
        self._last_thist = None
        self.compute()
        v = self._suggest_line()
        if v and int(round(float(self.var_line.get()))) != v:
            self.var_line.set(float(min(v, 25)))
            self.compute()
            pct_auto = self.last_pct
            if pct_auto <= 0.05 or pct_auto >= 60:
                self.var_line.set(1.0)          # 1 = 不排除刀缝
                self.compute()
                self._add_result(
                    "⚠ 按建议的 %d px 排除刀缝后结果变成 %.1f%%（不可信），已先关掉刀缝排除。"
                    "请到「量尺」页量一条真刀缝，再手动填阈值。\n" % (v, pct_auto))
        return True

    def _on_auto_mode(self):
        if not self.var_auto_mode.get():
            self.refresh_overlay()
            return
        if not self.rois:
            self._set_result("半自动已开：先在图上框出划格区，框完自动出结果。")
            return
        if not self.auto_run():
            self.var_auto_mode.set(False)
            messagebox.showinfo("半自动", "选区里没找到对比足够的两类颜色，已切回手动。")

    def _on_tol_change(self):
        if self.var_auto.get() and not getattr(self, "_auto_setting", False):
            self.var_auto.set(False)      # 手动动了滑条，自动容差自动取消
        self.refresh_overlay()

    def _on_auto_tol(self):
        if self.var_auto.get():
            self._apply_auto_tol()
        self.refresh_overlay()

    def _apply_auto_tol(self):
        span = self._sample_span()
        if span is None:
            return
        self._auto_setting = True
        try:
            self.var_tol.set(round(max(6.0, min(120.0, span * 0.45)), 1))
        finally:
            self._auto_setting = False

    def _sample_span(self):
        labs = self._sample_labs()
        roles = [r for _, r in self.samples]
        coat = [l for l, r in zip(labs, roles) if r == ROLE_COAT]
        sub = [l for l, r in zip(labs, roles) if r == ROLE_SUB]
        if not coat or not sub:
            return None
        return min(float(np.linalg.norm(c - s)) for c in coat for s in sub)

    def _bind(self):
        c = self.canvas
        c.bind("<Button-1>", self.on_press)
        c.bind("<B1-Motion>", self.on_drag)
        c.bind("<ButtonRelease-1>", self.on_release)
        c.bind("<Motion>", self.on_motion)
        c.bind("<ButtonPress-2>", self.on_mid_press)
        c.bind("<B2-Motion>", self.on_mid_drag)
        c.bind("<ButtonRelease-2>", lambda e: setattr(self, "_pan_anchor", None))
        c.bind("<Button-3>", self.on_right_click)
        c.bind("<MouseWheel>", self.on_wheel)
        c.bind("<Double-Button-1>", lambda e: self.fit_view())
        self.root.bind_all("<Control-Key-0>", lambda e: self.fit_view())
        self.root.bind_all("<Control-Key-1>", lambda e: self.zoom_one())
        c.bind("<Configure>", self.on_resize)
        for key, dx, dy in (("<Left>", -1, 0), ("<Right>", 1, 0),
                            ("<Up>", 0, -1), ("<Down>", 0, 1)):
            c.bind(key, lambda e, dx=dx, dy=dy: self.nudge_roi(dx, dy, e))
            c.bind(key.replace("<", "<Shift-"), lambda e, dx=dx, dy=dy: self.nudge_roi(dx * 10, dy * 10, e))

    def on_resize(self, e=None):
        if self.work is None:
            return
        if abs(self.zoom - 1.0) < 1e-6:
            self._center_pan()
        else:
            self._clamp_pan()
        self.render()

    def on_mid_press(self, e):
        if self.work is None:
            return
        self._pan_anchor = (e.x, e.y, self.panx, self.pany)

    def on_mid_drag(self, e):
        self.on_drag(e)

    # ---------------- 文件 ----------------
    def open_file(self, path=None):
        if not path:
            path = filedialog.askopenfilename(
                title="选择测试照片",
                filetypes=[("图片", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("所有文件", "*.*")])
        if not path:
            return
        try:
            self.full = load_image(path)
        except Exception as exc:
            messagebox.showerror("打开失败", str(exc))
            return
        self.path = path
        self.rot = 0.0
        self.full_rgb = None
        self._prepare_work(reset_rois=True)
        self.samples = []
        self.rois = []
        self.angles = []
        self.roi_sel = -1
        self.labels = None
        self.last_stats = []
        self.last_boxes = []
        self.ruler = None
        self.ruler_done = None
        self.ruler_result = None
        self.ruler_hist = []
        if hasattr(self, "lbl_ruler"):
            self.lbl_ruler.config(text="刀缝量尺：切到「量刀缝」后横跨刀缝拖一下", foreground="#5f6368")
            self.lbl_span.config(text="样本间距：—（先吸漆膜和基材）", foreground="#5f6368")
            self.lbl_sum.config(text="")
        self.lst.set_items([])
        self._set_result("")
        self.fit_view()

    def _prepare_work(self, reset_rois=False):
        """按当前 work_max 生成分析图；换算比例同时更新"""
        old_w = self.work.size[0] if self.work is not None else None
        self.work0 = fit_work(self.full, self.work_max)
        self.work = self._rotated(self.work0)
        self._full_rot = None
        self.rgb = np.asarray(self.work).astype(np.uint8)
        self.lab = rgb_to_lab(self.rgb)
        self._best = self._mind = None
        self._key = None
        self.labels = None
        self._ver += 1
        self._base_small = None
        self._base_small_src = None
        self.lab_small = None
        self._lab_small_ver = -1
        self._disp_img = None
        self._disp_key = None
        self._img_id = None
        self._zooming = False
        self._zoom_job = None
        self.scale2full = self.full.size[0] / float(self.work.size[0])
        if old_w and not reset_rois:
            f = self.work.size[0] / float(old_w)
            self.rois = [(r[0] * f, r[1] * f, r[2] * f, r[3] * f) for r in self.rois]
            # 「忽略细线」也是按分析图像素计的，跟着一起换算
            try:
                cur = int(self.var_line.get())
                if cur > 1:
                    self.var_line.set(max(1, int(round(cur * f))))
            except Exception:
                pass
        self._update_file_label()

    def _update_file_label(self):
        if self.full is None:
            return
        name = os.path.basename(self.path or "")
        self.root.title("百格附着力分析  —  %s" % name)

    def _full_rgb(self):
        """原图分辨率像素（量刀缝用：不受分析分辨率限制）"""
        if abs(self.rot) > 1e-6:
            if self._full_rot is None or self._full_rot[0] != self.rot:
                im = self._rotated(self.full)
                self._full_rot = (self.rot, np.asarray(im).astype(np.uint8))
            return self._full_rot[1]
        if self.full_rgb is None:
            self.full_rgb = np.asarray(self.full).astype(np.uint8)
        return self.full_rgb

    def _on_resolution(self, choice=None):
        if self.full is None:
            return
        c = self.var_res.get()
        mx = max(self.full.size) if c.startswith("全") else int(c)
        self.work_max = mx
        self._prepare_work(reset_rois=False)
        self.refresh_roi_list()
        self.fit_view()
        if self.samples:
            self.compute()

    def fit_view(self):
        self.zoom = 1.0
        self._center_pan()
        self.render()

    def _center_pan(self):
        if not self.work:
            return
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        s = self._fit_scale() * self.zoom
        dw, dh = self.work.size[0] * s, self.work.size[1] * s
        self.panx = (cw - dw) / 2.0
        self.pany = (ch - dh) / 2.0

    def zoom_one(self):
        """1:1 —— 图像 1 像素 = 屏幕 1 像素，并以当前画布中心为目标缩放"""
        if not self.work:
            return
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        base = self._fit_scale()
        cx, cy = self.to_img(cw / 2.0, ch / 2.0)
        self.zoom = 1.0 / base
        s = base * self.zoom
        self.panx = cw / 2.0 - cx * s
        self.pany = ch / 2.0 - cy * s
        self._clamp_pan()
        self.render()

    def zoom_by(self, factor, ax, ay):
        """以画布点 (ax,ay) 为不动点缩放"""
        if not self.work:
            return
        ix, iy = self.to_img(ax, ay)
        self.zoom = max(0.05, min(40.0, self.zoom * factor))
        s = self._fit_scale() * self.zoom
        self.panx = ax - ix * s
        self.pany = ay - iy * s
        self._clamp_pan()
        self.render()

    def _clamp_pan(self):
        """不让图像被拖到看不见（留 60px 边缘可见）"""
        if not self.work:
            return
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        s = self._fit_scale() * self.zoom
        dw, dh = self.work.size[0] * s, self.work.size[1] * s
        m = 60
        if dw <= cw:
            self.panx = (cw - dw) / 2.0
        else:
            self.panx = min(m, max(cw - dw - m, self.panx))
        if dh <= ch:
            self.pany = (ch - dh) / 2.0
        else:
            self.pany = min(m, max(ch - dh - m, self.pany))

    # ---------------- 坐标 ----------------
    def _fit_scale(self):
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        iw, ih = self.work.size
        return min(cw / float(iw), ch / float(ih))

    def _geom(self):
        s = self._fit_scale() * self.zoom
        return s, self.panx, self.pany, self.work.size[0] * s, self.work.size[1] * s

    def to_img(self, ex, ey):
        s, ox, oy, _, _ = self._geom()
        return (ex - ox) / s, (ey - oy) / s

    def to_canvas(self, x, y):
        s, ox, oy, _, _ = self._geom()
        return ox + x * s, oy + y * s

    # ---------------- 渲染 ----------------
    def _small_base(self):
        """显示用的降采样底图（长边≤2000）——渲染只用它，不再碰全分辨率图"""
        if self._base_small is None or self._base_small_src is not self.work:
            w, h = self.work.size
            s = min(1.0, 2000.0 / max(w, h))
            self._base_small = (self.work if s >= 1.0 else
                                self.work.resize((max(1, int(w * s)), max(1, int(h * s))),
                                                 Image.LANCZOS))
            self._base_small_src = self.work
            self.lab_small = None
        return self._base_small

    def _labels_small(self):
        """把标签图降到显示底图尺寸（最近邻，很快），覆盖层在显示图上算"""
        if self.labels is None:
            return None
        base = self._small_base()
        if self.lab_small is None or self._lab_small_ver != self._ver:
            code = np.clip(self.labels.astype(np.int16) + 2, 0, 255).astype(np.uint8)
            im = Image.fromarray(code).resize(base.size, Image.NEAREST)
            self.lab_small = np.asarray(im).astype(np.int16) - 2
            self._lab_small_ver = self._ver
        return self.lab_small

    def render(self):
        """只渲染可视区域——放大到 1:1 也不会整幅重采样"""
        if self.work is None:
            return
        c = self.canvas
        s, ox, oy, dw, dh = self._geom()
        cw, ch = max(1, c.winfo_width()), max(1, c.winfo_height())
        ix0, iy0 = max(0, ox), max(0, oy)
        ix1, iy1 = min(cw, ox + dw), min(ch, oy + dh)
        c.config(scrollregion=(0, 0, max(dw, 1), max(dh, 1)))
        if ix1 <= ix0 or iy1 <= iy0:
            c.delete("all")
            self.render_overlays()
            return
        vw, vh = max(1, int(ix1 - ix0)), max(1, int(iy1 - iy0))
        want_ov = (self.var_overlay.get() and bool(self.samples)
                   and not getattr(self, "_zooming", False))
        key = (int(ix0), int(iy0), vw, vh, self._ver, want_ov, s > 2.2)
        if self._disp_key != key or self._disp_img is None:
            base = self._small_base()
            k = base.size[0] / float(self.work.size[0])
            bx0 = max(0, int((ix0 - ox) / s * k))
            by0 = max(0, int((iy0 - oy) / s * k))
            bx1 = min(base.size[0], int(np.ceil((ix1 - ox) / s * k)))
            by1 = min(base.size[1], int(np.ceil((iy1 - oy) / s * k)))
            if bx1 <= bx0 or by1 <= by0:
                c.delete("all")
                self.render_overlays()
                return
            filt = Image.BOX if vw < (bx1 - bx0) else (
                Image.NEAREST if s > 2.2 else Image.BILINEAR)
            piece = base.crop((bx0, by0, bx1, by1)).resize((vw, vh), filt)
            if want_ov:
                lab = self._labels_small()
                if lab is not None:
                    code = np.clip(lab[by0:by1, bx0:bx1].astype(np.int16) + 2, 0, 255).astype(np.uint8)
                    l2 = np.asarray(Image.fromarray(code).resize((vw, vh), Image.NEAREST)
                                    ).astype(np.int16) - 2
                    arr = np.asarray(piece).astype(np.float32)
                    for idx, role in enumerate([r for _, r in self.samples]):
                        m = (l2 == idx)
                        if not m.any():
                            continue
                        if role == ROLE_SUB:
                            arr[m] = arr[m] * 0.35 + np.array([255, 40, 40], np.float32) * 0.65
                        elif role == ROLE_COAT:
                            arr[m] = arr[m] * 0.78 + np.array([50, 210, 90], np.float32) * 0.22
                    piece = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
            self._disp_img = piece
            self._disp_key = key
        if ImageTk is None:
            raise RuntimeError("桌面图形界面需要 tkinter：Linux 请装 python3-tk；网页版不需要它")
        self._imgtk = ImageTk.PhotoImage(self._disp_img)
        if self._img_id is None:
            c.delete("img")
            self._img_id = c.create_image(ix0, iy0, anchor="nw",
                                          image=self._imgtk, tags="img")
        else:
            c.itemconfig(self._img_id, image=self._imgtk)
            c.coords(self._img_id, ix0, iy0)
        self.render_overlays()

    def render_overlays(self):
        """只重绘框/手柄/标签/量尺——拖动时用它，不重采样底图，才不会卡"""
        if self.work is None:
            return
        c = self.canvas
        c.delete("ov")
        cw, ch = max(1, c.winfo_width()), max(1, c.winfo_height())
        canvas_boxes = []
        for i, r in enumerate(self.rois, 1):
            ang = self.angles[i - 1] if i - 1 < len(self.angles) else 0.0
            cs = [self.to_canvas(px, py) for (px, py) in rect_corners(r, ang)]
            xs = [p[0] for p in cs]
            ys = [p[1] for p in cs]
            canvas_boxes.append((min(xs), min(ys), max(xs), max(ys)))
            sel = (i - 1 == self.roi_sel)
            flat = [v for p in cs for v in p]
            c.create_polygon(flat, tags="ov", fill="", outline="#ffb400" if sel else "#00c8ff",
                             width=3 if sel else 2)
            if sel:
                ccx, ccy = sum(xs) / 4.0, sum(ys) / 4.0
                mids = [((cs[0][0] + cs[1][0]) / 2, (cs[0][1] + cs[1][1]) / 2),
                        ((cs[1][0] + cs[2][0]) / 2, (cs[1][1] + cs[2][1]) / 2),
                        ((cs[2][0] + cs[3][0]) / 2, (cs[2][1] + cs[3][1]) / 2),
                        ((cs[3][0] + cs[0][0]) / 2, (cs[3][1] + cs[0][1]) / 2)]
                for hx, hy in cs + mids:
                    c.create_rectangle(hx - 4, hy - 4, hx + 4, hy + 4, tags="ov",
                                       fill="#ffb400", outline="#5a3d00")
                for hx, hy in cs:                      # 四角外侧：旋转热区
                    dx, dy = hx - ccx, hy - ccy
                    d = math.hypot(dx, dy) or 1.0
                    rx, ry = hx + dx / d * 18, hy + dy / d * 18
                    c.create_oval(rx - 5, ry - 5, rx + 5, ry + 5, tags="ov",
                                  outline="#ffb400", width=1)
                    c.create_arc(rx - 4, ry - 4, rx + 4, ry + 4, start=40, extent=220,
                                 style="arc", outline="#ffb400", tags="ov")
        # 标签：贴在框外侧，避开彼此
        texts = []
        for i in range(len(self.rois)):
            if self.last_stats and len(self.last_stats) == len(self.rois):
                st = self.last_stats[i]
                ang = self.angles[i] if i < len(self.angles) else 0.0
                extra = ("  %+.0f°" % ang) if abs(ang) > 0.5 else ""
                texts.append("R%d  %.1f%%  %s%s" % (i + 1, st["pct"], astm_class(st["pct"]), extra))
            else:
                texts.append("R%d" % (i + 1))
        for (lx, ly, lw, lh, tx), box in zip(place_labels(canvas_boxes, texts, (0, 0, cw, ch)),
                                             canvas_boxes):
            c.create_rectangle(lx, ly, lx + lw, ly + lh, tags="ov",
                               fill="#111111", outline="#7a7a7a")
            c.create_text(lx + 5, ly + lh / 2, anchor="w", text=tx, fill="#ffd54a", tags="ov")
            ax = min(max(lx, box[0]), box[2])
            ay = min(max(ly, box[1]), box[3])
            c.create_line(lx + lw / 2, ly + lh / 2, ax, ay, tags="ov",
                          fill="#7a7a7a", dash=(2, 3))
        if self.drag:
            x0, y0, x1, y1 = self.drag
            a = self.to_canvas(min(x0, x1), min(y0, y1))
            b = self.to_canvas(max(x0, x1), max(y0, y1))
            c.create_rectangle(a[0], a[1], b[0], b[1], tags="ov",
                               outline="#ffcc00", width=2, dash=(4, 3))
        self._draw_ruler(c)

    def _draw_ruler(self, c):
        for seg, color in ((self.ruler_done, "#ff66ff"), (self.ruler, "#ffffff")):
            if not seg:
                continue
            a = self.to_canvas(seg[0], seg[1])
            b = self.to_canvas(seg[2], seg[3])
            c.create_line(a[0], a[1], b[0], b[1], fill=color, width=2, tags="ov")
            c.create_line(a[0] - 4, a[1], a[0] + 4, a[1], fill=color, width=2, tags="ov")
            c.create_line(b[0] - 4, b[1], b[0] + 4, b[1], fill=color, width=2, tags="ov")
        if self.ruler_done and self.ruler_result:
            a = self.to_canvas(self.ruler_done[0], self.ruler_done[1])
            b = self.to_canvas(self.ruler_done[2], self.ruler_done[3])
            wf = self.ruler_result.get("width_full")
            txt = ("%.1f px 原图" % wf) if wf else ("%.1f px" % self.ruler_result["width_px"])
            c.create_text((a[0] + b[0]) / 2 + 6, (a[1] + b[1]) / 2 - 10, anchor="w",
                          text=txt, fill="#ff66ff", tags="ov")

    def _overlay_image(self):
        base = self.rgb.astype(np.float32).copy()
        if not self.samples:                 # 还没吸色：直接显示原图
            return Image.fromarray(self.rgb)
        if self.labels is None:
            self.labels = self._labels_for(self.var_tol.get())
        lab = self.labels
        for idx, role in enumerate([r for _, r in self.samples]):
            if role == ROLE_SUB:
                m = lab == idx
                base[m] = base[m] * 0.35 + np.array([255, 40, 40], np.float32) * 0.65
            elif role == ROLE_COAT:
                m = lab == idx
                base[m] = base[m] * 0.78 + np.array([50, 210, 90], np.float32) * 0.22
        return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))

    def refresh_overlay(self):
        self.labels = None
        self._ver += 1
        if self.work is not None and self.var_overlay.get():
            self.render()
        self._schedule_compute()

    def _schedule_compute(self):
        if self._job:
            try:
                self.root.after_cancel(self._job)
            except Exception:
                pass
            self._job = None
        if self.work is not None and len(self.samples) >= 2:
            self._job = self.root.after(450, self._deferred_compute)

    def _deferred_compute(self):
        self._job = None
        if self.work is not None and len(self.samples) >= 2:
            self.compute()

    # ---------------- 鼠标 ----------------
    def on_press(self, e):
        if self.work is None:
            return
        mode = self.var_mode.get()
        if mode == "pan":
            self._pan_anchor = (e.x, e.y, self.panx, self.pany)
            self.canvas.config(cursor="fleur")
            return
        x, y = self.to_img(e.x, e.y)
        if mode == "pick":
            self.pick_color(int(x), int(y))
            return
        if mode == "ruler":
            self.ruler = (x, y, x, y)
            return
        hit = self._hit_test(x, y)
        if hit:
            i, part = hit
            self.roi_sel = i

            if part == "rot":
                box = self.rois[i]
                cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
                self._rotate_drag = {"idx": i,
                                     "a0": self.angles[i] if i < len(self.angles) else 0.0,
                                     "m0": math.degrees(math.atan2(y - cy, x - cx)),
                                     "moved": False}
                self.canvas.config(cursor="exchange")
                return
            self._roi_drag = {"idx": i, "part": part, "box": self.rois[i],
                              "sx": x, "sy": y, "moved": False}
            self.render()
            return
        self.drag = (x, y, x, y)

    def on_motion(self, e):
        if self.work is None or self.var_mode.get() != "rect":
            return
        x, y = self.to_img(e.x, e.y)
        hit = self._hit_test(x, y)
        cur = "arrow"
        if hit:
            cur = {"nw": "size_nw_se", "se": "size_nw_se",
                   "ne": "size_ne_sw", "sw": "size_ne_sw",
                   "n": "sb_v_double_arrow", "s": "sb_v_double_arrow",
                   "w": "sb_h_double_arrow", "e": "sb_h_double_arrow",
                   "inside": "fleur", "rot": "exchange"}[hit[1]]
        self.canvas.config(cursor=cur)

    def on_drag(self, e):
        if self._pan_anchor:
            ax, ay, px, py = self._pan_anchor
            nx, ny = px + (e.x - ax), py + (e.y - ay)
            self.canvas.move("all", nx - self.panx, ny - self.pany)
            self.panx, self.pany = nx, ny
            return
        if self._rotate_drag:
            x, y = self.to_img(e.x, e.y)
            self._apply_roi_rotate(x, y)
            self.render_overlays()
            return
        if self._roi_drag:
            x, y = self.to_img(e.x, e.y)
            self._apply_roi_drag(x, y)
            self.render_overlays()
            return
        if self.ruler:
            x, y = self.to_img(e.x, e.y)
            self.ruler = (self.ruler[0], self.ruler[1], x, y)
            self.render_overlays()
            return
        if self.drag is None:
            return
        x, y = self.to_img(e.x, e.y)
        self.drag = (self.drag[0], self.drag[1], x, y)
        self.render_overlays()

    def on_release(self, e):
        if self._pan_anchor:
            self._pan_anchor = None
            self._clamp_pan()
            self.render()
            self._mode_changed()
            return
        if self._rotate_drag:
            moved = self._rotate_drag.get("moved")
            self._rotate_drag = None
            self._mode_changed()
            if moved:
                self.refresh_roi_list()
                self.compute()
            return
        if self._roi_drag:
            moved = self._roi_drag.get("moved")
            self._roi_drag = None
            if moved:
                self.refresh_roi_list()
                self.compute()
            return
        if self.ruler:
            seg = self.ruler
            self.ruler = None
            self.measure_groove(seg)
            self.render()
            return
        if self.drag is None:
            return
        x0, y0, x1, y1 = self.drag
        self.drag = None
        if abs(x1 - x0) < 6 or abs(y1 - y0) < 6:
            hit = self.roi_at(x1, y1)
            if hit >= 0:
                self.roi_sel = hit

                self.canvas.focus_set()
            self.render()
            return
        self.rois.append((min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)))
        self.angles.append(0.0)
        self.roi_sel = len(self.rois) - 1
        self.refresh_roi_list()
        self.canvas.focus_set()
        if self.var_auto_mode.get():
            if not self.auto_run():
                self.compute()
                self._set_result(
                    "半自动没认出来：%s。\n请取消「半自动」后在图上手动吸两个色。\n"
                    % getattr(self, "_auto_reason", "未知原因"))
        else:
            self.compute()

    def on_wheel(self, e):
        if self.work is None:
            return
        self._zooming = True
        if self._zoom_job:
            try:
                self.root.after_cancel(self._zoom_job)
            except Exception:
                pass
        self._zoom_job = self.root.after(180, self._end_zoom)
        self.zoom_by(1.2 if e.delta > 0 else 1 / 1.2, e.x, e.y)

    def _end_zoom(self):
        self._zoom_job = None
        self._zooming = False
        self._disp_key = None
        self.render()

    # ---------------- 刀缝量尺 ----------------
    def measure_groove(self, seg):
        """在原图分辨率上量刀缝（分析图只是缩小版，量出来会偏细）"""
        x0, y0, x1, y1 = seg
        k = self.scale2full
        full = self._full_rgb()
        res = measure_line_width(full, (x0 * k, y0 * k), (x1 * k, y1 * k))
        if res is None:
            self.ruler_done = None
            self.ruler_result = None
            self.lbl_ruler.config(text="没测到明显的沟槽：让线段垂直于刀缝、长度 30px 以上", foreground="#b00020")
            return
        res["width_full"] = res["width_px"]
        res["width_px"] = res["width_px"] / k           # 折回分析图坐标（开运算用）
        self.ruler_done = seg
        self.ruler_result = res
        reliable = res["contrast"] >= 8
        if reliable:
            self.ruler_hist.append(res["width_px"])
        avg = sum(self.ruler_hist) / len(self.ruler_hist) if self.ruler_hist else 0.0
        if res["contrast"] < 8:
            self.lbl_ruler.config(
                text="刀缝 ≈ %.1f px（原图）… 但对比度只有 %.0f，不可靠\n换个位置重拖一条，或先放大到 100%% 以上"
                     % (res["width_full"], res["contrast"]), foreground="#b06000")
        else:
            self.lbl_ruler.config(
                text="刀缝 ≈ %.1f px（原图）= %.1f px（分析图）\n对比度 %.0f   共 %d 次，均值 %.1f px"
                     % (res["width_full"], res["width_px"], res["contrast"],
                        len(self.ruler_hist), avg), foreground="#0a6b3d")
        self.update_suggestion()
        self._append_text("\n[量尺] 刀缝 %.1f px（原图）/ %.1f px（分析图，缩放 %.2f×），对比度 %.0f\n"
                          % (res["width_full"], res["width_px"], k, res["contrast"]))

    def apply_ruler(self):
        if not self.ruler_result:
            messagebox.showinfo("提示", "先量一次刀缝")
            return
        self.var_line.set(max(1, int(round(self.ruler_result["width_px"]))))
        self.refresh_overlay()
        self.compute()

    def apply_ruler_mean(self):
        if not self.ruler_hist:
            messagebox.showinfo("提示", "还没有测量记录")
            return
        self.var_line.set(max(1, int(round(sum(self.ruler_hist) / len(self.ruler_hist)))))
        self.refresh_overlay()
        self.compute()

    def clear_ruler(self):
        self.ruler_hist = []
        self.ruler_done = None
        self.ruler_result = None
        self.lbl_ruler.config(text="刀缝量尺：切到「量刀缝」后横跨刀缝拖一下", foreground="#5f6368")
        self.render()

    def _append_text(self, s):
        if self.work is None:
            return
        self._add_result(s)

    def _set_result(self, text):
        self.txt.config(state="normal")
        self.txt.delete("1.0", "end")
        self.txt.insert("end", text)
        self.txt.config(state="disabled")

    def _add_result(self, text):
        self.txt.config(state="normal")
        self.txt.insert("end", text)
        self.txt.config(state="disabled")
        self.txt.see("end")

    # ---------------- 颜色取样 ----------------
    def pick_color(self, x, y):
        """取色：在 11x11 的小块里挑“最纯”的那个像素，避免取到混色边缘"""
        if not (0 <= x < self.rgb.shape[1] and 0 <= y < self.rgb.shape[0]):
            return
        r = 5
        y0, y1 = max(0, y - r), min(self.rgb.shape[0], y + r + 1)
        x0, x1 = max(0, x - r), min(self.rgb.shape[1], x + r + 1)
        patch = self.rgb[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32)
        patch_lab = rgb_to_lab(patch)
        existing = self._sample_labs()
        if existing:
            d = np.min(np.stack([np.linalg.norm(patch_lab - e, axis=1) for e in existing], 1),
                       axis=1)
            k = int(np.argmax(d))
            c = tuple(int(v) for v in patch[k])
            note = "（与本块其他色差 %.0f，取最纯的一点）" % d[k]
        else:
            c = tuple(int(v) for v in np.median(patch, axis=0))
            note = "（本块中位色）"
        role = ROLE_COAT if not self.samples else (
            ROLE_SUB if len(self.samples) == 1 else ROLE_IGN)
        for i, (cc, rr) in enumerate(self.samples):
            if max(abs(cc[0] - c[0]), abs(cc[1] - c[1]), abs(cc[2] - c[2])) <= 3:
                messagebox.showinfo("提示", "这个颜色已经取过了（第 %d 个）" % (i + 1))
                return
        self.samples.append((c, role))
        self._after_sample_change()
        self._append_text("\n[吸色] %s RGB(%d,%d,%d) %s\n" % (ROLE_NAME[role], c[0], c[1], c[2], note))

    def _after_sample_change(self):
        self._best = self._mind = self._t = None
        self._refresh_sample_list()
        span = self._sample_span()
        if span is None:
            self.lbl_span.config(text="样本间距：—（还缺漆膜或基材）", foreground="#5f6368")
        else:
            self.lbl_span.config(
                text="样本间距 %.0f（Lab 距离）→ 建议容差 %.0f"
                     % (span, max(6.0, min(120.0, span * 0.45))), foreground="#0a6b3d")
            if self.var_auto.get():
                self._apply_auto_tol()
        self.refresh_overlay()

    def _refresh_sample_list(self):
        items = []
        for c, r in self.samples:
            items.append((c, ROLE_NAME[r], "RGB %d,%d,%d" % (c[0], c[1], c[2])))
        self.lst.set_items(items)

    def set_role(self, role):
        i = self.lst.get_sel()
        if i is None or i < 0 or i >= len(self.samples):
            return
        c, _ = self.samples[i]
        self.samples[i] = (c, role)
        self._after_sample_change()
        self.lst.select(i)

    def del_sample(self):
        i = self.lst.get_sel()
        if i is None or i < 0 or i >= len(self.samples):
            return
        self.samples.pop(i)
        self._after_sample_change()

    def _sample_labs(self):
        return [rgb_to_lab(np.array(c, np.float32)) for c, _ in self.samples]

    # ---------------- 选区 ----------------
    def refresh_roi_list(self):
        if not hasattr(self, "lbl_roi"):
            return
        n = len(self.rois)
        if n == 0:
            self.lbl_roi.config(text="未框选（按图例整幅统计）")
        elif 0 <= self.roi_sel < n:
            self.lbl_roi.config(text="选中 R%d / 共 %d 个" % (self.roi_sel + 1, n))
        else:
            self.lbl_roi.config(text="%d 个选区" % n)

    def on_right_click(self, e):
        """右键：弹一个菜单，点「删除」才真删"""
        if self.work is None:
            return
        x, y = self.to_img(e.x, e.y)
        hit = self.roi_at(x, y)
        self._rc_hit = hit
        menu = tk.Menu(self.root, tearoff=0)
        if hit >= 0:
            menu.add_command(label="删除 R%d" % (hit + 1), command=self._delete_roi_direct)
        if len(self.rois) > 1:
            menu.add_command(label="清空全部选区", command=self.clear_roi)
        if hit < 0 and len(self.rois) <= 1:
            menu.add_command(label="（这里没有选区）", state="disabled")
        try:
            menu.tk_popup(e.x_root, e.y_root)
        finally:
            menu.grab_release()

    def _delete_roi_direct(self):
        i = getattr(self, "_rc_hit", -1)
        if not (0 <= i < len(self.rois)):
            return
        self.rois.pop(i)
        if i < len(self.angles):
            self.angles.pop(i)
        if self.roi_sel >= len(self.rois):
            self.roi_sel = len(self.rois) - 1
        self.refresh_roi_list()
        self.render_overlays()        # 先把框擦掉，不等重算
        self._schedule_compute()      # 0.45 秒后再算

    def nudge_roi(self, dx, dy, e=None):
        if not (0 <= self.roi_sel < len(self.rois)):
            return "break"
        r = self.rois[self.roi_sel]
        w, h = self.work.size
        nr = (max(0, min(w - (r[2] - r[0]), r[0] + dx)), max(0, min(h - (r[3] - r[1]), r[1] + dy)),
              min(w, max(r[2] - r[0], r[2] + dx)), min(h, max(r[3] - r[1], r[3] + dy)))
        self.rois[self.roi_sel] = nr
        self.refresh_roi_list()
        self.compute()
        return "break"

    def roi_at(self, x, y):
        for i in range(len(self.rois) - 1, -1, -1):
            box = self.rois[i]
            ang = self.angles[i] if i < len(self.angles) else 0.0
            cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
            lx, ly = _unrot2(x, y, cx, cy, ang)
            if box[0] <= lx <= box[2] and box[1] <= ly <= box[3]:
                return i
        return -1

    def _hit_test(self, x, y, tol_px=8):
        """返回 (选区序号, 部位)；部位为 nw/n/ne/e/se/s/sw/w/inside/rot"""
        if not self.rois:
            return None
        s = max(1e-6, self._fit_scale() * self.zoom)
        tol = tol_px / s
        for i in range(len(self.rois) - 1, -1, -1):
            box = self.rois[i]
            ang = self.angles[i] if i < len(self.angles) else 0.0
            x0, y0, x1, y1 = box
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            lx, ly = _unrot2(x, y, cx, cy, ang)
            ring = 18 / s
            for px, py in ((x0, y0), (x1, y0), (x0, y1), (x1, y1)):
                if abs(math.hypot(lx - px, ly - py) - ring) <= tol:
                    return i, "rot"
            if not (x0 - tol <= lx <= x1 + tol and y0 - tol <= ly <= y1 + tol):
                continue
            w_, e_ = abs(lx - x0) <= tol, abs(lx - x1) <= tol
            n_, s_ = abs(ly - y0) <= tol, abs(ly - y1) <= tol
            if w_ and n_:
                return i, "nw"
            if e_ and n_:
                return i, "ne"
            if w_ and s_:
                return i, "sw"
            if e_ and s_:
                return i, "se"
            if n_ and x0 <= lx <= x1:
                return i, "n"
            if s_ and x0 <= lx <= x1:
                return i, "s"
            if w_ and y0 <= ly <= y1:
                return i, "w"
            if e_ and y0 <= ly <= y1:
                return i, "e"
            if x0 <= lx <= x1 and y0 <= ly <= y1:
                return i, "inside"
        return None

    def _apply_roi_drag(self, x, y):
        st = self._roi_drag
        i, part = st["idx"], st["part"]
        bx0, by0, bx1, by1 = st["box"]
        ang = self.angles[i] if i < len(self.angles) else 0.0
        w, h = self.work.size
        dx, dy = x - st["sx"], y - st["sy"]
        mn = 12.0
        cx, cy = (bx0 + bx1) / 2.0, (by0 + by1) / 2.0
        if part == "inside":
            bw, bh = bx1 - bx0, by1 - by0
            nx0 = min(max(0.0, bx0 + dx), w - bw)
            ny0 = min(max(0.0, by0 + dy), h - bh)
            box = (nx0, ny0, nx0 + bw, ny0 + bh)
        elif abs(ang) < 1e-9:
            x0, y0, x1, y1 = bx0, by0, bx1, by1
            if "w" in part:
                x0 = min(max(0.0, bx0 + dx), x1 - mn)
            if "e" in part:
                x1 = max(min(float(w), bx1 + dx), x0 + mn)
            if "n" in part:
                y0 = min(max(0.0, by0 + dy), y1 - mn)
            if "s" in part:
                y1 = max(min(float(h), by1 + dy), y0 + mn)
            box = (x0, y0, x1, y1)
        else:
            lx, ly = _unrot2(x, y, cx, cy, ang)
            fx0, fy0, fx1, fy1 = bx0, by0, bx1, by1
            if "w" in part:
                fx0 = min(lx, bx1 - mn)
            if "e" in part:
                fx1 = max(lx, bx0 + mn)
            if "n" in part:
                fy0 = min(ly, by1 - mn)
            if "s" in part:
                fy1 = max(ly, by0 + mn)
            mcxl, mcyl = (fx0 + fx1) / 2.0, (fy0 + fy1) / 2.0
            ncx, ncy = _rot2(mcxl, mcyl, cx, cy, ang)
            hw, hh = (fx1 - fx0) / 2.0, (fy1 - fy0) / 2.0
            ncx = min(max(hw, ncx), w - hw)
            ncy = min(max(hh, ncy), h - hh)
            box = (ncx - hw, ncy - hh, ncx + hw, ncy + hh)
        self.rois[i] = box
        st["moved"] = True

    def _apply_roi_rotate(self, x, y):
        st = self._rotate_drag
        i = st["idx"]
        box = self.rois[i]
        cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
        cur = math.degrees(math.atan2(y - cy, x - cx))
        self.angles[i] = round(st["a0"] + (cur - st["m0"]), 2)
        st["moved"] = True

    def del_roi(self, i=None):
        if i is None:
            i = self.roi_sel
        if not (0 <= i < len(self.rois)):
            return
        self.rois.pop(i)
        if i < len(self.angles):
            self.angles.pop(i)
        self.roi_sel = min(i, len(self.rois) - 1)
        self.refresh_roi_list()
        self.compute()

    def clear_roi(self):
        self.rois = []
        self.angles = []
        self.roi_sel = -1
        self.refresh_roi_list()
        self.compute()

    # ---------------- 计算 ----------------
    def _classify_raw(self):
        """缓存 (best, mind, t)；样本一变（颜色/角色/数量）就重算"""
        key = tuple(self.samples)
        if self._key != key or self._best is None or self._mind is None:
            roles = [r for _, r in self.samples]
            self._best, self._mind, self._t = classify_parts(
                self.lab, self._sample_labs(), roles)
            self._key = key
        return self._best, self._mind

    def _labels_for(self, tol):
        best, mind = self._classify_raw()
        return labels_from_parts(best, mind, getattr(self, "_t", None),
                                 tol, float(self.var_purity.get()))

    def _stats_at(self, tol, masks, roles, want_hist=False):
        labels = self._labels_for(tol)
        rows = []
        tot = {"coat_px": 0, "sub_raw": 0, "sub_open": 0, "area": 0, "unassigned": 0,
               "n_line": 0, "line_area": 0, "trans": 0, "thist": None}
        for m in masks:
            st = roi_stats(labels, m, roles, self.var_line.get(),
                           self.var_exmode.get(), want_hist)
            rows.append(st)
            for k in ("coat_px", "sub_raw", "sub_open", "area", "unassigned",
                      "n_line", "line_area", "trans"):
                tot[k] += st[k]
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
        return labels, rows, tot, pct

    def compute(self):
        self._set_result("")
        self.lbl_sum.config(text="")
        self.refresh_roi_list()
        if self.work is None:
            return
        if len(self.samples) < 2:
            self._set_result("请先吸两个颜色：①完好漆膜 ②露底基材\n（操作模式切到「吸取颜色」，在图上各点一下）\n")
            return
        roles = [r for _, r in self.samples]
        if ROLE_COAT not in roles or ROLE_SUB not in roles:
            self._set_result("颜色角色不完整：至少需要一个「漆膜」和一个「基材」。\n")
            return
        boxes = self.rois or [(0, 0, self.work.size[0], self.work.size[1])]
        shape2 = self.lab.shape[:2]
        masks = [rot_rect_mask(shape2, r,
                               self.angles[i] if i < len(self.angles) else 0.0)
                 for i, r in enumerate(boxes)]
        tol = float(self.var_tol.get())
        labels, rows, tot, pct = self._stats_at(tol, masks, roles, want_hist=True)
        self.labels = labels
        self._ver += 1
        self.last_pct, self.last_stats, self.last_boxes = pct, rows, boxes
        self._last_thist = tot.get("thist")
        cls = astm_class(pct)

        # 敏感性：容差 ±25% 时结果怎么变（同一套距离图，几乎不额外花时间）
        lo_labels, _, _, lo = self._stats_at(tol * 0.75, masks, roles)
        _, _, _, hi = self._stats_at(tol * 1.25, masks, roles)

        # ---- 精简输出（默认只给人看得懂的结论）----
        lines = []
        for i, st in enumerate(rows, 1):
            lines.append("R%d   %5.2f%%   %s\n" % (i, st["pct"], astm_class(st["pct"])))
        if len(rows) > 1:
            lines.append("─" * 22 + "\n")
            lines.append("合计 %5.2f%%   →  ASTM %s / ISO %s\n" % (pct, cls, iso_class(pct)))
        lines.append(margin_hint(pct))
        if tot.get("thist"):
            seg = " | ".join(("%d-%dpx %.0f%%" % (a[0], a[1], a[3])) if a[1] < 10 ** 5
                             else ("≥%dpx %.0f%%" % (a[0], a[3])) for a in tot["thist"])
            lines.append("露底厚度：%s\n" % seg)
            thick = sum(a[3] for a in tot["thist"] if a[0] >= 7)
            if thick < 5 and tot["sub_raw"] > 0:
                lines.append("⚠ 几乎全部露底都薄于 7px —— 这些多半是刀缝，不是掉漆。\n"
                             "  把「刀缝宽度/阈值」调到 7 以上再看，或换个真正的露底点重新吸色。\n")

        # ---- 诊断明细（勾选才显示）----
        if self.var_detail.get():
            lines.append("\n—— 诊断明细 ——\n")
            if self.full is not None and self.work is not None:
                k = self.full.size[0] / float(self.work.size[0])
                lines.append("分析 %dx%d / 原图 %dx%d（%.2f×）\n"
                             % (self.work.size[0], self.work.size[1],
                                self.full.size[0], self.full.size[1], k))
            for i, st in enumerate(rows, 1):
                a = max(1, st["area"])
                cut = max(0, st["sub_raw"] - st["sub_open"])
                lines.append(
                    "R%d 面积 %d px\n   漆膜 %.1f%%  露底 %.1f%%  刀缝排除 %.1f%%  过渡 %.1f%%  未分类 %.1f%%\n"
                    % (i, st["area"],
                       100.0 * st["coat_px"] / a, 100.0 * st["sub_open"] / a,
                       100.0 * cut / a, 100.0 * st["trans"] / a,
                       100.0 * st["unassigned"] / a))
            lines.append("（上面五项按选区面积算，相加≈100%；脱落率另按「漆膜+露底」算）\n")
            raw = 100.0 * tot["sub_raw"] / (tot["coat_px"] + tot["sub_raw"]) if (tot["coat_px"] + tot["sub_raw"]) else 0
            lines.append("合计 漆膜 %d px / 基材 %d px；不排除刀缝时 %.2f%%\n"
                         % (tot["coat_px"], tot["sub_raw"], raw))
            if tot["trans"]:
                lines.append("过渡带（漆/基材混色，已排除）%d px，占选区 %.1f%%\n"
                             % (tot["trans"], 100.0 * tot["trans"] / max(1, tot["area"])))
            if tot["n_line"]:
                lines.append("按形状识别线状刀缝 %d 段，排除 %.0f%% 的露底面积\n"
                             % (tot["n_line"], 100.0 * tot["line_area"] / max(1, tot["sub_raw"])))
            lines.append("容差敏感性：%.1f%% ~ %.1f%%（容差 %.0f~%.0f）\n"
                         % (lo, hi, tol * 0.75, tol * 1.25))
            best, mind = self._classify_raw()
            m_all = np.zeros(mind.shape, bool)
            for m in masks:
                m_all |= m
            mv = mind[m_all]
            if mv.size:
                q = np.percentile(mv, [50, 90, 99])
                parts = []
                for k in (0.6, 1.0, 1.5, 2.0):
                    t = tol * k
                    parts.append("容差%.0f→未分类%.0f%%" % (t, 100.0 * (mv > t).mean()))
                lines.append("色差分布：中位 %.0f / 90分位 %.0f / 99分位 %.0f；%s\n"
                             % (q[0], q[1], q[2], "，".join(parts)))

        # ---- 永远要看的警告 ----
        if getattr(self, "_auto_warn", "") and self.var_auto_mode.get():
            lines.append(self._auto_warn)
        if astm_class(lo) != astm_class(hi):
            lines.append("⚠ 这一档不稳：容差稍动就会在 %s ~ %s 之间跳。\n"
                         % (astm_class(lo), astm_class(hi)))
        unass = 100.0 * tot["unassigned"] / max(1, tot["area"])
        if unass > 12:
            lines.append("⚠ 未分类 %.0f%%：容差偏小或颜色取样不全（多吸一两个漆膜明暗档）\n" % unass)
        if tot["sub_raw"] > 0 and tot["sub_open"] < 0.2 * tot["sub_raw"]:
            thick_share = None
            if tot.get("thist"):
                thick_share = sum(a[3] for a in tot["thist"] if a[0] >= 7)
            if thick_share is None or thick_share >= 5:
                raw = 100.0 * tot["sub_raw"] / max(1, tot["coat_px"] + tot["sub_raw"])
                lines.append("⚠ 刀缝阈值 %d px 可能太大：把细小的脱落也抹掉了（原始 %.1f%% → 现在 %.1f%%）\n"
                             % (self.var_line.get(), raw, pct))
        self._set_result("".join(lines))
        self.lbl_sum.config(text="脱落 %.2f%%  →  ASTM %s   ISO %s" % (pct, cls, iso_class(pct)),
                            style="Warn.TLabel" if cls in ("0B", "1B") else "Sum.TLabel")
        self.refresh_roi_list()
        self.update_suggestion()
        self.render()

    def _default_name(self, suffix):
        if not self.path:
            return "结果" + suffix
        base = os.path.splitext(os.path.basename(self.path))[0]
        return base + suffix

    # ---------------- 导出 ----------------
    def export_csv(self):
        if not self.last_stats:
            messagebox.showinfo("提示", "先点「计算」")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv",
                                            filetypes=[("CSV", "*.csv")],
                                            initialfile=self._default_name("_百格.csv"))
        if not path:
            return
        roles = [r for _, r in self.samples]
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            wr = csv.writer(f)
            wr.writerow(["ROI", "x0", "y0", "x1", "y1", "面积px", "漆膜px", "基材px",
                         "未分类px", "脱落%(原始)", "脱落%(忽略细线)", "ASTM", "ISO"])
            for i, st in enumerate(self.last_stats, 1):
                r = self.rois[i - 1] if i - 1 < len(self.rois) else (0, 0, 0, 0)
                wr.writerow([i, round(r[0]), round(r[1]), round(r[2]), round(r[3]), st["area"],
                             st["coat_px"], st["sub_raw"], st["unassigned"],
                             round(st["pct_raw"], 2), round(st["pct"], 2),
                             astm_class(st["pct"]), iso_class(st["pct"])])
            wr.writerow(["合计", "", "", "", "", "", "", "", "", "",
                         round(self.last_pct, 2), astm_class(self.last_pct), iso_class(self.last_pct)])
        messagebox.showinfo("完成", "已保存：" + path)

    def export_png(self):
        if not self.last_stats:
            messagebox.showinfo("提示", "先点「计算」")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG", "*.png")],
                                            initialfile=self._default_name("_标注.png"))
        if not path:
            return
        rows = [(i + 1, r, st) for i, (r, st) in enumerate(zip(self.last_boxes, self.last_stats))]
        save_annotated(self.work, self.labels, [r for _, r, _ in rows], rows,
                       self.last_pct, astm_class(self.last_pct),
                       [c for c, _ in self.samples], [r for _, r in self.samples], path)
        messagebox.showinfo("完成", "已保存：" + path)

    def _mode_changed(self):
        mode = self.var_mode.get()
        cur = {"pick": "crosshair", "pan": "fleur", "ruler": "tcross"}.get(mode, "arrow")
        self.canvas.config(cursor=cur)
        hint = {"rect": "拖动框选；右键删框",
                "pick": "点击取色（第一次漆膜，第二次基材）",
                "ruler": "横跨刀缝拖一下"}.get(mode, "")
        if hasattr(self, "lbl_mode"):
            self.lbl_mode.config(text=hint)


# --------------------------------------------------------------------------
# 自检
# --------------------------------------------------------------------------
def self_test():
    ok = True
    # 1. 分级表
    cases = [(0.0, "5B"), (0.3, "5B"), (4.9, "4B"), (5.1, "3B"), (14.9, "3B"),
             (15.1, "2B"), (34.9, "2B"), (35.1, "1B"), (64.9, "1B"), (65.1, "0B")]
    for p, want in cases:
        got = astm_class(p)
        if got != want:
            ok = False
            print("FAIL class %.1f -> %s want %s" % (p, got, want))
    print("分级表 OK" if ok else "分级表 FAIL")

    # 2. 合成图：白漆 + 深色基材，已知脱落比例
    h, w = 400, 600
    img = np.full((h, w, 3), 236, np.uint8)
    img[:, :, 2] = 226
    # 两个 ROI，各 25% 与 10% 的脱落块
    img[20:180, 20:280][:, :65] = (40, 40, 44)
    img[220:380, 20:280][:, :26] = (40, 40, 44)
    pil = Image.fromarray(img)
    lab = rgb_to_lab(np.asarray(pil))
    sl = [rgb_to_lab(np.array((236, 232, 226), np.float32)),
          rgb_to_lab(np.array((40, 40, 44), np.float32))]
    labels, _ = classify(lab, sl, 35.0)
    roles = [ROLE_COAT, ROLE_SUB]
    st1 = roi_stats(labels, rect_mask(labels.shape, (20, 20, 280, 180)), roles, 1)
    st2 = roi_stats(labels, rect_mask(labels.shape, (20, 220, 280, 380)), roles, 1)
    print("合成 ROI1 脱落 %.2f%% (期望 25.00)" % st1["pct"])
    print("合成 ROI2 脱落 %.2f%% (期望 10.00)" % st2["pct"])
    if abs(st1["pct"] - 25) > 0.6 or abs(st2["pct"] - 10) > 0.6:
        ok = False
        print("FAIL 面积统计")

    # 3. 忽略细线：一条 2px 细线应被抹掉，一块 40px 方块应保留
    m = np.zeros((200, 200), bool)
    m[100:102, :] = True
    m[20:60, 20:60] = True
    op = morph_open(m, 5)
    line_left = int(op[100:102, :].sum())
    blob_left = int(op[20:60, 20:60].sum())
    print("开运算：细线残留 %d px（期望 0），方块保留 %d px（期望 1600）" % (line_left, blob_left))
    if line_left != 0 or blob_left < 1500:
        ok = False
        print("FAIL 开运算")

    # 4. 刀缝量尺：一条 7px 宽的暗线
    img2 = np.full((200, 300, 3), 200, np.uint8)
    img2[:, 150:157] = 60
    res = measure_line_width(img2, (60, 100), (240, 100))
    if res is None:
        ok = False
        print("FAIL 量尺：没测到")
    else:
        print("量尺：测得 %.1f px（真值 7.0，极性=%s，对比度 %.0f）"
              % (res["width_px"], res["polarity"], res["contrast"]))
        if abs(res["width_px"] - 7.0) > 1.2:
            ok = False
            print("FAIL 量尺宽度")

    # 5. GUI 构建
    try:
        root = tk.Tk()
        root.withdraw()
        app = App(root)
        app.full = pil
        app.work = pil
        app.rgb = np.asarray(pil).astype(np.uint8)
        app.lab = rgb_to_lab(app.rgb)
        # 空样本 / 单样本也不能崩（曾经在这里 argmin 空数组）
        app.samples = []
        app.labels = None
        app._best = app._mind = app._t = None
        app.render()
        app.samples = [((236, 232, 226), ROLE_COAT)]
        app._after_sample_change()
        app.render()
        print("空样本/单样本渲染 OK")
        app.samples = [((236, 232, 226), ROLE_COAT), ((40, 40, 44), ROLE_SUB)]
        app.rois = [(20, 20, 280, 180)]
        app._refresh_sample_list()
        app.refresh_roi_list()
        app.compute()
        out = app.txt.get("1.0", "end")
        root.update()
        root.destroy()
        print("GUI 构建 OK，计算输出：")
        print("  " + "\n  ".join(out.strip().splitlines()[:4]))
    except Exception as exc:
        ok = False
        print("GUI FAIL:", exc)

    print("\nSELF-TEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description="百格附着力图像分析")
    ap.add_argument("--cli", metavar="IMAGE", help="命令行模式：要分析的图片")
    ap.add_argument("--roi", action="append", help="选区 x0,y0,x1,y1；全为 0~1 时按比例解释；可重复")
    ap.add_argument("--coating", action="append", help="漆膜颜色 R,G,B；可重复")
    ap.add_argument("--substrate", action="append", help="基材颜色 R,G,B；可重复")
    ap.add_argument("--ignore", action="append", help="忽略颜色 R,G,B；可重复")
    ap.add_argument("--tol", type=float, default=35.0, help="Lab 容差（默认 35）")
    ap.add_argument("--purity", type=float, default=0.75,
                    help="露底纯度（默认 0.75，越大越只认纯基材）")
    ap.add_argument("--line-px", type=int, default=1, help="刀缝阈值 px（默认 1 = 不排除刀缝）")
    ap.add_argument("--out", help="标注图路径")
    ap.add_argument("--csv", help="CSV 路径")
    ap.add_argument("--self-test", action="store_true", help="运行自检")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()
    if args.cli:
        return run_cli(args)

    root = tk.Tk()
    try:
        import tkinter.font as tkfont
        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont"):
            tkfont.nametofont(name).configure(family="Microsoft YaHei UI", size=9)
    except Exception:
        pass
    App(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
