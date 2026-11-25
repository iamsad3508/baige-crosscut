"""格网拟合 v3（实验）：不做二值化，直接在"暗度连续场"上做投影/自相关。

为什么不用阈值：槽的深浅在整幅图上差很多（扇贝边 -60，浅边 -10），任何全局阈值
都会一边漏一边多。投影是线性叠加，峰位不依赖阈值；只有量带宽时才用 半高（局部）判据。
纯 numpy + PIL，不引入 scipy/opencv。
"""
import math

import numpy as np
from PIL import Image, ImageFilter

GRID_MAX = 1400
GRID_MAX_ANGLE = 1.5
GRID_BAND_LEVEL = 0.40     # 带宽：以该线峰值为基准，取 40% 处


def _gauss_bg(gray, radius):
    im = Image.fromarray(gray.astype(np.uint8))
    return np.asarray(im.filter(ImageFilter.GaussianBlur(radius))).astype(np.float32)


def _smooth1d(v, sigma=2.0):
    r = int(max(1, round(sigma * 2.5)))
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    return np.convolve(v.astype(np.float64), k, mode="same")


def _peaks(prof, min_dist, thr):
    idx = [i for i in range(1, len(prof) - 1)
           if prof[i] >= prof[i - 1] and prof[i] > prof[i + 1] and prof[i] > thr]
    if not idx:
        return np.array([], int)
    keep = []
    for i in sorted(idx, key=lambda i: -prof[i]):
        if all(abs(i - j) >= min_dist for j in keep):   # 两侧都要比，之前只跟最后一个比→会互相挤掉
            keep.append(i)
    return np.array(sorted(keep), int)


def _period(prof, lo=24, hi=900, fund=0.80):
    """自相关定周期：取第一个"够强"的局部极大，避免落到 2× 谐波上。"""
    s = prof - prof.mean()
    n = len(s)
    f = np.fft.rfft(s, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    ac = ac / max(ac[0], 1e-9)
    hi = min(hi, n // 2)
    if hi <= lo + 2:
        return None, 0.0
    seg = ac[lo:hi]
    pk = _peaks(seg, 3, float(seg.max()) * fund)
    if len(pk) == 0:
        k = lo + int(np.argmax(seg))
    else:
        k = lo + int(pk[0])
    return k, float(ac[k])


def _detrend(v, win):
    win = max(31, int(win) | 1)
    k = np.ones(win, dtype=np.float64) / win
    pad = np.pad(v, (win // 2, win // 2), mode="edge")
    return v - np.convolve(pad, k, mode="same")[win // 2:win // 2 + len(v)]


def _fit_axis(prof):
    p = _smooth1d(prof, 2.0)
    p = _detrend(p, max(51, len(p) // 6))   # 去低频（照明梯度/背景），保留线峰
    p = np.clip(p, 0, None)
    per, strength = _period(p)
    if not per or strength < 0.15:
        return None
    lo_lvl = float(np.mean(p) + 0.22 * (p.max() - np.mean(p)))
    pk = _peaks(p, max(6, int(per * 0.5)), lo_lvl)
    a, b, res = per, 0.0, 0.0
    for _ in range(4):
        if len(pk) < 2:
            return None
        idx = np.arange(len(pk), dtype=np.float64)
        A = np.vstack([idx, np.ones_like(idx)]).T
        (a, b), *_ = np.linalg.lstsq(A, pk.astype(np.float64), rcond=None)
        res = np.abs(pk - (a * idx + b))
        keep = res <= max(2.0, 0.2 * a)
        if keep.all() or keep.sum() < 2:
            break
        pk = pk[keep]
    pos, k = [], int(np.ceil((0 - b) / a))
    while a * k + b < len(p):
        v = a * k + b
        if v >= 0:
            pos.append(float(v))
        k += 1
    return {"period": float(a), "lines": pos, "resid": float(np.median(res)),
            "strength": float(strength), "prof": p}


def _band_width(raw, positions, axis, span, samples=40):
    """带宽：沿线采样，垂直方向在 raw 上找 ≥40% 峰值且连续的那一段。"""
    ws = []
    for pos in positions:
        p = int(round(pos))
        lim = raw.shape[1] if axis == "x" else raw.shape[0]
        if not (0 <= p < lim):
            continue
        for t in np.linspace(0, span - 1, samples).astype(int):
            line = raw[t, :].astype(np.float64) if axis == "x" else raw[:, t].astype(np.float64)
            win = line[max(0, p - 3):p + 4]
            if win.size == 0:
                continue
            pkv = float(win.max())
            if pkv < 4:
                continue
            lvl = max(2.0, GRID_BAND_LEVEL * pkv)
            a = p
            while a > 0 and line[a - 1] >= lvl:
                a -= 1
            c = p
            while c < len(line) - 1 and line[c + 1] >= lvl:
                c += 1
            ws.append(c - a + 1)
    return float(np.median(ws)) if ws else 0.0


def _angle(dir_x, dir_y):
    """用两个方向的投影峰"尖锐度"判断图有没有歪；歪了就报让用户先转正。"""
    best, ba = -1.0, 0.0
    for ang in (0.0, 0.5, -0.5, 1.0, -1.0):
        r = np.asarray(Image.fromarray(dir_x.astype(np.float32) / max(dir_x.max(), 1) * 255)
                       .rotate(ang, resample=Image.BILINEAR)).astype(np.float64)
        sc = float(np.var(np.diff(_smooth1d(r.mean(0), 2.0))))
        if sc > best:
            best, ba = sc, ang
    return ba


def fit(gray, max_dim=GRID_MAX):
    h0, w0 = gray.shape
    f = 1.0
    if max(h0, w0) > max_dim:
        f = float(max_dim) / max(h0, w0)
        im = Image.fromarray(gray).resize((max(1, int(w0 * f)), max(1, int(h0 * f))), Image.LANCZOS)
        g = np.asarray(im).astype(np.float32)
    else:
        g = gray.astype(np.float32)
    h, w = g.shape
    bg = _gauss_bg(g, max(9.0, 0.06 * min(h, w)))
    raw = np.clip(bg - g, 0, None)

    out = {"ok": False, "warn": "", "angle": 0.0, "scale": 1.0 / f, "size": (w0, h0),
           "dark_frac": float((raw > 12).mean())}
    fx = _fit_axis(raw.mean(0))
    fy = _fit_axis(raw.mean(1))
    if not fx or not fy:
        out["warn"] = "grid.nolines"
        return out
    ang = _angle(fx["prof"], fy["prof"]) if abs(fx["period"] - fy["period"]) > 0.06 * fx["period"] else 0.0
    out["angle"] = ang
    if abs(ang) > GRID_MAX_ANGLE:
        out["warn"] = "grid.tilted"
        return out
    kx = _band_width(raw, fx["lines"], "x", h)
    ky = _band_width(raw, fy["lines"], "y", w)
    out.update({
        "ok": len(fx["lines"]) >= 2 and len(fy["lines"]) >= 2,
        "period_x": fx["period"] / f, "period_y": fy["period"] / f,
        "resid_x": fx["resid"] / f, "resid_y": fy["resid"] / f,
        "strength_x": fx["strength"], "strength_y": fy["strength"],
        "xs": [v / f for v in fx["lines"]], "ys": [v / f for v in fy["lines"]],
        "k_x": kx / f, "k_y": ky / f,
    })
    if out["ok"]:
        bad = (fx["resid"] > 0.25 * fx["period"]) or (fy["resid"] > 0.25 * fy["period"])
        out["warn"] = "grid.loose" if bad else ""
    else:
        out["warn"] = "grid.fewlines"
    return out


def cells(xs, ys, k_x, k_y, inset=0.0):
    """由线位置生成格子；槽宽两半分掉，再可选内缩。"""
    cs = []
    for i in range(len(xs) - 1):
        x0 = xs[i] + k_x / 2.0 + inset
        x1 = xs[i + 1] - k_x / 2.0 - inset
        for j in range(len(ys) - 1):
            y0 = ys[j] + k_y / 2.0 + inset
            y1 = ys[j + 1] - k_y / 2.0 - inset
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            cs.append({"i": i, "j": j, "box": [x0, y0, x1, y1], "area": (x1 - x0) * (y1 - y0)})
    return cs


def estimate_angle(gray, span=8.0, max_dim=700, coarse=0.5, fine=0.05):
    """估格网倾角（度）：旋转到「投影最尖锐」的那个角度。

    翻拍/手摆的照片常歪 2~8°，轴对齐的 fit() 在这种图上会直接报 nolines
    （实测一张 1280×960 的旧翻拍图歪 5.75°，fit 返回 nolines，转正后周期
    123.6×117.6 px 一目了然）。只用来给人一个初值，最终角度仍由人定。
    """
    from PIL import ImageFilter
    g = np.asarray(gray).astype(np.float32)
    h0, w0 = g.shape
    f = 1.0
    if max(h0, w0) > max_dim:
        f = float(max_dim) / max(h0, w0)
        g = np.asarray(Image.fromarray(g.astype(np.uint8)).resize(
            (max(1, int(w0 * f)), max(1, int(h0 * f))), Image.LANCZOS)).astype(np.float32)
    bg = _gauss_bg(g, max(9.0, 0.05 * min(g.shape)))
    raw = np.clip(bg - g, 0, None)

    def score(ang):
        a = np.asarray(Image.fromarray(raw, mode="F").rotate(
            ang, resample=Image.BILINEAR, fillcolor=0.0))
        y0, y1 = int(a.shape[0] * .08), int(a.shape[0] * .92)
        x0, x1 = int(a.shape[1] * .08), int(a.shape[1] * .92)
        a = a[y0:y1, x0:x1]
        px = np.clip(_detrend(_smooth1d(a.mean(0), 2.0), 120), 0, None)
        py = np.clip(_detrend(_smooth1d(a.mean(1), 2.0), 120), 0, None)
        return float(np.var(np.diff(px)) + np.var(np.diff(py)))

    best = max(((score(a), a) for a in np.arange(-span, span + 1e-9, coarse)))
    best = max((score(a), a) for a in np.arange(best[1] - coarse, best[1] + coarse + 1e-9, fine))
    return round(float(best[1]), 2)

def snap_lines(gray, angle, xb, yb, per_x, per_y, center, span=0.30, sigma=1.5):
    """把每根等分线各自吸附到它那一道刀缝上。返回 (dx, dy)，视图 px，内部线用。

    判据不是「这一列平均比较暗」——那样格中央一片阴影也会赢，线就被吸到格子中间去
    （「自动对线对在格子中间」就是这个原因）。真正的缝要同时满足两条：
      1) 连续性：这一列/行里绝大多数都暗（缝从一头贯到另一头；阴影斑块只占一小段）
      2) 暗度：这一列/行的典型暗度（p75，不用均值，免得一头亮一头暗把值拉低）
    得分 = 连续性 × 暗度；连续性 < 0.45 一律不动（宁可不挪，也不挪错）。
    """
    g = np.asarray(gray).astype(np.uint8)
    fill = int(np.median(g))
    r = np.asarray(Image.fromarray(g, "L").rotate(
        angle, resample=Image.BILINEAR, center=center, fillcolor=fill)).astype(np.float32)
    bg = _gauss_bg(r, max(9.0, 0.06 * min(r.shape)))
    raw = np.clip(bg - r, 0, None)

    # 只在框内统计：框外没有缝，拉进来只会把峰压平
    x0 = int(max(0, math.floor(min(xb)) + 1))
    x1 = int(min(raw.shape[1] - 1, math.ceil(max(xb)) - 1))
    y0 = int(max(0, math.floor(min(yb)) + 1))
    y1 = int(min(raw.shape[0] - 1, math.ceil(max(yb)) - 1))
    sub = raw[y0:y1, x0:x1] if (x1 - x0 > 8 and y1 - y0 > 8) else raw
    p95 = float(np.percentile(sub, 95))
    if p95 <= 0:
        return [0.0] * len(xb), [0.0] * len(yb)
    thr = 0.35 * p95

    cont_x, cont_y = (sub > thr).mean(axis=0), (sub > thr).mean(axis=1)
    dark_x, dark_y = np.percentile(sub, 75, axis=0), np.percentile(sub, 75, axis=1)

    def one(cont, dark, pos, per, off):
        n = cont.size
        full = cont * dark                    # 整条剖面的得分（找峰、比次峰都用它）
        p = pos - off
        lo = int(max(0, p - span * per))
        hi = int(min(n - 1, p + span * per))
        if hi - lo < 3:
            return 0.0
        seg = full[lo:hi]
        j = lo + int(np.argmax(seg))          # 全局下标（seg 只是切片）
        if cont[j] < 0.45 or full[j] <= 0:
            return 0.0                        # 不是一条贯通的线 → 保持等分
        # 缝有一定宽度，单看最暗那一列会在沟里左右抖 ±几像素 → 取峰附近平台的重心
        a = j
        while a > lo and full[a - 1] >= 0.6 * full[j]:
            a -= 1
        b = j
        while b < hi - 1 and full[b + 1] >= 0.6 * full[j]:
            b += 1
        w = full[a:b + 1]
        cen = a + (float((w * np.arange(a, b + 1)).sum() / w.sum()) - a) if w.sum() > 0 else float(j)
        shift = cen - p
        # 不动比乱动好：小于 8% 格距的偏差视作“本来就对”，一格刀宽内的抖动没有意义
        if abs(shift) < 0.08 * per:
            return 0.0
        if abs(shift) > 0.20 * per:           # 太远一定不是这道缝
            return 0.0
        return float(shift)

    dx = [0.0] * len(xb)
    dy = [0.0] * len(yb)
    for i in range(1, len(xb) - 1):
        dx[i] = one(cont_x, dark_x, xb[i], per_x, x0)
    for j in range(1, len(yb) - 1):
        dy[j] = one(cont_y, dark_y, yb[j], per_y, y0)
    return dx, dy
