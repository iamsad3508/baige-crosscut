"""格网拟合 v3（实验）：不做二值化，直接在"暗度连续场"上做投影/自相关。

为什么不用阈值：槽的深浅在整幅图上差很多（扇贝边 -60，浅边 -10），任何全局阈值
都会一边漏一边多。投影是线性叠加，峰位不依赖阈值；只有量带宽时才用 半高（局部）判据。
纯 numpy + PIL，不引入 scipy/opencv。
"""
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
