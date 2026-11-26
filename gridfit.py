"""格网拟合 v3（实验）：不做二值化，直接在"暗度连续场"上做投影/自相关。

为什么不用阈值：槽的深浅在整幅图上差很多（扇贝边 -60，浅边 -10），任何全局阈值
都会一边漏一边多。投影是线性叠加，峰位不依赖阈值；只有量带宽时才用 半高（局部）判据。
纯 numpy + PIL，不引入 scipy/opencv。
"""
import math
import os

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

def snap_lines(gray, angle, xb, yb, per_x, per_y, center, span=0.30, sigma=1.5,
               sign_x=0, sign_y=0):
    """把每根等分线各自吸附到它那一道刀缝上。返回 (dx, dy)，视图 px，内部线用。

    判据不是「这一列平均比较暗」——那样格中央一片阴影也会赢，线就被吸到格子中间去
    （「自动对线对在格子中间」就是这个原因）。真正的缝要同时满足两条：
      1) 连续性：这一列/行里绝大多数都带该极性偏差（缝从一头贯到另一头；阴影斑块只占一小段）
      2) 典型偏离量（p75，不用均值，免得一头亮一头暗把值拉低）
    得分 = 连续性 × 偏离量；连续性 < 0.45 一律不动（宁可不挪，也不挪错）。

    **极性必须两个都试**：早期只有「暗缝」一支，遇到亮刀缝（露出白底材）的图，
    暗的那一支会锁到漆膜块上——用户看到的就是「自动对线之后线全跑到漆膜上」。
    这里每个方向都试 ±，取「峰更细」的那组：刀缝是细线，漆膜是宽块（按宽度定，不看强度）。
    `sign_x/sign_y` 传 1/-1 可直接指定（trace_grid 已经认过极性时用）。
    """
    g = np.asarray(gray).astype(np.uint8)
    fill = int(np.median(g))
    r = np.asarray(Image.fromarray(g, "L").rotate(
        angle, resample=Image.BILINEAR, center=center, fillcolor=fill)).astype(np.float32)
    bg = _gauss_bg(r, max(9.0, 0.06 * min(r.shape)))
    signed = bg - r

    # 只在框内统计：框外没有缝，拉进来只会把峰压平
    x0 = int(max(0, math.floor(min(xb)) + 1))
    x1 = int(min(signed.shape[1] - 1, math.ceil(max(xb)) - 1))
    y0 = int(max(0, math.floor(min(yb)) + 1))
    y1 = int(min(signed.shape[0] - 1, math.ceil(max(yb)) - 1))
    per_x = float(per_x) if per_x and per_x > 2 else 1.0
    per_y = float(per_y) if per_y and per_y > 2 else 1.0

    def one(cont, full, pos, per, off):
        """返回 (shift, 峰宽)；不动的返回 (0.0, None)。"""
        n = cont.size
        p = pos - off
        lo = int(max(0, p - span * per))
        hi = int(min(n - 1, p + span * per))
        if hi - lo < 3:
            return 0.0, None
        seg = full[lo:hi]
        j = lo + int(np.argmax(seg))          # 全局下标（seg 只是切片）
        if cont[j] < 0.45 or full[j] <= 0:
            return 0.0, None                  # 不是一条贯通的线 → 保持等分
        # 缝有一定宽度，单看最强那一列会在沟里左右抖 ±几像素 → 取峰附近平台的重心
        a = j
        while a > lo and full[a - 1] >= 0.6 * full[j]:
            a -= 1
        b = j
        while b < hi - 1 and full[b + 1] >= 0.6 * full[j]:
            b += 1
        w = full[a:b + 1]
        cen = a + (float((w * np.arange(a, b + 1)).sum() / w.sum()) - a) if w.sum() > 0 else float(j)
        shift = cen - p
        wdt = float(b - a + 1)
        if abs(shift) > 0.20 * per:           # 太远一定不是这道缝
            return 0.0, wdt
        if abs(shift) < 0.08 * per:           # 本来就在缝上：不动，但把宽度记下来（选极性要用）
            return 0.0, wdt
        return float(shift), wdt

    def run_axis(sign, positions, per, axis):
        raw = np.clip(sign * signed, 0, None)
        sub = raw[y0:y1, x0:x1] if (x1 - x0 > 8 and y1 - y0 > 8) else raw
        p95 = float(np.percentile(sub, 95))
        if p95 <= 0:
            return [0.0] * len(positions), [], 0
        thr = 0.35 * p95
        if axis == "x":
            cont = (sub > thr).mean(axis=0)
            dark = np.percentile(sub, 75, axis=0)
            off = x0
        else:
            cont = (sub > thr).mean(axis=1)
            dark = np.percentile(sub, 75, axis=1)
            off = y0
        full = cont * dark
        out = [0.0] * len(positions)
        widths = []
        moved = 0
        for i in range(1, len(positions) - 1):
            sh, wd = one(cont, full, positions[i], per, off)
            out[i] = sh
            if wd is not None:
                widths.append(wd)
            if sh != 0.0:
                moved += 1
        return out, widths, moved

    def pick(positions, per, axis, hint):
        best = None
        for sign in ((float(hint),) if hint else (1.0, -1.0)):
            shifts, widths, moved = run_axis(sign, positions, per, axis)
            if not widths:
                continue
            med_w = float(np.median(widths))
            # 刀缝比漆膜块细得多：峰宽更小的那一支才是缝
            key = (med_w, -moved)
            if best is None or key < best[0]:
                best = (key, shifts)
        return best[1] if best else [0.0] * len(positions)

    dx = pick(xb, per_x, "x", sign_x)
    dy = pick(yb, per_y, "y", sign_y)
    return dx, dy

# ---------------------------------------------------------------- 认缝 → 按缝铺线
def _profiles(raw, thr_frac=0.35):
    """沿两个方向各给一对剖面：(连续性, 典型偏离量)。`raw` 必须已经裁成「单向」偏差（≥0）。

    连续性 = 这一列/行里出现该偏差的像素占比：
      * 刀缝是**长条**（从一头贯到另一头）→ 接近 1
      * 漆膜上的阴影/斑块是**孤岛** → 0.1~0.3
    注意：整块漆膜同样能给出接近 1 的连续性（它也是一整列），宽度那一条由 `_centroids` 兜。
    """
    q = float(np.percentile(raw, 95))
    if q <= 0:
        return None, None
    hit = raw > thr_frac * q
    return (hit.mean(axis=0), np.percentile(raw, 75, axis=0)), \
           (hit.mean(axis=1), np.percentile(raw, 75, axis=1))


def _axis_dev(dev, axis, n_need, pitch0, lo, hi):
    """在一个方向上认缝。`dev` 是单向偏差（≥0，二维）。返回 (线位, 间距, 质量分)。

    axis=0 按列统计（找竖缝），axis=1 按行统计（找横缝）。
    """
    q = float(np.percentile(dev, 95))
    if q <= 0:
        return [], 0.0, -1.0, 0.0
    hit = dev > 0.35 * q
    cont = hit.mean(axis=axis)
    dark = np.percentile(dev, 75, axis=axis)
    score = _smooth1d(cont * dark, 1.0)
    raw = [c for c in _centroids(cont, dark, score, 1, score.size - 1, 1e9)
           if cont[int(round(c[0]))] >= 0.45]
    # 宽度门限用**候选自己量出来的间距**：刀道宽约为间距的 1/3，整块漆膜约 2/3。
    # 用框宽/格数算出来的间距（pitch0）在框画松时会失准，实测会让 48px 宽的漆膜块（间距 77）
    # 刚好卡在 0.55*pitch0 之内混进来，于是又错半格。
    if len(raw) >= 3:
        _sp = float(np.median(np.diff([c[0] for c in sorted(raw)])))
        max_w = 0.5 * _sp if _sp > 4 else 0.55 * pitch0
    else:
        _sp = pitch0
        max_w = 0.55 * pitch0
    cands = [c for c in raw if c[2] <= max_w]
    # 步进间距不能只用「框宽/格数」：框画得松（比格阵大一圈）时 pitch0 会大 10% 以上，
    # _walk 走到第 10 条就累计偏出界，整轴认不出来（实测 860 px 的框装着 771 px 的格阵：
    # pitch0=86 vs 实测 77，竖缝 12 条候选全在、结果 0 条）。候选自己的中位间距更准，
    # 两个都试，取认全且最规整的那组。
    tries = [pitch0]
    if abs(_sp - pitch0) > 0.02 * max(pitch0, 1.0):
        tries.append(_sp)
    best = None
    for p0 in tries:
        p0 = float(p0)
        if p0 <= 1:
            continue
        keep, last = [], -1e9
        for c in sorted(cands):
            if c[0] - last >= 0.45 * p0:
                keep.append(c)
                last = c[0]
        ctrs, pit = _walk(keep, n_need, lo, hi, p0)
        if len(ctrs) > n_need:
            ctrs = _trim(ctrs, n_need, lo, hi)
        if len(ctrs) != n_need:
            continue
        d = np.diff(ctrs)
        uni = 1.0 - float(np.std(d) / max(1e-6, np.mean(d)))
        peak = max(1e-6, float(score.max()))
        ctr = float(np.median([score[int(round(c))] for c in ctrs])) / peak
        sc = uni * (0.25 + ctr)
        if best is None or sc > best[0]:
            best = (sc, ctrs, pit)
    if best is None:
        return [], pitch0, -1.0, 0.0
    _sc, centers, pitch = best
    # 认出多于 n_need 条时（框把格外的缝也圈进来了），取最规整的那 n_need 条
    if len(centers) > n_need:
        centers = _trim(centers, n_need, lo, hi)
    if len(centers) != n_need:
        return centers, pitch, -1.0, 0.0
    d = np.diff(centers)
    uni = 1.0 - float(np.std(d) / max(1e-6, np.mean(d)))
    peak = max(1e-6, float(score.max()))
    ctr = float(np.median([score[int(round(c))] for c in centers])) / peak
    # 顺手把刀缝实测宽度带回去：用户常把“刀宽”填成刀片刃宽(0.1mm≈3px)，
    # 而图上露白的那条带其实十几二十像素，差太多就会把大半个缝算成“脱落”
    wd = []
    for c in centers:
        near = [k[2] for k in cands if abs(k[0] - c) < 2.0]
        if near:
            wd.append(min(near))
    return centers, pitch, uni * (0.25 + ctr), (float(np.median(wd)) if wd else 0.0)


def _centroids(cont, dark, score, lo, hi, pitch):
    """把一段剖面里的「窄峰」找出来，返回 [(位置, 得分, 宽度)]。

    峰位用平台重心（沟有宽度，取最暗/最亮那一列会在沟里左右抖）。
    **宽度是命门**：孤岛（漆膜块）和刀道都是一整列同向偏离、连续性都接近 1，
    只有宽度不同——刀道窄（≈刀宽）、整块漆膜宽（≈格距）。宽度超过 0.55 格距的一律当
    漆膜块丢掉，否则线会整整错开半格压在漆膜中心（实测过）。
    """
    out = []
    n = score.size
    lo, hi = int(max(1, lo)), int(min(n - 2, hi))
    max_w = max(3.0, 0.55 * pitch)
    i = lo
    while i < hi:
        if score[i] <= 0 or cont[i] < 0.45:
            i += 1
            continue
        if score[i] >= score[i - 1] and score[i] > score[i + 1]:
            a = b = i
            thr = 0.6 * score[i]
            while a > lo and score[a - 1] >= thr:
                a -= 1
            while b < hi and score[b + 1] >= thr:
                b += 1
            if b - a + 1 > max_w:            # 太宽 = 漆膜块，不是刀道
                i = b + 1
                continue
            w = score[a:b + 1]
            cen = float((w * np.arange(a, b + 1)).sum() / w.sum()) if w.sum() > 0 else float(i)
            out.append((cen, float(score[i]), float(b - a + 1)))
            i = b + 1
        else:
            i += 1
    return out


def _trim(centers, n_need, lo, hi):
    """候选多于期望条数时，挑最规整的连续 n_need 条（间距方差最小，尽量落在框内）。"""
    best, best_cost = None, None
    for i in range(0, len(centers) - n_need + 1):
        w = centers[i:i + n_need]
        d = np.diff(w)
        if d.size == 0:
            continue
        cost = float(np.std(d) / max(1e-6, np.mean(d)))
        outside = sum(1 for c in w if c < lo - 2 or c > hi + 2) * 0.2
        cost += outside
        if best_cost is None or cost < best_cost:
            best, best_cost = w, cost
    return list(best) if best else centers


def _walk(cands, n_need, lo, hi, pitch):
    """从最强的一条缝开始，按「上一条 + 实测间距」预测下一条，一条条往两边走。

    不用等分的原因：等分把误差全堆到最后几根，透视/刀走偏时越走越歪，再靠吸附去救很容易
    滑到隔壁（用户看到的就是「对到格子中间」）。逐条跟踪时每一步的搜索窗只有 0.35 格距，
    前一条是准的，下一条就不会跑偏。
    """
    if not cands:
        return [], 0.0
    cands = sorted(cands, key=lambda c: -c[1])
    anchor = cands[0][0]
    pts = sorted(c[0] for c in cands)
    strength = {round(c[0]): c[1] for c in cands}
    centers = [anchor]
    for d in (-1, 1):
        cur = anchor
        prev = None
        while len(centers) < n_need + 2:
            step = pitch if prev is None else (cur - prev)
            if step <= 0:
                step = pitch
            pred = cur + d * step
            w = 0.35 * abs(step)
            hit = [c for c in pts if pred - w <= c <= pred + w]
            if not hit:
                break
            best = max(hit, key=lambda c: strength.get(round(c), 0.0))
            if d * (best - cur) <= 0:
                break
            prev, cur = cur, best
            centers.append(best)
            if cur < lo or cur > hi:
                break
    centers.sort()
    ded = []
    for c in centers:
        if not ded or abs(c - ded[-1]) >= 0.45 * pitch:
            ded.append(c)
    pitch_out = float(np.median(np.diff(ded))) if len(ded) > 1 else pitch
    return ded, pitch_out


def trace_grid(gray, angle, center, box, cols, rows):
    """先认缝、再按缝铺线。返回旋转坐标系下的线位与诊断信息。

    - 期望条数 = 格数 + 1（10 格 → 11 条刀缝，最外两条就是这片的闭合边界）
    - 只认「连续性 ≥ 0.45」的峰，所以漆膜孤岛、阴影块不会被当成缝
    - 认全了才用（条数必须正好是 cols+1 / rows+1），否则该方向退回等分，别硬凑
    """
    g = np.asarray(gray).astype(np.uint8)
    x0, y0, x1, y1 = [float(v) for v in box]
    fill = int(np.median(g))
    r = np.asarray(Image.fromarray(g, "L").rotate(
        angle, resample=Image.BILINEAR, center=center, fillcolor=fill)).astype(np.float32)
    bg = _gauss_bg(r, max(9.0, 0.06 * min(r.shape)))
    # 带符号的暗场：正=比背景暗（暗沟），负=比背景亮（露白底材的刀缝）。
    # 早期版本只留正的，遇到「亮缝+暗漆膜」的图会把整块漆膜误判成缝，正好错开半格。
    raw = bg - r

    # 往外多留一圈：框画得紧时，最外那两条缝正好压在框边上，剖面里贴边的峰会被告警掉
    # （实测 1.0 倍能认 11 条，缩到 0.5 倍只认 10 条，就是最外一条被裁掉了）
    pad = int(max(4.0, 0.05 * max(x1 - x0, y1 - y0)))
    ix0 = int(max(1, math.floor(x0) - pad))
    ix1 = int(min(raw.shape[1] - 2, math.ceil(x1) + pad))
    iy0 = int(max(1, math.floor(y0) - pad))
    iy1 = int(min(raw.shape[0] - 2, math.ceil(y1) + pad))
    if ix1 - ix0 < 40 or iy1 - iy0 < 40:
        return {"ok_x": False, "ok_y": False, "xb": [], "yb": [], "msg": "选区太小"}
    sub = raw[iy0:iy1, ix0:ix1]

    # 分辨率归一：同一张图在 371 px 和 1400 px 下认缝结果必须一致，否则「预览分辨率」一变
    # 就换了个结果。这里把剖面统一缩放/放大到长边 ~900 px 再认（放大不增信息，但把固定
    # sigma 的平滑、固定比例的窗口拉回同一尺度），认完再换算回去。
    norm = 1.0
    long_side = max(sub.shape)
    if long_side and (long_side < 520 or long_side > 1500):
        norm = 900.0 / long_side
        tw, th = max(8, int(round(sub.shape[1] * norm))), max(8, int(round(sub.shape[0] * norm)))
        # 注意：sub 是带符号的暗场，负值代表「亮缝」。这里曾经用 uint8 装它，
        # 负值被 clip 成 0 → 亮缝整类被抹掉，只剩「压暗」一支能出候选（半格偏移就是这么来的）。
        sub = np.asarray(Image.fromarray(np.ascontiguousarray(sub, dtype=np.float32), mode="F")
                         .resize((tw, th), Image.BICUBIC), dtype=np.float32)

    out = {"ok_x": False, "ok_y": False, "xb": [], "yb": [], "msg": ""}

    # 极性未知：有的图是暗沟（沟比漆膜暗），有的是亮缝（露出白底材，缝比漆膜亮）。
    # 两个方向各自试两种极性，谁认出来的格子更规整、对比更强就用谁——不猜、不做全局假设。
    for tag, axis, n_need, off, blo, bhi in (
            ("x", 0, cols + 1, ix0, ix0, ix1),
            ("y", 1, rows + 1, iy0, iy0, iy1)):
        pitch0 = (bhi - blo) * norm / max(1, n_need - 1)
        lo, hi = (blo - off) * norm, (bhi - off) * norm
        best = None
        for sign in (1.0, -1.0):
            dev = np.clip(sign * sub, 0, None)
            centers, pitch, qual, wdt = _axis_dev(dev, axis, n_need, pitch0, lo, hi)
            if best is None or qual > best[2]:
                best = (centers, pitch, qual, sign, wdt)
        centers, pitch, qual, sign, wdt = best
        ok = len(centers) == n_need
        out["ok_" + tag] = ok
        out[("xb" if tag == "x" else "yb")] = [c / norm + off for c in centers] if ok else []
        out["n_" + tag] = len(centers)
        out["pitch_" + tag] = round(pitch / norm, 1)
        out["sign_" + tag] = "dark" if sign > 0 else "bright"
        out["w_" + tag] = round(float(wdt) / norm, 1)
        out["qual_" + tag] = round(qual, 2)
    return out

    for tag, (cont, dark), n_need, off, span, blo, bhi in (
            ("x", px, cols + 1, ix0, ix1 - ix0, ix0, ix1),
            ("y", py, rows + 1, iy0, iy1 - iy0, iy0, iy1)):
        score = _smooth1d(cont * dark, 1.0)
        pitch0 = span / max(1, n_need - 1)
        cands = _centroids(cont, dark, score, 1, score.size - 1, pitch0)
        # 只留连续性够的、且不至于挤在一起
        cands = [c for c in cands if cont[int(round(c[0]))] >= 0.45]
        keep, last = [], -1e9
        for c in sorted(cands):
            if c[0] - last >= 0.45 * pitch0:
                keep.append(c)
                last = c[0]
        # 边界要用各自的轴（这里以前两个方向都传了 x 的边界，y 方向走两步就被判出界）
        centers, pitch = _walk(keep, n_need, (blo - off) * norm, (bhi - off) * norm, pitch0 * norm)
        ok = len(centers) == n_need
        out["ok_" + tag] = ok
        out[("xb" if tag == "x" else "yb")] = [c / norm + off for c in centers] if ok else []
        out["n_" + tag] = len(centers)
        out["pitch_" + tag] = round(pitch / norm, 1)
        out["cont_" + tag] = round(float(np.median([cont[int(round(c))] for c in centers])), 2) if centers else 0.0
    return out
