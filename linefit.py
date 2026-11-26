"""逐线检测（v2.52）：不假设等距、不假设平行、不假设正交。

思路（比纯 Hough 稳）：
  1. 选区里取「亮沟 / 暗沟」残差；
  2. Hough 只用来定**两个族的大致方向**（各取一个最强峰）；
  3. 每族把残差按该角度转正，沿垂直方向取剖面 → 一维找峰（同一条缝不会劈成两个），
     峰＝一道刀缝 → **线数就是格数+1**，不需要用户填行列；
  4. 每根线再单独在 ±10° 内搜一遍自己的角度 → 支持「不平行」；
  5. 间距各是各的，不做等分、不做周期拟合。

只用 numpy + PIL（项目没有 scipy/opencv）。
"""
import math

import numpy as np
from PIL import Image, ImageFilter


def _residual(crop, bright=True, radius=None):
    g = np.asarray(crop, np.float32)
    r = int(radius or max(7, min(g.shape) / 10.0))
    bg = np.asarray(Image.fromarray(np.clip(g, 0, 255).astype(np.uint8))
                    .filter(ImageFilter.GaussianBlur(r)), np.float32)
    d = (g - bg) if bright else (bg - g)
    return np.clip(d, 0, None)


def _hough_top(res, exclude=None):
    """只求一个方向：返回 (角度, 票数)。exclude=(基准角, 下限, 上限) 时排除该角度带。"""
    thr = max(float(np.percentile(res, 96.0)), 0.5 * float(res.max()))
    hit = res > thr
    ys, xs = np.nonzero(hit)
    if xs.size < 80:
        return None
    pts = np.stack([xs, ys], 1).astype(np.float32)
    half = float(np.hypot(*res.shape)) + 2.0
    nr = int(round(2 * half)) + 2
    best = (0, 0.0)
    for t in np.arange(0.0, 180.0, 1.0):
        if exclude:
            dd = abs(((t - exclude[0] + 90) % 180) - 90)
            if not (exclude[1] <= dd <= exclude[2]):
                continue
        r = np.radians(t)
        rho = pts[:, 0] * np.cos(r) + pts[:, 1] * np.sin(r)
        idx = np.clip((rho + half).astype(np.int32), 0, nr - 1)
        v = int(np.bincount(idx, minlength=nr).max())
        if v > best[0]:
            best = (v, float(t))
    return (best[1], best[0]) if best[0] > 0 else None


def _line_score(res, thr, ang, px, py, length):
    """沿这条线取样，返回 (连续性, 平均强度)。"""
    ca, sa = math.cos(math.radians(ang)), math.sin(math.radians(ang))
    n = max(12, int(length / 2))
    t = np.linspace(-length / 2, length / 2, n)
    xs = np.round(px + ca * t).astype(np.int32)
    ys = np.round(py + sa * t).astype(np.int32)
    ok = (xs >= 0) & (xs < res.shape[1]) & (ys >= 0) & (ys < res.shape[0])
    if ok.sum() < 6:
        return 0.0, 0.0
    v = res[ys[ok], xs[ok]]
    return float((v > thr).mean()), float(v.mean())


def _peaks1d(prof, cont, gap, max_lines):
    order = np.argsort(-prof)
    keep = []
    for k in order:
        if prof[k] < 0.35 * prof.max() or cont[k] < 0.45:
            break
        if all(abs(k - q) >= gap for q in keep):
            keep.append(int(k))
        if len(keep) >= max_lines:
            break
    keep.sort()
    merged = []
    for k in keep:
        if merged and abs(k - merged[-1]) < gap:
            if prof[k] > prof[merged[-1]]:
                merged[-1] = k
        else:
            merged.append(k)
    return merged


def _family(res, ang, max_lines=26):
    """把这个角度的刀缝转正 → 一维找峰 → 每根线单独修正角度。

    转正后线可能变横也可能变竖（PIL 的旋转正负跟 y 朝下有关，同一个角度翻转一次就换一种），
    所以两个方向都算，取连续性高的那个；这一步不能写死 axis，否则整族一个峰都出不来。
    """
    h, w = res.shape
    cx, cy = w / 2.0, h / 2.0
    rot = np.asarray(Image.fromarray(np.clip(res * 6, 0, 255).astype(np.uint8))
                     .rotate(ang, resample=Image.BILINEAR, center=(cx, cy)), np.float32) / 6.0
    p95 = float(np.percentile(rot, 95))
    if p95 <= 0:
        return []
    thr = 0.35 * p95
    hit = rot > thr
    gap = max(5.0, 0.30 * min(h, w) / max(3, max_lines))
    row_cont, col_cont = hit.mean(axis=1), hit.mean(axis=0)
    row_prof = row_cont * (np.percentile(rot, 75, axis=1) / max(1e-6, p95))
    col_prof = col_cont * (np.percentile(rot, 75, axis=0) / max(1e-6, p95))
    if float(row_cont.max()) >= float(col_cont.max()):
        idx = _peaks1d(row_prof, row_cont, gap, max_lines)
        horiz = True
    else:
        idx = _peaks1d(col_prof, col_cont, gap, max_lines)
        horiz = False
    if len(idx) < 2:
        return []

    ca, sa = math.cos(math.radians(-ang)), math.sin(math.radians(-ang))
    out = []
    rmax = max(1e-6, float(res.max()))
    for k in idx:
        if horiz:
            du, dv = 0.0, k - cy
        else:
            du, dv = k - cx, 0.0
        px, py = cx + du * ca - dv * sa, cy + du * sa + dv * ca
        c0, m0 = _line_score(res, thr, ang, px, py, min(h, w) * 0.9)
        base = c0 * (m0 / rmax)
        best = (base, ang, px, py)
        for da in np.arange(-4, 4.01, 1.0):          # 每根线自己的角度（只允许小幅偏离，
            a = ang + da                              # 放太宽会被旁边更宽的暗/亮块吸走）
            c2, s2 = math.cos(math.radians(a)), math.sin(math.radians(a))
            for off in (-2.0, 0.0, 2.0):
                qx, qy = px - s2 * off, py + c2 * off
                c_, m_ = _line_score(res, thr, a, qx, qy, min(h, w) * 0.9)
                sc = c_ * (m_ / rmax)
                if sc > best[0] * 1.20:               # 没有明显更好就不动角度
                    best = (sc, a, qx, qy)
        out.append({"ang": float(best[1]), "px": float(best[2]), "py": float(best[3]),
                    "score": float(best[0])})
    return out


def _clip(ang, rho, box):
    """直线与框求交，返回框内那一小段的两端。

    口径：ang 是**方向角**，法线 = (−sin ang, cos ang)，方程 −x·sin + y·cos = rho。
    （早期混用过「角度=法线角」，和 band 的 nx=−sin/ny=cos 打架，格子会退化成一个点。）
    """
    x0, y0, x1, y1 = [float(v) for v in box]
    ca, sa = math.cos(math.radians(ang)), math.sin(math.radians(ang))
    pts = []
    if abs(ca) > 1e-9:                      # 与左右两边求交
        for x in (x0, x1):
            y = (rho + x * sa) / ca
            if y0 - 1 <= y <= y1 + 1:
                pts.append((x, y))
    if abs(sa) > 1e-9:                      # 与上下两边求交
        for y in (y0, y1):
            x = (y * ca - rho) / sa
            if x0 - 1 <= x <= x1 + 1:
                pts.append((x, y))
    if len(pts) < 2:
        return None
    a, b = pts[0], pts[-1]
    if math.hypot(a[0] - b[0], a[1] - b[1]) < 10:
        return None
    return [[round(a[0], 1), round(a[1], 1)], [round(b[0], 1), round(b[1], 1)]]


def intersect(l1, l2):
    """两条 (ang, rho) 线的交点；平行返回 None。"""
    a1, r1 = math.radians(l1["ang"]), l1["rho"]
    a2, r2 = math.radians(l2["ang"]), l2["rho"]
    s1, c1, s2, c2 = math.sin(a1), math.cos(a1), math.sin(a2), math.cos(a2)
    det = -s1 * c2 + c1 * s2
    if abs(det) < 1e-9:
        return None
    return [(r1 * c2 - r2 * c1) / det, (r1 * s2 - r2 * s1) / det]


def _hough_peaks(res, n=60, sup_deg=4.0, sup_rho=6.0, min_votes=12):
    """Hough 取峰：(角度, ρ, 票)。只用 1° / 1px 的粗网格，够定线位。"""
    thr = max(float(np.percentile(res, 96.0)), 0.55 * float(res.max()))
    ys, xs = np.nonzero(res > thr)
    if xs.size < 80:
        return None
    pts = np.stack([xs, ys], 1).astype(np.float32)
    half = float(np.hypot(*res.shape)) + 2.0
    nr = int(round(2 * half)) + 2
    thetas = np.arange(0.0, 180.0, 1.0)
    acc = np.zeros((thetas.size, nr), np.int32)
    for i, t in enumerate(np.radians(thetas)):
        rho = pts[:, 0] * np.cos(t) + pts[:, 1] * np.sin(t)
        np.add.at(acc[i], np.clip((rho + half).astype(np.int32), 0, nr - 1), 1)
    out = []
    a = acc.copy()
    di, dj = int(sup_deg) + 1, int(sup_rho) + 1
    for _ in range(n):
        i, j = np.unravel_index(int(a.argmax()), a.shape)
        v = int(a[i, j])
        if v < min_votes:
            break
        # Hough 的 θ 是法线角：方向角 = θ+90；ρ 换成 −x·sin+y·cos 口径（正好反号）
        out.append((float(thetas[i]) + 90.0, float(-(j - half)), v))
        a[max(0, i - di):i + di + 1, max(0, j - dj):j + dj + 1] = 0
    return out


def _norm_family(pk, a0):
    """把峰统一到以 a0 为中心的方向（0°/180° 同一条线，需要时翻转 ρ）。"""
    out = []
    for ang, rho, v in pk:
        da, k = ang - a0, 0
        while da > 90:
            da -= 180
            k += 1
        while da < -90:
            da += 180
            k += 1
        out.append((a0 + da, (rho if k % 2 == 0 else -rho), v))
    return out


def _space_filter(fam, min_gap=5.0):
    """间距一致性：留下最长的一段「间距相近」的线，其它当假峰丢掉。"""
    if len(fam) < 3:
        return fam
    fam = sorted(fam, key=lambda q: q[1])
    best = []
    for it in range(3):
        gaps = [fam[i + 1][1] - fam[i][1] for i in range(len(fam) - 1)]
        med = float(np.median([g for g in gaps if g > 0] or [0]))
        if med <= 0:
            break
        lo, hi = max(min_gap, 0.45 * med), 2.2 * med
        runs, cur = [], [fam[0]]
        for i, g in enumerate(gaps):
            if lo <= g <= hi:
                cur.append(fam[i + 1])
            else:
                runs.append(cur)
                cur = [fam[i + 1]]
        runs.append(cur)
        best = max(runs, key=len)
        if best is fam:
            break
        fam = best
    return best


def detect_lines(gray, box, max_lines=26):
    """返回 {'ok','a','b','polarity','why'}；line={ang,rho,p0,p1,score}（视图 px）。"""
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(gray.shape[1], x1), min(gray.shape[0], y1)
    ins = 6
    xi0, yi0 = x0 + ins, y0 + ins
    if x1 - xi0 < 50 or y1 - yi0 < 50:
        return {"ok": False, "why": "选区太小", "a": [], "b": []}
    crop = gray[yi0:y1 - ins, xi0:x1 - ins]

    best = None
    for bright in (True, False):
        res = _residual(crop, bright=bright)
        if float(res.max()) < 4:
            continue
        pk = _hough_peaks(res)
        if not pk or len(pk) < 6:
            continue
        # 按角度分箱统计票数：取票最多的两个「方向簇」，且两者夹角 50~130°
        # （不能只看最强的那一根——斜向的假峰常常单根最强，但真刀缝是成簇的）
        bins = {}
        for ang, rho, v in pk:
            k = int(round(ang / 5.0)) % 36
            bins[k] = bins.get(k, 0) + v
        order = sorted(bins.items(), key=lambda kv: -kv[1])
        a1 = order[0][0] * 5.0
        a2 = None
        for k, _v in order[1:]:
            ang = k * 5.0
            if 70 <= abs(((ang - a1 + 90) % 180) - 90) <= 110:   # 交叉划线工具≈90°，放宽到 70~110
                a2 = ang
                break
        if a2 is None:
            continue
        f1 = [p for p in pk if abs(((p[0] - a1 + 90) % 180) - 90) <= 12]
        f2 = [p for p in pk if abs(((p[0] - a2 + 90) % 180) - 90) <= 12]
        f1, f2 = _norm_family(f1, a1), _norm_family(f2, a2)
        f1, f2 = _space_filter(f1), _space_filter(f2)
        if len(f1) < 3 or len(f2) < 3:
            continue
        key = (len(f1) + len(f2), sum(p[2] for p in f1) + sum(p[2] for p in f2))
        if best is None or key > best[0]:
            best = (key, f1, f2, bright)

    if best is None:
        return {"ok": False, "why": "没找到成对的两族刀缝", "a": [], "b": []}
    _, f1, f2, bright = best

    def to_wire(fam):
        out = []
        for ang, rho, v in fam:
            rho_g = rho - xi0 * math.sin(math.radians(ang)) + yi0 * math.cos(math.radians(ang))
            seg = _clip(ang, rho_g, (x0, y0, x1, y1))
            if not seg:
                continue
            out.append({"ang": round(ang, 2), "rho": round(rho_g, 2), "score": int(v),
                        "p0": seg[0], "p1": seg[1]})
        out.sort(key=lambda L: L["rho"])
        # 每根线的角度再精修一次（±4°，只在明显更好时才动）
        for L in out:
            mx = (L["p0"][0] + L["p1"][0]) / 2.0
            my = (L["p0"][1] + L["p1"][1]) / 2.0
            c0, m0 = _line_score(gray, 0.0, L["ang"], mx, my,
                                 min(x1 - x0, y1 - y0) * 0.9)
            bestA = (c0 * m0, L["ang"])
            for da in np.arange(-4, 4.01, 0.5):
                c_, m_ = _line_score(gray, 0.0, L["ang"] + da, mx, my,
                                     min(x1 - x0, y1 - y0) * 0.9)
                if c_ * m_ > bestA[0] * 1.05:
                    bestA = (c_ * m_, L["ang"] + da)
            if bestA[1] != L["ang"]:
                a = bestA[1]
                rho = mx * math.cos(math.radians(a)) + my * math.sin(math.radians(a))
                L["ang"], L["rho"] = round(a, 2), round(rho, 2)
                seg = _clip(a, rho, (x0, y0, x1, y1))
                if seg:
                    L["p0"], L["p1"] = seg[0], seg[1]
        return out[:max_lines]

    la, lb = to_wire(f1), to_wire(f2)
    if len(la) < 2 or len(lb) < 2:
        return {"ok": False, "why": "线太少（%d / %d）" % (len(la), len(lb)), "a": la, "b": lb}
    return {"ok": True, "a": la, "b": lb, "polarity": ("bright" if bright else "dark"), "why": ""}


# ---------------------------------------------------------------- 线集合 → 格子
def norm(a, b):
    """把两族的线各自排序（沿自身法线方向），并保证第二族方向与第一族大致垂直。"""
    def key(L):
        mx = (L["p0"][0] + L["p1"][0]) / 2.0
        my = (L["p0"][1] + L["p1"][1]) / 2.0
        return mx * math.cos(math.radians(L["ang"])) + my * math.sin(math.radians(L["ang"]))
    return sorted(a, key=key), sorted(b, key=key)


def _order_by(a, b):
    """用「与另一族中间那根线的交点」把本族排序。

    只按自身 ρ 排不行：每根线角度不同，ρ 大小不等于空间次序（这就是交错四边形的原因）。
    """
    if len(a) < 2 or len(b) < 1:
        return a
    ref = b[len(b) // 2]
    ca, sa = math.cos(math.radians(ref["ang"])), math.sin(math.radians(ref["ang"]))
    out = []
    for L in a:
        p = intersect(L, ref)
        if p is None:
            continue
        out.append((p[0] * ca + p[1] * sa, L))
    out.sort(key=lambda q: q[0])
    return [L for _, L in out]


def build_cells(a, b):
    """两族线 → 逐个格子的四边形（相邻两线求交）。返回 (cells, pts, ok)。"""
    if len(a) < 2 or len(b) < 2:
        return [], [], False
    a = _order_by(a, b)
    b = _order_by(b, a)
    pts = [[intersect(ai, bj) for bj in b] for ai in a]
    if any(p is None for row in pts for p in row):
        return [], pts, False
    # 交点必须排成网格：同一行/列的 x、y 单调，否则说明两族线次序乱了
    cells = []
    for i in range(len(a) - 1):
        for j in range(len(b) - 1):
            q = [pts[i][j], pts[i + 1][j], pts[i + 1][j + 1], pts[i][j + 1]]
            # 去掉退化四边形（面积太小 / 自交）
            ar = 0.0
            for t in range(4):
                x1, y1 = q[t]
                x2, y2 = q[(t + 1) % 4]
                ar += x1 * y2 - x2 * y1
            if abs(ar) < 40:
                continue
            cells.append({"i": i, "j": j, "quad": [[round(x, 1), round(y, 1)] for x, y in q]})
    return cells, pts, len(cells) > 0


def band_quad(L, k, box):
    """一根线 ±k/2 的带子（画出来所见即所扣）。"""
    a = math.radians(L["ang"])
    nx, ny = -math.sin(a), math.cos(a)
    p0, p1 = L["p0"], L["p1"]
    h = k / 2.0
    # 角点顺序必须与「按列行铺」的 band 一致：0 左上、1 右上、2 右下、3 左下。
    # 之前给的是 (p0+n, p1+n, p1-n, p0-n) = 左上、左下、右下、右上 —— 整体转了一格，
    # 前端 gmHitAt 按「上边中点→下边中点」找命中就永远够不着，拖线全落到"整体平移"。
    return [[round(p0[0] + nx * h, 1), round(p0[1] + ny * h, 1)],
            [round(p0[0] - nx * h, 1), round(p0[1] - ny * h, 1)],
            [round(p1[0] - nx * h, 1), round(p1[1] - ny * h, 1)],
            [round(p1[0] + nx * h, 1), round(p1[1] + ny * h, 1)]]


# ════════════════════════════════════════════════════════════════════════════
# v2.53 检测：投影剖面（Radon）直接数刀缝，不用 Hough 分簇、不假设等距/平行
#   1) 对每个候选角度，把残差投到法线方向 → 一维剖面（每道刀缝就是一个峰）
#   2) 峰数 = 这一族的线数；峰间距的离散度 = 间距一致性（不等距也能过，只要别乱）
#   3) 拿最好的两个角度当两族，夹角 55~125°
#   4) 每根线单独 ±3.5° 精修自己的角度（支持不平行）
# ════════════════════════════════════════════════════════════════════════════
def _proj_profile(img, ang, sign=1.0):
    """沿法线 −sin/cos 把**灰度**投成一维剖面（每道刀缝=一个峰/谷）。

    早先用「残差 + 小半径模糊」是错的：细格的 pitch 只有 10 px 上下，
    模糊半径≈pitch 时背景估计把刀缝本身也抹掉了，剖面几乎一条直线（实测只剩 90~110 的抖动）。
    直接投灰度：亮刀缝 → 峰，暗刀缝 → 谷（sign=-1 换成峰）。
    """
    a = np.asarray(img, np.float32)
    h, w = a.shape
    ys, xs = np.mgrid[0:h, 0:w]
    r = math.radians(ang)
    t = -xs.astype(np.float32) * math.sin(r) + ys.astype(np.float32) * math.cos(r)
    t0 = float(t.min())
    tb = np.round(t - t0).astype(np.int32)
    n = int(tb.max()) + 2
    s = np.bincount(tb.ravel(), weights=(a * sign).ravel(), minlength=n)[:n]
    c = np.bincount(tb.ravel(), minlength=n)[:n]
    return s / np.maximum(c, 1.0), t0


def _smooth(p, k=3):
    if p.size < k:
        return p
    ker = np.ones(k, np.float32) / k
    return np.convolve(p, ker, mode="same")


def _pitch(prof):
    """自相关定主周期（相邻刀缝的间距）。返回 0 = 没周期（不是格网）。"""
    if prof is None or prof.size < 24:
        return 0
    p = _smooth(prof.astype(np.float32), 3)
    p = p - float(p.mean())
    if float(np.abs(p).max()) < 1e-6:
        return 0
    f = np.correlate(p, p, mode="full")[p.size - 1:]
    f = f / max(float(f[0]), 1e-6)
    lo = max(6, int(p.size * 0.02))
    hi = max(lo + 6, p.size // 3)
    seg = f[lo:hi]
    if seg.size < 4:
        return 0
    mx = float(seg.max())
    if mx < 0.20:                      # 自相关太弱 = 这个方向没周期性
        return 0
    # 取「接近最强的那些 lag 里最小的一个」：
    # 不这么做的话，平滑剖面在 lag=5 附近恒有一个小峰，会被误当成 pitch（实测全是 P=5）
    cand = np.nonzero(seg >= 0.90 * mx)[0]
    if cand.size == 0:
        return 0
    return int(cand[0]) + lo


def _prof_peaks(prof, rel=0.40):
    """一维找峰：先用自相关定间距 → 再按 0.45×间距 合并 → 峰数就是线数。"""
    if prof is None or prof.size < 8:
        return np.array([], np.int32), 1.0
    P = _pitch(prof)
    if P < 4:
        return np.array([], np.int32), 1.0
    p = _smooth(prof, 3)
    lo, hi = float(p.min()), float(p.max())
    if hi - lo < 3.0:                     # 动态范围太小 = 这个方向没有格
        return np.array([], np.int32), 1.0
    p = (p - lo) / (hi - lo)              # 归一到 0..1，门槛才与绝对亮度无关
    # 局部极大，且高于 0.40×峰高
    a, b, c = p[:-2], p[1:-1], p[2:]
    cand = np.nonzero((b >= a) & (b >= c) & (b > rel))[0] + 1
    if cand.size == 0:
        return np.array([], np.int32), 1.0
    # 按强度优先、再按最小间距（自适应）去重
    order = cand[np.argsort(-p[cand])]
    mind = max(3.0, 0.45 * P)
    keep = []
    for k in order:
        if all(abs(int(k) - int(q)) >= mind for q in keep):
            keep.append(int(k))
    if len(keep) < 2:
        return np.array(sorted(keep), np.int32), 1.0
    keep = np.array(sorted(keep), np.int32)
    gaps = np.diff(keep).astype(np.float32)
    med = float(np.median(gaps))
    if med < 3:
        return np.array([], np.int32), 1.0
    # 峰太密说明抓到的是漆膜块内部噪声，不是刀缝
    if med < 0.62 * P:
        return np.array([], np.int32), 1.0
    # 真的太密（<0.55×中位间距）的当重复峰合并
    ok = [keep[0]]
    for k in keep[1:]:
        if k - ok[-1] >= max(3.0, 0.70 * med):
            ok.append(k)
        elif p[k] > p[ok[-1]]:
            ok[-1] = k
    keep = np.array(ok, np.int32)
    if len(keep) < 2:
        return keep, 1.0
    gaps = np.diff(keep).astype(np.float32)
    reg = float(np.std(gaps) / max(1e-6, float(np.mean(gaps))))
    return keep, reg


def _family_score(peaks, reg, prof):
    if len(peaks) < 3 or reg > 0.55:
        return -1.0
    strength = float(np.mean(prof[peaks]) / max(1e-6, float(prof.max())))
    return len(peaks) * (0.6 + 0.4 * strength) / (1.0 + 1.2 * reg)


def _best_angles(res):
    """扫 0..179°：每个角度算剖面峰数；取最好的两族（夹角 55~125°）。"""
    profs, scores = {}, {}
    for ang in range(0, 180, 2):
        for sign in (1.0, -1.0):
            prof, t0 = _proj_profile(res, ang, sign)
            pk, reg = _prof_peaks(prof)
            key = (ang, sign)
            profs[key] = (prof, t0, pk, reg)
            scores[key] = _family_score(pk, reg, prof)
    order = sorted(scores.items(), key=lambda kv: -kv[1])
    if not order or order[0][1] <= 0:
        return None
    a1 = order[0][0]
    a2, best2 = None, 0.0
    for (ang, sg), sc in order[1:]:
        d = abs(((ang - a1[0] + 90) % 180) - 90)
        if 55 <= d <= 125 and sc > best2:
            a2, best2 = (ang, sg), sc
            a2 = (ang, sg)
    if a2 is None or best2 <= 0:
        return None
    return a1, a2, profs


def _lines_from_profile(res, gray, fam, ang, t0, peaks, box):
    """一维峰 → 线（钉在视图坐标），每根线再单独精修角度。"""
    x0, y0, x1, y1 = box
    out = []
    rmax = max(1e-6, float(res.max()))
    ln = min(x1 - x0, y1 - y0)
    for k in peaks:
        t = t0 + float(k)
        # 法线 t 上的点：取该 t 与框中心的交点
        mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        # t 是「裁剪图」法线坐标，d 必须也在同一坐标系里算（早先用全局中心，        # 差了一个裁剪原点的投影，线全部被推到框外 → 一根都出不来）
        d = (-(mx - x0) * math.sin(math.radians(ang))
             + (my - y0) * math.cos(math.radians(ang)))
        off = t - d
        px = mx - math.sin(math.radians(ang)) * off
        py = my + math.cos(math.radians(ang)) * off
        bestA, bestS = ang, -1.0
        for da in np.arange(-3.5, 3.51, 0.5):
            a = ang + da
            c_, m_ = _line_score(res, 0.25 * rmax, a, px, py, ln)
            sc = c_ * (m_ / rmax)
            if sc > bestS:
                bestS, bestA = sc, a
        a = bestA
        rho = -px * math.sin(math.radians(a)) + py * math.cos(math.radians(a))
        seg = _clip(a, rho, box)
        if not seg:
            continue
        out.append({"ang": round(a, 2), "rho": round(rho, 2), "p0": seg[0], "p1": seg[1],
                    "score": round(bestS * 100, 1)})
    out.sort(key=lambda L: -(-L["p0"][0] * math.sin(math.radians(L["ang"]))
                             + L["p0"][1] * math.cos(math.radians(L["ang"]))))
    return out


def detect_lines_proj(gray, box, max_lines=26):
    """v2.53：投影剖面直接数刀缝。返回结构与 detect_lines 相同。"""
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(gray.shape[1], x1), min(gray.shape[0], y1)
    ins = 5
    if x1 - x0 < 45 or y1 - y0 < 45:
        return {"ok": False, "why": "选区太小", "a": [], "b": []}
    crop = gray[y0 + ins:y1 - ins, x0 + ins:x1 - ins]
    if crop.size == 0:
        return {"ok": False, "why": "选区太小", "a": [], "b": []}
    view = (x0 + ins, y0 + ins, x1 - ins, y1 - ins)

    got = _best_angles(crop)
    if not got:
        return {"ok": False, "why": "投影里没找到两族刀缝", "a": [], "b": []}
    a1, a2, profs = got
    p1, t1, k1, r1 = profs[a1]
    p2, t2, k2, r2 = profs[a2]
    bright = a1[1] > 0
    res = np.asarray(crop, np.float32)
    f1 = (a1[0], p1, t1, k1)
    f2 = (a2[0], p2, t2, k2)
    la = _lines_from_profile(res, crop, f1, a1[0], t1, k1, view)
    lb = _lines_from_profile(res, crop, f2, a2[0], t2, k2, view)
    la, lb = la[:max_lines], lb[:max_lines]
    if len(la) < 3 or len(lb) < 3:
        return {"ok": False, "why": "线太少（%d / %d）" % (len(la), len(lb)),
                "a": la, "b": lb, "polarity": "bright" if bright else "dark"}
    return {"ok": True, "a": la, "b": lb,
            "polarity": ("bright" if bright else "dark"), "why": ""}


def detect_lines2(gray, box, max_lines=26, span=45.0):
    """v2.53 定稿思路：**先定一个倾角**（大角度也管），再把两族当作「正」的方向各投一次剖面。

    不做 180° 全扫（全扫必然挑到斜向的伪周期）；倾角由 gridfit.estimate_angle 给，
    只在这个角 ±8° 内微调。每族：自相关定 pitch → 一维找峰 → 峰就是刀缝。
    """
    import gridfit as _gf

    x0, y0, x1, y1 = [int(round(v)) for v in box]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(gray.shape[1], x1), min(gray.shape[0], y1)
    ins = 4
    if x1 - x0 < 45 or y1 - y0 < 45:
        return {"ok": False, "why": "选区太小", "a": [], "b": []}
    crop = gray[y0 + ins:y1 - ins, x0 + ins:x1 - ins]
    view = (x0 + ins, y0 + ins, x1 - ins, y1 - ins)

    try:
        a0 = float(_gf.estimate_angle(crop, span=span))
    except Exception:
        a0 = 0.0

    def best_family(ang0):
        best = None
        for da in (0.0, -1.0, 1.0, -2.0, 2.0, -4.0, 4.0, -8.0, 8.0):
            for sign in (1.0, -1.0):
                ang = ang0 + da
                prof, t0 = _proj_profile(crop, ang, sign)
                P = _pitch(prof)
                if P < 4:
                    continue
                pk, reg = _prof_peaks(prof)
                if len(pk) < 3 or reg > 0.45:
                    continue
                ext = (crop.shape[1] * abs(math.sin(math.radians(ang)))
                       + crop.shape[0] * abs(math.cos(math.radians(ang))))
                n_exp = ext / P
                if not (0.6 <= len(pk) / max(n_exp, 1e-6) <= 1.35):
                    continue                    # 峰数与「长度/间距」对不上 → 假周期
                p = _smooth(prof.astype(np.float32), 3)
                p0 = p - p.mean()
                f = np.correlate(p0, p0, mode="full")[p0.size - 1:]
                f = f / max(float(f[0]), 1e-6)
                R = float(f[P]) if 0 < P < f.size else 0.0
                sc = R * (1.0 - reg)
                if best is None or sc > best[0]:
                    best = (sc, ang, sign, P, pk, reg, R)
        return best

    fa = best_family(a0)
    fb = best_family(a0 + 90.0)
    if fa is None or fb is None:
        return {"ok": False, "why": "两族里有一族没认出来（倾角 %.1f°）" % a0, "a": [], "b": []}

    res = np.asarray(crop, np.float32)
    la = _lines_from_profile(res, crop, None, fa[1], _proj_profile(crop, fa[1], fa[2])[1], fa[4], view)
    lb = _lines_from_profile(res, crop, None, fb[1], _proj_profile(crop, fb[1], fb[2])[1], fb[4], view)
    la, lb = la[:max_lines], lb[:max_lines]
    if len(la) < 3 or len(lb) < 3:
        return {"ok": False, "why": "线太少（%d / %d），倾角 %.1f°" % (len(la), len(lb), a0),
                "a": la, "b": lb}
    return {"ok": True, "a": la, "b": lb, "angle": a0,
            "polarity": ("bright" if fa[2] > 0 else "dark"), "why": ""}


def detect_lines3(gray, box, max_lines=26):
    """v2.54：全角度扫，但每个角度都必须通过「峰数 ≈ 长度/间距」的一致性校验，
    再从合格的角度里挑一对近正交的当两族。"""
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(gray.shape[1], x1), min(gray.shape[0], y1)
    ins = 4
    if x1 - x0 < 45 or y1 - y0 < 45:
        return {"ok": False, "why": "选区太小", "a": [], "b": []}
    crop = gray[y0 + ins:y1 - ins, x0 + ins:x1 - ins]
    view = (x0 + ins, y0 + ins, x1 - ins, y1 - ins)
    ch, cw = crop.shape

    cand = []
    for ang0 in range(0, 180, 3):
        for da in (-1.0, 1.0):
            for sign in (1.0, -1.0):
                ang = ang0 + da
                prof, t0 = _proj_profile(crop, ang, sign)
                P = _pitch(prof)
                if P < 4:
                    continue
                pk, reg = _prof_peaks(prof)
                if len(pk) < 3 or reg > 0.40:
                    continue
                ext = cw * abs(math.sin(math.radians(ang))) + ch * abs(math.cos(math.radians(ang)))
                n_exp = ext / P
                if not (0.65 <= len(pk) / max(n_exp, 1e-6) <= 1.30):
                    continue
                p = _smooth(prof.astype(np.float32), 3)
                p0 = p - p.mean()
                cv = float(p0.std() / max(1e-6, abs(p.mean())))
                f = np.correlate(p0, p0, mode="full")[p0.size - 1:]
                f = f / max(float(f[0]), 1e-6)
                R = float(f[P]) if 0 < P < f.size else 0.0
                # 主判据是「调制深度 CV」：真方向上刀缝是细而深的一串峰，
                # 斜向投影会把缝抹平（CV 掉一半），之前只按 R 打分会挑到斜向的伪周期。
                cand.append((cv * (0.5 + 0.5 * R) * (1.0 - reg), ang, sign, P, pk, t0, reg))
    if not cand:
        return {"ok": False, "why": "没有哪个方向能数出稳定的刀缝", "a": [], "b": []}
    cand.sort(key=lambda q: -q[0])
    top = cand[0]
    second = None
    for c in cand[1:]:
        d = abs(((c[1] - top[1] + 90) % 180) - 90)
        if 60 <= d <= 120:
            second = c
            break
    if second is None:
        return {"ok": False, "why": "只认出一族方向", "a": [], "b": []}

    res = np.asarray(crop, np.float32)
    la = _lines_from_profile(res, crop, None, top[1], top[5], top[4], view)[:max_lines]
    lb = _lines_from_profile(res, crop, None, second[1], second[5], second[4], view)[:max_lines]
    if len(la) < 3 or len(lb) < 3:
        return {"ok": False, "why": "线太少（%d / %d）" % (len(la), len(lb)), "a": la, "b": lb}
    return {"ok": True, "a": la, "b": lb, "angle": round(top[1], 1),
            "polarity": ("bright" if top[2] > 0 else "dark"), "why": ""}


# ── 谐波问题的正解：间距候选里挑「峰最齐、调制最深、跟长度对得上」的那个 ──────────
def _pitch_cands(prof, max_c=6):
    """列出自相关的所有局部极大（候选基础间距），按强度排序返回 lag 列表。

    一道刀缝有两条亮边，自相关在 P/2 处也会出一个峰；只取「最强峰」就会把线数算成两倍。
    """
    if prof is None or prof.size < 24:
        return []
    p = _smooth(prof.astype(np.float32), 3)
    p = p - float(p.mean())
    if float(np.abs(p).max()) < 1e-6:
        return []
    f = np.correlate(p, p, mode="full")[p.size - 1:]
    f = f / max(float(f[0]), 1e-6)
    lo = max(5, int(p.size * 0.02))
    hi = max(lo + 6, p.size // 2)
    out = []
    for i in range(lo + 1, min(hi, f.size - 1)):
        if f[i] >= f[i - 1] and f[i] >= f[i + 1] and f[i] > 0.15:
            out.append((float(f[i]), int(i)))
    out.sort(reverse=True)
    return [k for _, k in out[:max_c]]


def _peaks_at(prof, P, rel=0.35):
    """按指定间距 P 找峰。返回 (peaks, reg, unif, mod)。

    unif = 最矮峰 / 峰高中位 —— 子谐波会让「隔一个」的峰偏矮，unif 掉下来，这是分辨 P 和 P/2 的关键；
    mod  = 峰的调制深度（峰高相对两峰之间的谷底）。
    """
    if prof is None or prof.size < 8 or P < 4:
        return np.array([], np.int32), 1.0, 0.0, 0.0
    p = _smooth(prof, 3)
    lo, hi = float(p.min()), float(p.max())
    if hi - lo < 3.0:
        return np.array([], np.int32), 1.0, 0.0, 0.0
    p = (p - lo) / (hi - lo)
    a, b, c = p[:-2], p[1:-1], p[2:]
    cand = np.nonzero((b >= a) & (b >= c) & (b > rel))[0] + 1
    if cand.size == 0:
        return np.array([], np.int32), 1.0, 0.0, 0.0
    order = cand[np.argsort(-p[cand])]
    mind = max(3.0, 0.60 * P)
    keep = []
    for k in order:
        if all(abs(int(k) - int(q)) >= mind for q in keep):
            keep.append(int(k))
    keep = np.array(sorted(keep), np.int32)
    if keep.size < 3:
        return keep, 1.0, 0.0, 0.0
    h = p[keep]
    unif = float(h.min() / max(float(np.median(h)), 1e-6))
    gaps = np.diff(keep).astype(np.float32)
    reg = float(np.std(gaps) / max(1e-6, float(np.mean(gaps))))
    # 谷底：相邻两峰中间的位置
    mids = [int((keep[i] + keep[i + 1]) / 2) for i in range(len(keep) - 1)]
    base = float(np.median(p[mids])) if mids else 0.0
    mod = float(min(1.0, (float(np.median(h)) - base) / max(1e-6, 1.0 - base)))
    return keep, reg, unif, mod


def detect_lines4(gray, box, max_lines=26):
    """v2.55：间距走「候选择优」，解决一道缝被算成两道（线数翻倍）的问题。"""
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(gray.shape[1], x1), min(gray.shape[0], y1)
    ins = 4
    if x1 - x0 < 45 or y1 - y0 < 45:
        return {"ok": False, "why": "选区太小", "a": [], "b": []}
    crop = gray[y0 + ins:y1 - ins, x0 + ins:x1 - ins]
    view = (x0 + ins, y0 + ins, x1 - ins, y1 - ins)
    ch, cw = crop.shape

    cand = []
    for ang0 in range(0, 180, 3):
        for da in (-1.0, 1.0):
            ang = ang0 + da
            for sign in (1.0, -1.0):
                prof, t0 = _proj_profile(crop, ang, sign)
                if prof is None:
                    continue
                ext = cw * abs(math.sin(math.radians(ang))) + ch * abs(math.cos(math.radians(ang)))
                for P in _pitch_cands(prof):
                    pk, reg, unif, mod = _peaks_at(prof, P)
                    if len(pk) < 3 or reg > 0.40 or mod < 0.15:
                        continue
                    n_exp = ext / P
                    if not (0.55 <= len(pk) / max(n_exp, 1e-6) <= 1.45):
                        continue
                    cons = math.exp(-abs(len(pk) - n_exp) / 2.0)
                    sc = (0.45 + 0.55 * unif) * mod * cons * (1.0 - reg)
                    cand.append((sc, ang, sign, P, pk, t0, reg, unif, mod))
    if not cand:
        return {"ok": False, "why": "没有哪个方向能数出稳定的刀缝", "a": [], "b": []}
    cand.sort(key=lambda q: -q[0])
    top = cand[0]
    second = None
    for c in cand[1:]:
        d = abs(((c[1] - top[1] + 90) % 180) - 90)
        if 60 <= d <= 120:
            second = c
            break
    if second is None:
        return {"ok": False, "why": "只认出一族方向", "a": [], "b": []}

    res = np.asarray(crop, np.float32)
    la = _lines_from_profile(res, crop, None, top[1], top[5], top[4], view)[:max_lines]
    lb = _lines_from_profile(res, crop, None, second[1], second[5], second[4], view)[:max_lines]
    if len(la) < 3 or len(lb) < 3:
        return {"ok": False, "why": "线太少（%d / %d）" % (len(la), len(lb)), "a": la, "b": lb}
    return {"ok": True, "a": la, "b": lb, "angle": round(top[1], 1),
            "pitch": float(top[3]), "unif": round(top[7], 2),
            "polarity": ("bright" if top[2] > 0 else "dark"), "why": ""}


def _merge_doublets(p, cand, dip=0.55):
    """把「同一条缝的两条亮边」并成一个峰。

    判据是两峰之间的**谷有多深**：谷值 ≥ 0.55×较矮那个峰 → 中间没真正掉下去，
    说明这是同一道缝的两个边（不是两道缝）。这比按间距去重可靠，
    因为缝宽和缝间距是两回事，间距阈值永远调不准。
    """
    out = []
    for k in cand:
        if out:
            j = out[-1]
            v = float(p[j:k + 1].min())
            h = min(float(p[j]), float(p[k]))
            if v >= dip * max(h, 1e-6):
                if p[k] > p[j]:
                    out[-1] = int(k)
                continue
        out.append(int(k))
    return np.array(out, np.int32)


def _peaks_dip(prof, rel=0.35, dip=0.55):
    """找峰 + 合并双边缘。返回 (peaks, reg, unif, P)。"""
    if prof is None or prof.size < 16:
        return np.array([], np.int32), 1.0, 0.0, 0
    p = _smooth(prof, 3)
    lo, hi = float(p.min()), float(p.max())
    if hi - lo < 3.0:
        return np.array([], np.int32), 1.0, 0.0, 0
    p = (p - lo) / (hi - lo)
    a, b, c = p[:-2], p[1:-1], p[2:]
    cand = np.nonzero((b >= a) & (b >= c) & (b > rel))[0] + 1
    if cand.size < 3:
        return np.array([], np.int32), 1.0, 0.0, 0
    pk = _merge_doublets(p, cand, dip)
    if pk.size < 3:
        return pk, 1.0, 0.0, 0
    gaps = np.diff(pk).astype(np.float32)
    P = float(np.median(gaps))
    reg = float(np.std(gaps) / max(1e-6, float(np.mean(gaps))))
    h = p[pk]
    unif = float(h.min() / max(float(np.median(h)), 1e-6))
    return pk, reg, unif, P


def detect_lines5(gray, box, max_lines=26):
    """v2.56：双边缘合并（谷深判据）→ 峰数即线数；再挑一对近正交的方向。"""
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(gray.shape[1], x1), min(gray.shape[0], y1)
    ins = 4
    if x1 - x0 < 45 or y1 - y0 < 45:
        return {"ok": False, "why": "选区太小", "a": [], "b": []}
    crop = gray[y0 + ins:y1 - ins, x0 + ins:x1 - ins]
    view = (x0 + ins, y0 + ins, x1 - ins, y1 - ins)
    ch, cw = crop.shape

    cand = []
    for ang0 in range(0, 180, 2):
        for da in (-0.5, 0.5):
            ang = ang0 + da
            for sign in (1.0, -1.0):
                prof, t0 = _proj_profile(crop, ang, sign)
                if prof is None:
                    continue
                pk, reg, unif, P = _peaks_dip(prof)
                if pk.size < 3 or reg > 0.35 or unif < 0.45:
                    continue
                ext = cw * abs(math.sin(math.radians(ang))) + ch * abs(math.cos(math.radians(ang)))
                n_exp = ext / max(P, 1e-6)
                if not (0.70 <= pk.size / n_exp <= 1.30):
                    continue
                p = _smooth(prof.astype(np.float32), 3)
                p = (p - p.min()) / max(1e-6, float(p.max() - p.min()))
                mids = [int((pk[i] + pk[i + 1]) / 2) for i in range(len(pk) - 1)]
                base = float(np.median(p[mids])) if mids else 0.0
                mod = float(min(1.0, (float(np.median(p[pk])) - base) / max(1e-6, 1.0 - base)))
                sc = mod * unif * (1.0 - reg) * math.exp(-abs(pk.size - n_exp) / 3.0)
                cand.append((sc, ang, sign, P, pk, t0, reg, unif, mod))
    if not cand:
        return {"ok": False, "why": "没有哪个方向能数出稳定的刀缝", "a": [], "b": []}
    cand.sort(key=lambda q: -q[0])
    top = cand[0]
    second = None
    for c in cand[1:]:
        d = abs(((c[1] - top[1] + 90) % 180) - 90)
        if 60 <= d <= 120:
            second = c
            break
    if second is None:
        return {"ok": False, "why": "只认出一族方向", "a": [], "b": []}

    res = np.asarray(crop, np.float32)
    la = _lines_from_profile(res, crop, None, top[1], top[5], top[4], view)[:max_lines]
    lb = _lines_from_profile(res, crop, None, second[1], second[5], second[4], view)[:max_lines]
    if len(la) < 3 or len(lb) < 3:
        return {"ok": False, "why": "线太少（%d / %d）" % (len(la), len(lb)), "a": la, "b": lb}
    return {"ok": True, "a": la, "b": lb, "angle": round(top[1], 1), "pitch": float(top[3]),
            "unif": round(top[7], 2), "mod": round(top[8], 2),
            "polarity": ("bright" if top[2] > 0 else "dark"), "why": ""}


def _fundamental(prof):
    """定「一个格子多宽」并顺带定格数：在自相关峰 P0 的 ½×/1×/2× 里挑，
    判据是「按这个间距等分采样时，落在缝上 vs 落在两缝中间」的对比度最大。

    这样数格子不依赖找峰是否干净：数错了只会是因为周期本身错，而周期有三档互相比。
    """
    p = _smooth(prof.astype(np.float32), 3)
    p = (p - p.min()) / max(1e-6, float(p.max() - p.min()))
    n_prof = p.size
    P0 = _pitch(prof)
    if P0 < 4:
        return None
    sd = float(p.std()) or 1e-6
    best = None
    for mult in (0.5, 1.0, 2.0):
        P = P0 * mult
        if P < 5 or P > n_prof / 2.5:
            continue
        n = int(round((n_prof - 2) / P))
        if n < 2 or n > 30:
            continue
        ks = np.arange(n + 1) * P
        ks = ks[ks <= n_prof - 1.5]
        if ks.size < 3:
            continue
        on = float(np.mean(p[np.round(ks).astype(int)]))
        offs = np.round(ks + P / 2.0).astype(int)
        offs = offs[offs <= n_prof - 1]
        off = float(np.mean(p[offs])) if offs.size else 0.0
        cont = (on - off) / sd
        # 细微优势留给「更细的间距」会数多；所以同分时偏向大 P（格子更宽）
        sc = cont * (1.0 + 0.02 * P) if cont > 0 else -1
        if best is None or sc > best[0]:
            best = (sc, P, n, cont)
    if best is None or best[3] <= 0.35:
        return None
    return {"P": best[1], "n": best[2], "cont": best[3]}


def _positions_from_fund(prof, fund):
    """按定好的间距，在每格边界附近找最近的峰当真实缝位（找不到就用等分位）。"""
    p = _smooth(prof.astype(np.float32), 3)
    P, n = fund["P"], fund["n"]
    out = []
    w = max(1, int(0.30 * P))
    for k in range(n + 1):
        c = int(round(k * P))
        lo, hi = max(0, c - w), min(p.size - 1, c + w)
        if hi <= lo:
            out.append(c)
            continue
        out.append(int(lo + np.argmax(p[lo:hi + 1])))
    return np.array(out, np.int32)


def detect_lines6(gray, box, max_lines=26):
    """v2.57：**格数由周期定**（½×/1×/2× 对比度择优），缝位在等分位附近吸附。"""
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(gray.shape[1], x1), min(gray.shape[0], y1)
    ins = 4
    if x1 - x0 < 45 or y1 - y0 < 45:
        return {"ok": False, "why": "选区太小", "a": [], "b": []}
    crop = gray[y0 + ins:y1 - ins, x0 + ins:x1 - ins]
    view = (x0 + ins, y0 + ins, x1 - ins, y1 - ins)

    cand = []
    for ang0 in range(0, 180, 2):
        for da in (-0.5, 0.5):
            ang = ang0 + da
            for sign in (1.0, -1.0):
                prof, t0 = _proj_profile(crop, ang, sign)
                if prof is None or prof.size < 24:
                    continue
                fund = _fundamental(prof)
                if not fund or fund["n"] < 3 or fund["n"] > max_lines - 1:
                    continue
                cand.append((fund["cont"] * (1.0 + 0.02 * fund["P"]), ang, sign,
                             fund["P"], fund["n"], t0, fund))
    if not cand:
        return {"ok": False, "why": "没有哪个方向的周期性站得住", "a": [], "b": []}
    cand.sort(key=lambda q: -q[0])
    top = cand[0]
    second = None
    for c in cand[1:]:
        d = abs(((c[1] - top[1] + 90) % 180) - 90)
        if 55 <= d <= 125:                 # 只要求近正交；横竖格数不必相同（6×11 这种是正常的）
            second = c
            break
    if second is None:
        return {"ok": False, "why": "只认出一族方向（另一族格数差太多）", "a": [], "b": []}

    res = np.asarray(crop, np.float32)
    out = []
    for c in (top, second):
        ang, sign, t0, fund = c[1], c[2], c[5], c[6]
        prof, _ = _proj_profile(crop, ang, sign)
        pos = _positions_from_fund(prof, fund)
        out.append(_lines_from_profile(res, crop, None, ang, t0, pos, view)[:max_lines])
    la, lb = out
    if len(la) < 3 or len(lb) < 3:
        return {"ok": False, "why": "线太少（%d / %d）" % (len(la), len(lb)), "a": la, "b": lb}
    return {"ok": True, "a": la, "b": lb, "angle": round(top[1], 1), "pitch": float(top[3]),
            "cells": [top[4], second[4]], "cont": round(top[6]["cont"], 2),
            "polarity": ("bright" if top[2] > 0 else "dark"), "why": ""}
