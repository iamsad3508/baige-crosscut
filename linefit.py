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
    return [[round(p0[0] + nx * h, 1), round(p0[1] + ny * h, 1)],
            [round(p1[0] + nx * h, 1), round(p1[1] + ny * h, 1)],
            [round(p1[0] - nx * h, 1), round(p1[1] - ny * h, 1)],
            [round(p0[0] - nx * h, 1), round(p0[1] - ny * h, 1)]]
