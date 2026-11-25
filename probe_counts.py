import numpy as np, gridfit as gf
from PIL import Image

im = np.asarray(Image.open('/root/.hermes/cache/scratch/ref_erichsen.png').convert('L'))
for box in [(845, 370, 1230, 755), (855, 380, 1220, 745)]:
    x0, y0, x1, y1 = box
    sub = np.asarray(im[y0:y1, x0:x1]).astype(np.uint8)
    ang = float(gf.estimate_angle(sub))
    rot = np.asarray(Image.fromarray(sub, "L").rotate(ang, resample=Image.BILINEAR, fillcolor=int(np.median(sub))))
    r = gf.fit(rot)
    h, w = rot.shape
    px, py = float(r.get("period_x") or 0), float(r.get("period_y") or 0)
    print(box, "ok", r.get("ok"), "period", round(px, 2), round(py, 2),
          "resid", round(float(r.get("resid_x") or -1), 2), round(float(r.get("resid_y") or -1), 2),
          "resid/p", round(float(r.get("resid_x") or 0) / max(px, 1e-6), 3),
          "counts", round(w / max(px, 1e-6), 2), round(h / max(py, 1e-6), 2))
    t = gf.trace_grid(im, ang, ((x0 + x1) / 2, (y0 + y1) / 2), box, round(w / px), round(h / py))
    print("   trace ok_x/ok_y", t.get("ok_x"), t.get("ok_y"), "n", t.get("n_x"), t.get("n_y"),
          "qual", t.get("qual_x"), t.get("qual_y"))
