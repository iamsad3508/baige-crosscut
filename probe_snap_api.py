import json, urllib.request, uuid

B = "http://127.0.0.1:8799"
path = "/root/.hermes/cache/scratch/ref_erichsen.png"
b = "----b" + uuid.uuid4().hex
body = ("--%s\r\nContent-Disposition: form-data; name=\"file\"; filename=\"x.png\"\r\nContent-Type: image/png\r\n\r\n" % b).encode()
body += open(path, "rb").read() + b"\r\n" + ("--%s--\r\n" % b).encode()
up = json.load(urllib.request.urlopen(urllib.request.Request(
    B + "/api/upload", data=body, headers={"Content-Type": "multipart/form-data; boundary=" + b}), timeout=300))
sid = up["id"]


def post(p, d):
    r = urllib.request.Request(B + p, data=json.dumps(d).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=300))


for box in ([845, 370, 1230, 755], [855, 380, 1220, 745]):
    roi = {"box": box, "ang": 0}
    a = post("/api/autosample", {"id": sid, "res": "full", "rois": [roi]})
    base = {"id": sid, "res": "full", "rois": [roi], "samples": a["samples"], "tol": a["auto_tol"],
            "purity": 0.5, "line_px": 12, "mode": "width", "model": True, "cols": 10, "rows": 10,
            "angle": 0, "k_band": 12, "roi_native": True, "trace": True, "roles_auto": True}
    for tag, extra in (("plain", {}), ("snap", {"snap": True}),
                       ("angle", {"angle": -1.4}), ("ang+snap", {"angle": -1.4, "snap": True})):
        g = post("/api/grid", dict(base, **extra))
        tr = g.get("trace") or {}
        print(box, tag, "pct", round((g.get("total") or {}).get("pct") or -1, 2), g.get("grade"),
              "n_x/n_y", tr.get("n_x"), tr.get("n_y"), "ok", tr.get("ok_x"), tr.get("ok_y"))
