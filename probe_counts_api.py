import json, urllib.request, uuid

B = "http://127.0.0.1:8799"
path = "/root/.hermes/cache/scratch/ref_erichsen.png"
boundary = "----b" + uuid.uuid4().hex
body = ("--%s\r\nContent-Disposition: form-data; name=\"file\"; filename=\"x.png\"\r\nContent-Type: image/png\r\n\r\n" % boundary).encode()
body += open(path, "rb").read() + b"\r\n"
body += ("--%s--\r\n" % boundary).encode()
up = json.load(urllib.request.urlopen(urllib.request.Request(
    B + "/api/upload", data=body,
    headers={"Content-Type": "multipart/form-data; boundary=" + boundary}), timeout=300))
sid = up["id"]
print("upload", sid, up["w"], up["h"], "scale", up.get("scale"))


def post(p, d):
    r = urllib.request.Request(B + p, data=json.dumps(d).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=300))


for box in ([845, 370, 1230, 755], [855, 380, 1220, 745]):
    roi = {"box": box, "ang": 0}
    a = post("/api/autosample", {"id": sid, "res": "full", "rois": [roi]})
    body = {"id": sid, "res": "full", "rois": [roi], "samples": a["samples"], "tol": a["auto_tol"],
            "purity": 0.5, "line_px": 12, "mode": "width", "model": True, "cols": 10, "rows": 9,
            "angle": 0, "k_band": 12, "roi_native": True, "trace": True,
            "auto_cols": True, "auto_rows": True, "roles_auto": True}
    for nat in (True, False):
        body["roi_native"] = nat
        g = post("/api/grid", body)
        print(box, "roi_native", nat, "counts_est", g.get("counts_est"), "->", g.get("cols"), "x", g.get("rows"),
              "trace", (g.get("trace") or {}).get("n_x"), (g.get("trace") or {}).get("n_y"))
