# Third-party notices

Baige itself is licensed under AGPL-3.0 (`LICENSE`). It depends on the following
packages; none of them are modified or bundled, they are installed by `pip` at run time
from `requirements.txt`. Their licences apply to their own code only.

| Package | Licence | Used for |
| --- | --- | --- |
| [Pillow](https://python-pillow.org/) | HPND (MIT-style, permissive) | image decode / resize / masks |
| [numpy](https://numpy.org/) | BSD-3-Clause | array maths (Lab conversion, morphology, histograms) |
| [FastAPI](https://fastapi.tiangolo.com/) | MIT | HTTP API |
| [Starlette](https://www.starlette.io/) | BSD-3-Clause | ASGI toolkit underneath FastAPI |
| [Uvicorn](https://www.uvicorn.org/) | BSD-3-Clause | ASGI server |
| [python-multipart](https://github.com/Kludex/python-multipart) | Apache-2.0 | upload form parsing |
| [Pydantic](https://docs.pydantic.dev/) | MIT | request models |

Optional, **not** in `requirements.txt`:

| Package | Licence | Notes |
| --- | --- | --- |
| [rawpy](https://github.com/letmaik/rawpy) | MIT | only needed to read camera RAW/DNG. `import rawpy` is guarded: without it, DNG upload returns a clear 400 and JPEG/TIFF still work. |

`rawpy` wheels bundle **LibRaw**, which is dual-licensed **LGPL-2.1 / CDDL-1.0**.
Nothing is bundled here — the user installs `rawpy` from PyPI themselves, so no
LGPL relinking obligation applies to this repository. If you ever ship a frozen
executable that embeds `rawpy`, you become a distributor of LibRaw: the LGPL part
requires that the recipient can replace/relink LibRaw (dynamic linking plus the
LibRaw sources, or the CDDL-1.0 alternative). Keep that in mind before building a
single-file `.exe` with DNG support.

## Standards

ASTM D3359, ISO 2409 and GB/T 9286 are copyrighted documents. This project implements
the grading thresholds that are widely published in test-method summaries; it does not
reproduce the standard texts, and the licences above grant **no** rights to them.
