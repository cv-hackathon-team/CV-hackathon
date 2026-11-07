"""DepthGuard 360 web app.  Run:  uvicorn ui.server:app --reload  (from repo root)"""
import base64
import csv
import io
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

try:
    from . import pipeline as P
except ImportError:
    import pipeline as P

HERE = Path(__file__).parent
RESULTS_CSV = HERE.parent / "outputs" / "results.csv"
app = FastAPI(title="DepthGuard 360")


# ---------- helpers ----------
def png64(img):
    return "data:image/png;base64," + base64.b64encode(cv2.imencode(".png", img)[1]).decode()


def colour(depth, lo, hi):
    n = np.clip((depth - lo) / (hi - lo + 1e-6), 0, 1)
    return cv2.applyColorMap((255 - n * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def read_rgb(data):
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_depth(data, name):
    name = name.lower()
    if name.endswith(".npy"):
        return np.load(io.BytesIO(data)).astype(np.float32)
    if name.endswith(".pfm"):
        f = io.BytesIO(data)
        f.readline()
        w, h = map(int, f.readline().split())
        s = float(f.readline())
        d = np.frombuffer(f.read(), "<f4" if s < 0 else ">f4").reshape(h, w)
        d = np.flipud(d).astype(np.float32)
        d[~np.isfinite(d)] = 0
        return d
    d = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED).astype(np.float32)
    return d[..., 0] if d.ndim == 3 else d


def crop8(*arrs):
    h, w = arrs[0].shape[:2]
    h, w = h - h % 8, w - w % 8
    return [a[:h, :w] for a in arrs]


# ---------- API ----------
@app.get("/api/status")
def status():
    """Shows if the real model is loaded (model: "real") or the stand-in is used."""
    return P.STATUS


@app.get("/api/methods")
def methods():
    return [{"name": n, "group": g} for n, (_, g) in P.METHODS.items()]


async def load_inputs(sample, rgb, depth, methods):
    if sample or rgb is None or depth is None:
        img, gt = P.sample_scene()
    else:
        img, gt = read_rgb(await rgb.read()), read_depth(await depth.read(), depth.filename)
        if img.shape[:2] != gt.shape:
            img = cv2.resize(img, gt.shape[::-1])
        if max(gt.shape) > 960:  # keep the demo snappy
            f = 960 / max(gt.shape)
            gt = cv2.resize(gt, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST)
            img = cv2.resize(img, gt.shape[::-1])
    img, gt = crop8(img, gt)
    names = [m for m in methods.split(",") if m in P.METHODS]
    if "DepthGuard 360" not in names:
        names.append("DepthGuard 360")
    return img, gt, names


def scores(r, gt):
    return {"ms": round(r["ms"], 1), "MB": round(r["mb"], 1),
            **{k: round(x, 4) for k, x in P.metrics(r["depth"], gt).items()}}


@app.post("/api/scales")
async def scales(
    methods: str = Form("Bicubic,Joint bilateral,Guided filter,DepthGuard 360"),
    sample: bool = Form(False),
    rgb: UploadFile | None = File(None),
    depth: UploadFile | None = File(None),
):
    """Same input at 2x, 4x and 8x: how each method holds up as the factor grows."""
    img, gt, names = await load_inputs(sample, rgb, depth, methods)
    out = {}
    for s in (2, 4, 8):
        _, _, res = P.run_all(gt, img, s, names)
        out[s] = {n: scores(r, gt) for n, r in res.items()}
    return {"methods": names, "scales": out}


@app.post("/api/reconstruct")
async def reconstruct(
    scale: int = Form(4),
    methods: str = Form("Bicubic,Joint bilateral,Guided filter,DepthGuard 360"),
    sample: bool = Form(False),
    rgb: UploadFile | None = File(None),
    depth: UploadFile | None = File(None),
):
    img, gt, names = await load_inputs(sample, rgb, depth, methods)
    lr, mask, res = P.run_all(gt, img, scale, names)
    v = gt[gt > 0]
    lo, hi = float(np.percentile(v, 1)), float(np.percentile(v, 99))
    _, conf, kept, rejected = P.analyse(lr, img, scale, mask)

    edges = np.full((*gt.shape, 3), 245, np.uint8)
    edges[rejected] = (90, 90, 235)    # red-ish (BGR): texture edges we ignored
    edges[kept] = (90, 180, 40)        # green: real depth edges we kept
    lr_vis = colour(lr, lo, hi)
    lr_vis[mask] = 30

    return JSONResponse({
        "size": list(gt.shape),
        "scale": scale,
        "inputs": {
            "rgb": png64(cv2.cvtColor(img, cv2.COLOR_RGB2BGR)),
            "lowres": png64(cv2.resize(lr_vis, gt.shape[::-1], interpolation=cv2.INTER_NEAREST)),
            "truth": png64(colour(gt, lo, hi)),
        },
        "maps": {
            "confidence": png64(cv2.applyColorMap((conf * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)),
            "edges": png64(edges),
        },
        "results": [
            {"name": n, "group": P.METHODS[n][1], "image": png64(colour(r["depth"], lo, hi)), **scores(r, gt)}
            for n, r in res.items()
        ],
    })


@app.get("/api/results")
def results():
    """Serves the team's offline benchmark (outputs/results.csv) if it exists."""
    if not RESULTS_CSV.exists():
        return {"rows": []}
    with open(RESULTS_CSV, newline="") as f:
        return {"rows": list(csv.DictReader(f))}


@app.get("/")
def index():
    return FileResponse(HERE / "web" / "index.html")


app.mount("/", StaticFiles(directory=HERE / "web"), name="web")
# Trigger Vercel deployment
