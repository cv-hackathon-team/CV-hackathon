"""
DepthGuard 360 – method registry used by the UI.

TEAMMATES: to add a method, write a function and decorate it:

    @register("My method", group="hybrid")
    def my_method(lr_depth, rgb, scale, hole_mask):
        return hr_depth            # float32, same H x W as rgb

It shows up in the UI automatically.

"DepthGuard 360" runs the real model (src/inference.py + checkpoints/best.pt),
loaded once when the server starts. If the checkpoint or torch is missing, it
falls back to the rule-based stand-in below. Check GET /api/status.
"""
import hashlib
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parent.parent      # the depthguard360 repo root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
STATUS = {"model": "stand-in", "device": None, "epoch": None, "note": "", "classical": {}}

METHODS = {}  # name -> (fn, group)


def register(name, group="classical"):
    def deco(fn):
        METHODS[name] = (fn, group)
        return fn
    return deco


# ---------- damage simulation (same idea as src/datasets/corruption.py) ----------
def make_damaged(gt, scale, noise=0.02, holes=0.04, seed=0):
    rng = np.random.default_rng(seed)
    h, w = gt.shape
    lr = cv2.resize(gt, (w // scale, h // scale), interpolation=cv2.INTER_NEAREST)
    rng_ = float(gt.max() - gt.min()) or 1.0
    lr = lr + rng.normal(0, noise * rng_, lr.shape).astype(np.float32)
    mask = np.zeros(lr.shape, np.uint8)
    for _ in range(int(holes * lr.size / 20) + 1):
        y, x = rng.integers(0, lr.shape[0]), rng.integers(0, lr.shape[1])
        cv2.circle(mask, (int(x), int(y)), int(rng.integers(1, 4)), 1, -1)
    lr[mask > 0] = 0
    return lr.astype(np.float32), mask > 0


def fill_holes(lr, mask):
    if not mask.any():
        return lr
    lo, hi = float(lr[~mask].min()), float(lr[~mask].max())
    n = ((lr - lo) / (hi - lo + 1e-6) * 65535).astype(np.uint16)
    n8 = (n / 257).astype(np.uint8)
    f = cv2.inpaint(n8, mask.astype(np.uint8), 3, cv2.INPAINT_NS).astype(np.float32) / 255
    out = lr.copy()
    out[mask] = f[mask] * (hi - lo) + lo
    return out


def up(lr, size, interp):
    return cv2.resize(lr, size, interpolation=interp)


# ---------- baselines ----------
@register("Nearest", "interpolation")
def nearest(lr, rgb, s, m):
    return up(fill_holes(lr, m), rgb.shape[1::-1], cv2.INTER_NEAREST)


@register("Bilinear", "interpolation")
def bilinear(lr, rgb, s, m):
    return up(fill_holes(lr, m), rgb.shape[1::-1], cv2.INTER_LINEAR)


@register("Bicubic", "interpolation")
def bicubic(lr, rgb, s, m):
    return up(fill_holes(lr, m), rgb.shape[1::-1], cv2.INTER_CUBIC)


@register("Joint bilateral", "classical")
def jbf(lr, rgb, s, m):
    d = cv2.medianBlur(bicubic(lr, rgb, s, m), 5)
    return cv2.ximgproc.jointBilateralFilter(rgb.astype(np.float32), d, 9, 25, 4 + s)


@register("Guided filter", "classical")
def guided(lr, rgb, s, m):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    return cv2.ximgproc.guidedFilter(g, cv2.medianBlur(bicubic(lr, rgb, s, m), 5), 2 + s, 1e-3)


# ---------- DepthGuard 360 (rule-based stand-in until the model is wired) ----------
def analyse(lr, rgb, s, m):
    """Returns the pieces the UI visualises: depth, confidence, kept/rejected edges."""
    clean = cv2.medianBlur(fill_holes(lr, m), 3)          # DEPTH RESCUE: holes + noise first
    size = rgb.shape[1::-1]
    near = up(clean, size, cv2.INTER_NEAREST)
    base = up(clean, size, cv2.INTER_CUBIC)
    # TEXTURE TRAP: keep RGB edges only where depth also jumps
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    rgb_edges = cv2.Canny(gray, 60, 140) > 0
    gx, gy = cv2.Sobel(base, cv2.CV_32F, 1, 0), cv2.Sobel(base, cv2.CV_32F, 0, 1)
    jump = np.hypot(gx, gy)
    depth_edge = cv2.dilate((jump > np.percentile(jump, 93)).astype(np.uint8), np.ones((s + 1,) * 2)) > 0
    kept, rejected = rgb_edges & depth_edge, rgb_edges & ~depth_edge
    # guide = RGB only at real edges, flat elsewhere -> no texture copying
    guide = cv2.GaussianBlur(gray, (0, 0), 3).astype(np.float32)
    guide[depth_edge] = gray[depth_edge]
    smooth = cv2.ximgproc.jointBilateralFilter(guide, base, 2 * s + 3, 20, 2 * s + 2)
    # OCCLUSION-X: in the boundary band, pick front OR back - never blend
    k = np.ones((2 * s + 1,) * 2, np.uint8)
    lo, hi = cv2.erode(near, k), cv2.dilate(near, k)
    pick = np.where(np.abs(smooth - lo) < np.abs(smooth - hi), lo, hi)
    out = np.where(depth_edge & (hi - lo > 0.1 * (hi.max() - lo.min()) / 10), pick, smooth)
    out = cv2.medianBlur(out.astype(np.float32), 3)
    # DEPTHGUARD: confidence = low where holes were, where methods disagree, near edges
    hole_up = up(m.astype(np.float32), size, cv2.INTER_NEAREST)
    dis = np.abs(out - base)
    conf = 1 - np.clip(0.5 * hole_up + dis / (np.percentile(dis, 99) + 1e-6) * 0.5, 0, 1)
    return out, conf, kept, rejected


# ---------- DepthGuard 360: the real model ----------
def _load_predictor():
    """Load checkpoints/best.pt once. Returns None (stand-in is used) if anything is missing."""
    ckpt = ROOT / "checkpoints" / "best.pt"
    if not ckpt.is_file():
        STATUS["note"] = f"{ckpt} not found"
        return None
    try:
        from src.inference import Predictor
        pred = Predictor(ckpt)  # repo's get_device(): cuda -> mps -> cpu
        STATUS.update(model="real", device=str(pred.runner.device), epoch=pred.runner.epoch, note="")
        return pred
    except Exception as e:  # torch missing, bad checkpoint, import error...
        STATUS["note"] = f"{type(e).__name__}: {e}"
        return None


PREDICTOR = _load_predictor()
print(f"[DepthGuard 360] model={STATUS['model']} device={STATUS['device']} {STATUS['note']}", flush=True)
_CACHE = {}  # one entry: last model output, so analyse() does not run the model twice


def _key(lr, rgb, s):
    return (hashlib.md5(lr.tobytes()).hexdigest(), rgb.shape, s)


def run_model(lr, rgb, s):
    """Real model output dict (depth, confidence, boundary) at rgb size, or None if unavailable."""
    if PREDICTOR is None:
        return None
    k = _key(lr, rgb, s)
    if k in _CACHE:
        return _CACHE[k]
    valid = np.isfinite(lr) & (lr > 0)
    if not valid.any():
        return None
    # The model was trained on metres. Bring other units (mm PNGs, disparity) into a metre-like
    # range, then scale back. Safe because the model and refinement are linear in depth scale.
    med = float(np.median(lr[valid]))
    f = 3.0 / med if not (0.3 <= med <= 20) else 1.0
    try:
        out = PREDICTOR.predict(rgb, np.where(valid, lr * f, 0).astype(np.float32), s)
    except Exception as e:
        print(f"[DepthGuard 360] model failed, using stand-in: {e}", flush=True)
        return None
    out = {"depth": out["depth"] / f, "confidence": out["confidence"], "boundary": out["boundary"]}
    _CACHE.clear()
    _CACHE[k] = out
    return out


_stand_in_analyse = analyse


def analyse(lr, rgb, s, m):
    """Depth, confidence, kept/rejected RGB edges. Uses the real model when it is loaded."""
    out = run_model(lr, rgb, s)
    if out is None:
        return _stand_in_analyse(lr, rgb, s, m)
    rgb_edges = cv2.Canny(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), 60, 140) > 0
    real = cv2.dilate((out["boundary"] >= 0.5).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    return out["depth"], np.clip(out["confidence"], 0, 1), rgb_edges & real, rgb_edges & ~real


@register("DepthGuard 360", "hybrid")
def depthguard(lr, rgb, s, m):
    return analyse(lr, rgb, s, m)[0]


# ---------- use the repo's own classical methods (src/classical) when they work ----------
def _wrap_real(name, fn, settings):
    """fn(depth_lr, rgb01, scale, *settings, valid) -> hr depth. Same call style as src/inference.py."""
    def method(lr, rgb, s, m):
        valid = np.isfinite(lr) & (lr > 0)
        out = fn(np.where(valid, lr, 0).astype(np.float32), rgb.astype(np.float32) / 255, s, *settings, valid)
        out = np.asarray(out, np.float32)
        return out if out.shape == rgb.shape[:2] else cv2.resize(out, rgb.shape[1::-1])
    return method


def _use_repo_classical():
    try:
        from src.utils import load_config
        cfg = load_config()["classical"]
    except Exception as e:
        STATUS["classical"]["error"] = f"{type(e).__name__}: {e}"
        return
    candidates = [  # (UI name, group, module, function, config section, config keys)
        ("Guided filter", "classical", "src.classical.guided_filter", "guided_upsample",
         "guided_filter", ("radius_per_scale", "epsilon")),
        ("Joint bilateral", "classical", "src.classical.joint_bilateral", "joint_bilateral_upsample",
         "joint_bilateral", ("sigma_spatial_per_scale", "sigma_range")),
    ]
    rng = np.random.default_rng(0)
    test_rgb = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
    test_lr = rng.uniform(1, 3, (8, 8)).astype(np.float32)
    test_lr[2:4, 2:4] = 0
    for ui_name, group, mod, fn_name, section, keys in candidates:
        try:
            fn = getattr(__import__(mod, fromlist=[fn_name]), fn_name)
            method = _wrap_real(ui_name, fn, [cfg[section][k] for k in keys])
            method(test_lr, test_rgb, 4, test_lr == 0)  # quick self-test on a tiny image
            register(ui_name, group)(method)
            STATUS["classical"][ui_name] = f"repo: {mod}.{fn_name}"
        except Exception as e:
            STATUS["classical"][ui_name] = f"UI copy ({type(e).__name__}: {e})"


_use_repo_classical()


# ---------- metrics ----------
def depth_edges(d, thr=0.05):
    """Pixels where depth jumps by more than thr (5%) to a neighbour = depth boundaries."""
    d = np.maximum(d, 1e-6)
    e = np.zeros(d.shape, bool)
    jx = np.abs(np.diff(d, axis=1)) / np.minimum(d[:, 1:], d[:, :-1]) > thr
    jy = np.abs(np.diff(d, axis=0)) / np.minimum(d[1:], d[:-1]) > thr
    e[:, 1:] |= jx; e[:, :-1] |= jx; e[1:] |= jy; e[:-1] |= jy
    return e


def boundary_f1(pred, gt, tol=2):
    """Boundary accuracy: predicted edges within tol px of a true edge (precision) and vice versa (recall)."""
    pe, ge = depth_edges(pred), depth_edges(gt) & (gt > 0)
    if not ge.any():
        return 1.0
    k = np.ones((2 * tol + 1,) * 2, np.uint8)
    near_g = cv2.dilate(ge.astype(np.uint8), k) > 0
    near_p = cv2.dilate(pe.astype(np.uint8), k) > 0
    prec = (pe & near_g).sum() / max(pe.sum(), 1)
    rec = (ge & near_p).sum() / ge.sum()
    return float(2 * prec * rec / (prec + rec + 1e-9))


def metrics(pred, gt):
    v = gt > 0
    e = pred[v] - gt[v]
    rng_ = float(gt[v].max() - gt[v].min()) or 1.0
    rmse = float(np.sqrt(np.mean(e ** 2)))
    gx, gy = cv2.Sobel(gt, cv2.CV_32F, 1, 0), cv2.Sobel(gt, cv2.CV_32F, 0, 1)
    j = np.hypot(gx, gy)
    band = (cv2.dilate((j > np.percentile(j, 95)).astype(np.uint8), np.ones((7, 7))) > 0) & v
    return {
        "RMSE": rmse,
        "MAE": float(np.mean(np.abs(e))),
        "PSNR": float(20 * np.log10(rng_ / (rmse + 1e-9))),
        "SSIM": float(structural_similarity(gt, pred, data_range=rng_)),
        "Edge RMSE": float(np.sqrt(np.mean((pred[band] - gt[band]) ** 2))) if band.any() else 0.0,
        "Boundary F1": boundary_f1(pred, gt),
    }


class PeakMemory:
    """Peak extra memory (MB) while a method runs: max of numpy allocations and process RAM growth."""

    def __enter__(self):
        import threading, tracemalloc
        import psutil
        self._tm, self._proc = tracemalloc, psutil.Process()
        self._start = self._peak = self._proc.memory_info().rss
        self._stop = threading.Event()

        def sample():
            while not self._stop.wait(0.002):
                self._peak = max(self._peak, self._proc.memory_info().rss)
        self._t = threading.Thread(target=sample, daemon=True)
        tracemalloc.start()
        self._t.start()
        return self

    def __exit__(self, *exc):
        _, traced = self._tm.get_traced_memory()
        self._tm.stop()
        self._stop.set()
        self._t.join()
        self._peak = max(self._peak, self._proc.memory_info().rss)
        self.mb = max(traced, self._peak - self._start) / 2 ** 20


def run_all(gt, rgb, scale, names, lr=None, mask=None):
    if lr is None:
        lr, mask = make_damaged(gt, scale)
    results = {}
    for n in names:
        fn, _ = METHODS[n]
        with PeakMemory() as mem:
            t = time.perf_counter()
            pred = fn(lr, rgb, scale, mask).astype(np.float32)
            ms = (time.perf_counter() - t) * 1000
        results[n] = {"depth": pred, "ms": ms, "mb": mem.mb}
    return lr, mask, results


# ---------- demo scene: person in front of a brick wall ----------
def sample_scene(h=288, w=384):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    depth = 6.0 - xx / w * 1.5                      # slanted wall
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[:] = (178, 92, 70)                           # brick colour
    for r in range(0, h, 18):                        # mortar lines = texture, not depth
        rgb[r:r + 3] = (225, 215, 200)
        off = 0 if (r // 18) % 2 else 24
        for c in range(off, w, 48):
            rgb[r:r + 18, c:c + 3] = (225, 215, 200)
    box = (slice(170, 288), slice(250, 360))
    depth[box] = 3.2
    rgb[box] = (70, 110, 160)
    person = ((xx - 150) / 55) ** 2 + ((yy - 190) / 110) ** 2 < 1
    head = (xx - 150) ** 2 + (yy - 62) ** 2 < 30 ** 2
    p = person | head
    depth[p] = 2.0 + (yy[p] / h) * 0.2
    rgb[p] = (40, 45, 55)
    return rgb, depth.astype(np.float32)
