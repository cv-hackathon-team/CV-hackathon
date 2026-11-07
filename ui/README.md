# DepthGuard 360 – Web UI

Clean one-page app: hero, Studio (upload → reconstruct → scores), How it works, Benchmark.
No build step. Python backend (FastAPI) + plain HTML/CSS/JS.

## Run locally (2 commands)
Put this `ui/` folder inside `depthguard360/`, then from `depthguard360/`:
```
pip install -r ui/requirements-ui.txt
uvicorn ui.server:app --reload
```
Open http://localhost:8000 and click **Try the sample scene**.

## Where to add things
| You want to…                        | Edit                                   |
|-------------------------------------|----------------------------------------|
| Add / swap a method                 | `ui/pipeline.py` – add a function with `@register("Name", group)`. It appears in the UI automatically. |
| Plug in the trained model           | `ui/pipeline.py` – replace the body of `depthguard()` with a call to `src/inference.py` + `checkpoints/best.pt`. Keep the same inputs/outputs. |
| Change the six module texts         | `ui/web/app.js` – the `MODULES` list at the top. |
| Add a page section                  | `ui/web/index.html` – copy a `<section class="wrap section">` block. |
| Change colours / fonts              | `ui/web/styles.css` – the `:root` variables at the top. |
| Benchmark table                     | Auto-loads `outputs/results.csv`. |

Method signature: `fn(lr_depth, rgb, scale, hole_mask) -> hr_depth` (float32, same height/width as `rgb`).

## Deploy (free)
**Hugging Face Spaces (Docker):** create a Space → SDK "Docker" → upload the repo, and copy `ui/Dockerfile` to the repo root (change nothing else). It serves on port 7860.

**Render:** New → Web Service → connect GitHub repo → Runtime "Docker", Dockerfile path `ui/Dockerfile`. Render sets `$PORT` itself.

Note: the current "DepthGuard 360" method is a rule-based stand-in (hole fill → texture-aware joint bilateral → front/back pick at edges) so the demo works before the model is wired in.
