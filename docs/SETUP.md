# DepthGuard360

Edge-preserving depth map upsampling (Hackathon PS4).

**Status:** the data pipeline, classical baselines, model, training, hybrid refinement,
diagnosis, inference and evaluation are implemented. `main.py` and the Streamlit app are
still stubs that raise `NotImplementedError`. Final metrics are in `eval_P4_TXX.md`.

## Setup

Requires Python 3.14. All reported results were produced on Python 3.14.6 with the exact
versions in `requirements.txt`, on macOS arm64 using MPS.

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Datasets

Datasets are not included. Place them in the folder set by `paths.dataset_root` in
`configs/default.yaml`. The default, `../DEPTHGUARD360_DATASETS`, is resolved from
the project root, so it sits next to the `depthguard360/` folder.

| Dataset | Used files |
| --- | --- |
| NYU Depth v2 | Labeled `.mat` |
| Middlebury 2014 | Scene folders with `im0.png`, `disp0.pfm` and `calib.txt` |
| KITTI depth completion | `val_selection_cropped` (optional, skipped if missing) |

### Download

| Dataset | Where to get it | Put it in |
| --- | --- | --- |
| NYU Depth v2 (labeled, ~2.8 GB) | https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat | `02_NYU_DEPTH_V2/labeled/` |
| NYU official split (`splits.mat`, 795 train / 654 test) | linked from https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html | `02_NYU_DEPTH_V2/labeled/` |
| Middlebury 2014, 10 `-perfect.zip` scenes | https://vision.middlebury.edu/stereo/data/scenes2014/zip/ | `03_MIDDLEBURY/<Scene>/` (drop the `-perfect` suffix) |
| KITTI depth completion validation (~2 GB, needs registration) | https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion | optional |

Middlebury scenes: Adirondack, Jadeplant, Motorcycle, Piano, Pipes, Playroom, Playtable, Recycle, Shelves, Vintage.
Middlebury is used only as an unseen test set; it gives disparity, which is converted to metric depth with `calib.txt`.

Expected layout:

```
DEPTHGUARD360_DATASETS/
  02_NYU_DEPTH_V2/labeled/nyu_depth_v2_labeled.mat
  02_NYU_DEPTH_V2/labeled/splits.mat
  03_MIDDLEBURY/Adirondack/{im0.png, disp0.pfm, calib.txt, ...}
  03_MIDDLEBURY/...
```

Run `python scripts/check_datasets.py` to verify the files before training.

## Commands

```bash
python scripts/check_datasets.py        # verify datasets, print shapes and depth ranges
python src/evaluation/evaluate.py --methods classical --split test   # add --limit 20 for a quick run
python -m pytest tests                  # run from the project root
```

Inference on one image, loading `checkpoints/best.pt` once and using cuda, then mps, then CPU:

```python
from src.inference import predict

output = predict(rgb, lr_depth, scale)  # rgb (H, W, 3); lr_depth (h, w) in metres; scale 2, 4 or 8
output["depth"], output["confidence"], output["boundary"], output["diagnosis"], output["timings"]
```

A full classical test run writes `outputs/results.csv`, `outputs/results_per_image.csv`
and PNGs in `outputs/figures/`. Other runs add their methods, split and limit to the names.

```bash
python scripts/tune_classical.py                     # classical settings, NYU val only
python src/training/train.py --smoke                 # short training check
python src/training/train.py                         # full training
python src/evaluation/evaluate.py --methods ablation --split val
python scripts/calibrate_diagnosis.py                # diagnosis thresholds, NYU val only
python src/evaluation/evaluate.py --methods final --split test \
    --corruptions noisy holes sparse boundary mixed --append           # adds to outputs/results.csv
python src/evaluation/evaluate.py --methods ablation --split test --tag ablation_test_final
python scripts/benchmark.py                          # latency and memory
python scripts/build_eval_report.py                  # writes eval_P4_TXX.md from the CSVs
```

## Layout

| Path | Contents |
| --- | --- |
| `configs/` | Paths, scale factors, training parameters |
| `src/datasets/` | Dataset loaders and input corruption |
| `src/classical/` | Interpolation, joint bilateral and guided filter baselines |
| `src/models/` | DepthGuardNet |
| `src/hybrid/` | Corruption diagnosis and confidence-guided refinement |
| `src/training/` | Losses and training loop |
| `src/evaluation/` | Metrics and evaluation runner |
| `app/` | Streamlit demo |
| `docs/design.md` | Design document |
