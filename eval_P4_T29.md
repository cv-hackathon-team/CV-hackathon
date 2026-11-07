1RUA24CSE0069,1RUA24CSE0066,1RUA24CSE0080,1RUA24CSE0077,1RUA24CSE0074

# Evaluation results: P4 (Edge-Preserving Depth Map Upsampling and Refinement)

DRAFT. Accuracy numbers are real outputs of `src/evaluation/evaluate.py` (`outputs/results.csv`, `outputs/results_ablation_test_final.csv`). Latency is the per-image runtime logged during evaluation. Memory benchmark for the DL model is pending (`scripts/benchmark.py`).

Test sets: NYU Depth V2 official test split (654 images, never used for training or tuning) and Middlebury 2014 (10 scenes, unseen). Input: ground-truth depth downsampled by the given factor. `hybrid` is the final DepthGuard hybrid (fusion off, selected on NYU val).

## 1. Depth accuracy: RMSE and MAE (m, lower is better)

**RMSE (m), NYU test**

| Scale | nearest | bilinear | bicubic | joint_bilateral | guided_filter | dl | hybrid |
|---|---|---|---|---|---|---|---|
| x2 | 0.0307 | 0.0253 | 0.0216 | 0.0248 | 0.0250 | 0.0271 | 0.0290 |
| x4 | 0.0565 | 0.0448 | 0.0394 | 0.0427 | 0.0437 | 0.0381 | 0.0366 |
| x8 | 0.0899 | 0.0734 | 0.0658 | 0.0702 | 0.0721 | 0.0533 | 0.0519 |

**RMSE (m), Middlebury**

| Scale | nearest | bilinear | bicubic | joint_bilateral | guided_filter | dl | hybrid |
|---|---|---|---|---|---|---|---|
| x2 | 0.0443 | 0.0399 | 0.0349 | 0.0387 | 0.0391 | 0.0365 | 0.0375 |
| x4 | 0.0760 | 0.0672 | 0.0614 | 0.0625 | 0.0650 | 0.0579 | 0.0534 |
| x8 | 0.1139 | 0.1024 | 0.0951 | 0.0954 | 0.0995 | 0.0833 | 0.0801 |

**MAE (m), NYU test**

| Scale | nearest | bilinear | bicubic | joint_bilateral | guided_filter | dl | hybrid |
|---|---|---|---|---|---|---|---|
| x2 | 0.0077 | 0.0070 | 0.0063 | 0.0076 | 0.0077 | 0.0089 | 0.0077 |
| x4 | 0.0168 | 0.0146 | 0.0132 | 0.0154 | 0.0149 | 0.0129 | 0.0118 |
| x8 | 0.0321 | 0.0285 | 0.0261 | 0.0295 | 0.0288 | 0.0209 | 0.0211 |

**MAE (m), Middlebury**

| Scale | nearest | bilinear | bicubic | joint_bilateral | guided_filter | dl | hybrid |
|---|---|---|---|---|---|---|---|
| x2 | 0.0091 | 0.0100 | 0.0099 | 0.0108 | 0.0111 | 0.0116 | 0.0099 |
| x4 | 0.0206 | 0.0220 | 0.0212 | 0.0227 | 0.0224 | 0.0198 | 0.0177 |
| x8 | 0.0414 | 0.0434 | 0.0408 | 0.0442 | 0.0439 | 0.0340 | 0.0339 |

## 2. PSNR and SSIM (higher is better)

**PSNR (dB), NYU test**

| Scale | nearest | bilinear | bicubic | joint_bilateral | guided_filter | dl | hybrid |
|---|---|---|---|---|---|---|---|
| x2 | 42.93 | 44.67 | 46.03 | 44.76 | 44.71 | 43.95 | 43.41 |
| x4 | 37.61 | 39.67 | 40.81 | 40.03 | 39.88 | 41.12 | 41.44 |
| x8 | 33.57 | 35.38 | 36.32 | 35.72 | 35.53 | 38.23 | 38.39 |

**PSNR (dB), Middlebury**

| Scale | nearest | bilinear | bicubic | joint_bilateral | guided_filter | dl | hybrid |
|---|---|---|---|---|---|---|---|
| x2 | 36.10 | 37.03 | 38.17 | 37.38 | 37.38 | 37.80 | 38.11 |
| x4 | 31.26 | 32.46 | 33.25 | 33.26 | 32.83 | 34.01 | 34.91 |
| x8 | 27.67 | 28.77 | 29.45 | 29.57 | 29.10 | 30.92 | 31.28 |

**SSIM, NYU test**

| Scale | nearest | bilinear | bicubic | joint_bilateral | guided_filter | dl | hybrid |
|---|---|---|---|---|---|---|---|
| x2 | 0.9904 | 0.9933 | 0.9947 | 0.9934 | 0.9933 | 0.9925 | 0.9931 |
| x4 | 0.9674 | 0.9799 | 0.9825 | 0.9820 | 0.9812 | 0.9849 | 0.9866 |
| x8 | 0.9241 | 0.9572 | 0.9580 | 0.9643 | 0.9603 | 0.9708 | 0.9708 |

**SSIM, Middlebury**

| Scale | nearest | bilinear | bicubic | joint_bilateral | guided_filter | dl | hybrid |
|---|---|---|---|---|---|---|---|
| x2 | 0.9783 | 0.9797 | 0.9824 | 0.9803 | 0.9802 | 0.9800 | 0.9839 |
| x4 | 0.9363 | 0.9454 | 0.9476 | 0.9516 | 0.9484 | 0.9543 | 0.9605 |
| x8 | 0.8645 | 0.8927 | 0.8907 | 0.9132 | 0.9011 | 0.9152 | 0.9137 |

## 3. Edge preservation and boundary accuracy (boundary F-score, higher is better)

**Boundary F-score, NYU test**

| Scale | nearest | bilinear | bicubic | joint_bilateral | guided_filter | dl | hybrid |
|---|---|---|---|---|---|---|---|
| x2 | 0.9580 | 0.8062 | 0.8481 | 0.8354 | 0.8390 | 0.8830 | 0.9103 |
| x4 | 0.8692 | 0.5438 | 0.5981 | 0.7238 | 0.6289 | 0.7310 | 0.8379 |
| x8 | 0.5626 | 0.2411 | 0.2995 | 0.6265 | 0.4089 | 0.5908 | 0.7228 |

**Boundary F-score, Middlebury**

| Scale | nearest | bilinear | bicubic | joint_bilateral | guided_filter | dl | hybrid |
|---|---|---|---|---|---|---|---|
| x2 | 0.9783 | 0.8501 | 0.8837 | 0.8826 | 0.8768 | 0.9021 | 0.9288 |
| x4 | 0.8960 | 0.4437 | 0.5221 | 0.7729 | 0.6109 | 0.6462 | 0.8483 |
| x8 | 0.5594 | 0.0679 | 0.1053 | 0.6466 | 0.3551 | 0.4130 | 0.6848 |

## 4. Performance across upsampling factors and corrupted inputs (NYU test, RMSE m / boundary F)

Classical baselines were evaluated on clean low-resolution input only (see sections 1-3).

| Input | Scale | dl | hybrid |
|---|---|---|---|
| lr | x2 | 0.0271 / 0.883 | 0.0290 / 0.910 |
| lr | x4 | 0.0381 / 0.731 | 0.0366 / 0.838 |
| lr | x8 | 0.0533 / 0.591 | 0.0519 / 0.723 |
| noisy | x2 | 0.0364 / 0.904 | 0.0381 / 0.911 |
| noisy | x4 | 0.0443 / 0.736 | 0.0446 / 0.839 |
| noisy | x8 | 0.0580 / 0.607 | 0.0581 / 0.731 |
| holes | x2 | 0.0481 / 0.843 | 0.0471 / 0.875 |
| holes | x4 | 0.0548 / 0.699 | 0.0526 / 0.809 |
| holes | x8 | 0.0695 / 0.558 | 0.0679 / 0.695 |
| sparse | x2 | 0.0617 / 0.864 | 0.0485 / 0.866 |
| sparse | x4 | 0.0822 / 0.627 | 0.0751 / 0.726 |
| sparse | x8 | 0.1233 / 0.353 | 0.1207 / 0.506 |
| boundary | x2 | 0.0446 / 0.889 | 0.0360 / 0.894 |
| boundary | x4 | 0.0607 / 0.687 | 0.0542 / 0.792 |
| boundary | x8 | 0.0912 / 0.458 | 0.0883 / 0.616 |
| mixed | x2 | 0.0863 / 0.801 | 0.0745 / 0.807 |
| mixed | x4 | 0.1091 / 0.528 | 0.1035 / 0.633 |
| mixed | x8 | 0.1623 / 0.256 | 0.1603 / 0.396 |

## 5. Processing latency and memory

Mean runtime per image (ms), NYU test evaluation, Apple M5 MacBook Air (24 GB), DL on MPS.

| Scale | nearest | bilinear | bicubic | joint_bilateral | guided_filter | dl | hybrid |
|---|---|---|---|---|---|---|---|
| x2 | 0.1 | 0.1 | 0.2 | 1.4 | 2.7 | 73.9 | 139.7 |
| x4 | 0.1 | 0.1 | 0.2 | 3.4 | 2.6 | 73.8 | 138.4 |
| x8 | 0.1 | 0.1 | 0.2 | 10.7 | 2.9 | 74.0 | 139.6 |

Model size: 3,509,275 parameters. Peak process memory for the full classical evaluation: about 1.06 GB. DL/hybrid peak memory: pending benchmark.

## Known weakness

At x2, bicubic has lower RMSE than dl and hybrid; the learned model's gains are at x4 and x8.
