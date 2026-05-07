# flood-detection

A project for detecting flooded areas from Sentinel‑2 satellite imagery using the WorldFloods v2 dataset. The repository contains two baseline approaches, visualization utilities, and training code for a lightweight semantic‑segmentation CNN.

---

## Contents

- Notebooks
  - `baseline-classical.ipynb` — classical remote‑sensing baseline (NDWI + Otsu thresholding), metrics and visualizations.
  - `baseline-CNN.ipynb` — a compact exploration of a simple CNN baseline on a small sample of data (fast experiments).
  - `Full-CNN.ipynb` — end‑to‑end PyTorch baseline: data loader, ResNet‑backed U‑Net, training loop, visualization, and floodmap overlays.
  - `EDA1` & `EDA2` notebooks / cells described below (exploratory analyses).
- Core experiments and artifacts (in the workspace): Sentinel‑2 tiles, GT (2‑band quality + landcover), permanent water rasters, and GeoJSON flood polygons, available at [Data Folder on Google Drive](https://drive.google.com/drive/u/0/folders/1dqFYWetX614r49kuVE3CbZwVO6qHvRVH).

---

## Project overview

The goal is to extract water/flood information from multispectral Sentinel‑2 imagery and evaluate several baseline methods and EDA pipelines:

1. Classical index-based method using NDWI (Normalized Difference Water Index) computed from the Green (B3) and NIR (B8) bands, with an automatically chosen global per‑tile threshold using Otsu's method.
2. A learned baseline: a small U‑Net style segmentation network with a ResNet encoder (ResNet‑18 by default) trained in PyTorch to predict per‑pixel water (binary mask).
3. Exploratory analyses (EDA1 / EDA2) to understand temporal coverage, label quality and direct use of S2 imagery for flood signals.

Both baselines use the WorldFloods GT semantics: a 2‑band raster where band 1 is a quality mask (0 invalid, 1 clear, 2 cloud) and band 2 is landcover (0 nodata, 1 land, 2 water). All training and evaluation exclude invalid/cloud pixels by default (only use `quality == 1` and `landcover != 0`).

---

## Exploratory Data Analysis (EDA)

Two EDA efforts are included and described in the notebooks:

- EDA1 — direct satellite image exploration
  - Purpose: investigate whether per‑location Sentinel‑2 time series provide sufficient temporal resolution to detect flood events directly from imagery.
  - Key finding / bottleneck: Sentinel‑2 revisit in practice (the dataset slices used here) often yields only 1 usable clear image per location in a 6–12 day window due to cloud cover and acquisition scheduling. Because floods evolve on shorter timescales (hours–days) and cloud cover removes many observations, relying on a single S2 snapshot per event is a major limitation for direct time‑series flood detection. Use EDA1 to quantify how often pre/post flood pairs exist and prioritize regions with better temporal coverage.

- EDA2 — exploring directly downloaded WorldFloods v2 data
  - Purpose: validate and inspect the raw WorldFloods v2 artifacts as available in the online folder prior to training.
  - What it does: enumerates `S2/`, `gt/`, `PERMANENTWATERJRC/`, and `floodmaps/` contents; checks file CRS and basic metadata (shape, resolution, nodata); computes class distributions in GT (land vs water vs nodata) and basic label quality summaries (cloud fraction from quality band).
  - How to use: run the EDA2 notebook/cells early after downloading the dataset to adapt preprocessing choices (e.g., target size, which bands to use, whether to subtract permanent water).

Notes: both EDA notebooks are lightweight and designed to run quickly on a subset of the dataset for diagnostics before full training.

---

## Classical baseline: NDWI + Otsu

- NDWI = (Green − NIR) / (Green + NIR) computed per tile after scaling reflectance (scale factor used: ~3500.0 in these notebooks).
- Otsu's method is applied to valid (non‑NaN) NDWI pixels per tile to pick a threshold that separates land vs water in the NDWI histogram.
- Predictions: pixels with NDWI >= threshold are classified as water. Metrics (IoU, precision, recall) are computed over valid GT pixels only.

Observed (example) behavior in experiments:
- Per‑tile Otsu thresholds vary widely depending on scene clarity; an example threshold was approx `-0.341`.
- Example NDWI+Otsu performance on a small test set: mean IoU ≈ `0.139` (low but sometimes visually plausible on clearer scenes).

Limitations: unimodal NDWI histograms, turbid water, wet soils, shadows and mixed pixels often confuse the global threshold, producing both false positives and false negatives.

---

## CNN baseline: ResNet‑backed U‑Net (PyTorch)

Model details:
- Encoder: ResNet‑18 (optionally ResNet‑34) taken from torchvision; initial conv adapted for arbitrary input channel count.
- Decoder: 4 upsampling decoder blocks, bilinear upsampling + convs, U‑Net skip connections.
- Output: single‑channel logits (use BCEWithLogits + Dice loss).
- Input bands used in experiments: `B4` (red), `B3` (green), `B8` (NIR) — stacked and scaled by ~3500.

Training & evaluation details:
- Dataset class aligns GT (quality + landcover) with the S2 grid using rasterio's reprojection (nearest resampling) and returns `(img_t, mask_t, valid_t)` where `valid_t` encodes pixels to include in loss/metrics.
- Loss = masked BCEWithLogits + masked soft‑Dice computed only on pixels where `quality == 1` and `landcover != 0`.
- Metrics = masked IoU / precision / recall computed per sample on valid pixels.
- To avoid batch collation errors with variable tile sizes, samples are optionally resized to a fixed `target_size` (e.g. 512×512) using bilinear for images and nearest for masks/valid flags.

Observed behavior in full-CNN (full dataset loader, model training):

- Showed very low IoU due to label noise and class imbalance; reported validation metrics: IoU ≈ `0.0082`, precision ≈ `0.6100`, recall ≈ `0.0082`.
  - Interpretation: the model predicted very few positive pixels (high precision on the few predictions, but almost all true water pixels were missed as there is extremely low recall and IoU).

Known practical issues:

- Severe class imbalance (water is rare) requires strong loss balancing, sampling, or focal/Dice weighting.
- GT coarseness / registration mismatch between S2 pixels and GT labels reduces achievable pixel‑level IoU.
- Clouds/invalid pixels: ensure masking is applied consistently during training and evaluation.

---

## Notebooks & scripts

- `EDA1.ipynb` / EDA1 cells — run direct S2 temporal diagnostics and NDWI snapshots to understand revisit cadence and cloud impacts.
- `EDA2.ipynb` / EDA2 cells — run dataset integrity and label distribution checks on the downloaded WorldFloods v2 folder.
- `baseline-classical.ipynb` — run the NDWI pipeline. Quick steps:
  1. Set `DATA_ROOT` to your dataset location.
  2. Run discovery cell to list S2/GT pairs.
  3. Execute the main loop that computes NDWI, Otsu threshold and metrics, and shows visual overlays.

- `baseline-CNN.ipynb` — a compact notebook that explores a simple CNN baseline on a *small sample* for fast iteration. Use it to test training loops, masked losses, and quick architectural changes before running the larger `Full-CNN.ipynb` experiments.

- `Full-CNN.ipynb` — full dataset loader, model training and visualization pipeline. Use for longer experiments and final evaluations.

---

## Reproducibility / environment

Recommended environment (conda / virtualenv):
- Python 3.9+ (your notebooks were developed on macOS)
- Core packages: numpy, pandas, matplotlib, seaborn
- Geospatial: rasterio, geopandas, shapely
- Deep learning: torch, torchvision
- Optional utilities: geemap, ml4floods (some notebooks include helper imports that can be toggled/commented if ml4floods is not installed)

---

## Known issues & debugging checklist

- Empty or invisible flood overlays: verify the GeoJSON CRS and that polygons intersect the S2 tile window. Print `gdf.crs`, S2 CRS and polygon bounds to debug.
- No matching GT / permanent water files: the notebooks include robust filename heuristics but may fail if naming differs substantially; place files under the expected folders or adjust filename logic.
- Very low IoU for CNN: try lower decision thresholds, balanced sampling, stronger augmentation, and experiment with weighted BCE / focal loss.
- Memory/colation errors: use `target_size` or smaller batch sizes; set `num_workers=0` for debugging.

---

## Next steps and suggestions

- Improve the classical baseline with local/adaptive thresholding (CLAHE, local Otsu) and additional indices (MNDWI, AWEI).
- Integrate permanent water subtraction (JRC) to focus training on flood‑only targets.
- Use class‑balanced sampling, focal loss, or oversample flooded tiles for the CNN baseline.
- Explore larger backbone encoders pretrained on multispectral data, and temporal models that use pre/post flood image pairs.

---


