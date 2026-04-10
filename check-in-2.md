# flood-detection
Computer Vision flood detection using the WorldFloods (WorldFloodsv2) dataset

## Current status
- Notebooks implemented for both a classical index-based baseline (NDWI + Otsu) and a CNN baseline (U-Net with ResNet encoder in PyTorch).
- Data pipeline: a simple Dataset class (`WorldFloodsTilesDataset`) that discovers S2/GT pairs, reads selected Sentinel-2 bands, aligns GT to S2 grid, and returns tensors. Added an option to enforce a fixed `target_size` to avoid collation errors.
- Environment: created a conda environment (`wf-conda`) with geospatial binaries (rasterio, geopandas, shapely, etc.) and registered a Jupyter kernel. PyTorch installed separately into that environment for training.
- Quick-run results: 18 test samples discovered; deterministic train/val split used (15 train / 3 val) for quick experiments; training loop and metric reporting present.

## How this differs from the original proposal
- Original proposal focused on flood prediction 3 days ahead using multi-source features; current check-in concentrates on building and validating image-based segmentation baselines (index-based and U-Net) and the end-to-end data/experiment plumbing.
- Work so far emphasizes data engineering and environment reproducibility (band selection, CRS handling, reprojection, kernel registration) rather than full multi-modal modeling or forecasting pipelines.
- The project now has runnable baselines and diagnostics that will feed into the forecast-stage design in subsequent iterations.

## Key blockers and resolved issues
- Resolved: disk-space and pip build failures by switching to a conda-forge environment with prebuilt geospatial binaries; registered `wf-conda` kernel.
- Remaining: `ml4floods` and related pip packages could not be installed cleanly due to a GDAL/libgdal/proj version mismatch; this is optional for now but may simplify some utilities if resolved.
- Model runtime issues fixed during development:
  - Heterogeneous tile sizes caused DataLoader collate errors — addressed by adding `target_size` resizing in the Dataset.
  - Encoder downsampling caused logits/mask size mismatch — resolved by interpolating logits to mask size before loss/metrics.

## Failure analysis → what to improve (short list)
- Use the full training set (not just the test-derived quick split) and run cross-validation to get robust performance estimates.
- Address class imbalance: focal loss / weighted BCE / dice + sampling strategies.
- Preserve spatial detail: larger `target_size`, multi-scale supervision, or skip-connections/decoder adjustments to recover small objects.
- Improve labels / filtering: remove or downweight noisy samples, or apply soft labels and/or label-cleaning heuristics before training.

## Immediate next steps (practical)
1. Run smoke tests with `wf-conda` kernel (imports, rasterio read, one-sample Dataset __getitem__) to confirm end‑to‑end I/O.
2. Train a baseline U-Net on the full training set with conservative settings: target_size=(512,512) or (256,256), batch_size=1–2, BCE+Dice loss, and model checkpointing.
3. Run classical-improvements experiments: local/adaptive threshold and combined indices; add small postprocessing pipeline and compare IoU/precision/recall.
4. If label-noise is significant, I will run quick label-quality checks (visual sample inspection) and remove egregious cases or create a clean holdout.

**Note: AI was leveraged from helping fix environment issues, to preproccessing and helping me understand satellite data, to improving on baseline models e.g., BCE to BCE + Dice for loss**


