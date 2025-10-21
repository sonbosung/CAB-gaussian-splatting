# CAB-GS (Co-visibility-Aware and Similarity-Driven Gaussian Splatting)

Generated: 2025-10-21T13:33:02.632Z
Owner: Bosung Sonn (sonbosung@kakao.com)
Repo: https://github.com/sonbosung/CAB-gaussian-splatting

## Executive summary
CAB-GS improves 3D Gaussian Splatting by (1) densifying sparse SfM point clouds via geometry-consistent augmentation and (2) scheduling training with similarity-based camera grouping derived from the co-visibility adjacency matrix. This yields sharper reconstructions, more stable densification, and improved metrics on Mip-NeRF 360.

## Problem and motivation
- Sparse or uneven SfM reconstructions cause under-reconstruction and unstable optimization in neural rendering.
- Uniform camera sampling ignores view relationships; training can oscillate or overfit local modes.

## Key contributions
1) SfM Point Augmentation (augment.py, utils/aug_utils.py)
- Quadtree-based region identification of under-reconstructed areas.
- Multi-view triangulation/depth interpolation with geometric consistency checks.
- Occlusion/texture filtering for robust point proposals.

2) Similarity-based Camera Grouping (train_similarity.py, utils/scheduler_utils.py)
- Build a co-visibility adjacency matrix from SfM; compute camera similarity from it.
- Cluster cameras and schedule training with PartialGroupScheduler.
- Progressive, group-aware densification and pruning during optimization.

3) Structure-aware losses and optimization
- Optional inverse-depth smoothness and Laplacian pyramid losses (utils/loss_utils.py).
- Compatibility with standard and SparseAdam optimization.

## System overview (files of interest)
- Training: train_similarity.py (default), render.py, metrics.py
- Augmentation: augment.py, utils/aug_utils.py
- Scheduling/Clustering: utils/scheduler_utils.py, utils/bundle_utils.py
- Geometry/IO: utils/colmap_utils.py, utils/read_write_model.py, utils/camera_utils.py
- Utilities: utils/image_utils.py, utils/general_utils.py
- Environment: environment.yml
- Docs: README.md, docs/TRAINING.md
- Figures: comparison_360_vs_360_similarity_full.png, improvements_360_vs_360_similarity_full.png

## Usage (minimal)
1) Create environment
```bash
conda env create -f environment.yml
conda activate cab_gs
```
2) Augment SfM points
```bash
python augment.py \
  --colmap_path <path_to_colmap>/sparse/0 \
  --image_path <path_to_images> \
  --augment_path <output_path>/sparse/0/points3D.bin \
  --camera_order covisibility \
  --visibility_aware_culling \
  --compare_center_patch \
  --n_clusters 4
```
3) Train with similarity-based grouping (default)
```bash
python train_similarity.py \
  -s <path_to_augmented_data> \
  -m <output_model_path> \
  --eval \
  --bundle_training \
  --camera_order covisibility \
  --enable_ds_lap \
  --lambda_ds 1.2 \
  --lambda_lap 0.4 \
  --n_clusters 4
```
4) Render and evaluate
```bash
python render.py -m <output_model_path> --skip_train
python metrics.py -m <output_model_path>
```
5) Full pipeline (repro)
```bash
bash eval_similarity.sh
```

## Results (Mip-NeRF 360)
- Average: PSNR 27.69, SSIM 0.827, LPIPS 0.190.
- Visuals:
  - comparison_360_vs_360_similarity_full.png (comparison grid)
  - improvements_360_vs_360_similarity_full.png (per-scene improvements)

## Implementation details
- Similarity computation from co-visibility adjacency; clustering yields stable groups that preserve scene structure across densification phases.
- PartialGroupScheduler toggles group activation, triggers densify/prune and optional opacity resets at controlled iterations.
- Depth smoothness uses inverse-depth gradients; Laplacian pyramid emphasizes multi-scale structure preservation.
- Logging via TensorBoard (if available), PSNR/SSIM/LPIPS reporting in metrics.py.

## Talking points for portfolio generation
- Contrast to naive batching: grouping respects scene structure and co-visibility, reducing optimization noise.
- Augmentation reduces holes from sparse SfM; geometric checks prevent drift/ghosting.
- Modular design: swapping grouping strategy, losses, and optimizer without changing pipeline.
- Reproducibility: single-script eval, environment lock, committed figures and versions.

## FAQs
- Q: Why adjacency-based similarity instead of raw image similarity?
  A: Co-visibility encodes 3D structure/overlap; adjacency-derived similarity better aligns with reconstruction objectives than 2D appearance alone.
- Q: Does augmentation bias geometry?
  A: Points pass geometric consistency and occlusion/texture filters; densification/pruning further regularize during training.
- Q: How many clusters?
  A: Typically 4–6 for Mip-NeRF 360; tune n_clusters by scene size/overlap density.

## Citations
- Sonn, B. “SfM Point Augmentation with Scene Co-visibility-based Image Batching for Sharper 3D Gaussian Splatting.” arXiv preprint, 2025.
- Kerbl, B., et al. “3D Gaussian Splatting for Real-Time Radiance Field Rendering.” TOG 42(4), 2023.

## License and acknowledgments
- Derivative of 3D Gaussian Splatting (GraphDECO Inria). Research/evaluation only; see LICENSE.md.

## How to use this file with a web-based GPT
- Paste this file as project context.
- Ask the model to generate a portfolio page for a Vision AI Research Engineer, emphasizing contributions, figures, and reproducibility.
- Provide role/company-specific tone and desired sections (Summary, Contributions, Results, Figures, Pipeline, Code snippets).
