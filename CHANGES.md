# Modifications to Original 3D Gaussian Splatting

This document details all modifications and additions made to the original 3D Gaussian Splatting codebase.

## New Files (Original Contributions)

### Core Method Implementation

- **`augment.py`**: Main augmentation pipeline
  - Implements quadtree-based SfM point augmentation
  - Multi-view geometric consistency checking
  - Visibility-aware culling
  - Local texture comparison for occlusion filtering
  - Co-visibility-based camera ordering

- **`train_partialscheduler.py`**: Training script with partial scheduling
  - Co-visibility-based camera grouping
  - Adaptive training scheduler
  - Progressive densification per camera group
  - Integration of depth smoothness and Laplacian pyramid losses

- **`eval_covis_augment_partialscheduler.sh`**: Evaluation pipeline script
  - End-to-end evaluation on Mip-NeRF 360 dataset
  - Automated augmentation, training, rendering, and metric computation

### Utility Modules

- **`utils/aug_utils.py`**: Augmentation utilities
  - Quadtree data structure implementation
  - Image region subdivision algorithm
  - 3D point sampling and interpolation
  - Multi-view consistency verification functions
  - Local texture similarity computation

- **`utils/bundle_utils.py`**: Co-visibility analysis utilities
  - Co-visibility matrix construction from SfM
  - Co-visibility graph creation
  - Camera sequence generation based on co-visibility
  - Camera clustering for batching

- **`utils/scheduler_utils.py`**: Training scheduler implementation
  - `ImageClustering`: Camera grouping based on co-visibility
  - `PartialGroupScheduler`: Adaptive training schedule
  - Progressive densification per group
  - Warmup, densification, and fine-tuning stage management

- **`utils/colmap_utils.py`**: COLMAP data handling
  - Reading COLMAP reconstruction data
  - Intrinsic/extrinsic matrix computation
  - Point cloud I/O operations

- **`utils/experiment_utils.py`**: Experiment management utilities
  - Result aggregation and logging
  - Per-scene metric computation

## Modified Files

### Training Loop Modifications

**`train_partialscheduler.py`** (based on original `train.py`):
- Added co-visibility-based camera ordering (line 91)
- Integrated `ImageClustering` and `PartialGroupScheduler` (lines 91-95)
- Added support for depth smoothness loss (`--enable_ds_lap`, `--lambda_ds`)
- Added support for Laplacian pyramid loss (`--lambda_lap`)
- Modified camera sampling to use scheduler (throughout training loop)

### Loss Functions

**`utils/loss_utils.py`** (extended from original):
- Added `InvDepthSmoothnessLoss`: Inverse depth smoothness regularization
- Added `laplacian_pyramid_loss`: Multi-scale image pyramid loss
- Both integrated into training objective

## New Dependencies

Added to `environment.yml`:
- `scikit-learn`: For KMeans clustering in camera grouping
- `rtree`: For spatial indexing in quadtree operations
- `shapely`: For geometric operations in augmentation
- OpenCV (`cv2`): Already in original, but more heavily used for image processing

## Configuration Files

### New Arguments

**Augmentation (`augment.py`)**:
- `--colmap_path`: Path to COLMAP sparse reconstruction
- `--image_path`: Path to input images
- `--augment_path`: Output path for augmented points3D.bin
- `--camera_order`: Camera ordering strategy ("covisibility" or "sequential")
- `--visibility_aware_culling`: Enable visibility-based point culling
- `--compare_center_patch`: Enable local texture comparison
- `--n_clusters`: Number of camera clusters

**Training (`train_partialscheduler.py`)**:
- `--bundle_training`: Enable co-visibility-based batching
- `--camera_order`: Camera ordering strategy
- `--enable_ds_lap`: Enable depth smoothness and Laplacian losses
- `--lambda_ds`: Weight for depth smoothness loss (default: 1.2)
- `--lambda_lap`: Weight for Laplacian pyramid loss (default: 0.4)
- `--n_clusters`: Number of camera clusters for scheduling
- `--inv_affinity_matrix`: Use inverse affinity for clustering

## Algorithm Changes

### 1. Point Augmentation Strategy

**Purpose**: Densify sparse SfM reconstructions

**Method**:
1. Decompose each image into quadtree structure based on texture complexity
2. For leaf nodes without SfM points:
   - Find neighboring nodes with depth information
   - Interpolate depth using bilinear interpolation
   - Back-project to 3D world coordinates
3. Verify consistency across co-visible views:
   - Project candidate points to neighboring views
   - Check depth consistency (within 20% tolerance)
   - Compare local texture patches
4. Accept points that pass multi-view verification

**Files**: `augment.py`, `utils/aug_utils.py`

### 2. Co-visibility-based Training Schedule

**Purpose**: Improve training efficiency and quality through scene-aware batching

**Method**:
1. Build co-visibility matrix from SfM reconstruction
2. Cluster cameras into groups using K-means on co-visibility features
3. Order groups to maximize inter-group co-visibility
4. During training:
   - Warmup: Random camera sampling (iterations 0-500)
   - Densification: Progressive training on camera groups
   - Fine-tuning: Random sampling with refined Gaussians

**Files**: `train_partialscheduler.py`, `utils/bundle_utils.py`, `utils/scheduler_utils.py`

### 3. Additional Loss Terms

**Depth Smoothness Loss**:
- Regularizes inverse depth to be locally smooth
- Helps prevent geometric artifacts
- Weight: λ_ds = 1.2 (adaptive, decreases over training)

**Laplacian Pyramid Loss**:
- Multi-scale photometric loss
- Improves detail preservation
- Weight: λ_lap = 0.4

**Files**: `utils/loss_utils.py`, `train_partialscheduler.py`

## Unchanged Original Components

The following components remain from the original 3D Gaussian Splatting implementation:

- `scene/`: Scene representation and camera handling
- `gaussian_renderer/`: CUDA rasterization and rendering
- `submodules/`: Differential Gaussian rasterization
- Core optimization: Adam optimizer with density control
- Spherical harmonics for appearance
- Adaptive densification and pruning strategies

## Testing and Validation

All modifications have been tested on:
- Mip-NeRF 360 dataset
- Tanks and Temples dataset
- Custom scenes

Evaluation metrics:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- LPIPS (Learned Perceptual Image Patch Similarity)

## Future Work

Planned improvements:
- [ ] Multi-resolution augmentation
- [ ] Adaptive cluster number selection
- [ ] Integration with other SfM pipelines (HLOC, etc.)
- [ ] Real-time augmentation during training

## Version History

- **v1.0.0** (2025-01): Initial release
  - Point augmentation with co-visibility verification
  - Partial scheduler with camera grouping
  - Additional geometric losses

---

For questions about specific modifications, please open an issue on GitHub or contact sonbosung@kakao.com.
