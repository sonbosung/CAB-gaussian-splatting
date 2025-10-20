# Training Guide

This guide provides detailed instructions for training CAB-GS models.

## Prerequisites

1. COLMAP reconstruction of your scene (sparse/0 directory)
2. Input images
3. Conda environment activated (`conda activate cab_gs`)

## Training Pipeline

### Step 1: Point Augmentation

Augment the sparse SfM point cloud before training:

```bash
python augment.py \
    --colmap_path /path/to/scene/sparse/0 \
    --image_path /path/to/scene/images \
    --augment_path /path/to/output/sparse/0/points3D.bin \
    --camera_order covisibility \
    --visibility_aware_culling \
    --compare_center_patch \
    --n_clusters 4
```

**Parameters**:

- `--colmap_path`: Path to COLMAP sparse reconstruction directory
- `--image_path`: Path to input images
- `--augment_path`: Output path for augmented points3D.bin file
- `--camera_order`: Camera ordering strategy
  - `covisibility`: Order cameras by co-visibility (recommended)
  - `sequential`: Use sequential ordering
- `--visibility_aware_culling`: Enable visibility-based point culling (recommended)
- `--compare_center_patch`: Enable local texture comparison for filtering (recommended)
- `--n_clusters`: Number of camera clusters (default: 4)
  - Larger values: More fine-grained grouping
  - Smaller values: Faster processing, coarser grouping

### Step 2: Training with Partial Scheduler

Train using the augmented point cloud and co-visibility-based scheduling:

```bash
python train_partialscheduler.py \
    -s /path/to/augmented/scene \
    -m /path/to/output/model \
    -i images \
    --eval \
    --bundle_training \
    --camera_order covisibility \
    --enable_ds_lap \
    --lambda_ds 1.2 \
    --lambda_lap 0.4 \
    --n_clusters 4
```

**Parameters**:

- `-s, --source_path`: Path to scene directory (with augmented sparse/0)
- `-m, --model_path`: Output path for trained model
- `-i, --images`: Image folder name (e.g., "images", "images_4", "images_2")
- `--eval`: Create train/test split for evaluation
- `--bundle_training`: Enable co-visibility-based batching (required for CAB-GS)
- `--camera_order`: Must match augmentation setting
- `--enable_ds_lap`: Enable depth smoothness and Laplacian pyramid losses
- `--lambda_ds`: Weight for depth smoothness loss (default: 1.2)
- `--lambda_lap`: Weight for Laplacian pyramid loss (default: 0.4)
- `--n_clusters`: Number of camera clusters (should match augmentation)
- `--inv_affinity_matrix`: Use inverse affinity for clustering (optional)

**Additional Original 3DGS Parameters**:

- `--iterations`: Number of training iterations (default: 30000)
- `--position_lr_init`: Initial learning rate for positions (default: 0.00016)
- `--feature_lr`: Learning rate for features (default: 0.0025)
- `--opacity_lr`: Learning rate for opacity (default: 0.05)
- `--scaling_lr`: Learning rate for scaling (default: 0.005)
- `--rotation_lr`: Learning rate for rotation (default: 0.001)
- `--densify_from_iter`: Start densification (default: 500)
- `--densify_until_iter`: Stop densification (default: 15000)
- `--densification_interval`: Densify every N iterations (default: 100)

### Step 3: Rendering

Render test views:

```bash
python render.py -m /path/to/output/model --skip_train
```

**Parameters**:

- `-m, --model_path`: Path to trained model
- `--skip_train`: Skip rendering training views (recommended for evaluation)
- `--skip_test`: Skip rendering test views

### Step 4: Evaluation

Compute metrics on rendered images:

```bash
python metrics.py -m /path/to/output/model
```

This computes PSNR, SSIM, and LPIPS on the test set.

## Training Schedule Details

The partial scheduler implements a three-stage training pipeline:

### Stage 1: Warmup (Iterations 0-500)

- Random camera sampling
- No densification
- Establishes initial Gaussian distribution

### Stage 2: Densification (Iterations 500-15000)

Progressive training with camera grouping:

1. **Group-based batching**: Cameras grouped by co-visibility
2. **Sequential group processing**: Train on one group at a time
3. **Adaptive densification**: Densify and prune after each group
4. **Multi-pass**: Multiple passes through all groups

**Benefits**:
- Better convergence in sparse regions
- Improved multi-view consistency
- Reduced training artifacts

### Stage 3: Fine-tuning (Iterations 15000-30000)

- Random camera sampling
- No densification
- Refinement of appearance and geometry

## Hyperparameter Tuning

### Number of Clusters (`--n_clusters`)

Recommended values based on scene size:

- **Small scenes** (< 50 images): 2-3 clusters
- **Medium scenes** (50-150 images): 4-6 clusters
- **Large scenes** (> 150 images): 6-10 clusters

Trade-offs:
- More clusters: Finer control, longer training
- Fewer clusters: Faster training, less precise grouping

### Loss Weights

**Depth Smoothness (`--lambda_ds`)**:

- Default: 1.2
- Increase (1.5-2.0): Smoother geometry, may lose fine details
- Decrease (0.8-1.0): Preserve details, may introduce artifacts

**Laplacian Pyramid (`--lambda_lap`)**:

- Default: 0.4
- Increase (0.6-0.8): Sharper details, may overfit
- Decrease (0.2-0.3): Smoother results, may blur details

### Scene-Specific Adjustments

**Indoor scenes**:
```bash
--n_clusters 4 --lambda_ds 1.5 --lambda_lap 0.3
```

**Outdoor scenes**:
```bash
--n_clusters 6 --lambda_ds 1.0 --lambda_lap 0.5
```

**Object-centric**:
```bash
--n_clusters 2 --lambda_ds 0.8 --lambda_lap 0.4
```

## Troubleshooting

### Issue: Augmentation produces too few points

**Solutions**:
- Decrease `--n_clusters` to increase points per view
- Remove `--compare_center_patch` flag (less strict filtering)
- Check COLMAP reconstruction quality

### Issue: Training diverges or produces artifacts

**Solutions**:
- Reduce loss weights (`--lambda_ds`, `--lambda_lap`)
- Increase warmup iterations
- Check if augmented points are reasonable (visualize in CloudCompare)

### Issue: Slow training

**Solutions**:
- Reduce `--n_clusters` (fewer groups = faster)
- Disable `--enable_ds_lap` (removes additional losses)
- Use `--optimizer_type sparse_adam` (if available)

### Issue: Poor test view quality

**Solutions**:
- Increase `--n_clusters` for better coverage
- Enable `--compare_center_patch` for better filtering
- Train longer (`--iterations 40000`)

## Example Configurations

### Mip-NeRF 360 Dataset

```bash
# Augmentation
python augment.py \
    --colmap_path data/360_v2/bicycle/sparse/0 \
    --image_path data/360_v2/bicycle/images_4 \
    --augment_path output/bicycle_aug/sparse/0/points3D.bin \
    --camera_order covisibility \
    --visibility_aware_culling \
    --compare_center_patch \
    --n_clusters 4

# Training
python train_partialscheduler.py \
    -s output/bicycle_aug \
    -m output/models/bicycle \
    -i images_4 \
    --eval \
    --bundle_training \
    --camera_order covisibility \
    --enable_ds_lap \
    --lambda_ds 1.2 \
    --lambda_lap 0.4 \
    --n_clusters 4
```

### Custom Dataset

```bash
# Augmentation
python augment.py \
    --colmap_path my_scene/sparse/0 \
    --image_path my_scene/images \
    --augment_path my_scene_aug/sparse/0/points3D.bin \
    --camera_order covisibility \
    --visibility_aware_culling \
    --n_clusters 3

# Training
python train_partialscheduler.py \
    -s my_scene_aug \
    -m output/my_scene \
    -i images \
    --eval \
    --bundle_training \
    --camera_order covisibility \
    --n_clusters 3
```

## Monitoring Training

Training progress is logged to TensorBoard:

```bash
tensorboard --logdir output/models/your_scene
```

Key metrics to monitor:
- **Total Loss**: Should decrease smoothly
- **L1 Loss**: RGB reconstruction error
- **SSIM Loss**: Structural similarity
- **Depth Smoothness**: Geometric consistency
- **Number of Gaussians**: Should stabilize after densification

## Advanced: Custom Datasets

For best results with custom datasets:

1. **COLMAP reconstruction**: Use high-quality settings
   ```bash
   colmap feature_extractor --ImageReader.camera_model PINHOLE
   colmap exhaustive_matcher
   colmap mapper --Mapper.ba_refine_focal_length 0
   ```

2. **Image quality**: Use high-resolution images (> 1K pixels)

3. **Camera coverage**: Ensure good overlap between views

4. **Augmentation settings**: Start conservative, then adjust
   ```bash
   --n_clusters 3 --visibility_aware_culling
   ```

5. **Training settings**: Use standard configuration first
   ```bash
   --bundle_training --n_clusters 3 --enable_ds_lap
   ```

## Performance Tips

1. **GPU Memory**: Reduce image resolution if OOM
   ```bash
   # Add to training command
   --resolution 1
   ```

2. **Training Speed**: Use batch size optimization
   ```bash
   # The scheduler automatically adjusts batch sizes
   ```

3. **Quality vs Speed**: Disable extra losses for faster training
   ```bash
   # Remove --enable_ds_lap for 10-15% speedup
   ```

---

For more details, see [CHANGES.md](../CHANGES.md) for algorithm descriptions or open an issue on GitHub.
