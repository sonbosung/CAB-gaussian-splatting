# Quick Start Guide

Get up and running with CAB-GS in 5 minutes!

## Prerequisites

- NVIDIA GPU with CUDA support
- Linux/macOS (or Windows with WSL2)
- Git, Conda, and basic command line knowledge

## 1. Clone and Setup (5 minutes)

```bash
# Clone repository
git clone https://github.com/sonbosung/CAB-gaussian-splatting.git
cd CAB-gaussian-splatting

# Get original 3DGS base code
git clone https://github.com/graphdeco-inria/gaussian-splatting.git thirdparty/gaussian-splatting

# Copy necessary files
cp -r thirdparty/gaussian-splatting/scene .
cp -r thirdparty/gaussian-splatting/gaussian_renderer .
cp -r thirdparty/gaussian-splatting/arguments .

# Create environment
conda env create -f environment.yml
conda activate cab_gs

# Install Gaussian rasterizer
cd thirdparty/gaussian-splatting/submodules/diff-gaussian-rasterization
pip install -e .
cd ../../../../

# Install simple-knn (optional, for speed)
cd thirdparty/gaussian-splatting/submodules/simple-knn
pip install -e .
cd ../../../../
```

## 2. Download Sample Data

```bash
# Download Mip-NeRF 360 dataset (or use your own)
mkdir -p data
cd data
wget http://storage.googleapis.com/gresearch/refraw360/360_v2.zip
unzip 360_v2.zip
cd ..
```

## 3. Run Pipeline (30-60 minutes per scene)

```bash
# Set data paths
export DATA_ROOT=./data
export OUTPUT_ROOT=./output

# Run on a single scene (bicycle example)
python augment.py \
    --colmap_path data/360_v2/bicycle/sparse/0 \
    --image_path data/360_v2/bicycle/images_4 \
    --augment_path output/bicycle_aug/sparse/0/points3D.bin \
    --camera_order covisibility \
    --visibility_aware_culling \
    --compare_center_patch \
    --n_clusters 4

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

python render.py -m output/models/bicycle --skip_train
python metrics.py -m output/models/bicycle
```

## 4. View Results

```bash
# Check metrics
cat output/models/bicycle/results.json

# View with TensorBoard
tensorboard --logdir output/models/bicycle

# View rendered images
ls output/models/bicycle/test/ours_30000/renders/
```

## Using Your Own Data

1. **Prepare COLMAP reconstruction**:
```bash
# Extract features
colmap feature_extractor \
    --database_path database.db \
    --image_path images

# Match features
colmap exhaustive_matcher \
    --database_path database.db

# Reconstruct
colmap mapper \
    --database_path database.db \
    --image_path images \
    --output_path sparse/0
```

2. **Run CAB-GS**:
```bash
python augment.py \
    --colmap_path your_scene/sparse/0 \
    --image_path your_scene/images \
    --augment_path your_scene_aug/sparse/0/points3D.bin \
    --camera_order covisibility \
    --n_clusters 3

python train_partialscheduler.py \
    -s your_scene_aug \
    -m output/your_scene \
    --eval \
    --bundle_training \
    --n_clusters 3
```

## Troubleshooting

**Error: CUDA out of memory**
```bash
# Reduce resolution
python train_partialscheduler.py ... --resolution 2
```

**Error: Module not found**
```bash
# Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**Error: Rasterizer import failed**
```bash
# Reinstall rasterizer
cd thirdparty/gaussian-splatting/submodules/diff-gaussian-rasterization
pip uninstall diff-gaussian-rasterization
pip install -e .
```

## Next Steps

- 📖 Read [TRAINING.md](docs/TRAINING.md) for detailed parameter tuning
- 🔬 See [CHANGES.md](CHANGES.md) for algorithm details
- 📦 Check [INSTALLATION.md](INSTALLATION.md) for advanced setup options

## Support

- 📝 [Open an issue](https://github.com/sonbosung/CAB-gaussian-splatting/issues)
- 📧 Email: sonbosung@kakao.com
- 🌟 Star the repo if you find it useful!

## Citation

```bibtex
@article{sonn2025cab,
  title={SfM Point Augmentation with Scene Co-visibility-based Image Batching for Sharper 3D Gaussian Splatting},
  author={Sonn, Bosung},
  year={2025}
}
```
