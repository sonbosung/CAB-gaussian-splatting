# Installation Guide

## Prerequisites

- Linux or macOS (Windows with WSL2 also works)
- NVIDIA GPU with CUDA support (Compute Capability >= 7.0)
- CUDA Toolkit 11.8 or higher
- Python 3.8 or higher
- Conda or Mamba package manager

## Step 1: Clone Repository

```bash
git clone https://github.com/sonbosung/CAB-gaussian-splatting.git
cd CAB-gaussian-splatting
```

## Step 2: Setup Base 3D Gaussian Splatting

This repository requires the original 3D Gaussian Splatting codebase. You have two options:

### Option A: Add as Git Submodule (Recommended)

```bash
# Add original 3DGS as submodule
git submodule add https://github.com/graphdeco-inria/gaussian-splatting.git thirdparty/gaussian-splatting
git submodule update --init --recursive

# Copy necessary directories
cp -r thirdparty/gaussian-splatting/scene .
cp -r thirdparty/gaussian-splatting/gaussian_renderer .
cp -r thirdparty/gaussian-splatting/arguments .
cp thirdparty/gaussian-splatting/train.py train_base.py
```

### Option B: Manual Setup

```bash
# Clone original repo separately
git clone https://github.com/graphdeco-inria/gaussian-splatting.git ../gaussian-splatting-original

# Copy necessary directories
cp -r ../gaussian-splatting-original/scene .
cp -r ../gaussian-splatting-original/gaussian_renderer .
cp -r ../gaussian-splatting-original/arguments .
cp ../gaussian-splatting-original/train.py train_base.py
```

## Step 3: Create Conda Environment

```bash
# Create environment from file
conda env create -f environment.yml
conda activate cab_gs
```

If `environment.yml` doesn't work, create manually:

```bash
# Create new environment
conda create -n cab_gs python=3.9 -y
conda activate cab_gs

# Install PyTorch (adjust CUDA version as needed)
conda install pytorch==2.0.0 torchvision==0.15.0 pytorch-cuda=11.8 -c pytorch -c nvidia

# Install dependencies
pip install plyfile tqdm opencv-python scikit-learn rtree shapely
```

## Step 4: Install Gaussian Splatting Rasterizer

```bash
# Install the differentiable Gaussian rasterizer
cd thirdparty/gaussian-splatting/submodules/diff-gaussian-rasterization
pip install -e .
cd ../../../..

# Optional: Install simple-knn for faster neighbor search
cd thirdparty/gaussian-splatting/submodules/simple-knn
pip install -e .
cd ../../../..
```

If using Option B (manual setup):

```bash
cd ../gaussian-splatting-original/submodules/diff-gaussian-rasterization
pip install -e .
cd ../simple-knn
pip install -e .
cd ../../../CAB-gaussian-splatting
```

## Step 5: Install COLMAP (Optional but Recommended)

For preprocessing your own data:

### Linux:
```bash
sudo apt-get install colmap
```

### macOS:
```bash
brew install colmap
```

### From Source:
Follow instructions at: https://colmap.github.io/install.html

## Step 6: Verify Installation

```bash
# Test imports
python -c "import torch; print(torch.cuda.is_available())"
python -c "from scene import Scene; print('Scene import OK')"
python -c "from gaussian_renderer import render; print('Renderer import OK')"
python -c "import utils; print('Utils import OK')"
```

Expected output:
```
True
Scene import OK
Renderer import OK
Utils import OK
```

## Troubleshooting

### CUDA Version Mismatch

If you get CUDA errors:

```bash
# Check your CUDA version
nvcc --version

# Reinstall PyTorch with correct CUDA version
conda install pytorch torchvision pytorch-cuda=XX.X -c pytorch -c nvidia
```

### Import Errors

If modules are not found:

```bash
# Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

Or add to your `~/.bashrc`:

```bash
echo 'export PYTHONPATH="${PYTHONPATH}:/path/to/CAB-gaussian-splatting"' >> ~/.bashrc
source ~/.bashrc
```

### Rasterizer Compilation Issues

If rasterizer fails to compile:

```bash
# Install build dependencies
conda install -c conda-forge gxx_linux-64=9.5.0  # Linux
# or
xcode-select --install  # macOS

# Retry installation
cd submodules/diff-gaussian-rasterization
pip install -e . --no-build-isolation
```

### RTrees/Shapely Installation Issues

If rtree or shapely fail:

```bash
# Linux
sudo apt-get install libspatialindex-dev libgeos-dev

# macOS
brew install spatialindex geos

# Then reinstall
pip install rtree shapely
```

## Alternative: Docker Installation

We provide a Dockerfile for containerized deployment:

```bash
# Build Docker image
docker build -t cab-gs .

# Run container
docker run --gpus all -it -v $(pwd):/workspace cab-gs
```

(Note: Dockerfile to be added in future release)

## Minimal Installation (Core Functionality Only)

If you only want to use pre-trained models or skip augmentation:

```bash
# Install only core dependencies
conda create -n cab_gs_minimal python=3.9
conda activate cab_gs_minimal
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
pip install plyfile tqdm

# Copy base 3DGS files (as in Step 2)
# Skip scikit-learn, rtree, shapely if not using augmentation
```

## Directory Structure After Installation

```
CAB-gaussian-splatting/
├── thirdparty/
│   └── gaussian-splatting/      # Original 3DGS (if using Option A)
├── scene/                        # Copied from original
├── gaussian_renderer/            # Copied from original
├── arguments/                    # Copied from original
├── utils/                        # Our utilities
├── augment.py                    # Our augmentation code
├── train_partialscheduler.py    # Our training script
├── render.py                     # From original
├── metrics.py                    # From original
└── ...
```

## Next Steps

After installation, see:
- [TRAINING.md](docs/TRAINING.md) for training instructions
- [README.md](README.md) for quick start guide
- [CHANGES.md](CHANGES.md) for implementation details

## Hardware Requirements

**Minimum**:
- GPU: NVIDIA RTX 2080 or equivalent (8GB VRAM)
- RAM: 16GB
- Storage: 50GB free space

**Recommended**:
- GPU: NVIDIA RTX 3090 or better (24GB VRAM)
- RAM: 32GB or more
- Storage: 100GB+ free space (for datasets)

**Tested Configurations**:
- NVIDIA RTX 3090 (24GB) - Full resolution, all features
- NVIDIA RTX 3080 (10GB) - Reduced resolution recommended
- NVIDIA A100 (40GB) - Optimal performance

## Support

For installation issues:
1. Check [Issues](https://github.com/sonbosung/CAB-gaussian-splatting/issues) for similar problems
2. Consult original 3DGS installation guide
3. Open a new issue with your error message and system info

System info to include:
```bash
python --version
nvcc --version
pip list | grep torch
```
