# Setup Notes for Repository Maintainers

This file documents the repository structure and what needs to be done to make it fully functional.

## Current Status

✅ **Completed**:
- Core novel implementation files (augment.py, train_partialscheduler.py)
- Custom utilities (utils/ directory with augmentation and scheduling code)
- Complete documentation (README, LICENSE, CHANGES, TRAINING guide)
- Evaluation scripts
- .gitignore and basic configuration

⚠️ **Required Before Public Release**:

### 1. Add Original 3DGS Base Files

The following directories/files from the original 3D Gaussian Splatting repo need to be included:

```bash
# Option A: As git submodule (recommended)
git submodule add https://github.com/graphdeco-inria/gaussian-splatting.git thirdparty/gaussian-splatting
git submodule update --init --recursive

# Then copy necessary files:
cp -r thirdparty/gaussian-splatting/scene .
cp -r thirdparty/gaussian-splatting/gaussian_renderer .
cp -r thirdparty/gaussian-splatting/arguments .
cp -r thirdparty/gaussian-splatting/submodules .
```

### 2. Update Import Paths

Check all Python files for import statements and ensure they work with new structure:

Files to check:
- `train_partialscheduler.py` - imports from scene, gaussian_renderer, arguments
- `augment.py` - imports from utils
- `render.py` - imports from scene, gaussian_renderer
- `metrics.py` - imports from utils

### 3. Create environment.yml

Currently copied from original repo. Verify it includes all dependencies:
- torch, torchvision
- plyfile, tqdm
- opencv-python
- scikit-learn (for clustering)
- rtree, shapely (for augmentation)
- numpy, scipy

### 4. Test Installation

Before pushing, test complete installation on clean machine:

```bash
# Clone repo
git clone https://github.com/sonbosung/CAB-gaussian-splatting.git
cd CAB-gaussian-splatting

# Follow INSTALLATION.md
# Run a simple test scene
```

### 5. Add Example Data/Scripts

Consider adding:
- Small example dataset for testing
- Pre-computed augmented points for demo
- Jupyter notebook with visualization

## File Organization

### Files from Original 3DGS (need to be included):
- `scene/` - Scene management, camera handling
- `gaussian_renderer/` - CUDA rasterization code
- `arguments/` - Argument parsers
- `submodules/` - Differential Gaussian rasterization
- `render.py` - Rendering script ✅ (copied)
- `metrics.py` - Evaluation metrics ✅ (copied)
- `train.py` - Base training (for reference)

### Our Novel Files (already included):
- `augment.py` - Point augmentation ✅
- `train_partialscheduler.py` - Training with scheduler ✅
- `eval_covis_augment_partialscheduler.sh` - Evaluation pipeline ✅
- `utils/aug_utils.py` - Augmentation utilities ✅
- `utils/bundle_utils.py` - Co-visibility analysis ✅
- `utils/scheduler_utils.py` - Training scheduler ✅
- `utils/colmap_utils.py` - COLMAP data handling ✅
- `utils/experiment_utils.py` - Experiment logging ✅

### Documentation (already included):
- `README.md` ✅
- `LICENSE.md` ✅
- `CHANGES.md` ✅
- `ACKNOWLEDGMENTS.md` ✅
- `INSTALLATION.md` ✅
- `docs/TRAINING.md` ✅

## Known Issues to Address

1. **Import Path Dependencies**: 
   - Some utils files may have absolute paths that need to be made relative
   - Check `sys.path.append()` calls in utils/

2. **COLMAP Integration**:
   - `augment.py` imports from `colmap.scripts.python.read_write_model`
   - Need to ensure COLMAP python bindings are available or include the scripts

3. **Hardcoded Paths**:
   - Check for any hardcoded paths in scripts (especially eval_covis_augment_partialscheduler.sh)
   - Update to use relative paths or environment variables

## Pre-Release Checklist

- [ ] Add 3DGS base files (as submodule or copy)
- [ ] Test installation from scratch
- [ ] Run complete pipeline on sample scene
- [ ] Update README with actual results/images
- [ ] Add badges (license, paper status, etc.)
- [ ] Create GitHub Actions for CI (optional)
- [ ] Add CONTRIBUTING.md if accepting contributions
- [ ] Create issue templates
- [ ] Add pre-trained model links (if sharing models)
- [ ] Create project website/page (optional)
- [ ] Submit paper to arXiv
- [ ] Update citation in README with arXiv link

## GitHub Repository Setup

```bash
# After completing above steps:
cd /mnt/disk2/auggs/CAB-gaussian-splatting

# Add all files
git add .

# Initial commit
git commit -m "Initial commit: CAB-GS implementation

- SfM point augmentation with co-visibility analysis
- Partial scheduler for adaptive training
- Complete documentation and examples
"

# Add remote
git remote add origin https://github.com/sonbosung/CAB-gaussian-splatting.git

# Push
git branch -M main
git push -u origin main
```

## Post-Release Tasks

1. **Monitor Issues**: Respond to user issues and questions
2. **Update Documentation**: Based on user feedback
3. **Add Examples**: More datasets, configurations
4. **Performance Optimization**: Profile and optimize bottlenecks
5. **Paper Release**: Update with arXiv link and final citation

## Contact for Questions

Maintainer: Bosung Sonn (sonbosung@kakao.com)

## Notes on License Compliance

✅ Original license preserved (LICENSE.md)
✅ Copyright notice added for modifications
✅ Clear attribution in README and ACKNOWLEDGMENTS
✅ Citation information provided
✅ Non-commercial restriction maintained

The repository structure follows the guidelines in NEW_REPO_GUIDE.md from the original development repo.
