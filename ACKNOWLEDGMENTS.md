# Acknowledgments

## Base Implementation

This work builds upon the excellent **3D Gaussian Splatting** implementation by Kerbl et al.:

- **Original Authors**: Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, George Drettakis
- **Affiliation**: Inria, Université Côte d'Azur, and Max Planck Institut für Informatik
- **Publication**: SIGGRAPH 2023
- **Repository**: https://github.com/graphdeco-inria/gaussian-splatting
- **Paper**: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

We gratefully acknowledge their groundbreaking work in real-time radiance field rendering and their decision to share their implementation with the research community.

## Our Contributions

This repository introduces novel extensions focused on:

1. **SfM Point Augmentation**: A multi-view geometric consistency-based approach to densify sparse Structure-from-Motion reconstructions using quadtree decomposition and visibility analysis.

2. **Co-visibility-based Training**: An adaptive camera grouping and scheduling strategy that leverages scene co-visibility information to improve training efficiency and reconstruction quality.

3. **Additional Geometric Losses**: Integration of depth smoothness and Laplacian pyramid losses for improved geometric consistency.

## Related Work

Our method is inspired by and builds upon concepts from:

- **Structure from Motion**: COLMAP (Schönberger & Frahm, CVPR 2016)
- **Neural Radiance Fields**: NeRF (Mildenhall et al., ECCV 2020)
- **Multi-view Stereo**: Visibility-based reconstruction techniques
- **Training Strategies**: Curriculum learning and progressive training

## Third-Party Libraries

This project uses the following open-source libraries:

- **PyTorch**: Deep learning framework
- **CUDA**: GPU acceleration
- **NumPy & SciPy**: Numerical computing
- **scikit-learn**: Machine learning utilities (clustering)
- **OpenCV**: Computer vision operations
- **COLMAP**: Structure from Motion
- **Pillow**: Image processing
- **tqdm**: Progress bars
- **rtree**: Spatial indexing
- **Shapely**: Computational geometry

## Datasets

Results in this work use the following datasets:

- **Mip-NeRF 360**: Barron et al., CVPR 2022
- **Tanks and Temples**: Knapitsch et al., TOG 2017

We thank the authors for making these datasets publicly available.

## Funding and Support

[Add your funding sources or institutional support here if applicable]

## Community

We thank the 3D vision and graphics community for their continued support and feedback. Special thanks to:

- The original 3DGS authors for technical discussions and support
- Early adopters and testers of this method
- Contributors who have provided feedback and suggestions

## Contact

For questions, suggestions, or collaboration opportunities:

- **Author**: Bosung Sonn
- **Email**: sonbosung@kakao.com
- **GitHub**: https://github.com/sonbosung/CAB-gaussian-splatting

## Citation

If you use this work, please cite both our paper and the original 3D Gaussian Splatting:

```bibtex
@article{sonn2025cab,
  title={SfM Point Augmentation with Scene Co-visibility-based Image Batching for Sharper 3D Gaussian Splatting},
  author={Sonn, Bosung},
  journal={arXiv preprint},
  year={2025}
}

@article{kerbl3Dgaussians,
  author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
  title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
  journal      = {ACM Transactions on Graphics},
  number       = {4},
  volume       = {42},
  year         = {2023},
  url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}
```

---

**License Note**: This work is a derivative of 3D Gaussian Splatting and is subject to the same license restrictions. See [LICENSE.md](LICENSE.md) for details.
