# License

## Modifications and Enhancements

Modifications and Enhancements Copyright (c) 2025 Bosung Sonn

This work is a derivative of 3D Gaussian Splatting, which is subject to the license below. All modifications are subject to the same license terms.

**Novel Contributions by Bosung Sonn:**
- SfM point augmentation strategy (`augment.py`)
- Co-visibility-based camera grouping and training scheduling (`utils/bundle_utils.py`, `utils/scheduler_utils.py`)
- Partial scheduler training approach (`train_partialscheduler.py`)
- Augmentation utilities for multi-view consistency (`utils/aug_utils.py`)
- Geometric loss functions and evaluation tools

---

## Original 3D Gaussian Splatting License

Gaussian-Splatting Software License
Version 1.0, 27 July 2023

Copyright (C) 2023, Inria and the Max Planck Institut for Informatik (MPII)
All rights reserved.

The Software (defined below) is free to use for both academic and commercial purposes only for non-commercial research and evaluation purposes. Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

4. The source and binary forms and any modifications must only be used for research and evaluation purposes in the field of 3D scene reconstruction and rendering. Commercial use of the source and binary forms, or any modifications thereof, requires explicit permission from the copyright holders.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

In case you use this software in a scientific publication, please cite:

```bibtex
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

For inquiries contact george.drettakis@inria.fr

The above license applies to the original 3D Gaussian Splatting code. This derivative work maintains the same restrictions and adds novel contributions as listed above.
