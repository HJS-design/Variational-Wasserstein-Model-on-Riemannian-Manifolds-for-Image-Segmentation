# Variational Wasserstein Model on Riemannian Manifolds for Image Segmentation

## Environment

Representing a Riemannian manifold and computing optimal transport (OT) on it are both computationally expensive. Therefore, we use CuPy with CUDA to accelerate these computations. An NVIDIA GPU is required to run the code. You can use Codex to read the code and install all required dependencies automatically.

## Run a case

From this directory:

```bash
python ACDC/seg.py
python RESECT/seg.py
python BraTS/seg.py
```
