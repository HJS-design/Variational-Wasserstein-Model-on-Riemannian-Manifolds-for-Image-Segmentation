# Variational Wasserstein Model on Riemannian Manifolds for Image Segmentation



## Environment

Representation of Riemannian manifold, calculation of OT on it is both expensive. Therefore we use Cupy with Cuda to speed up. Please make sure you have a GPU, otherwise the code can not run. You can use Codex to rad the code and install all dependency automatically.


## Run a case

From this directory:

```bash
python ACDC/seg.py
python resect/seg.py
python brats/seg.py
```


