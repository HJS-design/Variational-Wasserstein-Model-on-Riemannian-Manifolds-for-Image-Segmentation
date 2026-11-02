
IMAGE_FILE = 'img.nii'
SEEDS_FILE = 'seeds.nii'
FEATURE_BETA = 30
FEATURE_SIGMA = 1.55
POSITION_ALPHA = 0.01
GAMMA = 1
PRIMAL_STEP = 0.1
TV_DUAL_STEP = 0.15
SUPERPIXEL_COUNT = 350
ITERATIONS = 44

import os
import sys
from pathlib import Path

CASE_ROOT = Path(__file__).resolve().parent


os.environ.update(
    OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
    PYTHONDONTWRITEBYTECODE="1",
)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.dont_write_bytecode = True
sys.path.insert(0, str(CASE_ROOT.parent))

from algorithm.segmentation import segment


def main():
    import itk
    import numpy as np

    image = itk.imread(str(CASE_ROOT / IMAGE_FILE))
    seeds = itk.imread(str(CASE_ROOT / SEEDS_FILE))
    image_array = itk.array_from_image(image).transpose(2, 1, 0)
    seeds_array = itk.array_from_image(seeds).transpose(2, 1, 0)
    parameters = dict(
        feature_beta=FEATURE_BETA, position_alpha=POSITION_ALPHA, gamma=GAMMA,
        superpixel_count=SUPERPIXEL_COUNT, iterations=ITERATIONS,
        feature_sigma=FEATURE_SIGMA, primal_step=PRIMAL_STEP, tv_dual_step=TV_DUAL_STEP,
    )
    output = CASE_ROOT / "output"
    output.mkdir(parents=True, exist_ok=True)
    u = segment(image_array, seeds_array, **parameters)

    mask = itk.image_from_array(np.ascontiguousarray((u > 0.5).astype(np.uint8).transpose(2, 1, 0)))
    mask.CopyInformation(image)
    itk.imwrite(mask, str(output / "segmentation.nii.gz"))

    ground_truth = itk.imread(str(CASE_ROOT / "gt.nii.gz"))
    truth = itk.array_from_image(ground_truth) > 0
    prediction = itk.array_from_image(mask) > 0
    total = int(np.count_nonzero(prediction)) + int(np.count_nonzero(truth))
    intersection = int(np.count_nonzero(prediction & truth))
    dice = 2.0 * intersection / total if total else 1.0
    print(f"Dice: {dice:.6f}")
    return u


if __name__ == "__main__":
    main()
