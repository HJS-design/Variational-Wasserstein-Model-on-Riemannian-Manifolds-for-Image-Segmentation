"""Array interface for the CuPy segmentation pipeline."""

import cupy as cp
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.exposure import rescale_intensity
from skimage.segmentation import slic

from . import python_ccc_gpu as gpu
from .python_ccc_graph import prepare_domain_graphs


def segment(
    image: np.ndarray,
    seeds: np.ndarray,
    feature_beta: float,
    position_alpha: float,
    gamma: float,
    superpixel_count: int,
    iterations: int,
    feature_sigma: float = 1.0,
    primal_step: float = 0.25,
    tv_dual_step: float = 0.25,
) -> np.ndarray:
 

    # Preserve the established SLIC axis order and feature preprocessing.
    regions = slic(
        np.ascontiguousarray(image.transpose(0, 2, 1)),
        n_segments=superpixel_count,
        compactness=0.1,
        start_label=1,
        channel_axis=None,
    ).transpose(0, 2, 1)
    feature = gaussian_filter(
        rescale_intensity(np.asarray(image, dtype=np.float64), out_range=(0., 1.)),
        sigma=feature_sigma,
    )
    graphs = prepare_domain_graphs(
        feature, regions, seeds,
        position_alpha=position_alpha, feature_beta=feature_beta,
    )
    parameters = gpu.ProductionSolverParameters(
        position_alpha=position_alpha, feature_beta=feature_beta, gamma=gamma,
        iters=iterations, theta=1.0, tau=primal_step, sigma=tv_dual_step,
        eta=1.0e-3, dual_step=0.5, dual_iters_init=300, dual_iters_refine=3,
    )
    shape = feature.shape
    voxel_count = feature.size
    feature_device = cp.asarray(np.ascontiguousarray(feature.ravel(order="F")))
    # Allocate the larger domain first, as in the existing solver.
    ordered = sorted(
        (("foreground", graphs.foreground), ("background", graphs.background)),
        key=lambda item: item[1].mass.size, reverse=True,
    )
    domains = {
        name: gpu._OTDomainGPUState(
            shape, domain.distance, domain.mass, feature_device, parameters,
        )
        for name, domain in ordered
    }
    foreground = domains["foreground"]
    background = domains["background"]
    rng = gpu._RandomSamples(sample_count=4 * voxel_count)
    primal = gpu._PrimalGPUState(shape, parameters)
    cdf = gpu._SourceCDFWorkspace(voxel_count)
    owner, u = gpu._pinned_float64_array(voxel_count)
    u.fill(1.0)

    for iteration in range(iterations):
        dual_iterations = parameters.dual_iters_init if iteration == 0 else parameters.dual_iters_refine
        foreground.enqueue(cdf.build(u, parameters.eta, foreground=True), dual_iterations, rng)
        background.enqueue(cdf.build(u, parameters.eta, foreground=False), dual_iterations, rng)
        background_constant = background.gather_dual_constant()
        foreground_constant = foreground.gather_dual_constant()
        u = primal.step(
            foreground.node_conv, foreground_constant,
            background.node_conv, background_constant, host_output=u,
        )

    return u.reshape(shape, order="F").copy()
