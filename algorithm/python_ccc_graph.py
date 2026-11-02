
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
import multiprocessing
from multiprocessing.shared_memory import SharedMemory
import os

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse
from scipy.sparse import csgraph

FloatArray = NDArray[np.float64]

@dataclass(slots=True)
class OTGraphGeometry:
    """Static data shared by foreground and background graph packages."""

    spatial_size: tuple[int, int, int]
    voxel_count: int
    label_count: int
    alpha: float
    beta: float
    labels_zero_f: NDArray[np.int32]
    scaled_image_feature_f: FloatArray
    target_features_full: FloatArray
    target_spatial_full: FloatArray
    region_counts: NDArray[np.int64]
    voxel_graph: sparse.csr_matrix


@dataclass(slots=True)
class OTDomainArrays:
    """Compact integration view consumed by the Python GPU solver."""

    distance: FloatArray
    mass: FloatArray
    active_ids: NDArray[np.int64]


@dataclass(slots=True)
class OTForegroundBackgroundGraphs:
    """Foreground/background result from one shared geometry build."""

    foreground: OTDomainArrays
    background: OTDomainArrays


def _region_means(labels: NDArray[np.int32], values: FloatArray,
                  counts: NDArray[np.int64]) -> FloatArray:
    return np.bincount(labels, weights=values, minlength=counts.size) / counts


def _region_feature_means(labels: NDArray[np.int32], values: FloatArray,
                          counts: NDArray[np.int64]) -> FloatArray:
    
    order = np.argsort(labels, kind="stable")
    starts = np.cumsum(counts) - counts
    positions = np.arange(labels.size) - np.repeat(starts, counts)
    groups = 4 * labels[order] + positions % 4
    sums = np.bincount(groups, weights=values[order], minlength=4 * counts.size)
    return sums.reshape(-1, 4).sum(axis=1) / counts


def _axis_edge_weights(feature: FloatArray, coordinates: tuple[FloatArray, ...],
                       axis: int) -> FloatArray:
    squared = np.diff(feature, axis=axis)
    np.square(squared, out=squared)
    shape = [1, 1, 1]
    shape[axis] = feature.shape[axis] - 1
    squared += np.square(np.diff(coordinates[axis])).reshape(shape)
    np.sqrt(squared, out=squared)
    return squared


def _insert_undirected_edges(
    source: NDArray[np.int32],
    offset: int,
    weights: FloatArray,
    cursor: NDArray[np.int32],
    indices: NDArray[np.int32],
    data: FloatArray,
) -> None:
    source = np.asarray(source, dtype=np.int32)
    weights = np.asarray(weights, dtype=np.float64)

    target = source + np.int32(offset)
    positions = cursor[source]
    indices[positions] = target
    data[positions] = weights
    cursor[source] = positions + 1

    positions = cursor[target]
    indices[positions] = source
    data[positions] = weights
    cursor[target] = positions + 1


def _build_voxel_graph(
    scaled_image_feature_f: FloatArray,
    scaled_coordinates: tuple[FloatArray, FloatArray, FloatArray],
) -> sparse.csr_matrix:
    """Build a symmetric six-neighbour CSR without a memory-heavy COO."""

    spatial_size = tuple(int(value) for value in scaled_image_feature_f.shape)
    n1, n2, n3 = spatial_size
    voxel_count = int(np.prod(spatial_size, dtype=np.int64))
    undirected_count = (
        (n1 - 1) * n2 * n3
        + n1 * (n2 - 1) * n3
        + n1 * n2 * (n3 - 1)
    )
    directed_count = 2 * undirected_count

    degree = np.full(spatial_size, 6, dtype=np.uint8, order="F")
    for axis, length in enumerate(spatial_size):
        first = [slice(None)] * 3
        last = [slice(None)] * 3
        first[axis] = 0
        last[axis] = length - 1
        degree[tuple(first)] -= 1
        degree[tuple(last)] -= 1
    degree_f = degree.ravel(order="F")

    indptr = np.empty(voxel_count + 1, dtype=np.int32)
    indptr[0] = 0
    np.cumsum(degree_f, dtype=np.int64, out=indptr[1:])

    indices = np.empty(directed_count, dtype=np.int32)
    data = np.empty(directed_count, dtype=np.float64)
    cursor = indptr[:-1].copy()
    node_ids = np.arange(voxel_count, dtype=np.int32).reshape(spatial_size, order="F")
    offsets = (1, n1, n1 * n2)

    for axis, length in enumerate(spatial_size):
        source_slice = [slice(None)] * 3
        source_slice[axis] = slice(0, -1)
        source = np.asarray(node_ids[tuple(source_slice)].ravel(order="F"), dtype=np.int32)
        weights = _axis_edge_weights(
            scaled_image_feature_f, scaled_coordinates, axis
        )
        weights_f = np.asarray(weights.ravel(order="F"), dtype=np.float64)
        _insert_undirected_edges(
            source,
            offsets[axis],
            weights_f,
            cursor,
            indices,
            data,
        )
        # Do not retain one axis's large temporary vectors while allocating
        # those for the next axis.
        del source, weights, weights_f


    graph = sparse.csr_matrix(
        (data, indices, indptr),
        shape=(voxel_count, voxel_count),
        dtype=np.float64,
        copy=False,
    )
    # Sorting six entries per row is inexpensive and avoids a conversion in
    # downstream SciPy graph routines.  Explicit zero-weight edges are kept.
    graph.sort_indices()
    return graph


def build_graph_geometry(smoothed_image: ArrayLike, regions: ArrayLike,
                         alpha: float, beta: float) -> OTGraphGeometry:
    image = np.asfortranarray(smoothed_image, dtype=np.float64)
    regions = np.asarray(regions, dtype=np.int32)
    shape = image.shape
    labels = regions.ravel(order="F") - 1
    counts = np.bincount(labels)
    coordinates = tuple(np.arange(1, size + 1, dtype=np.float64) for size in shape)
    scaled_coordinates = tuple(alpha * c for c in coordinates)
    feature = beta * image
    spatial = np.empty((counts.size, 3), dtype=np.float64)
    for axis, coordinate in enumerate(coordinates):
        component_shape = [1, 1, 1]
        component_shape[axis] = shape[axis]
        component = np.broadcast_to(coordinate.reshape(component_shape), shape)
        spatial[:, axis] = _region_means(labels, component.ravel(order="F"), counts)
    features = np.column_stack((alpha * spatial,
                                _region_feature_means(labels, feature.ravel(order="F"), counts)))
    return OTGraphGeometry(
        spatial_size=shape, voxel_count=image.size, label_count=counts.size,
        alpha=alpha, beta=beta, labels_zero_f=labels,
        scaled_image_feature_f=feature, target_features_full=features,
        target_spatial_full=spatial, region_counts=counts,
        voxel_graph=_build_voxel_graph(feature, scaled_coordinates),
    )


def _target_attachments(geometry: OTGraphGeometry, active_zero: NDArray[np.int64]
                        ) -> tuple[NDArray[np.int32], NDArray[np.int32], FloatArray]:
    """Connect each target to the unique corners of its enclosing voxel cell."""
    shape = np.asarray(geometry.spatial_size)
    spatial = geometry.target_spatial_full[active_zero]
    lower = np.clip(np.floor(spatial).astype(np.int64), 1, shape)
    upper = np.minimum(lower + 1, shape)
    offsets = (np.arange(8)[:, None] >> np.arange(3)) & 1
    corners = np.where(offsets[None, :, :], upper[:, None, :], lower[:, None, :])
    ids = np.ravel_multi_index((corners - 1).reshape(-1, 3).T, shape, order="F")
    targets = np.repeat(np.arange(active_zero.size), 8)

    # A boundary cell can repeat a corner. Remove duplicates before CSR
    # construction, which would otherwise SUM repeated edge weights.
    _, unique = np.unique(targets * geometry.voxel_count + ids, return_index=True)
    ids, targets = ids[unique], targets[unique]
    corners = corners.reshape(-1, 3)[unique]
    image_values = geometry.scaled_image_feature_f.ravel(order="F")[ids]
    corner_features = np.column_stack((geometry.alpha * corners, image_values))
    delta = geometry.target_features_full[active_zero[targets]] - corner_features
    weights = np.sqrt(np.square(delta).sum(axis=1))
    return ids.astype(np.int32), targets.astype(np.int32), weights


@contextmanager
def _shared_array(shape, dtype, name=None):
    """Own a shared array, or attach to one owned by the parent process."""
    dtype = np.dtype(dtype)
    shared = SharedMemory(name=name, create=name is None,
                          size=int(np.prod(shape)) * dtype.itemsize if name is None else 0)
    array = np.ndarray(shape, dtype=dtype, buffer=shared.buf)
    try:
        yield (shared.name, shape, dtype.str), array
    finally:
        del array
        shared.close()
        if name is None:
            shared.unlink()


def _dijkstra_worker(graph_specs, graph_shape, sources, output_spec, row_start):
    """Solve independent sources and write their rows without pickling arrays."""
    with ExitStack() as stack:
        arrays = []
        for name, shape, dtype in graph_specs:
            _, array = stack.enter_context(_shared_array(shape, dtype, name))
            array.flags.writeable = False
            arrays.append(array)
        name, shape, dtype = output_spec
        _, output = stack.enter_context(_shared_array(shape, dtype, name))
        graph = sparse.csr_matrix(tuple(arrays), shape=graph_shape, copy=False)
        distances = csgraph.dijkstra(graph, directed=True, indices=sources)
        output[row_start:row_start + sources.size] = distances[:, :output.shape[1]]


def _target_distances(graph, sources, voxel_count, *, workers=None):
    
    if workers is None:
        workers = min(4, os.cpu_count() or 1) if sources.size * voxel_count >= 100_000_000 else 1
    workers = min(workers, sources.size)
    if workers <= 1:
        distances = csgraph.dijkstra(graph, directed=True, indices=sources)
        return np.ascontiguousarray(distances[:, :voxel_count])

    with ExitStack() as stack:
        graph_specs = []
        for array in (graph.data, graph.indices, graph.indptr):
            spec, shared = stack.enter_context(_shared_array(array.shape, array.dtype))
            shared[:] = array
            graph_specs.append(spec)
        output_spec, output = stack.enter_context(
            _shared_array((sources.size, voxel_count), np.float64))
        with ProcessPoolExecutor(max_workers=workers,
                                 mp_context=multiprocessing.get_context("spawn")) as executor:
            futures = []
            row_start = 0
            for chunk in np.array_split(sources, workers):
                futures.append(executor.submit(_dijkstra_worker, graph_specs, graph.shape,
                                               chunk, output_spec, row_start))
                row_start += chunk.size
            for future in futures:
                future.result()
        return output.copy()


def _prepare_domain(geometry: OTGraphGeometry, target_mask: NDArray[np.bool_]
                    ) -> OTDomainArrays:
    selected = geometry.labels_zero_f[target_mask.ravel(order="F")]
    counts = np.bincount(selected, minlength=geometry.label_count)
    active = np.flatnonzero(counts)
    rows, columns, weights = _target_attachments(geometry, active)
    attachments = sparse.csr_matrix(
        (weights, (rows, columns)), shape=(geometry.voxel_count, active.size))
    # Explicit zero-cost edges are real connections and must be retained.
    graph = sparse.bmat(((geometry.voxel_graph, attachments),
                         (attachments.T, None)), format="csr")
    graph.sort_indices()
    sources = np.arange(geometry.voxel_count, graph.shape[0], dtype=np.int32)
    distances = _target_distances(graph, sources, geometry.voxel_count)
    return OTDomainArrays(
        distance=distances,
        mass=counts[active].astype(np.float64), active_ids=active + 1,
    )


def prepare_domain_graphs(smoothed_image: ArrayLike, regions: ArrayLike,
                          seeds: ArrayLike, *, position_alpha: float,
                          feature_beta: float) -> OTForegroundBackgroundGraphs:
    """Build the shared graph once, then compute both domains' distances.

    Masses are raw seed counts; the GPU solver normalizes them once.
    """
    geometry = build_graph_geometry(smoothed_image, regions, position_alpha, feature_beta)
    seeds = np.asarray(seeds)
    return OTForegroundBackgroundGraphs(
        foreground=_prepare_domain(geometry, seeds == 1),
        background=_prepare_domain(geometry, seeds == 2),
    )
