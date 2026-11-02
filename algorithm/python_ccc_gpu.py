

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import cupy as cp
import numpy as np
from cupyx import jit



class _RoundingOptions(dict):
    def __init__(self, kernel):
        self.kernel = kernel

    def get(self, key, default=None):
        from cupyx.jit import _compile, _cuda_types
        if key not in self:
            result = _compile.transpile(self.kernel._func, ['extern "C"', '__global__'],
                                        self.kernel._mode, key, _cuda_types.void)
            self[key] = result._replace(options=result.options + ('--fmad=false',))
        return self[key]


@jit.rawkernel()
def _nodes_kernel(sorted_distance, sorted_target, psi, maximum, values, ids):
    v = np.int32(jit.grid(1))
    if v >= values.size:
        return
    best = np.int32(sorted_target[0, v])
    d = sorted_distance[0, v]
    best_cost = 0.5 * (d * d) - psi[best]
    j = 1
    searching = True
    while j < psi.size and searching:
        d = sorted_distance[j, v]
        half = 0.5 * (d * d)
        searching = half - maximum[0] <= best_cost
        target = np.int32(sorted_target[j, v])
        cost = half - psi[target]
        if searching and (cost < best_cost or (cost == best_cost and target < best)):
            best, best_cost = target, cost
        j += 1
    values[v], ids[v] = best_cost, np.uint32(best)


@jit.rawkernel()
def _lookup_kernel(random, cdf, buckets, voxels):
    """Find one CDF bin per sample before evaluating neighboring voxels."""
    sample = np.int32(jit.grid(1))
    if sample >= voxels.size:
        return
    r = random[0, sample]
    voxel = np.int32(r * (cdf.size - 1))
    if r < cdf[voxel] or r >= cdf[voxel + 1]:
        bucket = np.int32(r * 16384.0)
        low = buckets[bucket]
        high = min(buckets[bucket + 1] + 1, np.int32(cdf.size - 1))
        while low + 1 < high:
            mid = (low + high) // 2
            if r >= cdf[mid]:
                low = mid
            else:
                high = mid
        voxel = low
    voxels[sample] = voxel


@jit.rawkernel()
def _assign_kernel(random, voxels, feature, distance, psi, candidates, sizes,
                   singletons, starts, ties, counts, n1, n2, n3, alpha, beta):
   
    thread = np.int32(jit.grid(1))
    sample, corner = thread // 8, thread & 7
    if sample >= random.shape[1]:
        return
    voxel = voxels[sample]
    x = min(max(np.float64(voxel % n1 + 1) + random[1, sample] - 0.5, 1.0), np.float64(n1))
    y = min(max(np.float64(voxel // n1 % n2 + 1) + random[2, sample] - 0.5, 1.0), np.float64(n2))
    z = min(max(np.float64(voxel // (n1 * n2) + 1) + random[3, sample] - 0.5, 1.0), np.float64(n3))
    lx, ly, lz = np.int32(x) - 1, np.int32(y) - 1, np.int32(z) - 1
    ux, uy, uz = min(lx + 1, n1 - 1), min(ly + 1, n2 - 1), min(lz + 1, n3 - 1)
    fx, fy, fz = x - (lx + 1), y - (ly + 1), z - (lz + 1)
    cx = ux if corner & 1 else lx
    cy = uy if corner & 2 else ly
    cz = uz if corner & 4 else lz
    v = cx + cy * n1 + cz * n1 * n2
    best = singletons[v]
    first = jit.shfl_sync(0xffffffff, best, 0, width=8)
    different = np.int32(best < 0 or best != first)
    for shift in jit.range(3, unroll=True):
        different |= jit.shfl_xor_sync(0xffffffff, different, 1 << shift, width=8)
    warp_needs_cost = different | jit.shfl_xor_sync(0xffffffff, different, 8)
    warp_needs_cost |= jit.shfl_xor_sync(0xffffffff, warp_needs_cost, 16)
    if warp_needs_cost:
        original = feature[v]
        sampled = original
        neighbor = jit.shfl_xor_sync(0xffffffff, sampled, 1, width=8)
        if corner % 2 == 0:
            sampled += fx * (neighbor - sampled)
        neighbor = jit.shfl_xor_sync(0xffffffff, sampled, 2, width=8)
        if corner % 4 == 0:
            sampled += fy * (neighbor - sampled)
        neighbor = jit.shfl_xor_sync(0xffffffff, sampled, 4, width=8)
        if corner == 0:
            sampled += fz * (neighbor - sampled)
        sampled = jit.shfl_sync(0xffffffff, sampled, 0, width=8)
        df = beta * (sampled - original)
        local = cp.abs(df)
        if alpha != 0.0:
            px, py, pz = alpha * (x - (cx + 1)), alpha * (y - (cy + 1)), alpha * (z - (cz + 1))
            local = cp.sqrt(px * px + py * py + pz * pz + df * df)
        value = np.float64(0.0)
        if different:
            best, value = _select_target(v, local, distance, psi, starts, candidates, sizes, ties)
        for shift in jit.range(3, unroll=True):
            other_value = jit.shfl_xor_sync(0xffffffff, value, 1 << shift, width=8)
            other_target = jit.shfl_xor_sync(0xffffffff, best, 1 << shift, width=8)
            if other_value < value or (other_value == value and other_target < best):
                value, best = other_value, other_target
    if corner == 0:
        jit.atomic_add(counts, best, np.uint32(1))



@jit.rawkernel()
def _dual_kernel(psi, mass, counts, sample_count, step, lanes):
    for target in range(psi.size):
        empirical = np.float64(counts[target]) / np.float64(sample_count)
        psi[target] += step * (mass[target] - empirical)
    total = np.float64(0.0)
    for lane in range(lanes):
        subtotal = np.float64(0.0)
        for target in range(lane, psi.size, lanes):
            subtotal += psi[target]
        total += subtotal
    mean = total / np.float64(psi.size)
    for target in range(psi.size):
        psi[target] -= mean


@jit.rawkernel(device=True)
def _twist(a, b):
    word = (a & np.uint32(0x80000000)) | (b & np.uint32(0x7fffffff))
    return (word >> np.uint32(1)) ^ ((word & np.uint32(1)) * np.uint32(0x9908b0df))


@jit.rawkernel(device=True)
def _temper(word):
    word ^= word >> np.uint32(11)
    word ^= (word << np.uint32(7)) & np.uint32(0x9d2c5680)
    word ^= (word << np.uint32(15)) & np.uint32(0xefc60000)
    word ^= word >> np.uint32(18)
    return word


@jit.rawkernel()
def _random_kernel(state, position, output):
    words = jit.shared_memory(np.uint32, 1248)
    lane = np.int32(jit.threadIdx.x)
    for i in range(lane, 624, jit.blockDim.x):
        words[i] = state[i]
    jit.syncthreads()
    current = 0
    offset = position[0]
    produced = 0
    while produced < output.size:
        if offset == 624:
            following = 624 - current
            for i in range(lane, 624, jit.blockDim.x):
                word = np.uint32(0)
                if i < 227:
                    word = words[current + i + 397] ^ _twist(words[current + i], words[current + i + 1])
                elif i < 454:
                    word = words[current + i + 170] ^ _twist(words[current + i - 227], words[current + i - 226]) ^ _twist(words[current + i], words[current + i + 1])
                elif i < 623:
                    word = words[current + i - 57] ^ _twist(words[current + i - 454], words[current + i - 453]) ^ _twist(words[current + i - 227], words[current + i - 226]) ^ _twist(words[current + i], words[current + i + 1])
                else:
                    first = words[current + 397] ^ _twist(words[current], words[current + 1])
                    preceding = words[current + 566] ^ _twist(words[current + 169], words[current + 170]) ^ _twist(words[current + 396], words[current + 397])
                    word = preceding ^ _twist(words[current + 623], first)
                words[following + i] = word
            jit.syncthreads()
            current = following
            offset = 0
        count = min(624 - offset, np.int32(output.size) - produced)
        for i in range(lane, count, jit.blockDim.x):
            output[produced + i] = words[current + offset + i]
        jit.syncthreads()
        produced += count
        offset += count
    for i in range(lane, 624, jit.blockDim.x):
        state[i] = words[current + i]
    if lane == 0:
        position[0] = offset


@jit.rawkernel()
def _uniform_kernel(words, output):
    i = np.int32(jit.grid(1))
    if i < output.size:
        high = _temper(words[2 * i]) >> np.uint32(5)
        low = _temper(words[2 * i + 1]) >> np.uint32(6)
        output[i] = (np.float64(high) * 67108864.0 + np.float64(low)) * np.float64(1.0 / 9007199254740992.0)


@jit.rawkernel()
def _primal_kernel(u, xi, fg, bg, output, fc, bc, theta, tau, n1, n2):
    v = np.int32(jit.grid(1))
    if v >= u.size:
        return
    n = u.size
    divergence = 0.0 + xi[v]
    if v % n1 > 0:
        divergence -= xi[v - 1]
    divergence += xi[n + v]
    if v // n1 % n2 > 0:
        divergence -= xi[n + v - n1]
    divergence += xi[2 * n + v]
    if v // (n1 * n2) > 0:
        divergence -= xi[2 * n + v - n1 * n2]
    force = theta * (fg[v] + fc - bg[v] - bc)
    value = u[v] - tau * (force - divergence)
    output[v] = value if value == 0.0 else min(max(value, 0.0), 1.0)


@jit.rawkernel()
def _tv_kernel(u, updated, xi, sigma, gamma, n1, n2, n3):
    v = np.int32(jit.grid(1))
    if v >= u.size:
        return
    n = u.size
    center = 2.0 * updated[v] - u[v]
    gx, gy, gz = np.float64(0.0), np.float64(0.0), np.float64(0.0)
    if v % n1 + 1 < n1:
        gx = (2.0 * updated[v + 1] - u[v + 1]) - center
    if v // n1 % n2 + 1 < n2:
        gy = (2.0 * updated[v + n1] - u[v + n1]) - center
    if v // (n1 * n2) + 1 < n3:
        gz = (2.0 * updated[v + n1 * n2] - u[v + n1 * n2]) - center
    a, b, c = xi[v] + sigma * gx, xi[n + v] + sigma * gy, xi[2 * n + v] + sigma * gz
    scale = max(1.0, cp.sqrt(a * a + b * b + c * c) / gamma)
    xi[v], xi[n + v], xi[2 * n + v] = a / scale, b / scale, c / scale


@jit.rawkernel(device=True)
def _density(value, eta, foreground):
    return value + eta if foreground else (-value + 1.0) + eta


@jit.rawkernel()
def _source_cdf_kernel(u, eta, foreground, cdf):
    tile = jit.shared_memory(np.float64, 1024)
    totals = jit.shared_memory(np.float64, 4)
    lane = np.int32(jit.threadIdx.x)
    subtotal = np.float64(0.0)
    for start in range(0, u.size, 1024):
        count = min(1024, np.int32(u.size) - start)
        for i in range(lane, count, 128):
            tile[i] = _density(u[start + i], eta, foreground)
        jit.syncthreads()
        if lane < 4:
            for i in range(lane, count, 4):
                subtotal += tile[i]
        jit.syncthreads()
    if lane < 4:
        totals[lane] = subtotal
    jit.syncthreads()
    total = totals[0] + totals[1] + totals[2] + totals[3]
    running = np.float64(0.0)
    if lane == 0:
        cdf[0] = 0.0
    for start in range(0, u.size, 1024):
        count = min(1024, np.int32(u.size) - start)
        for i in range(lane, count, 128):
            tile[i] = _density(u[start + i], eta, foreground) / total
        jit.syncthreads()
        if lane == 0:
            for slot in jit.range(1024, unroll=True):
                if slot < count:
                    running += tile[slot]
                    tile[slot] = running
        jit.syncthreads()
        for i in range(lane, count, 128):
            cdf[start + i + 1] = tile[i]
        jit.syncthreads()
    if lane == 0:
        cdf[u.size] = 1.0


@jit.rawkernel(device=True)
def _target_sum(values, mass, weighted, lanes):
    total = np.float64(0.0)
    for lane in range(lanes):
        subtotal = np.float64(0.0)
        for target in range(lane, values.size, lanes):
            value = values[target]
            if weighted:
                value *= mass[target]
            subtotal += value
        total += subtotal
    return total


@jit.rawkernel()
def _center_kernel(psi, mass, lanes):
    mean = _target_sum(psi, mass, False, lanes) / np.float64(psi.size)
    for target in range(psi.size):
        psi[target] -= mean


@jit.rawkernel()
def _constant_kernel(psi, mass, lanes, output):
    output[0] = _target_sum(psi, mass, True, lanes)


@jit.rawkernel(device=True)
def _cost(distance, local, potential):
    total = distance + local
    return 0.5 * (total * total) - potential


@jit.rawkernel()
def _build_hulls_kernel(sorted_distance, sorted_target, distance, psi, maximum,
                        starts, candidates, sizes, singletons, ties):
    v = np.int32(jit.grid(1))
    if v >= sizes.size:
        return
    best = np.int32(sorted_target[0, v])
    best_cost = _cost(sorted_distance[0, v], 0.0, psi[best])
    best_index = 0
    j = 1
    searching = True
    while j < psi.size and searching:
        d = sorted_distance[j, v]
        half = 0.5 * (d * d)
        searching = half - maximum[0] <= best_cost
        target = np.int32(sorted_target[j, v])
        value = half - psi[target]
        if searching and (value < best_cost or (value == best_cost and target < best)):
            best, best_cost, best_index = target, value, j
        j += 1
    candidates[0, v] = best
    starts[0, v] = -np.float64(np.inf)
    count = 1
    tie_write = np.int32(psi.size)
    for j in range(best_index - 1, -1, -1):
        slope = sorted_distance[j, v]
        target = np.int32(sorted_target[j, v])
        intercept = _cost(slope, 0.0, psi[target])
        keep, searching = True, True
        line_start = np.float64(0.0)
        while count > 0 and searching:
            previous = candidates[count - 1, v]
            previous_slope = distance[previous, v]
            previous_intercept = _cost(previous_slope, 0.0, psi[previous])
            if slope == previous_slope:
                if intercept < previous_intercept or (intercept == previous_intercept and target < previous):
                    count -= 1
                else:
                    keep, searching = False, False
            else:
                line_start = (intercept - previous_intercept) / (previous_slope - slope)
                if count == 1 or line_start > starts[count - 1, v]:
                    searching = False
                else:
                    if line_start == starts[count - 1, v]:
                        tie_write -= 1
                        candidates[tie_write, v] = previous
                    count -= 1
        if keep:
            candidates[count, v] = target
            starts[count, v] = line_start
            count += 1
    sizes[v] = count
    ties[v] = psi.size - tie_write
    singletons[v] = candidates[0, v] if count == 1 and tie_write == psi.size else -1


@jit.rawkernel(device=True)
def _select_target(v, local, distance, psi, starts, candidates, sizes, ties):
    low, high = 0, sizes[v]
    while low + 1 < high:
        mid = (low + high) // 2
        if starts[mid, v] <= local:
            low = mid
        else:
            high = mid
    target = candidates[low, v]
    value = _cost(distance[target, v], local, psi[target])
    fallback = False
    if low > 0:
        neighbor = candidates[low - 1, v]
        fallback = _cost(distance[neighbor, v], local, psi[neighbor]) <= value
    if low + 1 < sizes[v]:
        neighbor = candidates[low + 1, v]
        fallback = fallback or _cost(distance[neighbor, v], local, psi[neighbor]) <= value
    for j in range(psi.size - ties[v], psi.size):
        neighbor = candidates[j, v]
        fallback = fallback or _cost(distance[neighbor, v], local, psi[neighbor]) <= value
    if fallback:
        target = 0
        value = _cost(distance[0, v], local, psi[0])
        for j in range(1, psi.size):
            candidate = _cost(distance[j, v], local, psi[j])
            if candidate < value:
                target, value = j, candidate
    return target, value


for _kernel in (
    _nodes_kernel,
    _assign_kernel,
    _lookup_kernel,
    _dual_kernel,
    _random_kernel,
    _uniform_kernel,
    _primal_kernel,
    _tv_kernel,
    _source_cdf_kernel,
    _center_kernel,
    _constant_kernel,
    _build_hulls_kernel,
):
    _kernel._cached_codes = _RoundingOptions(_kernel)


@dataclass(frozen=True)
class ProductionSolverParameters:

    position_alpha: float = 0.05
    feature_beta: float = 32.0
    theta: float = 0.5
    iters: int = 60
    tau: float = 0.25
    sigma: float = 0.25
    gamma: float = 0.1
    eta: float = 1.0e-3
    dual_step: float = 0.5
    dual_iters_init: int = 50
    dual_iters_refine: int = 2
    random_batch_iters: int = 1


# Bound the temporary sorting workspace.
_SAMPLE_BATCH = 65_536


def estimate_gpu_bytes(voxel_count: int, target_counts: list[int]) -> int:
    largest = max(target_counts)
    batch = min(4 * voxel_count, _SAMPLE_BATCH)
    resident = voxel_count * (40 * sum(target_counts) + 336)
    temporary = max(256 * batch * largest, 24 * voxel_count * largest)
    return resident + temporary


class _RandomSamples:

    def __init__(self, sample_count):
        _, words, position, _, _ = np.random.RandomState().get_state()
        self.state = cp.asarray(words)
        self.position = cp.asarray([position], dtype=cp.int32)
        self.values = cp.empty((4, sample_count), dtype=cp.float64)
        self.buffers = [self.values, cp.empty_like(self.values)]
        self.words = cp.empty(8 * sample_count, dtype=cp.uint32)
        self.stream = cp.cuda.Stream(non_blocking=True)
        self.ready = [cp.cuda.Event(disable_timing=True) for _ in range(2)]
        self.consumed = [cp.cuda.Event(disable_timing=True) for _ in range(2)]

    def _fill(self, values):
        _random_kernel[1, 624](self.state, self.position, self.words)
        _uniform_kernel[(values.size + 255) // 256, 256](self.words, values.ravel())

    def generate(self):
        self._fill(self.values)
        return self.values

    def batches(self, count):
        current_stream = cp.cuda.get_current_stream()
        initialized = cp.cuda.Event(disable_timing=True)
        initialized.record(current_stream)
        self.stream.wait_event(initialized)
        with self.stream:
            self._fill(self.buffers[0])
            self.ready[0].record()
        for iteration in range(count):
            index = iteration % 2
            current_stream.wait_event(self.ready[index])
            if iteration + 1 < count:
                other = 1 - index
                self.stream.wait_event(self.consumed[other])
                with self.stream:
                    self._fill(self.buffers[other])
                    self.ready[other].record()
            yield self.buffers[index]
            self.consumed[index].record(current_stream)


class _OTDomainGPUState:
    """Target costs, Monte Carlo mass assignments and warm-started dual state."""

    def __init__(self, shape, distance, target_mass, feature_gpu,
                 parameters: ProductionSolverParameters) -> None:
        self.shape = shape
        self.voxel_count = int(np.prod(shape))
        self.sample_count = 4 * self.voxel_count
        self.parameters = parameters
        mass = np.asarray(target_mass, dtype=np.float64)
        self.target_count = mass.size
        self.target_mass_host = mass / mass.sum()
        self.target_mass = cp.asarray(self.target_mass_host)
        self.distance = cp.asarray(distance, dtype=cp.float64, order="C")
        self.sorted_target = cp.empty(self.distance.shape, dtype=cp.int32)
        self.sorted_distance = cp.empty_like(self.distance)
        for start in range(0, self.voxel_count, _SAMPLE_BATCH):
            stop = min(start + _SAMPLE_BATCH, self.voxel_count)
            block = cp.ascontiguousarray(self.distance[:, start:stop].T)
            order = block.argsort(axis=1)
            self.sorted_target[:, start:stop] = order.T
            self.sorted_distance[:, start:stop] = cp.take_along_axis(block, order, axis=1).T
        self.candidate_target = cp.empty(self.distance.shape, dtype=cp.int32)
        self.hull_starts = cp.empty_like(self.distance)
        self.hull_ties = cp.empty(self.voxel_count, dtype=cp.int32)
        self.candidate_count = cp.empty(self.voxel_count, dtype=cp.int32)
        self.singletons = cp.empty(self.voxel_count, dtype=cp.int32)
        self.feature = feature_gpu
        self.psi = cp.zeros(self.target_count, dtype=cp.float64)
        self.voxels = cp.empty(self.sample_count, dtype=cp.int32)
        self.counts = cp.zeros(self.target_count, dtype=cp.uint32)
        self.node_conv = cp.empty(self.voxel_count, dtype=cp.float64)
        self.node_assignment = cp.empty(self.voxel_count, dtype=cp.uint32)
        self.sum_lanes = 4 if self.target_count >= 44 else 1

    def load_dynamic(self, cdf_edges: np.ndarray) -> None:
        _center_kernel[1, 1](self.psi, self.target_mass, self.sum_lanes)
        self.cdf_edges = cp.asarray(cdf_edges)
        boundaries = cp.arange(16385, dtype=cp.float64) / 16384.0
        self.cdf_buckets = cp.clip(cp.searchsorted(self.cdf_edges, boundaries,
            side="right") - 1, 0, self.voxel_count - 1).astype(cp.int32)

    def _build_candidates(self):
        _build_hulls_kernel[(self.voxel_count + 127) // 128, 128](
            self.sorted_distance, self.sorted_target, self.distance, self.psi,
            self.psi.max().reshape(1), self.hull_starts, self.candidate_target,
            self.candidate_count, self.singletons, self.hull_ties)

    def _assign_samples(self, random_values):
        self.counts.fill(0)
        self._build_candidates()
        _lookup_kernel[(self.sample_count + 255) // 256, 256](
            random_values, self.cdf_edges, self.cdf_buckets, self.voxels)
        _assign_kernel[(self.sample_count * 8 + 255) // 256, 256](
            random_values, self.voxels, self.feature, self.distance, self.psi,
            self.candidate_target, self.candidate_count, self.singletons,
            self.hull_starts, self.hull_ties, self.counts, *self.shape, np.float64(self.parameters.position_alpha),
            np.float64(self.parameters.feature_beta))

    def _update_dual(self):
        _dual_kernel[1, 1](self.psi, self.target_mass, self.counts,
                           self.sample_count, np.float64(self.parameters.dual_step), self.sum_lanes)

    def _evaluate_nodes(self):
        _nodes_kernel[(self.voxel_count + 127) // 128, 128](
            self.sorted_distance, self.sorted_target, self.psi, self.psi.max().reshape(1),
            self.node_conv, self.node_assignment)

    def enqueue(self, cdf_edges, dual_iters, rng) -> None:
        self.load_dynamic(cdf_edges)
        for samples in rng.batches(dual_iters):
            self._assign_samples(samples)
            self._update_dual()
        self._evaluate_nodes()

    def gather_dual_constant(self) -> float:
        value = cp.empty(1, dtype=cp.float64)
        _constant_kernel[1, 1](self.psi, self.target_mass, self.sum_lanes, value)
        return float(value.get()[0])


class _PrimalGPUState:
    """Primal and TV-dual updates with one GPU thread per voxel."""

    def __init__(self, shape, parameters: ProductionSolverParameters) -> None:
        self.shape = shape
        self.voxel_count = int(np.prod(shape))
        self.parameters = parameters
        self.u = cp.ones(self.voxel_count, dtype=cp.float64)
        self.xi = cp.zeros(3 * self.voxel_count, dtype=cp.float64)

    def step(self, foreground_conv, foreground_constant, background_conv,
             background_constant, host_output):
        updated = cp.empty_like(self.u)
        blocks = (self.voxel_count + 255) // 256
        p = self.parameters
        _primal_kernel[blocks, 256](self.u, self.xi, foreground_conv, background_conv,
            updated, np.float64(foreground_constant), np.float64(background_constant),
            np.float64(p.theta), np.float64(p.tau), *self.shape[:2])
        _tv_kernel[blocks, 256](self.u, updated, self.xi, np.float64(p.sigma),
                               np.float64(p.gamma), *self.shape)
        self.u = updated
        self.u.get(out=host_output, blocking=True)
        return host_output


def _pinned_float64_array(element_count: int) -> tuple[Any, np.ndarray]:
    owner = cp.cuda.alloc_pinned_memory(int(element_count) * 8)
    array = np.frombuffer(owner, dtype=np.float64, count=int(element_count))
    return owner, array


class _SourceCDFWorkspace:

    def __init__(self, voxel_count):
        self.foreground_cdf = cp.empty(voxel_count + 1, dtype=cp.float64)
        self.background_cdf = cp.empty_like(self.foreground_cdf)

    def build(self, u_flat, eta, foreground):
        output = self.foreground_cdf if foreground else self.background_cdf
        _source_cdf_kernel[1, 128](cp.asarray(u_flat), np.float64(eta), foreground, output)
        return output
