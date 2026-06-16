"""NVIDIA Warp nearest-neighbor particle search prototypes."""

import numpy as np

try:
    import warp as wp
except ImportError:  # pragma: no cover
    wp = None

from cyarray.carray import UIntArray

from pysph.base.warp_device_helper import WarpDeviceHelper


if wp is not None:
    @wp.kernel
    def _copy_i32(src: wp.array(dtype=wp.int32),
                  dst: wp.array(dtype=wp.int32)):
        i = wp.tid()
        dst[i] = src[i]


    @wp.kernel
    def _zero_i32(dst: wp.array(dtype=wp.int32)):
        i = wp.tid()
        dst[i] = wp.int32(0)


    @wp.kernel
    def _neighbor_flags_f64(
            s_x: wp.array(dtype=wp.float64),
            s_y: wp.array(dtype=wp.float64),
            s_z: wp.array(dtype=wp.float64),
            s_h: wp.array(dtype=wp.float64),
            d_x: wp.array(dtype=wp.float64),
            d_y: wp.array(dtype=wp.float64),
            d_z: wp.array(dtype=wp.float64),
            d_h: wp.array(dtype=wp.float64),
            d_idx: wp.int32,
            dim: wp.int32,
            radius_scale: wp.float64,
            flags: wp.array(dtype=wp.uint8),
    ):
        j = wp.tid()
        dx = d_x[d_idx] - s_x[j]
        dy = wp.float64(0.0)
        dz = wp.float64(0.0)
        if dim > 1:
            dy = d_y[d_idx] - s_y[j]
        if dim > 2:
            dz = d_z[d_idx] - s_z[j]
        dist2 = dx*dx + dy*dy + dz*dz
        hi = radius_scale * d_h[d_idx]
        hj = radius_scale * s_h[j]
        if dist2 < hi*hi or dist2 < hj*hj:
            flags[j] = wp.uint8(1)
        else:
            flags[j] = wp.uint8(0)


    @wp.kernel
    def _neighbor_flags_f32(
            s_x: wp.array(dtype=wp.float32),
            s_y: wp.array(dtype=wp.float32),
            s_z: wp.array(dtype=wp.float32),
            s_h: wp.array(dtype=wp.float32),
            d_x: wp.array(dtype=wp.float32),
            d_y: wp.array(dtype=wp.float32),
            d_z: wp.array(dtype=wp.float32),
            d_h: wp.array(dtype=wp.float32),
            d_idx: wp.int32,
            dim: wp.int32,
            radius_scale: wp.float32,
            flags: wp.array(dtype=wp.uint8),
    ):
        j = wp.tid()
        dx = d_x[d_idx] - s_x[j]
        dy = wp.float32(0.0)
        dz = wp.float32(0.0)
        if dim > 1:
            dy = d_y[d_idx] - s_y[j]
        if dim > 2:
            dz = d_z[d_idx] - s_z[j]
        dist2 = dx*dx + dy*dy + dz*dz
        hi = radius_scale * d_h[d_idx]
        hj = radius_scale * s_h[j]
        if dist2 < hi*hi or dist2 < hj*hj:
            flags[j] = wp.uint8(1)
        else:
            flags[j] = wp.uint8(0)


    @wp.kernel
    def _neighbor_lengths_f64(
            s_x: wp.array(dtype=wp.float64),
            s_y: wp.array(dtype=wp.float64),
            s_z: wp.array(dtype=wp.float64),
            s_h: wp.array(dtype=wp.float64),
            d_x: wp.array(dtype=wp.float64),
            d_y: wp.array(dtype=wp.float64),
            d_z: wp.array(dtype=wp.float64),
            d_h: wp.array(dtype=wp.float64),
            nsrc: wp.int32,
            dim: wp.int32,
            radius_scale: wp.float64,
            lengths: wp.array(dtype=wp.int32),
    ):
        i = wp.tid()
        count = wp.int32(0)
        for j in range(nsrc):
            dx = d_x[i] - s_x[j]
            dy = wp.float64(0.0)
            dz = wp.float64(0.0)
            if dim > 1:
                dy = d_y[i] - s_y[j]
            if dim > 2:
                dz = d_z[i] - s_z[j]
            dist2 = dx*dx + dy*dy + dz*dz
            hi = radius_scale * d_h[i]
            hj = radius_scale * s_h[j]
            if dist2 < hi*hi or dist2 < hj*hj:
                count += wp.int32(1)
        lengths[i] = count


    @wp.kernel
    def _neighbor_lengths_f32(
            s_x: wp.array(dtype=wp.float32),
            s_y: wp.array(dtype=wp.float32),
            s_z: wp.array(dtype=wp.float32),
            s_h: wp.array(dtype=wp.float32),
            d_x: wp.array(dtype=wp.float32),
            d_y: wp.array(dtype=wp.float32),
            d_z: wp.array(dtype=wp.float32),
            d_h: wp.array(dtype=wp.float32),
            nsrc: wp.int32,
            dim: wp.int32,
            radius_scale: wp.float32,
            lengths: wp.array(dtype=wp.int32),
    ):
        i = wp.tid()
        count = wp.int32(0)
        for j in range(nsrc):
            dx = d_x[i] - s_x[j]
            dy = wp.float32(0.0)
            dz = wp.float32(0.0)
            if dim > 1:
                dy = d_y[i] - s_y[j]
            if dim > 2:
                dz = d_z[i] - s_z[j]
            dist2 = dx*dx + dy*dy + dz*dz
            hi = radius_scale * d_h[i]
            hj = radius_scale * s_h[j]
            if dist2 < hi*hi or dist2 < hj*hj:
                count += wp.int32(1)
        lengths[i] = count


    @wp.kernel
    def _neighbor_fill_f64(
            s_x: wp.array(dtype=wp.float64),
            s_y: wp.array(dtype=wp.float64),
            s_z: wp.array(dtype=wp.float64),
            s_h: wp.array(dtype=wp.float64),
            d_x: wp.array(dtype=wp.float64),
            d_y: wp.array(dtype=wp.float64),
            d_z: wp.array(dtype=wp.float64),
            d_h: wp.array(dtype=wp.float64),
            starts: wp.array(dtype=wp.int32),
            nsrc: wp.int32,
            dim: wp.int32,
            radius_scale: wp.float64,
            neighbors: wp.array(dtype=wp.uint32),
    ):
        i = wp.tid()
        k = wp.int32(0)
        start = starts[i]
        for j in range(nsrc):
            dx = d_x[i] - s_x[j]
            dy = wp.float64(0.0)
            dz = wp.float64(0.0)
            if dim > 1:
                dy = d_y[i] - s_y[j]
            if dim > 2:
                dz = d_z[i] - s_z[j]
            dist2 = dx*dx + dy*dy + dz*dz
            hi = radius_scale * d_h[i]
            hj = radius_scale * s_h[j]
            if dist2 < hi*hi or dist2 < hj*hj:
                neighbors[start + k] = wp.uint32(j)
                k += wp.int32(1)


    @wp.kernel
    def _neighbor_fill_f32(
            s_x: wp.array(dtype=wp.float32),
            s_y: wp.array(dtype=wp.float32),
            s_z: wp.array(dtype=wp.float32),
            s_h: wp.array(dtype=wp.float32),
            d_x: wp.array(dtype=wp.float32),
            d_y: wp.array(dtype=wp.float32),
            d_z: wp.array(dtype=wp.float32),
            d_h: wp.array(dtype=wp.float32),
            starts: wp.array(dtype=wp.int32),
            nsrc: wp.int32,
            dim: wp.int32,
            radius_scale: wp.float32,
            neighbors: wp.array(dtype=wp.uint32),
    ):
        i = wp.tid()
        k = wp.int32(0)
        start = starts[i]
        for j in range(nsrc):
            dx = d_x[i] - s_x[j]
            dy = wp.float32(0.0)
            dz = wp.float32(0.0)
            if dim > 1:
                dy = d_y[i] - s_y[j]
            if dim > 2:
                dz = d_z[i] - s_z[j]
            dist2 = dx*dx + dy*dy + dz*dz
            hi = radius_scale * d_h[i]
            hj = radius_scale * s_h[j]
            if dist2 < hi*hi or dist2 < hj*hj:
                neighbors[start + k] = wp.uint32(j)
                k += wp.int32(1)


    @wp.kernel
    def _cell_ids_counts_f64(
            x: wp.array(dtype=wp.float64),
            y: wp.array(dtype=wp.float64),
            z: wp.array(dtype=wp.float64),
            xmin: wp.float64,
            ymin: wp.float64,
            zmin: wp.float64,
            cell_size: wp.float64,
            nx: wp.int32,
            ny: wp.int32,
            nz: wp.int32,
            dim: wp.int32,
            cell_ids: wp.array(dtype=wp.int32),
            counts: wp.array(dtype=wp.int32),
    ):
        i = wp.tid()
        ix = wp.int32(wp.floor((x[i] - xmin) / cell_size))
        iy = wp.int32(0)
        iz = wp.int32(0)
        if dim > 1:
            iy = wp.int32(wp.floor((y[i] - ymin) / cell_size))
        if dim > 2:
            iz = wp.int32(wp.floor((z[i] - zmin) / cell_size))
        ix = wp.clamp(ix, wp.int32(0), nx - wp.int32(1))
        iy = wp.clamp(iy, wp.int32(0), ny - wp.int32(1))
        iz = wp.clamp(iz, wp.int32(0), nz - wp.int32(1))
        cid = ix + iy * nx + iz * nx * ny
        cell_ids[i] = cid
        wp.atomic_add(counts, cid, wp.int32(1))


    @wp.kernel
    def _cell_ids_counts_f32(
            x: wp.array(dtype=wp.float32),
            y: wp.array(dtype=wp.float32),
            z: wp.array(dtype=wp.float32),
            xmin: wp.float32,
            ymin: wp.float32,
            zmin: wp.float32,
            cell_size: wp.float32,
            nx: wp.int32,
            ny: wp.int32,
            nz: wp.int32,
            dim: wp.int32,
            cell_ids: wp.array(dtype=wp.int32),
            counts: wp.array(dtype=wp.int32),
    ):
        i = wp.tid()
        ix = wp.int32(wp.floor((x[i] - xmin) / cell_size))
        iy = wp.int32(0)
        iz = wp.int32(0)
        if dim > 1:
            iy = wp.int32(wp.floor((y[i] - ymin) / cell_size))
        if dim > 2:
            iz = wp.int32(wp.floor((z[i] - zmin) / cell_size))
        ix = wp.clamp(ix, wp.int32(0), nx - wp.int32(1))
        iy = wp.clamp(iy, wp.int32(0), ny - wp.int32(1))
        iz = wp.clamp(iz, wp.int32(0), nz - wp.int32(1))
        cid = ix + iy * nx + iz * nx * ny
        cell_ids[i] = cid
        wp.atomic_add(counts, cid, wp.int32(1))


    @wp.kernel
    def _scatter_cell_particles(
            cell_ids: wp.array(dtype=wp.int32),
            cursor: wp.array(dtype=wp.int32),
            cell_particles: wp.array(dtype=wp.uint32),
    ):
        i = wp.tid()
        cid = cell_ids[i]
        out = wp.atomic_add(cursor, cid, wp.int32(1))
        cell_particles[out] = wp.uint32(i)


    @wp.kernel
    def _grid_neighbor_lengths_f64(
            s_x: wp.array(dtype=wp.float64),
            s_y: wp.array(dtype=wp.float64),
            s_z: wp.array(dtype=wp.float64),
            s_h: wp.array(dtype=wp.float64),
            d_x: wp.array(dtype=wp.float64),
            d_y: wp.array(dtype=wp.float64),
            d_z: wp.array(dtype=wp.float64),
            d_h: wp.array(dtype=wp.float64),
            cell_starts: wp.array(dtype=wp.int32),
            cell_counts: wp.array(dtype=wp.int32),
            cell_particles: wp.array(dtype=wp.uint32),
            xmin: wp.float64,
            ymin: wp.float64,
            zmin: wp.float64,
            cell_size: wp.float64,
            nx: wp.int32,
            ny: wp.int32,
            nz: wp.int32,
            ncells: wp.int32,
            dim: wp.int32,
            radius_scale: wp.float64,
            lengths: wp.array(dtype=wp.int32),
    ):
        i = wp.tid()
        ix0 = wp.int32(wp.floor((d_x[i] - xmin) / cell_size))
        iy0 = wp.int32(0)
        iz0 = wp.int32(0)
        if dim > 1:
            iy0 = wp.int32(wp.floor((d_y[i] - ymin) / cell_size))
        if dim > 2:
            iz0 = wp.int32(wp.floor((d_z[i] - zmin) / cell_size))
        count = wp.int32(0)
        for dzc in range(-1, 2):
            for dyc in range(-1, 2):
                for dxc in range(-1, 2):
                    ix = ix0 + wp.int32(dxc)
                    iy = iy0 + wp.int32(dyc)
                    iz = iz0 + wp.int32(dzc)
                    if ix >= 0 and ix < nx and iy >= 0 and iy < ny and iz >= 0 and iz < nz:
                        cid = ix + iy * nx + iz * nx * ny
                        if cid >= 0 and cid < ncells:
                            start = cell_starts[cid]
                            stop = start + cell_counts[cid]
                            for pos in range(start, stop):
                                j = cell_particles[pos]
                                dx = d_x[i] - s_x[j]
                                dy = wp.float64(0.0)
                                dz = wp.float64(0.0)
                                if dim > 1:
                                    dy = d_y[i] - s_y[j]
                                if dim > 2:
                                    dz = d_z[i] - s_z[j]
                                dist2 = dx*dx + dy*dy + dz*dz
                                hi = radius_scale * d_h[i]
                                hj = radius_scale * s_h[j]
                                if dist2 < hi*hi or dist2 < hj*hj:
                                    count += wp.int32(1)
        lengths[i] = count


    @wp.kernel
    def _grid_neighbor_lengths_f32(
            s_x: wp.array(dtype=wp.float32),
            s_y: wp.array(dtype=wp.float32),
            s_z: wp.array(dtype=wp.float32),
            s_h: wp.array(dtype=wp.float32),
            d_x: wp.array(dtype=wp.float32),
            d_y: wp.array(dtype=wp.float32),
            d_z: wp.array(dtype=wp.float32),
            d_h: wp.array(dtype=wp.float32),
            cell_starts: wp.array(dtype=wp.int32),
            cell_counts: wp.array(dtype=wp.int32),
            cell_particles: wp.array(dtype=wp.uint32),
            xmin: wp.float32,
            ymin: wp.float32,
            zmin: wp.float32,
            cell_size: wp.float32,
            nx: wp.int32,
            ny: wp.int32,
            nz: wp.int32,
            ncells: wp.int32,
            dim: wp.int32,
            radius_scale: wp.float32,
            lengths: wp.array(dtype=wp.int32),
    ):
        i = wp.tid()
        ix0 = wp.int32(wp.floor((d_x[i] - xmin) / cell_size))
        iy0 = wp.int32(0)
        iz0 = wp.int32(0)
        if dim > 1:
            iy0 = wp.int32(wp.floor((d_y[i] - ymin) / cell_size))
        if dim > 2:
            iz0 = wp.int32(wp.floor((d_z[i] - zmin) / cell_size))
        count = wp.int32(0)
        for dzc in range(-1, 2):
            for dyc in range(-1, 2):
                for dxc in range(-1, 2):
                    ix = ix0 + wp.int32(dxc)
                    iy = iy0 + wp.int32(dyc)
                    iz = iz0 + wp.int32(dzc)
                    if ix >= 0 and ix < nx and iy >= 0 and iy < ny and iz >= 0 and iz < nz:
                        cid = ix + iy * nx + iz * nx * ny
                        if cid >= 0 and cid < ncells:
                            start = cell_starts[cid]
                            stop = start + cell_counts[cid]
                            for pos in range(start, stop):
                                j = cell_particles[pos]
                                dx = d_x[i] - s_x[j]
                                dy = wp.float32(0.0)
                                dz = wp.float32(0.0)
                                if dim > 1:
                                    dy = d_y[i] - s_y[j]
                                if dim > 2:
                                    dz = d_z[i] - s_z[j]
                                dist2 = dx*dx + dy*dy + dz*dz
                                hi = radius_scale * d_h[i]
                                hj = radius_scale * s_h[j]
                                if dist2 < hi*hi or dist2 < hj*hj:
                                    count += wp.int32(1)
        lengths[i] = count


    @wp.kernel
    def _grid_neighbor_fill_f64(
            s_x: wp.array(dtype=wp.float64),
            s_y: wp.array(dtype=wp.float64),
            s_z: wp.array(dtype=wp.float64),
            s_h: wp.array(dtype=wp.float64),
            d_x: wp.array(dtype=wp.float64),
            d_y: wp.array(dtype=wp.float64),
            d_z: wp.array(dtype=wp.float64),
            d_h: wp.array(dtype=wp.float64),
            cell_starts: wp.array(dtype=wp.int32),
            cell_counts: wp.array(dtype=wp.int32),
            cell_particles: wp.array(dtype=wp.uint32),
            starts: wp.array(dtype=wp.int32),
            xmin: wp.float64,
            ymin: wp.float64,
            zmin: wp.float64,
            cell_size: wp.float64,
            nx: wp.int32,
            ny: wp.int32,
            nz: wp.int32,
            ncells: wp.int32,
            dim: wp.int32,
            radius_scale: wp.float64,
            neighbors: wp.array(dtype=wp.uint32),
    ):
        i = wp.tid()
        ix0 = wp.int32(wp.floor((d_x[i] - xmin) / cell_size))
        iy0 = wp.int32(0)
        iz0 = wp.int32(0)
        if dim > 1:
            iy0 = wp.int32(wp.floor((d_y[i] - ymin) / cell_size))
        if dim > 2:
            iz0 = wp.int32(wp.floor((d_z[i] - zmin) / cell_size))
        k = wp.int32(0)
        out_start = starts[i]
        for dzc in range(-1, 2):
            for dyc in range(-1, 2):
                for dxc in range(-1, 2):
                    ix = ix0 + wp.int32(dxc)
                    iy = iy0 + wp.int32(dyc)
                    iz = iz0 + wp.int32(dzc)
                    if ix >= 0 and ix < nx and iy >= 0 and iy < ny and iz >= 0 and iz < nz:
                        cid = ix + iy * nx + iz * nx * ny
                        if cid >= 0 and cid < ncells:
                            start = cell_starts[cid]
                            stop = start + cell_counts[cid]
                            for pos in range(start, stop):
                                j = cell_particles[pos]
                                dx = d_x[i] - s_x[j]
                                dy = wp.float64(0.0)
                                dz = wp.float64(0.0)
                                if dim > 1:
                                    dy = d_y[i] - s_y[j]
                                if dim > 2:
                                    dz = d_z[i] - s_z[j]
                                dist2 = dx*dx + dy*dy + dz*dz
                                hi = radius_scale * d_h[i]
                                hj = radius_scale * s_h[j]
                                if dist2 < hi*hi or dist2 < hj*hj:
                                    neighbors[out_start + k] = j
                                    k += wp.int32(1)


    @wp.kernel
    def _grid_neighbor_fill_f32(
            s_x: wp.array(dtype=wp.float32),
            s_y: wp.array(dtype=wp.float32),
            s_z: wp.array(dtype=wp.float32),
            s_h: wp.array(dtype=wp.float32),
            d_x: wp.array(dtype=wp.float32),
            d_y: wp.array(dtype=wp.float32),
            d_z: wp.array(dtype=wp.float32),
            d_h: wp.array(dtype=wp.float32),
            cell_starts: wp.array(dtype=wp.int32),
            cell_counts: wp.array(dtype=wp.int32),
            cell_particles: wp.array(dtype=wp.uint32),
            starts: wp.array(dtype=wp.int32),
            xmin: wp.float32,
            ymin: wp.float32,
            zmin: wp.float32,
            cell_size: wp.float32,
            nx: wp.int32,
            ny: wp.int32,
            nz: wp.int32,
            ncells: wp.int32,
            dim: wp.int32,
            radius_scale: wp.float32,
            neighbors: wp.array(dtype=wp.uint32),
    ):
        i = wp.tid()
        ix0 = wp.int32(wp.floor((d_x[i] - xmin) / cell_size))
        iy0 = wp.int32(0)
        iz0 = wp.int32(0)
        if dim > 1:
            iy0 = wp.int32(wp.floor((d_y[i] - ymin) / cell_size))
        if dim > 2:
            iz0 = wp.int32(wp.floor((d_z[i] - zmin) / cell_size))
        k = wp.int32(0)
        out_start = starts[i]
        for dzc in range(-1, 2):
            for dyc in range(-1, 2):
                for dxc in range(-1, 2):
                    ix = ix0 + wp.int32(dxc)
                    iy = iy0 + wp.int32(dyc)
                    iz = iz0 + wp.int32(dzc)
                    if ix >= 0 and ix < nx and iy >= 0 and iy < ny and iz >= 0 and iz < nz:
                        cid = ix + iy * nx + iz * nx * ny
                        if cid >= 0 and cid < ncells:
                            start = cell_starts[cid]
                            stop = start + cell_counts[cid]
                            for pos in range(start, stop):
                                j = cell_particles[pos]
                                dx = d_x[i] - s_x[j]
                                dy = wp.float32(0.0)
                                dz = wp.float32(0.0)
                                if dim > 1:
                                    dy = d_y[i] - s_y[j]
                                if dim > 2:
                                    dz = d_z[i] - s_z[j]
                                dist2 = dx*dx + dy*dy + dz*dz
                                hi = radius_scale * d_h[i]
                                hj = radius_scale * s_h[j]
                                if dist2 < hi*hi or dist2 < hj*hj:
                                    neighbors[out_start + k] = j
                                    k += wp.int32(1)


    @wp.kernel
    def _neighbor_sum_f64(
            values: wp.array(dtype=wp.float64),
            starts: wp.array(dtype=wp.int32),
            lengths: wp.array(dtype=wp.int32),
            neighbors: wp.array(dtype=wp.uint32),
            out: wp.array(dtype=wp.float64),
    ):
        i = wp.tid()
        total = wp.float64(0.0)
        start = starts[i]
        stop = start + lengths[i]
        for pos in range(start, stop):
            j = wp.int32(neighbors[pos])
            total += values[j]
        out[i] = total


    @wp.kernel
    def _neighbor_sum_f32(
            values: wp.array(dtype=wp.float32),
            starts: wp.array(dtype=wp.int32),
            lengths: wp.array(dtype=wp.int32),
            neighbors: wp.array(dtype=wp.uint32),
            out: wp.array(dtype=wp.float32),
    ):
        i = wp.tid()
        total = wp.float32(0.0)
        start = starts[i]
        stop = start + lengths[i]
        for pos in range(start, stop):
            j = wp.int32(neighbors[pos])
            total += values[j]
        out[i] = total


class BruteForceWarpNNPS(object):
    """Brute-force NNPS using Warp arrays for the distance test.

    This class is intentionally small and compatibility-oriented. It provides
    the public query contract used by PySPH's NNPS tests while establishing a
    Warp-native correctness baseline before a cell-list implementation.
    """

    def __init__(self, dim, particles, radius_scale=2.0, ghost_layers=1,
                 domain=None, cache=False, sort_gids=False, backend='warp',
                 device=None):
        if wp is None:  # pragma: no cover
            raise ImportError("warp is required for BruteForceWarpNNPS")
        self.dim = dim
        self.particles = particles
        self.radius_scale = radius_scale
        self.ghost_layers = ghost_layers
        self.domain = domain
        self.use_cache = cache
        self.sort_gids = sort_gids
        self.backend = backend
        self.device = wp.get_device(device)
        self.narrays = len(particles)
        self.src_index = -1
        self.dst_index = -1
        self.src = None
        self.dst = None
        self._flags = {}
        self._cache = {}

        for pa in self.particles:
            if pa.gpu is None or getattr(pa.gpu, 'backend', None) != 'warp':
                pa.set_device_helper(
                    WarpDeviceHelper(pa, backend='warp', device=self.device)
                )

        self.update_domain()
        self.update()

    def update_domain(self):
        if self.domain is not None:
            self.domain.update()

    def update(self, push=True):
        if push:
            for pa in self.particles:
                pa.gpu.push('x', 'y', 'z', 'h')
        self._flags.clear()
        self._cache.clear()

    def set_use_cache(self, use_cache):
        self.use_cache = use_cache

    def set_context(self, src_index, dst_index):
        self.src_index = src_index
        self.dst_index = dst_index
        self.src = self.particles[src_index]
        self.dst = self.particles[dst_index]

    def _get_flags(self, src_index):
        src = self.particles[src_index]
        size = src.gpu.get_number_of_particles()
        flags = self._flags.get(src_index)
        if flags is None or flags.shape[0] != size:
            flags = wp.empty(size, dtype=wp.uint8, device=self.device)
            self._flags[src_index] = flags
        return flags

    def _launch_flags(self, src_index, dst_index, d_idx):
        src = self.particles[src_index].gpu
        dst = self.particles[dst_index].gpu
        flags = self._get_flags(src_index)
        nsrc = src.get_number_of_particles()
        if nsrc == 0:
            return np.array([], dtype=np.uint8)

        if src.x.dtype == np.float32:
            kernel = _neighbor_flags_f32
            radius_scale = np.float32(self.radius_scale)
        else:
            kernel = _neighbor_flags_f64
            radius_scale = np.float64(self.radius_scale)
        wp.launch(
            kernel,
            dim=nsrc,
            inputs=[
                src.x.dev, src.y.dev, src.z.dev, src.h.dev,
                dst.x.dev, dst.y.dev, dst.z.dev, dst.h.dev,
                np.int32(d_idx), np.int32(self.dim), radius_scale, flags
            ],
            device=self.device,
        )
        wp.synchronize_device(self.device)
        return flags.numpy()

    def _kernels_for(self, src):
        if src.x.dtype == np.float32:
            return (
                _neighbor_flags_f32,
                _neighbor_lengths_f32,
                _neighbor_fill_f32,
                np.float32(self.radius_scale),
            )
        else:
            return (
                _neighbor_flags_f64,
                _neighbor_lengths_f64,
                _neighbor_fill_f64,
                np.float64(self.radius_scale),
            )

    def _build_cache(self, src_index, dst_index):
        src = self.particles[src_index].gpu
        dst = self.particles[dst_index].gpu
        nsrc = src.get_number_of_particles()
        ndst = dst.get_number_of_particles()
        lengths = wp.empty(ndst, dtype=wp.int32, device=self.device)
        starts = wp.empty(ndst, dtype=wp.int32, device=self.device)
        _, lengths_kernel, fill_kernel, radius_scale = self._kernels_for(src)

        if ndst > 0:
            wp.launch(
                lengths_kernel,
                dim=ndst,
                inputs=[
                    src.x.dev, src.y.dev, src.z.dev, src.h.dev,
                    dst.x.dev, dst.y.dev, dst.z.dev, dst.h.dev,
                    np.int32(nsrc), np.int32(self.dim), radius_scale, lengths
                ],
                device=self.device,
            )
            wp.utils.array_scan(lengths, starts, inclusive=False)
            wp.synchronize_device(self.device)

        lengths_cpu = lengths.numpy() if ndst > 0 else np.array([], np.int32)
        starts_cpu = starts.numpy() if ndst > 0 else np.array([], np.int32)
        total = 0
        if ndst > 0:
            total = int(starts_cpu[-1] + lengths_cpu[-1])
        neighbors = wp.empty(total, dtype=wp.uint32, device=self.device)
        if ndst > 0 and total > 0:
            wp.launch(
                fill_kernel,
                dim=ndst,
                inputs=[
                    src.x.dev, src.y.dev, src.z.dev, src.h.dev,
                    dst.x.dev, dst.y.dev, dst.z.dev, dst.h.dev,
                    starts, np.int32(nsrc), np.int32(self.dim), radius_scale,
                    neighbors
                ],
                device=self.device,
            )
            wp.synchronize_device(self.device)
        neighbors_cpu = (
            neighbors.numpy() if total > 0 else np.array([], dtype=np.uint32)
        )
        cache = {
            'lengths': lengths_cpu,
            'starts': starts_cpu,
            'neighbors': neighbors_cpu,
        }
        self._cache[(src_index, dst_index)] = cache
        return cache

    def _get_cached_neighbors(self, src_index, dst_index, d_idx):
        cache = self._cache.get((src_index, dst_index))
        if cache is None:
            cache = self._build_cache(src_index, dst_index)
        start = int(cache['starts'][d_idx])
        stop = start + int(cache['lengths'][d_idx])
        return cache['neighbors'][start:stop].astype(np.uint32, copy=False)

    def get_nearest_particles(self, src_index, dst_index, d_idx, nbrs):
        if self.use_cache:
            indices = self._get_cached_neighbors(src_index, dst_index, d_idx)
        else:
            flags = self._launch_flags(src_index, dst_index, d_idx)
            indices = np.nonzero(flags)[0].astype(np.uint32)
        if self.sort_gids and len(indices) > 0:
            gids = self.particles[src_index].properties['gid'].get_npy_array()
            if gids[0] == np.iinfo(np.uint32).max:
                indices.sort()
            else:
                order = np.argsort(gids[indices], kind='stable')
                indices = indices[order].astype(np.uint32)
        nbrs.reset()
        for index in indices:
            nbrs.append(int(index))

    def brute_force_neighbors(self, src_index, dst_index, d_idx, nbrs):
        self.get_nearest_particles(src_index, dst_index, d_idx, nbrs)

    def get_nearest_particles_gpu(self, src_index, dst_index):
        raise NotImplementedError(
            "BruteForceWarpNNPS does not yet build cached GPU neighbor lists"
        )

    def spatially_order_particles(self, pa_index):
        raise NotImplementedError(
            "BruteForceWarpNNPS does not define a spatial ordering"
        )


class UniformGridWarpNNPS(BruteForceWarpNNPS):
    """Uniform-grid Warp NNPS using device-side cell lists.

    The implementation builds a per-source flat cell list on the device and
    then builds cached flat neighbor lists by scanning adjacent cells. It keeps
    the same host-facing `get_nearest_particles()` contract as other PySPH NNPS
    implementations.
    """

    def __init__(self, dim, particles, radius_scale=2.0, ghost_layers=1,
                 domain=None, cache=True, sort_gids=False, backend='warp',
                 device=None):
        self._grid = {}
        self._bounds = None
        self.cell_size = 1.0
        super(UniformGridWarpNNPS, self).__init__(
            dim=dim, particles=particles, radius_scale=radius_scale,
            ghost_layers=ghost_layers, domain=domain, cache=cache,
            sort_gids=sort_gids, backend=backend, device=device
        )
        self.use_cache = True

    def update(self, push=True):
        if push:
            for pa in self.particles:
                pa.gpu.push('x', 'y', 'z', 'h')
        self._flags.clear()
        self._cache.clear()
        self._grid.clear()
        self._compute_bounds_and_cell_size()

    def set_use_cache(self, use_cache):
        if not use_cache:
            raise ValueError("UniformGridWarpNNPS requires cached queries")
        self.use_cache = True

    def _compute_bounds_and_cell_size(self):
        xmin = ymin = zmin = np.inf
        xmax = ymax = zmax = -np.inf
        hmax = 0.0
        for pa in self.particles:
            x = pa.gpu.x.get()
            y = pa.gpu.y.get()
            z = pa.gpu.z.get()
            h = pa.gpu.h.get()
            if len(x) == 0:
                continue
            xmin = min(xmin, float(np.min(x)))
            xmax = max(xmax, float(np.max(x)))
            if self.dim > 1:
                ymin = min(ymin, float(np.min(y)))
                ymax = max(ymax, float(np.max(y)))
            else:
                ymin = ymax = 0.0
            if self.dim > 2:
                zmin = min(zmin, float(np.min(z)))
                zmax = max(zmax, float(np.max(z)))
            else:
                zmin = zmax = 0.0
            hmax = max(hmax, float(np.max(h)))

        if not np.isfinite(xmin):
            xmin = ymin = zmin = -0.5
            xmax = ymax = zmax = 0.5
        self.cell_size = self.radius_scale * hmax
        if self.cell_size <= 1e-14:
            self.cell_size = 1.0

        pad = self.cell_size
        xmin -= pad
        xmax += pad
        ymin -= pad
        ymax += pad
        zmin -= pad
        zmax += pad

        nx = max(1, int(np.ceil((xmax - xmin) / self.cell_size)))
        ny = 1
        nz = 1
        if self.dim > 1:
            ny = max(1, int(np.ceil((ymax - ymin) / self.cell_size)))
        if self.dim > 2:
            nz = max(1, int(np.ceil((zmax - zmin) / self.cell_size)))

        self._bounds = {
            'xmin': xmin, 'ymin': ymin, 'zmin': zmin,
            'nx': nx, 'ny': ny, 'nz': nz,
            'ncells': nx * ny * nz,
        }

    def _scalar(self, value, gpu):
        if gpu.x.dtype == np.float32:
            return np.float32(value)
        return np.float64(value)

    def _grid_kernels_for(self, gpu):
        if gpu.x.dtype == np.float32:
            return (
                _cell_ids_counts_f32,
                _grid_neighbor_lengths_f32,
                _grid_neighbor_fill_f32,
                np.float32(self.radius_scale),
            )
        return (
            _cell_ids_counts_f64,
            _grid_neighbor_lengths_f64,
            _grid_neighbor_fill_f64,
            np.float64(self.radius_scale),
        )

    def _build_grid(self, src_index):
        grid = self._grid.get(src_index)
        if grid is not None:
            return grid

        src = self.particles[src_index].gpu
        nsrc = src.get_number_of_particles()
        bounds = self._bounds
        ncells = bounds['ncells']
        cell_ids = wp.empty(nsrc, dtype=wp.int32, device=self.device)
        counts = wp.empty(ncells, dtype=wp.int32, device=self.device)
        starts = wp.empty(ncells, dtype=wp.int32, device=self.device)
        cursor = wp.empty(ncells, dtype=wp.int32, device=self.device)
        cell_particles = wp.empty(nsrc, dtype=wp.uint32, device=self.device)
        ids_kernel, _, _, _ = self._grid_kernels_for(src)

        if ncells > 0:
            wp.launch(_zero_i32, dim=ncells, inputs=[counts],
                      device=self.device)
        if nsrc > 0:
            wp.launch(
                ids_kernel,
                dim=nsrc,
                inputs=[
                    src.x.dev, src.y.dev, src.z.dev,
                    self._scalar(bounds['xmin'], src),
                    self._scalar(bounds['ymin'], src),
                    self._scalar(bounds['zmin'], src),
                    self._scalar(self.cell_size, src),
                    np.int32(bounds['nx']), np.int32(bounds['ny']),
                    np.int32(bounds['nz']), np.int32(self.dim), cell_ids,
                    counts
                ],
                device=self.device,
            )
        if ncells > 0:
            wp.utils.array_scan(counts, starts, inclusive=False)
            wp.launch(_copy_i32, dim=ncells, inputs=[starts, cursor],
                      device=self.device)
        if nsrc > 0:
            wp.launch(
                _scatter_cell_particles,
                dim=nsrc,
                inputs=[cell_ids, cursor, cell_particles],
                device=self.device,
            )
            wp.synchronize_device(self.device)

        grid = {
            'cell_ids': cell_ids,
            'counts': counts,
            'starts': starts,
            'cell_particles': cell_particles,
        }
        self._grid[src_index] = grid
        return grid

    def build_neighbor_cache_gpu(self, src_index, dst_index):
        """Build and return a device-resident neighbor cache.

        This avoids the host-facing per-particle `UIntArray` query path. A
        small lengths readback is retained to size the flat neighbor array and
        report average neighbor count.
        """
        src = self.particles[src_index].gpu
        dst = self.particles[dst_index].gpu
        grid = self._build_grid(src_index)
        bounds = self._bounds
        ndst = dst.get_number_of_particles()
        lengths = wp.empty(ndst, dtype=wp.int32, device=self.device)
        starts = wp.empty(ndst, dtype=wp.int32, device=self.device)
        _, lengths_kernel, fill_kernel, radius_scale = \
            self._grid_kernels_for(src)

        if ndst > 0:
            wp.launch(
                lengths_kernel,
                dim=ndst,
                inputs=[
                    src.x.dev, src.y.dev, src.z.dev, src.h.dev,
                    dst.x.dev, dst.y.dev, dst.z.dev, dst.h.dev,
                    grid['starts'], grid['counts'], grid['cell_particles'],
                    self._scalar(bounds['xmin'], src),
                    self._scalar(bounds['ymin'], src),
                    self._scalar(bounds['zmin'], src),
                    self._scalar(self.cell_size, src),
                    np.int32(bounds['nx']), np.int32(bounds['ny']),
                    np.int32(bounds['nz']), np.int32(bounds['ncells']),
                    np.int32(self.dim), radius_scale, lengths
                ],
                device=self.device,
            )
            wp.utils.array_scan(lengths, starts, inclusive=False)
            wp.synchronize_device(self.device)

        lengths_cpu = lengths.numpy() if ndst > 0 else np.array([], np.int32)
        total = int(np.sum(lengths_cpu, dtype=np.int64))
        neighbors = wp.empty(total, dtype=wp.uint32, device=self.device)
        if ndst > 0 and total > 0:
            wp.launch(
                fill_kernel,
                dim=ndst,
                inputs=[
                    src.x.dev, src.y.dev, src.z.dev, src.h.dev,
                    dst.x.dev, dst.y.dev, dst.z.dev, dst.h.dev,
                    grid['starts'], grid['counts'], grid['cell_particles'],
                    starts,
                    self._scalar(bounds['xmin'], src),
                    self._scalar(bounds['ymin'], src),
                    self._scalar(bounds['zmin'], src),
                    self._scalar(self.cell_size, src),
                    np.int32(bounds['nx']), np.int32(bounds['ny']),
                    np.int32(bounds['nz']), np.int32(bounds['ncells']),
                    np.int32(self.dim), radius_scale, neighbors
                ],
                device=self.device,
            )
            wp.synchronize_device(self.device)
        return {
            'lengths_dev': lengths,
            'starts_dev': starts,
            'neighbors_dev': neighbors,
            'lengths': lengths_cpu,
            'total_neighbors': total,
        }

    def compute_neighbor_sum(self, src_index, dst_index, prop):
        """Sum a scalar source property over neighbors on the device.

        This is a minimal equation-like consumer for the device-resident
        neighbor cache. It returns one Warp array with a value per destination
        particle and does not materialize per-particle neighbors on the host.
        """
        src_pa = self.particles[src_index]
        dst = self.particles[dst_index].gpu
        if prop not in src_pa.properties:
            raise KeyError("Unknown source particle property: %s" % prop)
        if src_pa.stride.get(prop, 1) != 1:
            raise ValueError(
                "compute_neighbor_sum only supports scalar properties"
            )

        src_pa.gpu.push(prop)
        values = src_pa.gpu.get_device_array(prop)
        cache = self.build_neighbor_cache_gpu(src_index, dst_index)
        ndst = dst.get_number_of_particles()

        if values.dtype == np.float32:
            kernel = _neighbor_sum_f32
            out = wp.empty(ndst, dtype=wp.float32, device=self.device)
        elif values.dtype == np.float64:
            kernel = _neighbor_sum_f64
            out = wp.empty(ndst, dtype=wp.float64, device=self.device)
        else:
            raise TypeError(
                "compute_neighbor_sum only supports float properties"
            )

        if ndst > 0:
            wp.launch(
                kernel,
                dim=ndst,
                inputs=[
                    values.dev, cache['starts_dev'], cache['lengths_dev'],
                    cache['neighbors_dev'], out
                ],
                device=self.device,
            )
            wp.synchronize_device(self.device)
        return out

    def _build_cache(self, src_index, dst_index):
        device_cache = self.build_neighbor_cache_gpu(src_index, dst_index)
        starts = device_cache['starts_dev']
        neighbors = device_cache['neighbors_dev']
        starts_cpu = (
            starts.numpy() if len(device_cache['lengths']) > 0
            else np.array([], np.int32)
        )
        total = device_cache['total_neighbors']
        neighbors_cpu = (
            neighbors.numpy() if total > 0 else np.array([], dtype=np.uint32)
        )
        cache = {
            'lengths': device_cache['lengths'],
            'starts': starts_cpu,
            'neighbors': neighbors_cpu,
        }
        self._cache[(src_index, dst_index)] = cache
        return cache
