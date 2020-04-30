#cython: embedsignature=True
import numpy as np
cimport numpy as np
from pysph.base.tree.point_tree import PointTree

cdef class OctreeGPUNNPS(GPUNNPS):
    def __init__(self, int dim, list particles, double radius_scale=2.0,
                 int ghost_layers=1, domain=None, bint fixed_h=False,
                 bint cache=True, bint sort_gids=False,
                 allow_sort=False, leaf_size=32,
                 bint use_elementwise=False, bint use_partitions=False,
                 backend=None):
        GPUNNPS.__init__(
            self, dim, particles, radius_scale, ghost_layers, domain,
            cache, sort_gids, backend=backend
        )

        self.src_index = -1
        self.dst_index = -1
        self.sort_gids = sort_gids
        self.leaf_size = leaf_size

        cdef NNPSParticleArrayWrapper pa_wrapper
        cdef int i, num_particles

        self.octrees = []

        if self.use_double:
            from warnings import warn
            warn("Octree NNPS by default uses single precision arithmetic for"
                 "finding neighbors. A few particles outside of the original "
                 "search radius might be included.")

        for i in range(self.narrays):
            self.octrees.append(PointTree(pa=self.pa_wrappers[i].pa,
                                          radius_scale=radius_scale,
                                          use_double=self.use_double,
                                          leaf_size=leaf_size, dim=dim))
        self.use_elementwise = use_elementwise
        self.use_partitions = use_partitions
        self.allow_sort = allow_sort
        self.domain.update()
        self.update()

        # Check if device supports required workgroup size,
        # else default to elementwise nnps
        if not self.octrees[0]._is_valid_nnps_wgs():
            from warnings import warn
            warn("Octree NNPS with given leaf size (%d) is "
                 "not supported for given device. Switching to a elementwise "
                 "version of the Octree NNPS" % leaf_size)

            self.use_elementwise = True

    cpdef _bin(self, int pa_index):
        self.octrees[pa_index].refresh(self.xmin, self.xmax,
                                       self.domain.manager.hmin)
        self.octrees[pa_index].set_node_bounds()

        if self.allow_sort:
            self.spatially_order_particles(pa_index)

    def get_spatially_ordered_indices(self, int pa_index):
        def update():
            self.octrees[pa_index]._sort()

        return self.octrees[pa_index].pids, update

    cpdef _refresh(self):
        pass

    cpdef set_context(self, int src_index, int dst_index):
        """Setup the context before asking for neighbors.  The `dst_index`
        represents the particles for whom the neighbors are to be determined
        from the particle array with index `src_index`.

        Parameters
        ----------

         src_index: int: the source index of the particle array.
         dst_index: int: the destination index of the particle array.
        """
        GPUNNPS.set_context(self, src_index, dst_index)

        self.src_index = src_index
        self.dst_index = dst_index

        octree_src = self.octrees[src_index]
        octree_dst = self.octrees[dst_index]
        self.dst_src = src_index != dst_index

        self.neighbor_cid_counts, self.neighbor_cids = octree_dst.find_neighbor_cids(
            octree_src)

    cdef void find_neighbor_lengths(self, nbr_lengths):
        octree_src = self.octrees[self.src_index]
        octree_dst = self.octrees[self.dst_index]

        args = []
        if self.use_elementwise:
            find_neighbor_lengths = octree_dst.find_neighbor_lengths_elementwise
        else:
            find_neighbor_lengths = octree_dst.find_neighbor_lengths

        find_neighbor_lengths(
            self.neighbor_cid_counts, self.neighbor_cids, octree_src,
            nbr_lengths, *args
        )

    cdef void find_nearest_neighbors_gpu(self, nbrs, start_indices):
        octree_src = self.octrees[self.src_index]
        octree_dst = self.octrees[self.dst_index]

        args = []
        if self.use_elementwise:
            find_neighbors = octree_dst.find_neighbors_elementwise
        else:
            find_neighbors = octree_dst.find_neighbors
            args.append(self.use_partitions)

        find_neighbors(
            self.neighbor_cid_counts, self.neighbor_cids, octree_src,
            start_indices, nbrs, *args
        )

    cpdef get_kernel_args(self, c_type):
        octree_dst = self.octrees[self.dst_index]
        octree_src = self.octrees[self.src_index]
        pa_gpu_dst = octree_dst.pa.gpu
        pa_gpu_src = octree_src.pa.gpu

        return [
                   octree_dst.unique_cids.dev.data,
                   octree_src.pids.dev.data,
                   octree_dst.pids.dev.data,
                   octree_dst.cids.dev.data,
                   octree_src.pbounds.dev.data, octree_dst.pbounds.dev.data,
                   np.float32(octree_dst.radius_scale),
                   self.neighbor_cid_counts.dev.data,
                   self.neighbor_cids.dev.data,
                   pa_gpu_dst.x.dev.data, pa_gpu_dst.y.dev.data,
                   pa_gpu_dst.z.dev.data, pa_gpu_dst.h.dev.data,
                   pa_gpu_src.x.dev.data, pa_gpu_src.y.dev.data,
                   pa_gpu_src.z.dev.data, pa_gpu_src.h.dev.data
               ], [
                   (self.leaf_size * octree_dst.unique_cid_count,),
                   (self.leaf_size,)
               ]
