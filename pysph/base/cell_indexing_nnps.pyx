
cdef class CellIndexing(NNPS):

    """Find nearest neighbors using cell indexing"""

    def __init__(self, int dim, list particles, double radius_scale = 2.0,
            int ghost_layers = 1, domain=None, bint fixed_h = False,
            bint cache = False, bint sort_gids = False):
        NNPS.__init__(
            self, dim, particles, radius_scale, ghost_layers, domain,
            cache, sort_gids
        )

        self.radius_scale2 = radius_scale*radius_scale

        self.keys = <LL_INT**> malloc(self.narrays*sizeof(LL_INT*))
        self.key_indices = \
                <key_to_idx_t**> malloc(self.narrays*sizeof(key_to_idx_t*))

        self.I = <LL_INT*> malloc(self.narrays*sizeof(LL_INT))
        self.J = <LL_INT*> malloc(self.narrays*sizeof(LL_INT))
        self.K = <LL_INT*> malloc(self.narrays*sizeof(LL_INT))

        cdef NNPSParticleArrayWrapper pa_wrapper
        cdef int i, num_particles

        for i from 0<=i<self.narrays:
            pa_wrapper = <NNPSParticleArrayWrapper> self.pa_wrappers[i]
            num_particles = pa_wrapper.get_number_of_particles()

            self.keys[i] = <LL_INT*> malloc(num_particles*sizeof(LL_INT))
            self.key_indices[i] = new key_to_idx_t()

        self.src_index = 0
        self.dst_index = 0
        self.current_keys = NULL
        self.current_indices = NULL
        self.sort_gids = sort_gids
        self.domain.update()
        self.update()

    cdef inline LL_INT get_key(self, LL_INT n, LL_INT i, LL_INT j,
            LL_INT k, int pa_index) nogil:
        return  n + \
                (1 << self.I[pa_index])*i + \
                (1 << (self.I[pa_index] + self.J[pa_index]))*j + \
                (1 << (self.I[pa_index] + self.J[pa_index] + self.K[pa_index]))*k

    @cython.cdivision(True)
    cdef inline int _get_id(self, LL_INT key, int pa_index) nogil:
        return key % (1 << self.I[pa_index])

    @cython.cdivision(True)
    cdef inline int _get_x(self, LL_INT key, int pa_index) nogil:
        return (key >> self.I[pa_index]) % (1 << self.J[pa_index])

    @cython.cdivision(True)
    cdef inline int _get_y(self, LL_INT key, int pa_index) nogil:
        return (key >> (self.I[pa_index] + self.J[pa_index])) % (1 << self.K[pa_index])

    @cython.cdivision(True)
    cdef inline int _get_z(self, LL_INT key, int pa_index) nogil:
        return key >> (self.I[pa_index] + self.J[pa_index] + self.K[pa_index])

    cpdef set_context(self, int src_index, int dst_index):
        """Set context for nearest neighbor searches.

        Parameters
        ----------
        src_index: int
            Index in the list of particle arrays to which the neighbors belong

        dst_index: int
            Index in the list of particle arrays to which the query point belongs

        """
        NNPS.set_context(self, src_index, dst_index)
        self.current_keys = self.keys[src_index]
        self.current_indices = self.key_indices[src_index]

        self.dst = <NNPSParticleArrayWrapper> self.pa_wrappers[dst_index]
        self.src = <NNPSParticleArrayWrapper> self.pa_wrappers[src_index]

    cdef void find_nearest_neighbors(self, size_t d_idx, UIntArray nbrs) nogil:
        """Low level, high-performance non-gil method to find neighbors.
        This requires that `set_context()` be called beforehand.  This method
        does not reset the neighbors array before it appends the
        neighbors to it.

        """
        cdef double* dst_x_ptr = self.dst.x.data
        cdef double* dst_y_ptr = self.dst.y.data
        cdef double* dst_z_ptr = self.dst.z.data
        cdef double* dst_h_ptr = self.dst.h.data

        cdef double* src_x_ptr = self.src.x.data
        cdef double* src_y_ptr = self.src.y.data
        cdef double* src_z_ptr = self.src.z.data
        cdef double* src_h_ptr = self.src.h.data

        cdef double x = dst_x_ptr[d_idx]
        cdef double y = dst_y_ptr[d_idx]
        cdef double z = dst_z_ptr[d_idx]
        cdef double h = dst_h_ptr[d_idx]

        cdef unsigned int* s_gid = self.src.gid.data
        cdef int orig_length = nbrs.length

        cdef int c_x, c_y, c_z
        cdef double* xmin = self.xmin.data
        cdef int i, j, k

        find_cell_id_raw(
                x - xmin[0],
                y - xmin[1],
                z - xmin[2],
                self.cell_size,
                &c_x, &c_y, &c_z
                )

        cdef double xij2 = 0
        cdef double hi2 = self.radius_scale2*h*h
        cdef double hj2 = 0

        cdef map[LL_INT, int].iterator it

        cdef int n, idx
        cdef LL_INT next_cell
        for i from -1<=i<2:
            for j from -1<=j<2:
                for k from -1<=k<2:
                    next_cell = c_x+k
                    it = self.current_indices.find(self.get_key(0, c_x+k, c_y+j, \
                        c_z+i, self.src_index))
                    if it == self.current_indices.end():
                        continue
                    n = deref(it).second
                    while next_cell == c_x+k:
                        idx = self._get_id(self.current_keys[n], self.src_index)

                        hj2 = self.radius_scale2*src_h_ptr[idx]*src_h_ptr[idx]

                        xij2 = norm2(
                            src_x_ptr[idx] - x,
                            src_y_ptr[idx] - y,
                            src_z_ptr[idx] - z
                            )

                        if (xij2 < hi2) or (xij2 < hj2):
                            nbrs.c_append(idx)

                        n += 1
                        next_cell = self._get_x(self.current_keys[n], self.src_index)

        if self.sort_gids:
            self._sort_neighbors(
                &nbrs.data[orig_length], nbrs.length - orig_length, s_gid
            )


    cpdef get_nearest_particles_no_cache(self, int src_index, int dst_index,
            size_t d_idx, UIntArray nbrs, bint prealloc):
        """Find nearest neighbors for particle id 'd_idx' without cache

        Parameters
        ----------
        src_index: int
            Index in the list of particle arrays to which the neighbors belong

        dst_index: int
            Index in the list of particle arrays to which the query point belongs

        d_idx: size_t
            Index of the query point in the destination particle array

        nbrs: UIntArray
            Array to be populated by nearest neighbors of 'd_idx'

        """
        self.set_context(src_index, dst_index)

        if prealloc:
            nbrs.length = 0
        else:
            nbrs.c_reset()

        self.find_nearest_neighbors(d_idx, nbrs)


    cdef void fill_array(self, NNPSParticleArrayWrapper pa_wrapper, int pa_index,
            UIntArray indices, LL_INT* current_keys, key_to_idx_t* current_indices) nogil:
        cdef double* x_ptr = pa_wrapper.x.data
        cdef double* y_ptr = pa_wrapper.y.data
        cdef double* z_ptr = pa_wrapper.z.data
        cdef double* h_ptr = pa_wrapper.h.data

        cdef double* xmin = self.xmin.data

        cdef int i, n
        cdef int c_x, c_y, c_z
        for i from 0<=i<indices.length:
            n = indices.data[i]
            find_cell_id_raw(
                    x_ptr[i] - xmin[0],
                    y_ptr[i] - xmin[1],
                    z_ptr[i] - xmin[2],
                    self.cell_size,
                    &c_x, &c_y, &c_z)
            current_keys[i] = self.get_key(n, c_x, c_y, c_z, pa_index)

        sort(current_keys, current_keys + indices.length)

        c_x = -1
        c_y = -1
        c_z = -1

        cdef int id_x, id_y, id_z

        cdef pair[LL_INT, int] temp

        for i from 0<=i<indices.length:
            id_x = self._get_x(current_keys[i], pa_index)
            id_y = self._get_y(current_keys[i], pa_index)
            id_z = self._get_z(current_keys[i], pa_index)

            if(id_x != c_x or id_y != c_y or id_z != c_z):
                temp.first = self.get_key(0, id_x, id_y, id_z, pa_index)
                temp.second = i
                current_indices.insert(temp)

                c_x = id_x
                c_y = id_y
                c_z = id_z

    cpdef _refresh(self):
        cdef NNPSParticleArrayWrapper pa_wrapper

        # Only necessary if number of particles in a ParticleArray changes
        cdef int i, num_particles
        for i from 0<=i<self.narrays:
            free(self.keys[i])
            del self.key_indices[i]

            pa_wrapper = <NNPSParticleArrayWrapper> self.pa_wrappers[i]
            num_particles = pa_wrapper.get_number_of_particles()

            self.keys[i] = <LL_INT*> malloc(num_particles*sizeof(LL_INT))
            self.key_indices[i] = new key_to_idx_t()

        self.current_keys = self.keys[self.src_index]
        self.current_indices = self.key_indices[self.src_index]

    @cython.cdivision(True)
    cpdef _bin(self, int pa_index, UIntArray indices):
        cdef NNPSParticleArrayWrapper pa_wrapper = self.pa_wrappers[pa_index]
        cdef int num_particles = pa_wrapper.get_number_of_particles()

        cdef double* xmax = self.xmax.data
        cdef double* xmin = self.xmin.data

        self.I[pa_index] = <LL_INT> (1 + log2(pa_wrapper.get_number_of_particles()))
        self.J[pa_index] = <LL_INT> (1 + log2(ceil((xmax[0] - xmin[0])/self.cell_size)))
        self.K[pa_index] = <LL_INT> (1 + log2(ceil((xmax[1] - xmin[1])/self.cell_size)))

        cdef LL_INT* current_keys = self.keys[pa_index]
        cdef key_to_idx_t* current_indices = self.key_indices[pa_index]

        self.fill_array(pa_wrapper, pa_index, indices, current_keys, current_indices)

    def __dealloc__(self):
        cdef int i
        for i from 0<=i<self.narrays:
            free(self.keys[i])
            del self.key_indices[i]
        free(self.keys)
        free(self.key_indices)

        free(self.I)
        free(self.J)
        free(self.K)


