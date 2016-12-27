#cython: embedsignature=True

# malloc and friends
from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector
from libcpp.map cimport map

from cython.operator cimport dereference as deref, preincrement as inc

# Cython for compiler directives
cimport cython

cdef extern from "<algorithm>" namespace "std" nogil:
    void sort[Iter, Compare](Iter first, Iter last, Compare comp)
    void sort[Iter](Iter first, Iter last)

IF UNAME_SYSNAME == "Windows":
    @cython.cdivision(True)
    cdef inline double log2(double n) nogil:
        return log(n)/log(2)

#############################################################################

cdef class CellIndexingNNPS(NNPS):

    """Find nearest neighbors using cell indexing.

    Uses a sorted array to access particles in a cell. Gives better cache performance
    than spatial hash.

    Ref: J. Onderik, R. Durikovic, Efficient Neighbor Search for Particle-based Fluids,
    Journal of the Applied Mathematics, Statistics and Informatics 4 (1) (2008) 29–43.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.105.6732&rep=rep1&type=pdf
    """

    def __init__(self, int dim, list particles, double radius_scale = 2.0,
            int ghost_layers = 1, domain=None, bint fixed_h = False,
            bint cache = False, bint sort_gids = False):
        NNPS.__init__(
            self, dim, particles, radius_scale, ghost_layers, domain,
            cache, sort_gids
        )

        self.radius_scale2 = radius_scale*radius_scale
        cdef NNPSParticleArrayWrapper pa_wrapper
        cdef int i, num_particles

        for i from 0<=i<self.narrays:
            pa_wrapper = <NNPSParticleArrayWrapper> self.pa_wrappers[i]
            num_particles = pa_wrapper.get_number_of_particles()

            self.keys[i] = <u_int*> malloc(num_particles*sizeof(u_int))
            self.key_indices[i] = new key_to_idx_t()

        self.src_index = 0
        self.dst_index = 0
        self.sort_gids = sort_gids
        self.domain.update()
        self.update()

    def __cinit__(self, int dim, list particles, double radius_scale = 2.0,
            int ghost_layers = 1, domain=None, bint fixed_h = False,
            bint cache = False, bint sort_gids = False):
        cdef int narrays = len(particles)

        self.keys = <u_int**> malloc(narrays*sizeof(u_int*))
        self.key_indices = <key_to_idx_t**> malloc(narrays*sizeof(key_to_idx_t*))

        self.I = <u_int*> malloc(narrays*sizeof(u_int))

        self.current_keys = NULL
        self.current_indices = NULL

    def __dealloc__(self):
        cdef int i
        for i from 0<=i<self.narrays:
            free(self.keys[i])
            del self.key_indices[i]
        free(self.keys)
        free(self.key_indices)
        free(self.I)


    #### Public protocol ################################################

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
        cdef int i, j

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

        cdef map[u_int, pair[u_int, u_int]].iterator it

        cdef int x_boxes[27]
        cdef int y_boxes[27]
        cdef int z_boxes[27]
        cdef int num_boxes = self._neighbor_boxes(c_x, c_y, c_z,
                x_boxes, y_boxes, z_boxes)

        cdef pair[u_int, u_int] candidate

        cdef u_int n, idx
        for i from 0<=i<num_boxes:
            it = self.current_indices.find(self._get_key(0, x_boxes[i], y_boxes[i],
                z_boxes[i], self.src_index))
            if it == self.current_indices.end():
                continue
            candidate = deref(it).second
            n = candidate.first
            candidate_length = candidate.second

            for j from 0<=j<candidate_length:
                idx = self._get_id(self.current_keys[n+j], self.src_index)

                hj2 = self.radius_scale2*src_h_ptr[idx]*src_h_ptr[idx]

                xij2 = norm2(
                    src_x_ptr[idx] - x,
                    src_y_ptr[idx] - y,
                    src_z_ptr[idx] - z
                    )

                if (xij2 < hi2) or (xij2 < hj2):
                    nbrs.c_append(idx)

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

    cpdef get_spatially_ordered_indices(self, int pa_index, LongArray indices):
        indices.reset()
        cdef NNPSParticleArrayWrapper pa_wrapper = self.pa_wrappers[pa_index]
        cdef int num_particles = pa_wrapper.get_number_of_particles()

        cdef u_int* current_keys = self.keys[pa_index]

        cdef int j
        for j from 0<=j<num_particles:
            indices.c_append(<long>self._get_id(current_keys[j], pa_index))


    cdef void fill_array(self, NNPSParticleArrayWrapper pa_wrapper, int pa_index,
            UIntArray indices, u_int* current_keys, key_to_idx_t* current_indices) nogil:
        cdef double* x_ptr = pa_wrapper.x.data
        cdef double* y_ptr = pa_wrapper.y.data
        cdef double* z_ptr = pa_wrapper.z.data

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
                    &c_x, &c_y, &c_z
                    )
            current_keys[i] = self._get_key(n, c_x, c_y, c_z, pa_index)

        sort(current_keys, current_keys + indices.length)

        cdef int id_x, id_y, id_z

        cdef pair[u_int, pair[u_int, u_int]] temp
        cdef pair[u_int, u_int] cell

        c_x = self._get_x(current_keys[0], pa_index)
        c_y = self._get_y(current_keys[0], pa_index)
        c_z = self._get_z(current_keys[0], pa_index)

        temp.first = self._get_key(0, c_x, c_y, c_z, pa_index)
        cell.first = 0

        cdef u_int length = 0

        for i from 0<i<indices.length:
            id_x = self._get_x(current_keys[i], pa_index)
            id_y = self._get_y(current_keys[i], pa_index)
            id_z = self._get_z(current_keys[i], pa_index)

            length += 1

            if(id_x != c_x or id_y != c_y or id_z != c_z):
                cell.second = length
                temp.second = cell
                current_indices.insert(temp)

                temp.first = self._get_key(0, id_x, id_y, id_z, pa_index)
                cell.first = i

                length = 0

                c_x = id_x
                c_y = id_y
                c_z = id_z

        cell.second = length + 1
        temp.second = cell
        current_indices.insert(temp)

    #### Private protocol ################################################

    cdef inline u_int _get_key(self, u_int n, u_int i, u_int j,
            u_int k, int pa_index) nogil:
        return  n + \
                (1 << self.I[pa_index])*i + \
                (1 << (self.I[pa_index] + self.J))*j + \
                (1 << (self.I[pa_index] + self.J + self.K))*k

    @cython.cdivision(True)
    cdef inline int _get_id(self, u_int key, int pa_index) nogil:
        return key % (1 << self.I[pa_index])

    @cython.cdivision(True)
    cdef inline int _get_x(self, u_int key, int pa_index) nogil:
        return (key >> self.I[pa_index]) % (1 << self.J)

    @cython.cdivision(True)
    cdef inline int _get_y(self, u_int key, int pa_index) nogil:
        return (key >> (self.I[pa_index] + self.J)) % (1 << self.K)

    @cython.cdivision(True)
    cdef inline int _get_z(self, u_int key, int pa_index) nogil:
        return key >> (self.I[pa_index] + self.J + self.K)

    cdef inline int _neighbor_boxes(self, int i, int j, int k,
            int* x, int* y, int* z) nogil:
        cdef int length = 0
        cdef int p, q, r
        for p from -1<=p<2:
            for q from -1<=q<2:
                for r from -1<=r<2:
                    if i+r>=0 and j+q>=0 and k+p>=0:
                        x[length] = i+r
                        y[length] = j+q
                        z[length] = k+p
                        length += 1
        return length

    cpdef _refresh(self):
        cdef NNPSParticleArrayWrapper pa_wrapper

        # Only necessary if number of particles in a ParticleArray changes
        cdef int i, num_particles

        cdef double* xmax = self.xmax.data
        cdef double* xmin = self.xmin.data

        self.J = <u_int> (1 + log2(ceil((xmax[0] - xmin[0])/self.cell_size)))
        self.K = <u_int> (1 + log2(ceil((xmax[1] - xmin[1])/self.cell_size)))

        for i from 0<=i<self.narrays:
            free(self.keys[i])
            del self.key_indices[i]

            pa_wrapper = <NNPSParticleArrayWrapper> self.pa_wrappers[i]
            num_particles = pa_wrapper.get_number_of_particles()

            self.keys[i] = <u_int*> malloc(num_particles*sizeof(u_int))
            self.key_indices[i] = new key_to_idx_t()

        self.current_keys = self.keys[self.src_index]
        self.current_indices = self.key_indices[self.src_index]

    @cython.cdivision(True)
    cpdef _bin(self, int pa_index, UIntArray indices):
        cdef NNPSParticleArrayWrapper pa_wrapper = self.pa_wrappers[pa_index]
        cdef int num_particles = pa_wrapper.get_number_of_particles()

        self.I[pa_index] = <u_int> (1 + log2(pa_wrapper.get_number_of_particles()))

        cdef u_int* current_keys = self.keys[pa_index]
        cdef key_to_idx_t* current_indices = self.key_indices[pa_index]

        self.fill_array(pa_wrapper, pa_index, indices, current_keys, current_indices)

