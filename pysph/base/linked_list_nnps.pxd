from libcpp.map cimport map
from libcpp.vector cimport vector

from nnps_base cimport *

# NNPS using the linked list approach
cdef class LinkedListNNPS(NNPS):
    ############################################################################
    # Data Attributes
    ############################################################################
    cdef public IntArray ncells_per_dim  # number of cells in each direction
    cdef public int ncells_tot           # total number of cells
    cdef public bint fixed_h             # Constant cell sizes
    cdef public list heads               # Head arrays for the cells
    cdef public list nexts               # Next arrays for the particles

    cdef NNPSParticleArrayWrapper src, dst # Current source and destination.
    cdef UIntArray next, head              # Current next and head arrays.

    cpdef long _count_occupied_cells(self, long n_cells) except -1
    cpdef long _get_number_of_cells(self) except -1
    cdef long _get_flattened_cell_index(self, cPoint pnt, double cell_size)
    cdef long _get_valid_cell_index(self, int cid_x, int cid_y, int cid_z,
            int* ncells_per_dim, int dim, int n_cells) nogil
    cdef void find_nearest_neighbors(self, size_t d_idx, UIntArray nbrs) nogil


