from libcpp.map cimport map

from nnps_base cimport *
from linked_list_nnps cimport *

# NNPS using the original gridding algorithm
cdef class DictBoxSortNNPS(NNPS):
    cdef public dict cells               # lookup table for the cells
    cdef list _cell_keys

    cpdef get_nearest_particles_no_cache(self, int src_index, int dst_index,
            size_t d_idx, UIntArray nbrs, bint prealloc)

    cpdef _refresh(self)

    cpdef _bin(self, int pa_index, UIntArray indices)

# NNPS using the linked list approach
cdef class BoxSortNNPS(LinkedListNNPS):
    ############################################################################
    # Data Attributes
    ############################################################################
    cdef public map[long, int] cell_to_index  # Maps cell ID to an index


