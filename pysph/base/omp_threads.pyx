# cython: language_level=3, embedsignature=True
# distutils: language=c++
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

from cython.parallel import parallel, prange
cimport openmp


cpdef int get_number_of_threads():
    cdef int i, n
    with nogil, parallel():
        for i in prange(1):
            n = openmp.omp_get_num_threads()
    return n
    
cpdef set_number_of_threads(int n):
    openmp.omp_set_num_threads(n)