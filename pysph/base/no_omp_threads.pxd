# cython: language_level=3, embedsignature=True
# distutils: language=c++
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

cpdef int get_number_of_threads()
cpdef set_number_of_threads(int)