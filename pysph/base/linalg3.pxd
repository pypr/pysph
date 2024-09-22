# cython: language_level=3, embedsignature=True
# distutils: language=c++

"""Routines for eigen decomposition of symmetric 3x3 matrices.
"""
cdef double det(double [3][3]a) noexcept nogil
cdef void get_eigenvalues(double [3][3]a, double *result) noexcept nogil
cdef void eigen_decomposition(double [3][3]A, double [3][3]V, double *d) noexcept nogil
cdef void transform(double [3][3]A, double [3][3]P, double [3][3]res) noexcept nogil
cdef void transform_diag(double *A, double [3][3]P, double [3][3]res) noexcept nogil
cdef void transform_diag_inv(double *A, double [3][3]P, double [3][3]res) nogil

cdef void get_eigenvalvec(double [3][3]A, double *R, double *e)
