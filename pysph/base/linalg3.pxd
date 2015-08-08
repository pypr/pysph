"""Routines for eigen decomposition of symmetric 3x3 matrices.
"""
cdef double det(double a[3][3]) nogil
cdef void get_eigenvalues(double a[3][3], double *result) nogil
cdef void eigen_decomposition(double A[3][3], double V[3][3], double *d) nogil
cdef void transform(double A[3][3], double P[3][3], double res[3][3]) nogil
cdef void transform_diag(double *A, double P[3][3], double res[3][3]) nogil
cdef void transform_diag_inv(double *A, double P[3][3], double res[3][3]) nogil

cdef void get_eigenvalvec(double A[3][3], double *R, double *e)
