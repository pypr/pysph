from pysph.base.point cimport cPoint

cdef double det(cPoint d, cPoint s)
cdef cPoint get_eigenvalues(cPoint d, cPoint s)
cdef cPoint get_eigenvector(cPoint d, cPoint s, double r)
cdef void transform(double A[3][3], double P[3][3], double res[3][3])
cdef void transform2(cPoint A, double P[3][3], double res[3][3])


cdef void eigen_decomposition(double A[3][3], double V[3][3], double d[3])

# return the Eigenvalue and Eigenvectors
cdef cPoint get_eigenvalvec(cPoint d, cPoint s, double * R)
cdef _get_eigenvalvec(double d[3], double s[3], double R[3][3], double eigenvalues[3])

# compute the transformation P*A*P.T
cdef void transform2inv(double A[3], double P[3][3], double res[3][3])
# cdef void _transform2inv(double *_A, double *P0, double *P1, double * P2,
#                         double *R0, double *R1, double * R2)
