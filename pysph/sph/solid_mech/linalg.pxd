from pysph.base.point cimport cPoint


cdef double det(cPoint d, cPoint s)
cdef cPoint get_eigenvalues(cPoint d, cPoint s)
cdef cPoint get_eigenvector(cPoint d, cPoint s, double r)
cdef void transform(double A[3][3], double P[3][3], double res[3][3])
cdef void transform2(cPoint A, double P[3][3], double res[3][3])
cdef void transform2inv(cPoint A, double P[3][3], double res[3][3])



cdef cPoint get_eigenvalvec(cPoint d, cPoint s, double * R)
cdef void eigen_decomposition(double A[3][3], double V[3][3], double d[3])

