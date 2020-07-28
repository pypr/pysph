#cython: boundscheck=False

#
# Eigen decomposition code for symmetric 3x3 matrices, some code taken
# from the public domain Java Matrix library JAMA

from libc.math cimport sqrt, cos, acos, sin, atan2, M_PI
from libc.string cimport memcpy

from numpy.linalg import eigh
cimport numpy
import numpy

cdef extern:
    double fabs(double) nogil

# this is cython substitute for const values
cdef enum:
    n=3

cdef double EPS = numpy.finfo(float).eps

cdef inline double MAX(double a, double b) nogil:
    return a if a>b else b

cdef inline double SQR(double a) nogil:
    return a*a

cdef inline double hypot2(double x, double y) nogil:
    return sqrt(x*x+y*y)


cdef double det(double a[3][3]) nogil:
    '''Determinant of symmetrix matrix
    '''
    return (a[0][0]*a[1][1]*a[2][2] + 2*a[1][2]*a[0][2]*a[0][1] -
            a[0][0]*a[1][2]*a[1][2] - a[1][1]*a[0][2]*a[0][2] -
            a[2][2]*a[0][1]*a[0][1])

cpdef double py_det(double[:,:] m):
    '''Determinant of symmetrix matrix
    '''
    return det(<double(*)[3]>&m[0][0])

# d,s are diagonal and off-diagonal elements of a symmetric 3x3 matrix
# d:11,22,33; s:23,13,12
# d:00,11,22; s:12,02,01

cdef void get_eigenvalues(double a[3][3], double *result) nogil:
    '''Compute the eigenvalues of symmetric matrix a and return in
    result array.
    '''
    cdef double m = (a[0][0]+ a[1][1]+a[2][2])/3.0
    cdef double K[3][3]
    memcpy(&K[0][0], &a[0][0], sizeof(double)*9)
    K[0][0], K[1][1], K[2][2] = a[0][0] - m, a[1][1] - m, a[2][2] - m
    cdef double q = det(K)*0.5
    cdef double p = 0
    p += K[0][0]*K[0][0] + 2*K[1][2]*K[1][2]
    p += K[1][1]*K[1][1] + 2*K[0][2]*K[0][2]
    p += K[2][2]*K[2][2] + 2*K[0][1]*K[0][1]
    p /= 6.0
    cdef double pi = M_PI
    cdef double phi = 0.5*pi
    cdef double tmp = p**3 - q**2

    if q == 0.0 and p == 0.0:
        # singular zero matrix
        result[0] = result[1] = result[2] = m
        return
    elif tmp < 0.0 or fabs(tmp) < EPS: # eliminate roundoff error
        phi = 0
    else:
        phi = atan2(sqrt(tmp), q)/3.0
    if phi == 0 and q < 0:
        phi = pi

    result[0] = m + 2*sqrt(p)*cos(phi)
    result[1] = m - sqrt(p)*(cos(phi) + sqrt(3)*sin(phi))
    result[2] = m - sqrt(p)*(cos(phi) - sqrt(3)*sin(phi))

cpdef py_get_eigenvalues(double[:,:] m):
    '''Return the eigenvalues of symmetric matrix.
    '''
    res = numpy.empty(3, float)
    cdef double[:] _res  = res
    get_eigenvalues(<double(*)[3]>&m[0][0], &_res[0])
    return res


##############################################################################
cdef void get_eigenvector_np(double A[n][n], double r, double *res):
    ''' eigenvector of symmetric matrix for given eigenvalue `r` using numpy
    '''
    cdef numpy.ndarray[ndim=2, dtype=numpy.float64_t] mat=numpy.empty((3,3)), evec
    cdef numpy.ndarray[ndim=1, dtype=numpy.float64_t] evals
    cdef double[:,:] _mat = mat
    cdef int i, j
    for i in range(3):
        for j in range(3):
            _mat[i,j] = A[i][j]
    evals, evec = eigh(mat)
    cdef int idx=0
    cdef double di = fabs(evals[0]-r)
    if fabs(evals[1]-r) < di:
        idx = 1
    if fabs(evals[2]-r) < di:
        idx = 2
    for i in range(3):
        res[i] = evec[idx,i]

cdef void get_eigenvector(double A[n][n], double r, double *res):
    ''' get eigenvector of symmetric 3x3 matrix for given eigenvalue `r`

    uses a fast method to get eigenvectors with a fallback to using numpy
    '''
    res[0] = A[0][1]*A[1][2] - A[0][2]*(A[1][1]-r) # a_01 * a_12 - a_02 * (a_11 - r)
    res[1] = A[0][1]*A[0][2] - A[1][2]*(A[0][0]-r) # a_01 * a_02 - a_12 * (a_00 - r)
    res[2] = (A[0][0]-r)*(A[1][1]-r) - A[0][1]*A[0][1] # (a_00 - r) * (a_11 - r) - a_01^2
    cdef double norm = sqrt(SQR(res[0]) + SQR(res[1]) + SQR(res[2]))

    if norm *1e7 <= fabs(r):
        # its a zero, let numpy get the answer
        get_eigenvector_np(A, r, res)
    else:
        res[0] /= norm
        res[1] /= norm
        res[2] /= norm

cpdef py_get_eigenvector(double[:,:] A, double r):
    ''' get eigenvector of a symmetric matrix for given eigenvalue `r` '''
    d = numpy.empty(3, dtype=float)
    cdef double[:] _d = d
    get_eigenvector(<double(*)[n]>&A[0,0], r, &_d[0])
    return d

cdef void get_eigenvec_from_val(double A[n][n], double *R, double *e):
    cdef int i, j
    cdef double res[3]
    for i in range(3):
        get_eigenvector(A, e[i], &res[0])
        for j in range(3):
            R[j*3+i] = res[j]


cdef bint _nearly_diagonal(double A[n][n]) nogil:
    return (
        (SQR(A[0][0]) + SQR(A[1][1]) + SQR(A[2][2])) >
        1e8*(SQR(A[0][1]) + SQR(A[0][2]) + SQR(A[1][2]))
    )

cdef void get_eigenvalvec(double A[n][n], double *R, double *e):
    '''Get the eigenvalues and eigenvectors of symmetric 3x3 matrix.

    A is the input 3x3 matrix.
    R is the output eigen matrix
    e are the output eigenvalues
    '''
    cdef bint use_iter = False
    cdef int i,j
    if A[0][1] == A[0][2] == A[1][2] == 0.0:
        # diagonal matrix.
        e[0] = A[0][0]
        e[1] = A[1][1]
        e[2] = A[2][2]
        for i in range(3):
            for j in range(3):
                R[i*3+j] = (i==j)
        return

    # FIXME: implement fast version
    get_eigenvalues(A, e)
    if e[0] != e[1] and e[1] != e[2] and e[0] != e[2]:
        # no repeated eigenvalues
        use_iter = True
    if _nearly_diagonal(A):
        # nearly diagonal matrix
        use_iter = True
    if not use_iter:
        get_eigenvec_from_val(A, R, e)
    else:
        eigen_decomposition(
            A, <double(*)[n]>&R[0], &e[0]
        )

def py_get_eigenvalvec(double[:,:] A):
    v = numpy.empty((3,3), dtype=float)
    d = numpy.empty(3, dtype=float)
    cdef double[:,:] _v = v
    cdef double[:] _d = d
    get_eigenvalvec(<double(*)[n]>&A[0,0], &_v[0,0], &_d[0])
    return d, v


##############################################################################

cdef void transform(double A[3][3], double P[3][3], double res[3][3]) nogil:
    '''Compute the transformation P.T*A*P and add it into res.
    '''
    cdef int i, j, k, l
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    res[i][j] += P[k][i]*A[k][l]*P[l][j] # P.T*A*P

cdef void transform_diag(double *A, double P[3][3],
                         double res[3][3]) nogil:
    '''Compute the transformation P.T*A*P and add it into res.

    A is diagonal and contains the diagonal entries alone.
    '''
    cdef int i, j, k
    for i in range(3):
        for j in range(3):
            for k in range(3):
                res[i][j] += P[k][i]*A[k]*P[k][j] # P.T*A*P

cdef void transform_diag_inv(double *A, double P[3][3],
                             double res[3][3]) nogil:
    '''Compute the transformation P*A*P.T and set it into res.
    A is diagonal and contains just the diagonal entries.
    '''
    cdef int i, j, k
    for i in range(3):
        for j in range(3):
            res[i][j] = 0.0

    for i in range(3):
        for j in range(3):
            for k in range(3):
                res[i][j] += P[i][k]*A[k]*P[j][k] # P*A*P.T

def py_transform(double[:,:] A, double[:,:] P):
    res = numpy.zeros((3,3), dtype=float)
    cdef double[:,:] _res = res
    transform(
        <double(*)[3]>&A[0][0], <double(*)[3]>&P[0][0],
        <double(*)[3]>&_res[0][0]
    )
    return res

def py_transform_diag(double[:] A, double[:,:] P):
    res = numpy.zeros((3,3), dtype=float)
    cdef double[:,:] _res = res
    transform_diag(
        &A[0], <double(*)[3]>&P[0][0], <double(*)[3]>&_res[0][0]
    )
    return res

def py_transform_diag_inv(double[:] A, double[:,:] P):
    res = numpy.empty((3,3), dtype=float)
    cdef double[:,:] _res = res
    transform_diag_inv(
        &A[0], <double(*)[3]>&P[0][0], <double(*)[3]>&_res[0][0]
    )
    return res


cdef double * tred2(double V[n][n], double *d, double *e) nogil:
    '''Symmetric Householder reduction to tridiagonal form

    This is derived from the Algol procedures tred2 by
    Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
    Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
    Fortran subroutine in EISPACK.

    d contains the diagonal elements of the tridiagonal matrix.

    e contains the subdiagonal elements of the tridiagonal matrix in its
    last n-1 positions.  e[0] is set to zero.
    '''

    cdef:
        double scale, f, g, h, hh
        int i, j, k

    for j in range(n):
        d[j] = V[n-1][j]

    # Householder reduction to tridiagonal form.

    for i in range(n-1,0,-1):

        # Scale to avoid under/overflow.
        scale = 0.0
        h = 0.0
        for k in range(i):
            scale += fabs(d[k])

        if (scale == 0.0):
            e[i] = d[i-1]
            for j in range(i):
                d[j] = V[i-1][j]
                V[i][j] = 0.0
                V[j][i] = 0.0

        else:
            # Generate Householder vector.
            for k in range(i):
                d[k] /= scale
                h += d[k] * d[k]

            f = d[i-1]
            g = sqrt(h)
            if f > 0:
                g = -g

            e[i] = scale * g
            h = h - f * g
            d[i-1] = f - g
            for j in range(i):
                e[j] = 0.0

        # Apply similarity transformation to remaining columns.

            for j in range(i):
                f = d[j]
                V[j][i] = f
                g = e[j] + V[j][j] * f
                for k in range(j+1, i):
                    g += V[k][j] * d[k]
                    e[k] += V[k][j] * f

                e[j] = g

            f = 0.0
            for j in range(i):
                e[j] /= h
                f += e[j] * d[j]

            hh = f / (h + h)
            for j in range(i):
                e[j] -= hh * d[j]

            for j in range(i):
                f = d[j]
                g = e[j]
                for k in range(j,i):
                    V[k][j] -= (f * e[k] + g * d[k])

                d[j] = V[i-1][j];
                V[i][j] = 0.0;

        d[i] = h

    # Accumulate transformations.

    for i in range(n-1):
        V[n-1][i] = V[i][i]
        V[i][i] = 1.0
        h = d[i+1]
        if h != 0.0:
            for k in range(i+1):
                d[k] = V[k][i+1] / h

            for j in range(i+1):
                g = 0.0
                for k in range(i+1):
                    g += V[k][i+1] * V[k][j]

                for k in range(i+1):
                    V[k][j] -= g * d[k]


        for k in range(i+1):
            V[k][i+1] = 0.0

    for j in range(n):
        d[j] = V[n-1][j]
        V[n-1][j] = 0.0

    V[n-1][n-1] = 1.0
    e[0] = 0.0

    return d


cdef void tql2(double V[n][n], double *d, double *e) nogil:
    '''Symmetric tridiagonal QL algo for eigendecomposition

    This is derived from the Algol procedures tql2, by
    Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
    Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
    Fortran subroutine in EISPACK.

    d contains the eigenvalues in ascending order.  if an error exit is
    made, the eigenvalues are correct but unordered for indices
    1,2,...,ierr-1.

    e has been destroyed.
    '''

    cdef:
        double f, tst1, eps, g, h, p, r, dl1, c, c2, c3, el1, s, s2
        int i, j, k, l, m, iter
        bint cont

    for i in range(1, n):
        e[i-1] = e[i]

    e[n-1] = 0.0

    f = 0.0
    tst1 = 0.0
    eps = 2.0**-52.0
    for l in range(n):

        # Find small subdiagonal element
        tst1 = MAX(tst1,fabs(d[l]) + fabs(e[l]))
        m = l
        while m < n:
            if fabs(e[m]) <= eps*tst1:
                break
            m += 1

        # If m == l, d[l] is an eigenvalue,
        # otherwise, iterate.
        if m > l:
            iter = 0
            cont = True
            while cont:
                iter = iter + 1        # (Could check iteration count here.)

                # Compute implicit shift
                g = d[l]
                p = (d[l+1] - g) / (2.0 * e[l])
                r = hypot2(p,1.0)
                if p < 0:
                    r = -r

                d[l] = e[l] / (p + r)
                d[l+1] = e[l] * (p + r)
                dl1 = d[l+1]
                h = g - d[l]
                for i in range(l+2,n):
                    d[i] -= h

                f += h

                # Implicit QL transformation.
                p = d[m]
                c = 1.0
                c2 = c
                c3 = c
                el1 = e[l+1]
                s = 0.0
                s2 = 0.0
                for i in range(m-1,l-1,-1):
                    c3 = c2
                    c2 = c
                    s2 = s
                    g = c * e[i]
                    h = c * p
                    r = hypot2(p,e[i])
                    e[i+1] = s * r
                    s = e[i] / r
                    c = p / r
                    p = c * d[i] - s * g
                    d[i+1] = h + s * (c * g + s * d[i])

                    # Accumulate transformation
                    for k in range(n):
                        h = V[k][i+1]
                        V[k][i+1] = s * V[k][i] + c * h
                        V[k][i] = c * V[k][i] - s * h

                p = -s * s2 * c3 * el1 * e[l] / dl1
                e[l] = s * p
                d[l] = c * p

                # Check for convergence
                cont = bool(fabs(e[l]) > eps*tst1)
        d[l] += f
        e[l] = 0.0

    # Sort eigenvalues and corresponding vectors.
    for i in range(n-1):
        k = i
        p = d[i]
        for j in range(i+1,n):
            if d[j] < p:
                k = j
                p = d[j]

        if k != i:
            d[k] = d[i]
            d[i] = p
            for j in range(n):
                p = V[j][i]
                V[j][i] = V[j][k]
                V[j][k] = p


cdef void zero_matrix_case(double V[n][n], double *d) nogil:
    cdef int i, j
    for i in range(3):
        d[i] = 0.0
        for j in range(3):
            V[i][j] = (i==j)

cdef void eigen_decomposition(double A[n][n], double V[n][n], double *d) nogil:
    '''Get eigenvalues and eigenvectors of matrix A.
    V is output eigenvectors and d are the eigenvalues.
    '''
    cdef double e[n]
    cdef int i, j
    # Scale the matrix, as if the matrix is tiny, floating point errors
    # creep up leading to zero division errors in tql2.  This is
    # specifically tested for with a tiny matrix.
    cdef double s = 0.0
    for i in range(n):
        for j in range(n):
            V[i][j] = A[i][j]
            s += fabs(V[i][j])

    if s == 0:
        zero_matrix_case(V, d)
    else:
        for i in range(n):
            for j in range(n):
                V[i][j] /= s

        d = tred2(V, d, &e[0])
        tql2(V, d, &e[0])
        for i in range(n):
            d[i] *= s


def py_eigen_decompose_eispack(double[:,:] a):
    v = numpy.empty((3,3), dtype=float)
    d = numpy.empty(3, dtype=float)
    cdef double[:,:] _v = v
    cdef double[:] _d = d
    eigen_decomposition(
        <double(*)[n]>&a[0,0], <double(*)[n]>&_v[0,0], &_d[0]
    )
    return d, v
