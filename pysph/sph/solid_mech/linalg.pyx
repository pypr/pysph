# Eigen decomposition code for symmetric 3x3 matrices, copied from the public
#   domain Java Matrix library JAMA

from libc.math cimport sqrt, cos, acos, sin, atan2
cimport cython

from numpy import empty
from numpy.linalg import eigh
cimport numpy
import numpy

from pysph.base.point cimport cPoint, cPoint_new, cPoint_sub, cPoint_dot, \
        cPoint_norm, cPoint_scale, cPoint_length, Point


cdef extern:
    double fabs(double)

# this is cython substitute for const values
cdef enum:
    n=3

cdef inline MAX(double a, double b):
    return a if a>b else b

cdef inline double hypot2(double x, double y):
    return sqrt(x*x+y*y)


# d,s are diagonal and off-diagonal elements of a symmetric 3x3 matrix
# d:11,22,33; s:23,13,12

cdef double det(cPoint d, cPoint s):
    ''' determinant of symmetrix matrix '''
    return d.x*d.y*d.z + 2*s.x*s.y*s.z - d.x*s.x*s.x - d.y*s.y*s.y - d.z*s.z*s.z

cpdef double py_det(diag, side):
    ''' determinant of symmetrix matrix '''
    cdef cPoint d
    d.x, d.y, d.z = diag
    cdef cPoint s
    s.x, s.y, s.z = side
    return det(d, s)

cdef cPoint get_eigenvalues(cPoint d, cPoint s):
    ''' eigenvalues of symmetric matrix '''
    cdef cPoint ret
    cdef double m = (d.x+d.y+d.z)/3
    cdef cPoint Kd = cPoint_sub(d, cPoint_new(m,m,m)) # M-m*eye(3);
    cdef cPoint Ks = s
    
    cdef double q = det(Kd, Ks)/2
    
    cdef double p = 0
    p += Kd.x*Kd.x + 2*Ks.x*Ks.x
    p += Kd.y*Kd.y + 2*Ks.y*Ks.y
    p += Kd.z*Kd.z + 2*Ks.z*Ks.z
    p /= 6.0
    cdef double pi = acos(-1.0)
    cdef double phi = 0.5*pi

    if q == 0 and p == 0:
        # singular zero matrix
        ret.x = ret.y = ret.z = m
        return ret
    elif fabs(q) >= fabs(p**(3.0/2)): # eliminate roundoff error
        phi = 0
    else:
        phi = atan2(sqrt(p**3 - q**2), q)/3.0
    if phi == 0 and q < 0:
        phi = pi

    ret.x = m + 2*sqrt(p)*cos(phi)
    ret.y = m - sqrt(p)*(cos(phi) + sqrt(3)*sin(phi))
    ret.z = m - sqrt(p)*(cos(phi) - sqrt(3)*sin(phi))
    
    return ret

cdef cPoint get_eigenvalues_trig(cPoint d, cPoint s):
    ''' eigenvalues of symmetric non-singular matrix '''
    cdef cPoint ret
    # characteristic equation is: ax^3+bx^2+cx+d=0
    cdef double A = 1.0
    cdef double B = -(d.x+d.y+d.z)
    cdef double C = d.x*d.y + d.x*d.z + d.y*d.z -s.x**2 - s.y**2 - s.z**2
    cdef double D = d.x*s.x**2 + d.y*s.y**2 + d.z*s.z**2 - d.x*d.y*d.z - 2*s.x*s.y*s.z
    print 'ABCD:', A, B, C, D

    cdef double m = -B/3/A
    cdef double p = ((3*C/A) - (B**2/A**2))/3
    cdef double q = ((2*B**3/A**3) - (9*B*C/A**2) + (27*D/A))/27
    
    if p==0==q:
        ret.x = ret.y = ret.z = m
        return ret

    cdef double z = q**2/4 + p**3/27
 
    cdef double u = 2*sqrt(-3/p)
    cdef double k = acos(3*q*u/4/p)/3

    cdef double pi = acos(-1.0)
    
    # Define Eig1, Eig2, Eig3
    ret.x = m + u*cos(k)
    ret.y = m + u*cos(k-2*pi/3)
    ret.z = m + u*cos(k-4*pi/3)

    return ret

cpdef py_get_eigenvalues(diag, side):
    ''' eigenvalues of symmetric matrix '''
    cdef cPoint d
    d.x, d.y, d.z = diag
    cdef cPoint s
    s.x, s.y, s.z = side
    cdef cPoint ret = get_eigenvalues(d, s)
    return ret.x, ret.y, ret.z


@cython.boundscheck(False)
cdef cPoint get_eigenvector_np(cPoint d, cPoint s, double r):
    ''' eigenvector of symmetric matrix for given eigenvalue `r` using numpy '''
    cdef numpy.ndarray[ndim=2,dtype=numpy.float64_t] mat=empty((3,3)), evec
    cdef numpy.ndarray[ndim=1, dtype=numpy.float64_t] evals
    mat[0,0] = d.x
    mat[1,1] = d.y
    mat[2,2] = d.z
    mat[0,1] = mat[1,0] = s.z
    mat[0,2] = mat[2,0] = s.y
    mat[2,1] = mat[1,2] = s.x
    evals, evec = eigh(mat)
    cdef int idx=0
    cdef double di = fabs(evals[0]-r)
    if fabs(evals[1]-r) < di:
        idx = 1
    if fabs(evals[2]-r) < di:
        idx = 2
    cdef cPoint ret
    ret.x = evec[idx,0]
    ret.y = evec[idx,1]
    ret.z = evec[idx,2]
    return ret

cdef cPoint get_eigenvector(cPoint d, cPoint s, double r):
    ''' get eigenvector of symmetric 3x3 matrix for given eigenvalue `r`

    uses a fast method to get eigenvectors with a fallback to using numpy '''
    cdef cPoint ret
    ret.x = s.z*s.x - s.y*(d.y-r) # a_12 * a_23 - a_13 * (a_22 - r)
    ret.y = s.z*s.y - s.x*(d.x-r) # a_12 * a_13 - a_23 * (a_11 - r)
    ret.z = (d.x-r)*(d.y-r) - s.z*s.z # (a_11 - r) * (a_22 - r) - a_12^2
    cdef double norm = cPoint_length(ret)
    
    if norm *1e7 <= fabs(r):
        # its a zero, let numpy get the answer
        return get_eigenvector_np(d, s, r)
        
    return cPoint_scale(ret, 1.0/norm)

cpdef py_get_eigenvector(diag, side, double r):
    ''' get eigenvector of a symmetric matrix for given eigenvalue `r` '''
    cdef cPoint d
    d.x, d.y, d.z = diag
    cdef cPoint s
    s.x, s.y, s.z = side
    cdef cPoint ret = get_eigenvector(d, s, r)
    return ret.x, ret.y, ret.z

cdef cPoint get_eigenvalvec_np(cPoint d, cPoint s, double * R):
    ''' get eigenvector/eigenvalues using numpy (slow) '''
    cdef int i,j
    cdef numpy.ndarray[ndim=2,dtype=numpy.float64_t] mat=empty((3,3)), evec
    cdef numpy.ndarray[ndim=1, dtype=numpy.float64_t] evals

    mat[0,0] = d.x
    mat[1,1] = d.y
    mat[2,2] = d.z
    mat[0,1] = mat[1,0] = s.z
    mat[0,2] = mat[2,0] = s.y
    mat[2,1] = mat[1,2] = s.x
    try:
        evals, evec = eigh(mat)
    except numpy.linalg.linalg.LinAlgError, e:
        print mat
        raise

    for i in range(3):
        for j in range(3):
            R[i*3+j] = evec[i,j]

    cdef cPoint ret
    ret.x = evals[0]
    ret.y = evals[1]
    ret.z = evals[2]
    return ret

cdef cPoint get_eigenvalvec(cPoint d, cPoint s, double * R):
    ''' get the eigenvalues and eigenvectors of symm 3x3 matrix

    d,s are diagonal/off-diagonal elements
    R is output eigenmatrix, eigenvalues are returned
    '''
    cdef cPoint ret, tmp
    cdef bint use_iter = False
    cdef int i,j
    if s.x==s.y==s.z==0:
        # diagonal matrix
        ret = d
        for i in range(3):
            for j in range(3):
                R[i*3+j] = (i==j)
        return ret
    # FIXME: implement fast version
    ret = get_eigenvalues(d,s)
    if ret.x!=ret.y and ret.y!=ret.z and ret.x!=ret.z:
        # repeated eigenvalues
        use_iter = True
    if cPoint_norm(d) > 1e8*cPoint_norm(s):
        # nearly diagonal matrix
        use_iter = True
    if not use_iter:
        for i in range(3):
            tmp = get_eigenvector(d, s, (&ret.x)[i])
            for j in range(3):
                R[j*3+i] = (&tmp.x)[j]
        return ret
    else:
        return get_eigenvalvec_iter(d, s, R)

def py_get_eigenvalvec(d, s):
    cdef double vec[3][3]
    cdef Point dp = Point(*d)
    cdef Point ds = Point(*s)
    cdef cPoint r = get_eigenvalvec(dp.data, ds.data, &vec[0][0])
    cdef ret = numpy.empty((3,3))
    for i in range(3):
        for j in range(3):
            ret[i][j] = vec[i][j]
    return (r.x,r.y,r.z), ret

cdef void transform(double A[3][3], double P[3][3], double res[3][3]):
    ''' compute the transformation P.T*A*P and add it into result '''
    cdef int i, j, k, l
    for i in range(3):
        for j in range(3):
            #res[i][j] = 0
            for k in range(3):
                for l in range(3):
                    res[i][j] += P[k][i]*A[k][l]*P[l][j] # P.T*A*P

cdef void transform2(cPoint A, double P[3][3], double res[3][3]):
    ''' compute the transformation P.T*A*P and add it into result
    
    A is diagonal '''
    cdef int i, j, k
    for i in range(3):
        for j in range(3):
            #res[i][j] = 0
            for k in range(3):
                # l = k
                #for l in range(3):
                res[i][j] += P[k][i]*(&A.x)[k]*P[k][j] # P.T*A*P

cdef void transform2inv(cPoint A, double P[3][3], double res[3][3]):
    ''' compute the transformation P*A*P.T and add it into result
    
    A is diagonal '''
    cdef int i, j, k
    for i in range(3):
        for j in range(3):
            res[i][j] = 0.0

    for i in range(3):
        for j in range(3):
            #res[i][j] = 0
            for k in range(3):
                # l = k
                #for l in range(3):
                res[i][j] += P[i][k]*(&A.x)[k]*P[j][k] # P.T*A*P

def py_transform2(A, P):
    cdef double res[3][3]
    cdef double cP[3][3]
    for i in range(3):
        for j in range(3):
            res[i][j] = 0
            cP[i][j] = P[i][j]
    cdef Point cA = Point(*A)
    
    transform2(cA.data, cP, res)
    ret = empty((3,3))
    for i in range(3):
        for j in range(3):
            ret[i][j] = res[i][j]
    return ret

def py_transform2inv(A, P):
    cdef double res[3][3]
    cdef double cP[3][3]
    for i in range(3):
        for j in range(3):
            res[i][j] = 0
            cP[i][j] = P[i][j]
    cdef Point cA = Point(*A)
    
    transform2inv(cA.data, cP, res)
    ret = empty((3,3))
    for i in range(3):
        for j in range(3):
            ret[i][j] = res[i][j]
    return ret





cdef double * tred2(double V[n][n], double * d, double e[n]):
    ''' Symmetric Householder reduction to tridiagonal form '''

    #  This is derived from the Algol procedures tred2 by
    #  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
    #  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
    #  Fortran subroutine in EISPACK.
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


cdef void tql2(double V[n][n], double d[n], double e[n]):
    ''' symmetric tridiagonal QL algo for eigendecomposition '''

    #  This is derived from the Algol procedures tql2, by
    #  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
    #  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
    #  Fortran subroutine in EISPACK.

    cdef:
        double f, tst1, eps, g, h, p, r, dl1, c, c2, c3, el1, s, s2
        int i, j, k, l, m, iter
        bint cont

    for i in range(n):
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
            
 

cdef void eigen_decomposition(double A[n][n], double V[n][n], double d[n]):
    ''' get eigenvalues and eigenvectors of matrix A
    V is output eigenvectors and d is eigenvalue '''
    cdef double e[n]
    cdef int i, j
    for i in range(n):
        for j in range(n):
            V[i][j] = A[i][j]
    
    d = tred2(V, d, e)
    tql2(V, d, e)




cdef cPoint get_eigenvalvec_iter(cPoint d, cPoint s, double * R):
    ''' get eigenvalues and eigenvectors using an iterative method
    d, s are diagonal/off-diagonal elements
    R is the output eigenvectors and the eigenvalues are returned as cPoint
    '''
    cdef cPoint ret
    cdef double V[3][3]
    cdef int i, j
    for i in range(3):
        V[i][i] = (&d.x)[i]
        for j in range(i):
            V[i][j] = V[j][i] = (&s.x)[3-i-j]

    eigen_decomposition(V, <double(*)[n]>R, &ret.x)
    return ret

cpdef test_main():    
    cdef double A[3][3]
    cdef double d[3]
    print "Diagonals:"
    for i in range(3):
        A[i][i] = 1
    
    print "Off-diagonals:"
    for p in range(3):
        A[p==0][2-(p==2)] = (p==0)*1e-2
        A[2-(p==2)][p==0] = A[p==0][2-(p==2)]

    for i in range(3):
        for j in range(3):
            print A[i][j], "\t",
        print
    print

    cdef double V[3][3]
    eigen_decomposition(A, V, d)
    print "Eigenvalues:\n"
    for i in range(3):
        print d[i], "\t",
    
    print "\n\nEigenvectors:"
    for i in range(3):
        for j in range(3):
            print V[i][j], "\t",
        
        print
    
    print
