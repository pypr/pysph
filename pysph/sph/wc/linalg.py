from compyle.api import declare


def identity(a=[0.0, 0.0], n=3):
    """Initialize an identity matrix.
    """
    i, j = declare('int', 2)
    for i in range(n):
        for j in range(n):
            if i == j:
                a[n*i + j] = 1.0
            else:
                a[n*i + j] = 0.0


def dot(a=[0.0, 0.0], b=[0.0, 0.0], n=3):
    i = declare('int')
    result = 0.0
    for i in range(n):
        result += a[i]*b[i]
    return result


def deteminant_2d(A=[0.0, 0.0]):
    return A[0]*A[3] - A[1]*A[2]


def deteminant_3d(A=[0.0, 0.0]):
    c1 = deteminant_2d([A[4], A[5], A[7], A[8]])
    c2 = deteminant_2d([A[3], A[5], A[6], A[8]])
    c3 = deteminant_2d([A[3], A[4], A[6], A[7]])
    result = A[0] * c1 -  A[1] * c2 + A[2] * c3
    return result


def deteminant_4d(A=[0.0, 0.0]):
    c1 = deteminant_3d([A[5], A[6], A[7],
                   A[9], A[10], A[11],
                   A[13], A[14], A[15]])
    c2 = deteminant_3d([A[4], A[6], A[7],
                   A[8], A[10], A[11],
                   A[12], A[14], A[15]])
    c3 = deteminant_3d([A[4], A[5], A[7],
                   A[8], A[9], A[11],
                   A[12], A[13], A[15]])
    c4 = deteminant_3d([A[4], A[5], A[6],
                   A[8], A[9], A[10],
                   A[12], A[13], A[14]])
    result = A[0] * c1 - A[1] * c2 + A[2] * c3 - A[3] * c4
    return result


def replace_vector_in_matrix(A=[0.0, 0.0], b=[0.0, 0.0],
                             pos=2, dim=3, result=[0.0, 0.0]):
    i, index, next = declare('int', 3)
    if pos + 1 <= dim:
        index = 0
        next = 0
        for i in range(dim*dim):
            if index == pos:
                result[i] = b[next]
                next += 1
            else:
                result[i] = A[i]

            index += 1
            if index == dim:
                index = 0
        return 0.0
    else:
        return -1.0


def mat_mult(a=[1.0, 0.0], b=[1.0, 0.0], n=3, result=[0.0, 0.0]):
    """Multiply two square matrices (not element-wise).

    Stores the result in `result`.

    Parameters
    ----------

    a: list
    b: list
    n: int : number of rows/columns
    result: list
    """
    i, j, k = declare('int', 3)
    for i in range(n):
        for k in range(n):
            s = 0.0
            for j in range(n):
                s += a[n*i + j] * b[n*j + k]
            result[n*i + k] = s


def mat_vec_mult(a=[1.0, 0.0], b=[1.0, 0.0], n=3, result=[0.0, 0.0]):
    """Multiply a square matrix with a vector.

    Parameters
    ----------

    a: list
    b: list
    n: int : number of rows/columns
    result: list
    """
    i, j = declare('int', 2)
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += a[n*i + j] * b[j]
        result[i] = s


def linear_solver_2d(A=[0.0, 0.0], b=[0.0, 0.0], result=[0.0, 0.0]):
    """Linear solver using Cramers's rule

    Given flattened matrix `A` and vector `b`. Cramer's rule is used to
    determine the result. In case the deteminant is zero, values considering
    the diagonal terms is returned. Doing this is equivalent to Shepard
    interpolation in the context of SPH.

    Parameters
    -----------
    A: list: given matrix.
    b: list: given vector
    result: list: x = inv(A)b
    """

    detA = A[0]*A[3] - A[1]*A[2]
    if abs(detA) > 1e-14:
        detA1 = b[0]*A[3] - A[1]*b[1]
        detA2 = A[0]*b[1] - b[0]*A[2]
        result[0] = detA1/detA
        result[1] = detA2/detA
    else:
        result[0] = b[0]/A[0]
        result[1] = b[1]/A[3]


def linear_solver_3d(A=[0.0, 0.0], b=[0.0, 0.0], result=[0.0, 0.0]):
    """Linear solver using Cramers's rule

    Given flattened matrix `A` and vector `b`. Cramer's rule is used to
    determine the result. In case the deteminant is zero, values considering
    the diagonal terms is returned. Doing this is equivalent to Shepard
    interpolation in the context of SPH.

    Parameters
    -----------
    A: list: given matrix.
    b: list: given vector
    result: list: x = inv(A)b
    """

    A1, A2, A3 = declare('matrix(9)', 3)
    i = declare('int')
    for i in range(9):
        A1[i] = 0.0
        A2[i] = 0.0
        A3[i] = 0.0

    detA = deteminant_3d(A)
    if abs(detA) > 1e-14:
        replace_vector_in_matrix(A, b, 0, 3, A1)
        detA1 = deteminant_3d(A1)
        replace_vector_in_matrix(A, b, 1, 3, A2)
        detA2 = deteminant_3d(A2)
        replace_vector_in_matrix(A, b, 2, 3, A3)
        detA3 = deteminant_3d(A3)
        result[0] = detA1/detA
        result[1] = detA2/detA
        result[2] = detA3/detA
    else:
        result[0] = b[0]/A[0]
        result[1] = b[1]/A[4]
        result[2] = b[2]/A[8]


def linear_solver_4d(A=[0.0, 0.0], b=[0.0, 0.0], result=[0.0, 0.0]):
    """Linear solver using Cramers's rule

    Given flattened matrix `A` and vector `b`. Cramer's rule is used to
    determine the result. In case the deteminant is zero, values considering
    the diagonal terms is returned. Doing this is equivalent to Shepard
    interpolation in the context of SPH.

    Parameters
    -----------
    A: list: given matrix.
    b: list: given vector
    result: list: x = inv(A)b
    """

    A1, A2, A3, A4 = declare('matrix(16)', 4)
    i = declare('int')
    for i in range(16):
        A1[i] = 0.0
        A2[i] = 0.0
        A3[i] = 0.0
        A4[i] = 0.0

    detA = deteminant_4d(A)
    if abs(detA) > 1e-14:
        replace_vector_in_matrix(A, b, 0, 4, A1)
        detA1 = deteminant_4d(A1)
        replace_vector_in_matrix(A, b, 1, 4, A2)
        detA2 = deteminant_4d(A2)
        replace_vector_in_matrix(A, b, 2, 4, A3)
        detA3 = deteminant_4d(A3)
        replace_vector_in_matrix(A, b, 3, 4, A4)
        detA4 = deteminant_4d(A4)
        result[0] = detA1/detA
        result[1] = detA2/detA
        result[2] = detA3/detA
        result[3] = detA4/detA
    else:
        result[0] = b[0]/A[0]
        result[1] = b[1]/A[5]
        result[2] = b[2]/A[10]
        result[3] = b[3]/A[15]


def augmented_matrix(A=[0.0, 0.0], b=[0.0, 0.0], n=3, na=1, nmax=3,
                     result=[0.0, 0.0]):
    """Create augmented matrix.

    Given flattened matrix, `A` of max rows/columns `nmax`, and flattened
    columns `b` with `n` rows of interest and `na` additional columns, put
    these in `result`. Result must be already allocated and be flattened.
    The `result` will contain `(n + na)*n` first entries as the
    augmented_matrix.


    Parameters
    ----------
    A: list: given matrix.
    b: list: additional columns to be augmented.
    n: int : number of rows/columns to use from `A`.
    na: int: number of added columns in `b`.
    nmax: int: the maximum dimension 'A'
    result: list: must have size of (nmax + na)*n.
    """
    i, j, nt = declare('int', 3)
    nt = n + na
    for i in range(n):
        for j in range(n):
            result[nt*i + j] = A[nmax * i + j]
        for j in range(na):
            result[nt*i + n + j] = b[na*i + j]


def gj_solve(m=[1., 0.], n=3, nb=1, result=[0.0, 0.0]):
    r"""A gauss-jordan method to solve an augmented matrix.

    The routine is given the augmented matrix, the number of rows/cols in the
    original matrix and the number of added columns. The result is stored in
    the result array passed.

    Parameters
    ----------

    m : list: a flattened list representing the augmented matrix [A|b].
    n : int: number of columns/rows used from A in augmented_matrix.
    nb: int: number of columns added to A.
    result: list: with size n*nb

    References
    ----------
    https://ricardianambivalence.com/2012/10/20/pure-python-gauss-jordan
    -solve-ax-b-invert-a/

    """

    i, j, eqns, colrange, augCol, col, row, bigrow, nt = declare('int', 9)
    eqns = n
    colrange = n
    augCol = n + nb
    nt = n + nb

    for col in range(colrange):
        bigrow = col
        for row in range(col + 1, colrange):
            if abs(m[nt*row + col]) > abs(m[nt*bigrow + col]):
                bigrow = row
                temp = m[nt*row + col]
                m[nt*row + col] = m[nt*bigrow + col]
                m[nt*bigrow + col] = temp

    rr, rrcol, rb, rbr, kup, kupr, kleft, kleftr = declare('int', 8)
    for rrcol in range(0, colrange):
        for rr in range(rrcol + 1, eqns):
            dnr = float(m[nt*rrcol + rrcol])
            if abs(dnr) < 1e-12:
                return 1.0
            cc = -float(m[nt*rr + rrcol]) / dnr
            for j in range(augCol):
                m[nt*rr + j] = m[nt*rr + j] + cc * m[nt*rrcol + j]

    backCol, backColr = declare('int', 2)
    tol = 1.0e-12
    for rbr in range(eqns):
        rb = eqns - rbr - 1
        if (m[nt*rb + rb] == 0):
            if abs(m[nt*rb + augCol - 1]) > tol:
                # Error, singular matrix.
                return 1.0
        else:
            for backColr in range(rb, augCol):
                backCol = rb + augCol - backColr - 1
                m[nt*rb + backCol] = m[nt*rb + backCol] / m[nt*rb + rb]
            if not (rb == 0):
                for kupr in range(rb):
                    kup = rb - kupr - 1
                    for kleftr in range(rb, augCol):
                        kleft = rb + augCol - kleftr - 1
                        kk = -m[nt*kup + rb] / m[nt*rb + rb]
                        m[nt*kup + kleft] = (m[nt*kup + kleft] +
                                             kk * m[nt*rb + kleft])

    for i in range(n):
        for j in range(nb):
            result[nb*i + j] = m[nt*i + n + j]

    return 0.0
