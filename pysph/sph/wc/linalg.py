from pysph.cpy.api import declare


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


def mat_mult(a=[1.0, 0.0], b=[1.0, 0.0], n=3, result=[0.0, 0.0]):
    """Multiply two square matrices (not element-wise).

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


def gj_solve(A=[1., 0.], b=[1., 0.], n=3, result=[0., 1.]):
    r""" A gauss-jordan method to solve an augmented matrix for the
        unknown variables, x, in Ax = b.

    Parameters
    ----------

    A : list: a flattened list representing the square matrix
    b : list: right-hand side
    n : int: number of columns/rows of A.
    result: list: should have n elements

    References
    ----------
    https://ricardianambivalence.com/2012/10/20/pure-python-gauss-jordan
    -solve-ax-b-invert-a/
    """
    gj_solve_m(A, b, n, 1, result)


def gj_solve_m(A=[1., 0.], b=[1., 0.], n=3, nb=1, result=[0., 1.]):
    r"""A gauss-jordan method to solve an augmented matrix.

    This variant solves it for multiple right-hand sides in b.

    Parameters
    ----------

    A : list: a flattened list representing the square matrix
    b : list: right-hand sides, nb of them.
    n : int: number of columns/rows of A.
    nb: int: number of columns in b
    result: list: should have n x nb elements

    References
    ----------
    https://ricardianambivalence.com/2012/10/20/pure-python-gauss-jordan
    -solve-ax-b-invert-a/

    """

    i, j, eqns, colrange, augCol, col, row, bigrow = declare('int', 8)
    m = declare('matrix((4, 8))')
    for i in range(n):
        for j in range(n):
            m[i][j] = A[n * i + j]
        for j in range(nb):
            m[i][n + j] = b[nb*i + j]

    eqns = n
    colrange = n
    augCol = n + nb

    for col in range(colrange):
        bigrow = col
        for row in range(col + 1, colrange):
            if abs(m[row][col]) > abs(m[bigrow][col]):
                bigrow = row
                temp = m[row][col]
                m[row][col] = m[bigrow][col]
                m[bigrow][col] = temp

    rr, rrcol, rb, rbr, kup, kupr, kleft, kleftr = declare('int', 8)
    for rrcol in range(0, colrange):
        for rr in range(rrcol + 1, eqns):
            cc = -(float(m[rr][rrcol]) / float(m[rrcol][rrcol]))
            for j in range(augCol):
                m[rr][j] = m[rr][j] + cc * m[rrcol][j]

    backCol, backColr = declare('int', 2)
    tol = 1.0e-05
    for rbr in range(eqns):
        rb = eqns - rbr - 1
        if (m[rb][rb] == 0):
            if abs(m[rb][augCol - 1]) >= tol:
                return 0.0
        else:
            for backColr in range(rb, augCol):
                backCol = rb + augCol - backColr - 1
                m[rb][backCol] = float(m[rb][backCol]) / float(m[rb][rb])
            if not (rb == 0):
                for kupr in range(rb):
                    kup = rb - kupr - 1
                    for kleftr in range(rb, augCol):
                        kleft = rb + augCol - kleftr - 1
                        kk = -float(m[kup][rb]) / float(m[rb][rb])
                        m[kup][kleft] = m[kup][kleft] + \
                            kk * float(m[rb][kleft])
    for i in range(n):
        for j in range(nb):
            result[nb*i + j] = m[i][n + j]
