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
