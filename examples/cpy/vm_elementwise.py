import numpy as np
from math import pi
import time

from pysph.cpy.config import get_config
from pysph.cpy.api import declare, annotate
from pysph.cpy.parallel import Elementwise
from pysph.cpy.array import wrap


@annotate(double='xi, yi, xj, yj, gamma', result='doublep')
def point_vortex(xi, yi, xj, yj, gamma, result):
    xij = xi - xj
    yij = yi - yj
    r2ij = xij*xij + yij*yij
    if r2ij < 1e-14:
        result[0] = 0.0
        result[1] = 0.0
    else:
        tmp = gamma/(2.0*pi*r2ij)
        result[0] = -tmp*yij
        result[1] = tmp*xij


@annotate(int='i, nv', gdoublep='x, y, gamma, u, v')
def velocity(i, x, y, gamma, u, v, nv):
    j = declare('int')
    tmp = declare('matrix(2)')
    xi = x[i]
    yi = y[i]
    u[i] = 0.0
    v[i] = 0.0
    for j in range(nv):
        point_vortex(xi, yi, x[j], y[j], gamma[j], tmp)
        u[i] += tmp[0]
        v[i] += tmp[1]


def make_vortices(nv, backend):
    x = np.linspace(-1, 1, nv)
    y = x.copy()
    gamma = np.ones(nv)
    u = np.zeros_like(x)
    v = np.zeros_like(x)
    x, y, gamma, u, v = wrap(x, y, gamma, u, v, backend=backend)
    return x, y, gamma, u, v, nv


def run(nv, backend):
    e = Elementwise(velocity, backend=backend)
    args = make_vortices(nv, backend)
    t1 = time.time()
    e(*args)
    print(time.time() - t1)
    u = args[-3]
    u.pull()
    return e, args


if __name__ == '__main__':
    from argparse import ArgumentParser
    p = ArgumentParser()
    p.add_argument(
        '-b', '--backend', action='store', dest='backend', default='cython',
        help='Choose the backend.'
    )
    p.add_argument(
        '--openmp', action='store_true', dest='openmp', default=False,
        help='Use OpenMP.'
    )
    p.add_argument(
        '--use-double', action='store_true', dest='use_double',
        default=False,  help='Use double precision on the GPU.'
    )
    p.add_argument('-n', action='store', type=int, dest='n',
                   default=10000, help='Number of particles.')
    o = p.parse_args()
    get_config().use_openmp = o.openmp
    get_config().use_double = o.use_double
    run(o.n, o.backend)
