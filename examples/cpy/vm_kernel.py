"""Shows the use of a raw opencl Kernel but written using pure Python. It
makes use of local memory allocated on the host.

Note that the local memory is allocated as a multiple of workgroup size times
the size of the data type automatically.

This is a raw opencl kernel so will not work on Cython!

"""
import numpy as np
from math import pi
import time

from pysph.cpy.config import get_config
from pysph.cpy.api import declare, annotate
from pysph.cpy.low_level import (Kernel, LocalMem, local_barrier,
                                 LID_0, LDIM_0, GDIM_0)
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


@annotate(nv='int', gdoublep='x, y, gamma, u, v', ldoublep='xc, yc, gc')
def velocity(x, y, gamma, u, v, xc, yc, gc, nv):
    i, gid, nb = declare('int', 3)
    j, ti, nt, jb = declare('int', 4)
    ti = LID_0
    nt = LDIM_0
    gid = GID_0
    i = gid*nt + ti
    idx = declare('int')
    tmp = declare('matrix(2)')
    uj, vj = declare('double', 2)
    nb = GDIM_0

    if i < nv:
        xi = x[i]
        yi = y[i]
    uj = 0.0
    vj = 0.0
    for jb in range(nb):
        idx = jb*nt + ti
        if idx < nv:
            xc[ti] = x[idx]
            yc[ti] = y[idx]
            gc[ti] = gamma[idx]
        else:
            gc[ti] = 0.0
        local_barrier()

        if i < nv:
            for j in range(nt):
                point_vortex(xi, yi, xc[j], yc[j], gc[j], tmp)
                uj += tmp[0]
                vj += tmp[1]

        local_barrier()

    if i < nv:
        u[i] = uj
        v[i] = vj


def make_vortices(nv, backend):
    x = np.linspace(-1, 1, nv)
    y = x.copy()
    gamma = np.ones(nv)
    u = np.zeros_like(x)
    v = np.zeros_like(x)
    x, y, gamma, u, v = wrap(x, y, gamma, u, v, backend=backend)
    xc, yc, gc = (LocalMem(1, backend), LocalMem(1, backend),
                  LocalMem(1, backend))
    return x, y, gamma, u, v, xc, yc, gc, nv


def run(nv, backend):
    e = Kernel(velocity, backend=backend)
    args = make_vortices(nv, backend)
    t1 = time.time()
    gs = ((nv + 128 - 1)//128)*128
    e(*args, global_size=(gs,))
    print(time.time() - t1)
    u = args[3]
    u.pull()
    print(u.data)
    return e, args


if __name__ == '__main__':
    from argparse import ArgumentParser
    p = ArgumentParser()
    p.add_argument(
        '-b', '--backend', action='store', dest='backend',
        default='opencl', help='Choose the backend.'
    )
    p.add_argument(
        '--use-double', action='store_true', dest='use_double',
        default=False,  help='Use double precision on the GPU.'
    )
    p.add_argument('-n', action='store', type=int, dest='n',
                   default=10000, help='Number of particles.')
    o = p.parse_args()
    get_config().use_double = o.use_double
    assert o.backend in ['opencl', 'cuda'], ("Only OpenCL/CUDA backend is "
                                             "supported.")
    run(o.n, o.backend)
