import numpy as np
from math import pi
import time

from numba import jit


@jit
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


@jit
def velocity(x, y, gamma, u, v, nv):
    tmp = np.zeros(2)
    for i in range(nv):
        xi = x[i]
        yi = y[i]
        u[i] = 0.0
        v[i] = 0.0
        for j in range(nv):
            point_vortex(xi, yi, x[j], y[j], gamma[j], tmp)
            u[i] += tmp[0]
            v[i] += tmp[1]


def make_vortices(nv):
    x = np.linspace(-1, 1, nv)
    y = x.copy()
    gamma = np.ones(nv)
    u = np.zeros_like(x)
    v = np.zeros_like(x)
    return x, y, gamma, u, v, nv


def run(nv):
    args = make_vortices(nv)
    t1 = time.time()
    velocity(*args)
    print(time.time() - t1)
    u = args[-3]
    print(u)
    return velocity, args


if __name__ == '__main__':
    from argparse import ArgumentParser
    p = ArgumentParser()
    p.add_argument('-n', action='store',
                   type=int, dest='n', default=10000)
    o = p.parse_args()
    run(o.n)
