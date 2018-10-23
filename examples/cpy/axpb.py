from pysph.cpy.api import Elementwise, annotate, wrap, get_config
import numpy as np
from numpy import sin
import time


@annotate(i='int', doublep='x, y, a, b')
def axpb(i, x, y, a, b):
    y[i] = a[i]*sin(x[i]) + b[i]


def setup(backend, openmp=False):
    get_config().use_openmp = openmp
    e = Elementwise(axpb, backend=backend)
    return e


def data(n, backend):
    x = np.linspace(0, 1, n)
    y = np.zeros_like(x)
    a = x*x
    b = np.sqrt(x + 1)
    return wrap(x, y, a, b, backend=backend)


def compare(m=20):
    N = 2**np.arange(1, 25)
    backends = [['cython', False], ['cython', True]]
    try:
        import pyopencl
        backends.append(['opencl', False])
    except ImportError as e:
        pass

    try:
        import pycuda
        backends.append(['cuda', False])
    except ImportError as e:
        pass

    timing = []
    for backend in backends:
        e = setup(*backend)
        times = []
        for n in N:
            args = data(n, backend[0])
            t = []
            for j in range(m):
                start = time.time()
                e(*args)
                secs = time.time() - start
                t.append(secs)
            times.append(np.average(t))
        timing.append(times)

    return N, backends, np.array(timing)


def plot_timing(n, timing, backends):
    from matplotlib import pyplot as plt
    backends[1][0] = 'openmp'
    for time, backend in zip(timing[1:], backends[1:]):
        plt.semilogx(n, timing[0]/time, label='serial/' + backend[0], marker='+')
    plt.grid()
    plt.xlabel('N')
    plt.ylabel('Speedup')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    n, backends, times = compare()
    plot_timing(n, times, backends)
