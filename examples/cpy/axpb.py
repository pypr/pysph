from pysph.cpy.api import Elementwise, annotate, wrap, get_config
import numpy as np
from numpy import sin
import time
import pycuda.autoinit


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
    backends = [['cython', False], ['cython', True], ['opencl', False],
            ['cuda', False]]
    timing = []
    for backend in backends:
        e = setup(*backend)
        times = []
        for n in N:
            args = data(n, backend[0])
            t = []
            for j in range(m):
                if backend[0] == 'cuda':
                    import pycuda.driver as drv
                    start = drv.Event()
                    end = drv.Event()
                    start.record()
                    #start.synchronize()
                    e(*args)
                    end.record()
                    end.synchronize()
                    secs = start.time_till(end) * 1e-3
                    t.append(secs)
                else:
                    start = time.time()
                    e(*args)
                    secs = time.time() - start
                    t.append(secs)
            y = args[1]
            print "ANS =", y.dev[0], y.dev[1]
            times.append(np.average(t))
        timing.append(times)

    return N, np.array(timing)


def plot_timing(n, timing):
    from matplotlib import pyplot as plt
    plt.semilogx(n, timing[0]/timing[1], label='serial/openmp', marker='+')
    plt.semilogx(n, timing[0]/timing[2], label='serial/opencl', marker='+')
    plt.semilogx(n, timing[0]/timing[3], label='serial/cuda', marker='+')
    plt.grid()
    plt.xlabel('N')
    plt.ylabel('Speedup')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    n, times = compare()
    plot_timing(n, times)
