import numpy as np
import time

from pysph.cpy.config import get_config
import vm_numba as VN
import vm_elementwise as VE
import vm_kernel as VK


def setup(mod, backend, openmp):
    get_config().use_openmp = openmp
    if mod == VE:
        e = VE.Elementwise(VE.velocity, backend)
    elif mod == VN:
        e = VN.velocity
    elif mod == VK:
        e = VK.Kernel(VK.velocity, backend)

    return e


def data(n, mod, backend):
    if mod == VN:
        args = mod.make_vortices(n)
    else:
        args = mod.make_vortices(n, backend)
    return args


def compare(m=5):
    # Warm up the jit to prevent the timing from going off for the first point.
    VN.velocity(*VN.make_vortices(100))
    N = np.array([10, 50, 100, 200, 500, 1000, 2000, 4000, 6000,
                  8000, 10000, 12000])
    backends = [(VN, '', False), (VE, 'cython', False), (VE, 'cython', True),
                (VE, 'opencl', False), (VK, 'opencl', False)]
    timing = []
    for backend in backends:
        e = setup(*backend)
        times = []
        for n in N:
            args = data(n, backend[0], backend[1])
            t = []
            for j in range(m):
                start = time.time()
                e(*args)
                t.append(time.time() - start)
            times.append(np.average(t))
        timing.append(times)

    return N, np.array(timing)


def plot_timing(n, timing):
    import matplotlib
    matplotlib.use('Qt5Agg')
    from matplotlib import pyplot as plt
    plt.plot(n, timing[0]/timing[1], label='numba/cython', marker='+')
    plt.plot(n, timing[0]/timing[2], label='numba/openmp', marker='+')
    plt.plot(n, timing[0]/timing[3], label='numba/opencl', marker='+')
    plt.plot(n, timing[0]/timing[4], label='numba/opencl local', marker='+')
    plt.grid()
    plt.xlabel('N')
    plt.ylabel('Speedup')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    n, t = compare()
    plot_timing(n, t)
