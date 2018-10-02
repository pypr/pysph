from pysph.cpy.api import Elementwise, annotate, wrap, get_config
from pysph.cpy.jit import jit, ElementwiseJIT
import numpy as np

@jit
def axpb(i, x, y, a, b):
    xi = declare('double')
    xi = x[i]
    y[i] = a*sin(xi) + b

x = np.linspace(0, 1, 10000)
y = np.zeros_like(x)
a = 2.0
b = 3.0

backend = 'opencl'
get_config().use_openmp = True
x, y = wrap(x, y, backend=backend)
e = ElementwiseJIT(axpb, backend=backend)
e(x, y, a, b)
