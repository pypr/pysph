CPy
====

CPy (pronounced spy) allows users to execute a restricted subset of Python
(almost similar to C) on a variety of HPC platforms. Currently we support
multi-core execution using Cython, and OpenCL for GPU devices. CUDA will be
supported very soon.

Users start with code implemented in a very restricted Python syntax, this
code is then automatically transpiled, compiled and executed to run on either
one CPU core, or multiple CPU cores or on a GPU. CPy offers source-to-source
transpilation, making it a very convenient tool for writing HPC libraries.

CPy is not a magic bullet,

- Do not expect that you may get a tremendous speedup.
- Performance optimization can be hard and is platform specific. What works on
  the CPU may not work on the GPU and vice-versa. CPy does not do anything to
  make this aspect easier. All the issues with memory bandwidth, cache, false
  sharing etc. still remain. Differences between memory architectures of CPUs
  and GPUs are not avoided at all -- you still have to deal with it. But you
  can do so from the comfort of one simple programming language, Python.
- CPy makes it easy to write everything in pure Python and generate the
  platform specific code from Python. It provides a low-level tool to make it
  easy for you to generate whatever appropriate code.
- The restrictions CPy imposes make it easy for you to think about your
  algorithms in that context and thereby allow you to build functionality that
  exploits the hardware as you see fit.
- CPy hides the details of the backend to the extent possible. You can write
  your code in Python, you can reuse your functions and decompose your problem
  to maximize reuse. Traditionally you would end up implementing some code in
  C, some in Python, some in OpenCL/CUDA, some in string fragments that you
  put together. Then you'd have to manage each of the runtimes yourself, worry
  about compilation etc. CPy minimizes that pain.
- By being written in Python, we make it easy to assemble these building
  blocks together to do fairly sophisticated things relatively easily from the
  same language.
- CPy is fairly simple and does source translation making it generally easier
  to understand and debug. The code-base is less than 5k lines of code
  (including the tests).
- CPy has relatively simple dependencies, for CPU support it requires Cython_
  and a C-compiler which supports OpenMP_. On the GPU you need either PyOpenCL_
  or PyCUDA_.  In addition it depends on NumPy_ and Mako_.


.. _Cython: http://www.cython.org
.. _OpenMP: http://openmp.org/
.. _PyOpenCL: https://documen.tician.de/pyopencl/
.. _PyCUDA: https://documen.tician.de/pycuda/
.. _OpenCL: https://www.khronos.org/opencl/
.. _NumPy: http://numpy.scipy.org
.. _Mako: https://pypi.python.org/pypi/Mako

While CPy is simple and modest, it is quite powerful and convenient. In fact,
CPy has its origins in PySPH_ which is a powerful Python package supporting
SPH, molecular dynamics, and other particle-based algorithms. The basic
elements of CPy are used in PySPH_ to automatically generate HPC code from
code written in pure Python and execute it on multiple cores, and on GPUs
without the user having to change any of their code. CPy generalizes this code
generation to make it available as a general tool.

.. _PySPH: http://pysph.readthedocs.io


These are the restrictions on the Python language that CPy poses:

- Functions with a C-syntax.
- Function arguments must be declared using either type annotation or with a
  decorator or with default arguments.
- No Python data structures, i.e. no lists, tuples, or dictionaries.
- Contiguous Numpy arrays are supported but must be one dimensional.
- No memory allocation is allowed inside these functions.
- On opencl no recursion is supported.

Basically think of it as good old FORTRAN.

Technically we do support structs but this is not yet exposed and is likely to
be supported in the future.

Simple example
--------------

Enough talk, lets look at some code.  Here is a very simple example::

   from pysph.cpy.api import Elementwise, annotate, wrap, get_config
   import numpy as np

   @annotate(i='int', x='doublep', y='doublep', double='a,b')
   def axpb(i, x, y, a, b):
       y[i] = a*sin(x[i]) + b

   x = np.linspace(0, 1, 10000)
   y = np.zeros_like(x)
   a = 2.0
   b = 3.0

   backend = 'cython'
   get_config().use_openmp = True
   x, y = wrap(x, y, backend=backend)
   e = Elementwise(axpb, backend=backend)
   e(x, y, a, b)

This will execute the elementwise operation in parallel using OpenMP with
Cython. The code is auto-generated, compiled and called for you transparently.
The first time this runs, it will take a bit of time to compile everything but
the next time, this is cached and will run much faster.

If you just change the ``backend = 'opencl'``, the same exact code will be
executed using PyOpenCL_.

More complex examples (but still fairly simple) are available in the examples
directory.

- ``axpb.py``: the above example but for openmp and opencl compared with
  serial showing that in some cases serial is actually faster than parallel!

- ``vm_elementwise.py``: shows a simple N-body code with two-dimensional point
  vortices. The code uses a simple elementwise operation and works with OpenMP
  and OpenCL.

- ``vm_numba.py``: shows the same code written in numba for comparison. In our
  benchmarks, CPy is actually faster even in serial and in parallel it can be
  much faster when you use all cores.

- ``vm_kernel.py``: shows how one can write a low-level OpenCL kernel in pure
  Python and use that. This also shows how you can allocate and use local (or
  shared) memory which is often very important for performance on GPGPUs. This
  code will only run via PyOpenCL.
