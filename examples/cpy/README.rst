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
- On OpenCL no recursion is supported.
- All function calls must not use dotted names, i.e. don't use ``math.sin``,
  instead just use ``sin``. This is because we do not perform any kind of name
  mangling of the generated code to make it easier to read.

Basically think of it as good old FORTRAN.

Technically we do support structs internally (we use it heavily in PySPH_) but
this is not yet exposed at the high-level and is very likely to be supported
in the future.


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

- ``bench_vm.py``: Benchmarks the various vortex method results above for a
  comparison with numba.


An overview of functionality
-----------------------------

The functionality provided falls into two broad categories,

- Common parallel algorithms that will work across backends. This includes,
  elementwise operations, reductions, and prefix-sums/scans.

- Specific support to run code on a particular backend. This is for code that
  will only work on one backend by definition. This is necessary in order to
  best use different hardware and also use differences in the particular
  backend implementations. For example, the notion of local (or shared) memory
  only has meaning on a GPGPU. In this category we provide support to compile
  and execute Cython code, and also create and execute a GPU kernel.

In addition there is common functionality to perform type annotations. At a
lower level, there are code translators (transpilers) that handle generation
of Cython and C code from annotated Python code. Technically these transpilers
can be reused by users to do other things but we only go over the higher level
tools in this documentation. All the code is fairly extensively tested and
developed using a test-driven approach. In fact, a good place to see examples
are the tests.

We now go into the details of each of these so as to provide a high-level
overview of what can be done with CPy.

Annotating functions
---------------------

The first step in getting started using CPy is to annotate your functions and
also declare variables in code.

Annotation is provided by a simple decorator, called ``annotate``. One can
declare local variables inside these functions using ``declare``. A simple
example serves to illustrate these::


  @annotate(i='int', x='floatp', return_='float')
  def f(i, x):
      return x[i]*2.0

  @annotate(i='int', floatp='x, y', return_='float')
  def g(i, x, y):
      return f(i, x)*y[i]


Note that for convenience ``annotate``, accepts types and variable names in
two different ways, which you can use interchangeably.

1. You can simply use ``var_name=type_str``, or ``var_name=type`` where the
   type is from the ``cpy.types`` module.

2. You can instead use ``type_name='x, y, z'``, which is often very
   convenient. The order of the variables is not important and need not match
   with the order in which they are declared.

You can use ``return_=type``, where ``type`` is an appropriate type or
standard string representing one of the types. If the return type is not
specified it assumes a ``void`` return.


The definitions of the various standard types is in ``cpy.types.TYPES``. Some
are listed below:

- ``'float', 'double', 'int', 'long', 'uint', 'ulong'``: etc. are exactly as
  you would expect.

- ``'doublep'`` would refer to a double pointer, i.e. ``double*`` and
  similarly for anything with a ``p`` at the end.

- ``gdoublep`` would be a ``global doublep``, which makes sense with OpenCL
  where you would have ``__global double* xx``. The global address space
  specification is ignored when Cython code is generated, so this is safe to
  use with Cython code too.

- ``ldoublep`` would be equivalent to ``__local double*`` in OpenCL, for local
  memory. Again this address space qualifier is ignored in Cython.

All these types are available in the ``cpy.types`` module namespace also for
your convenience. The ``int, float, long`` types are accessible as ``int_,
float_, long_`` so as not to override the default Python types. For example
the function ``f`` in the above could also have been declared like so::

  from pysph.cpy.types import floatp, float_, int_

  @annotate(i=int_, x=floatp, return_=float_)
  def f(i, x):
      return x[i]*2.0


One can also use custom types (albeit with care) by using the
``cpy.typs.KnownType`` class. This is convenient in other scenarios where you
could potentially pass instances/structs to a function. We will discuss this
later but all of the basic types discussed above are all instances of
``KnownType``.

CPy actually supports Python3 style annotations but only for the function
arguments and NOT for the local variables. The only caveat is you must use the
types in ``pysph.cpy.types``, i.e. you must use ``KnownType`` instances as the
types for things to work.


Declaring variables
-------------------

In addition to annotating the function arguments and return types, it is
important to be able to declare the local variables. We provide a simple
``declare`` function that lets us do this. One again, a few examples serve to
illustrate this::

  i = declare('int')
  x = declare('float')
  u, v = declare('double', 2)

Notice the last one where we passed an additional argument of the number of
types we want. This is really done to keep this functional in pure Python so
that your code executes on Python also.  In Cython these would produce::

  cdef int i
  cdef float x
  cdef double u, v

On OpenCL this would produce the equivalent::

  int i;
  float x;
  double u, v;

Technically one could also write::

  f = declare('float4')

but clearly this would only work on OpenCL, however, you can definitely
declare other variables too!

Note that in OpenCL/Cython code if you do not declare a variable, it is
automatically declared as a ``double`` to prevent compilation errors.

We often also require arrays, ``declare`` also supports this, for example
consider these examples::

  r = declare('matrix(3)')
  a = declare('matrix((2, 2))')
  u, v = declare('matrix(2)', 2)

This reduces to the following on OpenCL::

  double r[3];
  double a[3][3];
  double u[2], v[2];

Note that this will only work with fixed sizes, and not with dynamic sizes. As
we pointed out earlier, dynamic memory allocation is not allowed. Of course
you could easily do this with Cython code but the declare syntax does not
allow this.

If you want non-double matrices, you can simply pass a type as in::

  a = declare('matrix((2, 2), "int")')

Which would result in::

  int a[2][2];

As you can imagine, just being able to do this opens up quite a few
possibilities.  You could also do things like this::

  xloc = declare('LOCAL_MEM matrix(128)')

which will become in OpenCL::

  LOCAL_MEM double xloc[128];

The ``LOCAL_MEM`` is a special variable that expands to the appropriate flag
on OpenCL or CUDA to allow you to write kernels once and have them run on
either OpenCL or CUDA. These special variables are discussed later below.

Writing the functions
----------------------

All of basic Python is supported. As you may have seen in the examples, you
can write code that uses the following:

- Indexing (only positive indices please).
- Conditionals with if/elif/else.
- While loops.
- For loops with the ``for var in range(...)`` construction.
- Nested fors.
- Ternary operators.

This allows us to write most numerical code. Fancy slicing etc. are not
supported, numpy based slicing and striding are not supported. You are
supposed to write these out elementwise. The idea is to keep things simple.
Yes, this may make things verbose but it does keep our life simple and
eventually yours too.

Do not create any Python data structures in the code unless you do not want to
run the code on a GPU. No numpy arrays can be created, also avoid calling any
numpy functions as these will NOT translate to any GPU code. You have to write
what you need by hand. Having said that, all the basic math functions and
symbols are automatically available. Essentially all of ``math`` is available.
All of the ``math.h`` constants are also available for use.

If you declare a global constant it will be automatically defined in the
generated code.  For example::

  MY_CONST = 42

  @annotate(x='double', return_='double')
  def f(x):
     return x + MY_CONST


The ``MY_CONST`` will be automatically injected in your generated code.

Now you may wonder about how you can call an external library that is not in
``math.h``. Lets say you have an external CUDA library, how do you call that?
We have a simple approach for this which we discuss later. We call this an
``Extern`` and discuss it later.


Common parallel algorithms
---------------------------

CPy provides a few very powerful parallel algorithms. These are all directly
motivated by Andreas Kloeckner's PyOpenCL_ package. On the GPU they are
wrappers on top of the functionality provided there. These algorithms make it
possible to implement scalable algorithms for a variety of common numerical
problems. In PySPH_ for example all of the GPU based nearest neighbor finding
algorithms are written with these fundamental primitives and scale very well.

All of the following parallel algorithms allow choice of a suitable backend
and take a keyword argument to specify this backend. If no backend is provided
a default is chosen from the ``cpy.config`` module. You can get the global
config using::

  from pysph.cpy.config import get_config

  cfg = get_config()
  cfg.use_openmp = True
  cfg.use_opencl = True

etc. The following are the parallel algorithms available from the
``pysph.cpy.parallel`` module.

``Elementwise``
~~~~~~~~~~~~~~~

This is also available as a decorator ``elementwise``. One can pass it an
annotated function and an optional backend. The elementwise processes every
element in the second argument to the function. The elementwise basically
passes the function an index of the element it is processing and parallelizes
the calls to this automatically. If you are familiar with writing GPU kernels,
this is the same thing except the index is passed along to you.

Here is a very simple example that shows how this works for a case where we
compute ``y = a*sin(x) + b`` where ``y, a, x, b`` are all numpy arrays but let
us say we want to do this in parallel::

  import numpy as np
  from pysph.cpy.api import annotate, Elementwise, get_config

  @annotate(i='int', doublep='x, y, a, b')
  def axpb(i, x, y, a, b):
      y[i] = a[i]*sin(x[i]) + b[i]

  # Setup the input data
  n = 1000000
  x = np.linspace(0, 1, n)
  y = np.zeros_like(x)
  a = np.random.random(n)
  b = np.random.random(n)

  # Use OpenMP
  get_config().use_openmp = True

  # Now run this in parallel with Cython.
  backend = 'cython'
  e = Elementwise(axpb, backend=backend)
  e(x, y, a, b)

This will call the ``axpb`` function in parallel and if your problem is large
enough will effectively scale on all your cores.  Its as simple as that.

Now let us say we want to run this with OpenCL. The only issue with OpenCL is
that the data needs to be sent to the GPU. This is transparently handled by a
simple ``Array`` wrapper that handles this for us automatically. Here is a
simple example building on the above::

  from pysph.cpy.api import wrap

  backend = 'opencl'
  x, y, a, b = wrap(x, y, a, b, backend=backend)

What this does is to wrap each of the arrays and also sends the data to the
device. ``x`` is now an instance of ``pypsh.cpy.array.Array``, this simple
class has two attributes, ``data`` and ``dev``. The first is the original data
and the second is a suitable device array from PyOpenCL/PyCUDA depending on
the backend. To get data from the device to the host you can call ``x.pull()``
to push data to the device you can call ``x.push()``.

Now that we have this wrapped we can simply do::

  e = Elementwise(axpb, backend=backend)
  e(x, y, a, b)

We do not need to change any of our other code.  As you can see this is very convenient.

Here is all the code put together::

  import numpy as np
  from pysph.cpy.api import annotate, Elementwise, get_config, wrap

  @annotate(i='int', doublep='x, y, a, b')
  def axpb(i, x, y, a, b):
      y[i] = a[i]*sin(x[i]) + b[i]

  # Setup the input data
  n = 1000000
  x = np.linspace(0, 1, n)
  y = np.zeros_like(x)
  a = np.random.random(n)
  b = np.random.random(n)

  # Turn on OpenMP for Cython.
  get_config().use_openmp = True

  for backend in ('cython', 'opencl'):
      xa, ya, aa, ba = wrap(x, y, a, b, backend=backend)
      e = Elementwise(axpb, backend=backend)
      e(xa, ya, aa, ba)

This will run the code on both backends! We use the for loop just to show that
this will run on all backends! The ``axpb.py`` example shows this for a
variety of array sizes and plots the performance.


``Reduction``
~~~~~~~~~~~~~~~

The ``cpy.parallel`` module also provides a ``Reduction`` class which can be
used fairly easily. Using it is a bit complex, a good starting point for this
is the documentation of PyOpenCL_, here
https://documen.tician.de/pyopencl/algorithm.html#module-pyopencl.reduction

The difference from the PyOpenCL implementation is that the ``map_expr`` is a
function rather than a string.

We provide a couple of simple examples to illustrate the above. The first
example is to find the sum of all elements of an array::

  x = np.linspace(0, 1, 1000)/1000
  x = wrap(x, backend=backend)

  r = Reduction('a+b', backend=backend)
  result = r(x)

Here is an example of a function to find the minimum of an array::

  x = np.linspace(0, 1, 1000)/1000
  x = wrap(x, backend=backend)

  r = Reduction('min(a, b)', neutral='INFINITY', backend=backend)
  result = r(x)

Here is a final one with a map expression thrown in::

  from math import cos, sin
  x = np.linspace(0, 1, 1000)/1000
  y = x.copy()
  x, y = wrap(x, y, backend=backend)

  @annotate(i='int', doublep='x, y')
  def map(i=0, x=[0.0], y=[0.0]):
      return cos(x[i])*sin(y[i])

  r = Reduction('a+b', map_func=map, backend=backend)
  result = r(x, y)

As you can see this is faithful to the PyOpenCL implementation with the only
difference that the ``map_expr`` is actually a nice function. Further, this
works on all backends, even on Cython.


``Scan``
~~~~~~~~~~

Scans are generalizations of prefix sums / cumulative sums and can be used as
building blocks to construct a number of parallel algorithms. These include but
not are limited to sorting, polynomial evaluation, and tree
operations. Blelloch's literature on prefix sums (`Prefix Sums and Their
Applications <https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf>`_) has many more
examples and is a recommended read before using scans. The ``cpy.parallel``
module provides a ``Scan`` class which can be used to develop and execute such
scans. The scans can be run on GPUs using the OpenCL backend or on CPUs using
either the OpenCL or Cython backend. A CUDA backend is not yet supported.

The scan semantics in cpy are similar to those of the GenericScanKernel in
PyOpenCL
(https://documen.tician.de/pyopencl/algorithm.html#pyopencl.scan.GenericScanKernel). Similar
to the case for reduction, the main differences from the PyOpenCL implementation
are that the expressions (`input_expr`, `segment_expr`, `output_expr`) are all
functions rather than strings.

The following examples demonstrate how scans can be used in cpy. The first
example is to find the cumulative sum of all elements of an array::

  ary = np.arange(10000, dtype=np.int32)
  ary = wrap(ary, backend=backend)
  
  @annotate(i='int', ary='intp', return_='int')
  def input_expr(i, ary):
      return ary[i]

  @annotate(int='i, item', ary='intp')
  def output_expr(i, item, ary):
      ary[i] = item

  scan = Scan(input_expr, output_expr, 'a+b', dtype=np.int32,
              backend=backend)
  scan(ary=ary)
  ary.pull()

  # Result = ary.data

Here is a more complex example of a function that finds the unique elements in
an array::

  ary = np.random.randint(0, 100, 1000, dtype=np.int32)
  unique_ary = np.zeros(len(ary.data), dtype=np.int32)
  unique_ary = wrap(unique_ary, backend=backend)
  unique_count = np.zeros(1, dtype=np.int32)
  unique_count = wrap(unique_count, backend=backend)
  ary = wrap(ary, backend=backend)
  
  @annotate(i='int', ary='intp', return_='int')
  def input_expr(i, ary):
      if i == 0 or ary[i] != ary[i - 1]:
          return 1
      else:
          return 0

  @annotate(int='i, prev_item, item, N', ary='intp',
            unique='intp', unique_count='intp')
  def output_expr(i, prev_item, item, N, ary, unique, unique_count):
      if item != prev_item:
          unique[item - 1] = ary[i]
      if i == N - 1:
          unique_count[0] = item

  scan = Scan(input_expr, output_expr, 'a+b', dtype=np.int32, backend=backend)
  scan(ary=ary, unique=unique_ary, unique_count=unique_count)
  unique_ary.pull()
  unique_count.pull()
  unique_count = unique_count.data[0]
  unique_ary = unique_ary.data[:unique_count]

  # Result = unique_ary
  
The following points highlight some important details and quirks about using
scans in cpy:

1. The scan call does not return anything. All output must be handled manually.
   Usually this involves writing the results available in ``output_expr``
   (``prev_item``, ``item`` and ``last_item``) to an array.
2. ``input_expr`` might be evaluated multiple times. However, it can be assumed
   that ``input_expr`` for an element or index ``i`` is not evaluated again
   after the output expression ``output_expr`` for that element is
   evaluated. Therefore, it is safe to write the output of a scan back to an
   array also used for the input like in the first example.
3. (For PyOpenCL users) If a segmented scan is used, unlike PyOpenCL where the
   ``across_seg_boundary`` is used to handle the segment logic in the scan
   expression, in cpy the logic is handled automatically. More specifically,
   using ``a + b`` as the scan expression in cpy is equivalent to using
   ``(across_seg_boundary ? b : a + b)`` in PyOpenCL.

Abstracting out arrays
-----------------------

As discussed in the section on Elementwise operations, different backends need
to do different things with arrays. With OpenCL/CUDA one needs to send the
array to the device. This is transparently managed by the
``pysph.cpy.array.Array`` class. It is easiest to use this transparently with
the ``wrap`` convenience function as below::

  x = np.linspace(0, 1, 1000)/1000
  y = x.copy()
  x, y = wrap(x, y, backend=backend)

Thus these, new arrays can be passed to any operation and is handled transparently.


Choice of backend and configuration
------------------------------------

The ``pysph.cpy.config`` module provides a simple ``Configuration`` class that
is used internally in CPy to set things like the backend (Cython,
OpenCL/CUDA), and some common options like profiling, turning on OpenMP, using
double on the GPU etc.  Here is an example of the various options::

  from pysph.cpy.config import get_config
  cfg = get_config()
  cfg.use_double
  cfg.profile
  cfg.use_opencl
  cfg.use_openmp

If one wants to temporarily set an option and perform an action, one can do::

  from pysph.cpy.config import use_config

  with use_config(use_openmp=False):
     ...

Here everything within the ``with`` clause will be executed using the
specified option and once the clause is exited, the previous settings will be
restored.  This can be convenient.


Low level functionality
-----------------------

In addition to the above, there are also powerful low-level functionality that
is provided in ``pysph.cpy.low_level``.


``Kernel``
~~~~~~~~~~~

The ``Kernel`` class allows one to execute a pure GPU kernel. Unlike the
Elementwise functionality above, this is specific to OpenCL/CUDA and will not
execute via Cython. What this class lets one do is write low-level kernels
which are often required to extract the best performance from your hardware.

Most of the functionality is exactly the same, one declares functions and
annotates them and then passes a function to the ``Kernel`` which calls this
just as we would a normal OpenCL kernel for example. The major advantage is
that all your code is pure Python. Here is a simple example::

   from pysph.cpy.api import annotate, wrap, get_config
   from pysph.cpy.low_level import Kernel, LID_0, LDIM_0, GID_0
   import numpy as np

   @annotate(x='doublep', y='doublep', double='a,b')
   def axpb(x, y, a, b):
       i = declare('int')
       i = LDIM_0*GID_0 + LID_0
       y[i] = a*sin(x[i]) + b

   x = np.linspace(0, 1, 10000)
   y = np.zeros_like(x)
   a = 2.0
   b = 3.0

   get_config().use_opencl = True
   x, y = wrap(x, y)

   k = Kernel(axpb)
   k(x, y, a, b)

This is the same Elementwise kernel equivalent from the first example at the
top but written as a raw kernel. Notice that ``i`` is not passed but computed
using ``LDIM_0, GID_0 and LID_0`` which are automatically made available on
OpenCL/CUDA. In addition to these the function ``local_barrier`` is also
available. Internally these are ``#defines`` that are like so on OpenCL::

   #define LID_0 get_local_id(0)
   #define LID_1 get_local_id(1)
   #define LID_2 get_local_id(2)

   #define GID_0 get_group_id(0)
   #define GID_1 get_group_id(1)
   #define GID_2 get_group_id(2)

   #define LDIM_0 get_local_size(0)
   #define LDIM_1 get_local_size(1)
   #define LDIM_2 get_local_size(2)

   #define GDIM_0 get_num_groups(0)
   #define GDIM_1 get_num_groups(1)
   #define GDIM_2 get_num_groups(2)

   #define local_barrier() barrier(CLK_LOCAL_MEM_FENCE);

On CUDA, these are mapped to the equivalent ::

   #define LID_0 threadIdx.x
   #define LID_1 threadIdx.y
   #define LID_2 threadIdx.z

   #define GID_0 blockIdx.x
   #define GID_1 blockIdx.y
   #define GID_2 blockIdx.z

   #define LDIM_0 blockDim.x
   #define LDIM_1 blockDim.y
   #define LDIM_2 blockDim.z

   #define GDIM_0 gridDim.x
   #define GDIM_1 gridDim.y
   #define GDIM_2 gridDim.z

   #define local_barrier() __syncthreads();


In fact these are all provided by the ``_cluda.py`` in PyOpenCL and PyCUDA.
These allow us to write CUDA/OpenCL agnostic code from Python.

One may also pass local memory to such a kernel, this trivial example
demonstrates this::


   from pysph.cpy.api import annotate
   from pysph.cpy.low_level import (
       Kernel, LID_0, LDIM_0, GID_0, LocalMem, local_barrier
   )
   import numpy as np

   @annotate(gdoublep='x',  ldoublep='xl')
   def f(x, xl):
       i, thread_id = declare('int', 2)
       thread_id = LID_0
       i = GID_0*LDIM_0 + thread_id

       xl[thread_id] = x[i]
       local_barrier()


   x = np.linspace(0, 1, 10000)

   get_config().use_opencl = True
   x = wrap(x)
   xl = LocalMem(1)

   k = Kernel(f)
   k(x, xl)

This kernel does nothing useful and is just meant to demonstrate how one can
allocate and use local memory. Note that here we "allocated" the local memory
on the host and are passing it in to the Kernel. The local memory is allocated
as ``LocalMem(1)``, this implicitly means allocate the required size in
multiples of the size of the type and the work group size. Thus the allocated
memory is ``work_group_size * sizeof(double) * 1``. This is convenient as very
often the exact work group size is not known.

A more complex and meaningful example is the ``vm_kernel.py`` example that is
included with CPy.


``Cython``
~~~~~~~~~~~

Just like the ``Kernel`` we also have a ``Cython`` class to run pure Cython
code. Here is an example of its usage::

  from pysph.cpy.config import use_config
  from pysph.cpy.types import annotate
  from pysph.cpy.low_level import Cython, nogil, parallel, prange

  import numpy as np

  @annotate(n='int', doublep='x, y', a='double')
  def cy_ex(x, y, a, n):
      i = declare('int')
      with nogil, parallel():
          for i in prange(n):
              y[i] = x[i]*a

  n = 1000
  x = np.linspace(0, 1, n)
  y = np.zeros_like(x)
  a = 2.0

  with use_config(use_openmp=True):
      cy = Cython(cy_ex)

   cy(x, y, a, n)

If you look at the above code, we are effectively writing Cython code but
compiling it and calling it in the last two lines. Note the use of the
``nogil, parallel`` and ``prange`` functions which are also provided in the
``low_level`` module. As you can see it is just as easy to write Cython code
and have it execute in parallel.


Externs
~~~~~~~

The ``nogil, parallel`` and ``prange`` functions we see in the previous
section are examples of external functionality. Note that these have no
straight-forward Python analog or implementation. They are implemented as
Externs. This functionality allows us to link to external code opening up many
interesting possibilities.

Note that as far as CPy is concerned, we need to know if a function needs to
be wrapped or somehow injected. Externs offer us a way to cleanly inject
external function definitions and use them. This is useful for example when
you need to include an external CUDA library.

Let us see how the ``prange`` extern is internally defined::

  from pysph.cpy.extern import Extern

  class _prange(Extern):
      def link(self, backend):
          # We don't need to link to anything to get prange working.
          return []

      def code(self, backend):
          if backend != 'cython':
              raise NotImplementedError('prange only available with Cython')
          return 'from cython.parallel import prange'

      def __call__(self, *args, **kw):
          # Ignore the kwargs.
          return range(*args)

  prange = _prange()


The Extern object has two important methods, ``link`` and ``code``. The
``__call__`` interface is provided just so this can be executed with pure
Python. The link returns a list of link args, these are currently ignored
until we figure out a good test/example for this. The ``code`` method returns
a suitable line of code inserted into the generated code. Note that in this
case it just performs a suitable import.

Thus, with this feature we are able to connect CPy with other libraries. This
functionality will probably evolve a little more as we gain more experience
linking with other libraries. However, we have a clean mechanism for doing so
already in-place.
