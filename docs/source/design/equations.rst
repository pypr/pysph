.. _writing_equations:

==================
Writing equations
==================

This document puts together all the essential information on how to write
equations. We assume that you have already read the section
:ref:`design_overview`. Some information is repeated from there as well.

The PySPH equations are written in a very restricted way. The reason for this
is that if you do follow the suggestions and the conventions below you will
benefit from:

 - a high-performance serial implementation.
 - support for using your equations with OpenMP.
 - support for running on a GPU.

These are the main motivations for the severe restrictions we impose when you
write your equations.

Overview
--------

PySPH takes the equations you write and converts them on the fly to a
high-performance implementation suitable for the particular backend you
request.

.. py:currentmodule:: pysph.sph.equation

It is important to understand the overall structure of how the equations are
used when the high-performance code is generated. Let us look at the different
methods of a typical :py:class:`Equation` subclass::

  class YourEquation(Equation):
      def __init__(self, dest, sources):
          # Overload this only if you need to pass additional constants
          # Otherwise, no need to override __init__

      def initialize(self, d_idx, ...):
          # Called once per destination before loop.

      def loop_all(self, d_idx, ..., NBRS, N_NBRS, ...):
          # Called once before the loop and can be used
          # for non-pairwise interactions as one can pass the neighbors
          # for particle d_idx.

      def loop(self, d_idx, s_idx, ...):
          # loop over neighbors for all sources,
          # called once for each pair of particles!

      def post_loop(self, d_idx ...):
          # called after all looping is done.

      def reduce(self, dst, t, dt):
          # Called once for the destination array.
          # Any Python code can go here.

      def converged(self):
          # return > 0 for convergence < 0 for lack of convergence


It is easier to understand this if we take a specific example. Let us say, we
have a case where we have two particle arrays ``'fluid', 'solid'``. Let us say
the equation is used as ``YourEquation(dest='fluid', sources=['fluid',
'solid'])``. Now given this context, let us see what happens when this
equation is used.  What happens is as follows:

- for each fluid particle, the ``initialize`` method is called with the
  required arrays.

- the *fluid* neighbors for each fluid particle are found for each particle
  and can be passed en-masse to the ``loop_all`` method. One can pass ``NBRS``
  which is an array of unsigned ints with indices to the neighbors in the
  source particles. ``N_NBRS`` is the number of neighbors (an integer). This
  method is ideal for any non-pairwise computations or more complex
  computations.

- the *fluid* neighbors for each fluid particle are found and for each pair,
  the ``loop`` method is called with the required properties/values.

- the *solid* neighbors for each fluid particle are found and for each pair,
  the ``loop`` method is called with the required properties/values.

- for each fluid particle, the ``post_loop`` method is called with the
  required properties.

- If a reduce method exists, it is called for the destination (only once, not
  once per particle). It is passed the destination particle array.

The ``initialize, loop_all, loop, post_loop`` methods all may be called in
separate threads (both on CPU/GPU) depending on the implementation of the
backend.

It is possible to set a scalar value in the equation as an instance attribute,
i.e. by setting ``self.something = value`` but remember that this is just one
value for the equation. This value must also be initialized in the
``__init__`` method. Also make sure that the attributes are public and not
private. There is only one equation instance used in the code, not one
equation per thread or particle. So if you wish to calculate a temporary
quantity for each particle, you should create a separate property for it and
use that instead of assuming that the initialize and loop functions run in
serial. They do not when you use OpenMP or OpenCL. So do not create temporary
arrays inside the equation for these sort of things.

Now, if the group containing the equation has ``iterate`` set to True, then
the group will be iterated until convergence is attained for all the equations
(or sub-groups) contained by it. The ``converged`` method is called once and
not once per particle.

If you wish to compute something like a convergence condition, like the
maximum error or the average error, you should do it in the reduce method.

The reduce function is called only once every time the accelerations are
evaluated. As such you may write any Python code there. The only caveat is
that when using the CPU, one will have to declare any variables used a little
carefully -- ideally declare any variables used in this as
``declare('object')``. On the GPU, this function is not called via OpenCL and
is a pure Python function.



Conventions followed
--------------------

There are a few important conventions that are to be followed when writing the
equations. When passing arguments to the ``initialize, loop, post_loop``
methods,

    - ``d_*`` indicates a destination array.

    - ``s_*`` indicates a source array.

    - ``d_idx`` and ``s_idx`` represent the destination and source index
      respectively.

    - Each function can take any number of arguments as required, these are
      automatically supplied internally when the application runs.

    - All the standard math symbols from ``math.h`` are also available.

The following precomputed quantites are available and may be passed into any
equation:

    - ``HIJ = 0.5*(d_h[d_idx] + s_h[s_idx])``.

    - ``XIJ[0] = d_x[d_idx] - s_x[s_idx]``,
      ``XIJ[1] = d_y[d_idx] - s_y[s_idx]``,
      ``XIJ[2] = d_z[d_idx] - s_z[s_idx]``

    - ``R2IJ = XIJ[0]*XIJ[0] + XIJ[1]*XIJ[1] + XIJ[2]*XIJ[2]``

    - ``RIJ = sqrt(R2IJ)``

    - ``WIJ = KERNEL(XIJ, RIJ, HIJ)``

    - ``WJ = KERNEL(XIJ, RIJ, s_h[s_idx])``

    - ``RHOIJ = 0.5*(d_rho[d_idx] + s_rho[s_idx])``

    - ``WI = KERNEL(XIJ, RIJ, d_h[d_idx])``

    - ``RHOIJ1 = 1.0/RHOIJ``

    - ``DWIJ``: ``GRADIENT(XIJ, RIJ, HIJ, DWIJ)``
    - ``DWJ``: ``GRADIENT(XIJ, RIJ, s_h[s_idx], DWJ)``
    - ``DWI``: ``GRADIENT(XIJ, RIJ, d_h[d_idx], DWI)``

    - ``VIJ[0] = d_u[d_idx] - s_u[s_idx]``
      ``VIJ[1] = d_v[d_idx] - s_v[s_idx]``
      ``VIJ[2] = d_w[d_idx] - s_w[s_idx]``

    - ``EPS = 0.01 * HIJ * HIJ``

    - ``SPH_KERNEL``: the kernel being used and one can call the kernel as
      ``SPH_KERNEL.kernel(xij, rij, h)`` the gradient as
      ``SPH_KERNEL.gradient(...)``, ``SPH_KERNEL.gradient_h(...)`` etc. The
      kernel is any one of the instances of the kernel classes defined in
      :py:mod:`pysph.base.kernels`

In addition if one requires the current time or the timestep in an equation,
the following may be passed into any of the methods of an equation:

    - ``t``: is the current time.

    - ``dt``: the current time step.

For the ``loop_all`` method and the ``loop`` method, one may also pass the
following:

 - ``NBRS``: an array of unsigned ints with neighbor indices.
 - ``N_NBRS``: an integer denoting the number of neighbors for the current
   destination particle with index, ``d_idx``.



.. note::

   Note that all standard functions and constants in ``math.h`` are available
   for use in the equations. ``pi`` is defined. Please avoid using functions
   from ``numpy`` as these are Python functions and are slow. They also will
   not allow PySPH to be run with OpenMP. Similarly, do not use functions or
   constants from ``sympy`` and other libraries inside the equation methods as
   these will significantly slow down your code.


In an equation, any undeclared variables are automatically declared to be
doubles in the high-performance Cython code that is generated.  In addition
one may declare a temporary variable to be a ``matrix`` or a ``cPoint`` by
writing:

.. code-block:: python

    vec, vec1 = declare("matrix(3)", 2)
    mat = declare("matrix((3,3))")
    i, j = declare('int')

When the Cython code is generated, this gets translated to:

.. code-block:: cython

    cdef double vec[3], vec1[3]
    cdef double mat[3][3]
    cdef int i, j

One can also declare any valid c-type using the same approach, for example if
one desires a ``long`` data type, one may use ``i = declare("long")``.

Note that the additional (optional) argument in the declare specifies the
number of variables. While this is ignored during transpilation, this is
useful when writing functions in pure Python, the
:py:func:`pysph.base.cython_generator.declare` function provides a pure Python
implementation of this so that the code works both when compiled as well as
when run from pure Python. For example::

.. code-block:: python

   i, j = declare("int", 2)

In this case, the declare function call returns two integers so that the code
runs correctly in pure Python also. The second argument is optional and
defaults to 1. If we defined a matrix, then this returns two NumPy arrays of
the appropriate shape.

.. code-block:: python

   >>> declare("matrix(2)", 2)
   (array([ 0.,  0.]), array([ 0.,  0.]))

Thus the code one writes can be used in pure Python and can also be safely
transpiled into other languages.

Writing the reduce method
-------------------------

One may also perform any reductions on properties.  Consider a trivial example
of calculating the total mass and the maximum ``u`` velocity in the following
equation:

.. code-block:: python

    class FindMaxU(Equation):
        def reduce(self, dst, t, dt):
            m = serial_reduce_array(dst.m, 'sum')
            max_u = serial_reduce_array(dst.u, 'max')
            dst.total_mass[0] = parallel_reduce_array(m, 'sum')
            dst.max_u[0] = parallel_reduce_array(u, 'max')

where:

    - ``dst``: refers to a destination ``ParticleArray``.

    - ``t, dt``: are the current time and timestep respectively.

    - ``serial_reduce_array``: is a special function provided that performs
      reductions correctly in serial. It currently supports ``sum, prod, max``
      and ``min`` operations.  See
      :py:func:`pysph.base.reduce_array.serial_reduce_array`.  There is also a
      :py:func:`pysph.base.reduce_array.parallel_reduce_array` which is to be
      used to reduce an array across processors.  Using
      ``parallel_reduce_array`` is expensive as it is an all-to-all
      communication.  One can reduce these by using a single array and use
      that to reduce the communication.

We recommend that for any kind of reductions one always use the
``serial_reduce_array`` function and the ``parallel_reduce_array`` inside a
``reduce`` method.  One should not worry about parallel/serial modes in this
case as this is automatically taken care of by the code generator.  In serial,
the parallel reduction does nothing.

With this machinery, we are able to write complex equations to solve almost
any SPH problem.  A user can easily define a new equation and instantiate the
equation in the list of equations to be passed to the application.  It is
often easiest to look at the many existing equations in PySPH and learn the
general patterns.

If you wish to use adaptive time stepping, see the code
:py:class:`pysph.sph.integrator.Integrator`. The integrator uses information
from the arrays ``dt_cfl``, ``dt_force``, and ``dt_visc`` in each of the
particle arrays to determine the most suitable time step.

Illustration of the ``loop_all`` method
----------------------------------------

The ``loop_all`` is a powerful method we show how we can use the above to
perform what the ``loop`` method usually does ourselves.

.. code-block:: python

   class LoopAllEquation(Equation):
       def initialize(self, d_idx, d_rho):
           d_rho[d_idx] = 0.0

       def loop_all(self, d_idx, d_x, d_y, d_z, d_rho, d_h,
                    s_m, s_x, s_y, s_z, s_h,
                    SPH_KERNEL, NBRS, N_NBRS):
           i = declare('int')
           s_idx = declare('long')
           xij = declare('matrix(3)')
           rij = 0.0
           sum = 0.0
           for i in range(N_NBRS):
               s_idx = NBRS[i]
               xij[0] = d_x[d_idx] - s_x[s_idx]
               xij[1] = d_y[d_idx] - s_y[s_idx]
               xij[2] = d_z[d_idx] - s_z[s_idx]
               rij = sqrt(xij[0]*xij[0], xij[1]*xij[1], xij[2]*xij[2])
               sum += s_m[s_idx]*SPH_KERNEL.kernel(xij, rij, 0.5*(s_h[s_idx] + d_h[d_idx]))
           d_rho[d_idx] += sum

This seems a bit complex but let us look at what is being done. ``initialize``
is called once per particle and each of their densities is set to zero. Then
when ``loop_all`` is called it is called once per destination particle (unlike
``loop`` which is called pairwise for each destination and source particle).
The ``loop_all`` is passed arrays as is typical of most equations but is also
passed the ``SPH_KERNEL`` itself, the list of neighbors, and the number of
neighbors.

The code first declares the variables, ``i, s_idx`` as an integer and long,
and then ``x_ij`` as a 3-element array. These are important for performance in
the generated code. The code then loops over all neighbors and computes the
summation density. Notice how the kernel is computed using
``SPH_KERNEL.kernel(...)``. Notice also how the source index, ``s_idx`` is found
from the neighbors.

This above ``loop_all`` code does exactly what the following single line of
code does.

.. code-block:: python

       def loop(self, d_idx, d_rho, s_m, s_idx, WIJ):
           d_rho[d_idx] += s_m[s_idx]*WIJ

However, ``loop`` is only called pairwise and there are times when we want to
do more with the neighbors. For example if we wish to setup a matrix and solve
it per particle, we could do it in ``loop_all`` efficiently. This is also very
useful for non-pairwise interactions which are common in other particle
methods like molecular dynamics.

Calling user-defined functions from equations
----------------------------------------------

Sometimes we may want to call a user-defined function from the equations. Any
pure Python function defined using the same conventions as listed above (with
suitable type hints) can be called from the equations. Here is a simple
example from one of the tests in PySPH.

.. code-block:: python

    def helper(x=1.0):
        return x*1.5

    class SillyEquation(Equation):
        def initialize(self, d_idx, d_au, d_m):
            d_au[d_idx] += helper(d_m[d_idx])

        def _get_helpers_(self):
            return [helper]

Notice that ``initialize`` is calling the ``helper`` function defined above.
The helper function has a default argument to indicate to our code generation
that x is a floating point number. We could have also set the default argument
to a list and this would then be passed an array of values. The
``_get_helpers_`` method returns a list of functions and these functions are
automatically transpiled into high-performance C or OpenCL/CUDA code and can
be called from your equations.

Here is a more complex helper function.

.. code-block:: python

    def trace(x=[1.0, 1.0], nx=1):
        i = declare('int')
        result = 0.0
        for i in range(nx):
            result += x[i]
        return result

    class SillyEquation(Equation):
        def loop(self, d_idx, d_au, d_m, XIJ):
            d_au[d_idx] += trace(XIJ, 3)

        def _get_helpers_(self):
            return [trace]

The trace function effectively is converted into a function with signature
``double trace(double* x, int nx)`` and thus can be called with any
one-dimensional array.

Calling arbitrary Python functions from a Group
------------------------------------------------

Sometimes, you may need to implement something that is hard to write (at least
initially) with the constraints that PySPH places. For example if you need to
implement an algorithm that requires more complex data structures and you want
to do it easily in Python. There are ways to call arbitrary Python code from
the application already but sometimes you need to do this during every
acceleration evaluation. To support this the :py:class:`Group` class supports
two additional keyword arguments called ``pre`` and ``post``. These can be any
Python callable that take no arguments. Any callable passed as ``pre`` will be
called *before* any equation related code is executed and ``post`` will be
executed after the entire group is finished. If the group is iterated, it
should call those functions repeatedly.

Now these functions are pure Python functions so you may choose to do anything
in them. These are not called within an OpenMP context and if you are using
the OpenCL or CUDA backends again this will simply be a Python function call
that has nothing to do with the particular backend. However, since it is
arbitrary Python, you can choose to implement the code using any approach you
choose to do. This should be flexible enough to customize PySPH greatly.



Examples to study
------------------

The following equations provide good examples for how one could use/write the
``reduce`` method:

- :py:class:`pysph.sph.gas_dynamics.basic.SummationDensityADKE`: relatively simple.
- :py:class:`pysph.sph.rigid_body.RigidBodyMoments`: this is pretty complex.
- :py:class:`pysph.sph.iisph.PressureSolve`: relatively straight-forward.

The equations that demonstrate the ``converged`` method are:

- :py:class:`pysph.sph.gas_dynamics.basic.SummationDensity`: relatively simple.
- :py:class:`pysph.sph.iisph.PressureSolve`.
