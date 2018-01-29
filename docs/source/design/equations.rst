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

      def loop(self, d_idx, s_idx, ...):
          # loop over neighbors for all sources.

      def post_loop(self, d_idx ...):
          # called after all looping is done.

      def reduce(self, dst):
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

- the *fluid* neighbors for each fluid particle are found and for each pair,
  the ``loop`` method is called with the required properties/values.

- the *solid* neighbors for each fluid particle are found and for each pair,
  the ``loop`` method is called with the required properties/values.

- for each fluid particle, the ``post_loop`` method is called with the
  required properties.

- If a reduce method exists, it is called for the destination (only once, not
  once per particle). It is passed the destination particle array.

The ``initialize, loop, post_loop`` methods all may be called in separate
threads (both on CPU/GPU) depending on the implementation of the backend.

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
``declare('object')``.



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


In addition if one requires the current time or the timestep in an equation,
the following may be passed into any of the methods of an equation:

    - ``t``: is the current time.

    - ``dt``: the current time step.


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

    mat = declare("matrix((3,3))")
    ii = declare('int')

When the Cython code is generated, this gets translated to:

.. code-block:: cython

    cdef double[3][3] mat
    cdef int ii

One can also declare any valid c-type using the same approach, for example if
one desires a ``long`` data type, one may use ``ii = declare("long")``.

Writing the reduce method
-------------------------

One may also perform any reductions on properties.  Consider a trivial example
of calculating the total mass and the maximum ``u`` velocity in the following
equation:

.. code-block:: python

    class FindMaxU(Equation):
        def reduce(self, dst):
            m = serial_reduce_array(dst.m, 'sum')
            max_u = serial_reduce_array(dst.u, 'max')
            dst.total_mass[0] = parallel_reduce_array(m, 'sum')
            dst.max_u[0] = parallel_reduce_array(u, 'max')

where:

    - ``dst``: refers to a destination ``ParticleArray``.

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
