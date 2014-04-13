.. _tutorials:

In the tutorials, we will introduce the PySPH framework in the context
of the examples provided. Read this if you are a casual user and want
to use the framework *as is*. If you want to add new functions and
capabilities to PySPH, you should read :ref:`design_overview`. If you
are new to PySPH however, we highly recommend that you go through this
document.

Recall that PySPH is a framework for parallel SPH-like simulations in
Python. The idea therefore, is to provide a user friendly mechanism to
set-up problems while leaving the internal details to the
framework. *All* examples follow the following steps:

.. figure:: ../../Images/pysph-examples-common-steps.png
   :align: center

The tutorials address each of the steps in this flowchart for problems
with increasing complexity.


Learning the ropes
==================

The first example we consider is a "patch" test for SPH formulations
for incompressible fluids in ``examples/elliptical_drop.py``. This
problem simulates the evolution of a 2D circular patch of fluid under
the influence of an initial velocity field given by:

.. math::

   u &= -100 x \\
   v &= 100 y

The kinematical constraint of incompressibility causes the initially
circular patch of fluid to deform into an ellipse such that the volume
(area) is conserved. An expression can be derived for this deformation
which makes it an ideal test to verify codes.

Imports
~~~~~~~~~~~~~~~~~~~~~~~~

Taking a look at the example, the first several lines are imports of
various modules:

.. code-block:: python

   numpy import ones_like, mgrid, sqrt, array, savez
   from time import time

   # PySPH base and carray imports
   from pysph.base.utils import get_particle_array_wcsph
   from pysph.base.kernels import CubicSpline
   from pyzoltan.core.carray import LongArray

   # PySPH solver and integrator
   from pysph.solver.application import Application
   from pysph.solver.solver import Solver
   from pysph.sph.integrator import WCSPHStep, Integrator

   # PySPH sph imports
   from pysph.sph.basic_equations import ContinuityEquation, XSPHCorrection
   from pysph.sph.wc.basic import TaitEOS, MomentumEquation

.. note::

    This is common for all examples and it is worth noting the pattern of the
    PySPH imports. Fundamental SPH constructs like the kernel and particle
    containers are imported from the ``base`` subpackage. The framework
    related objects like the solver and integrator are imported from the
    ``solver`` subpackage. Finally, we import from the ``sph`` subpackage, the
    physics related part for this problem.

Functions for loading/generating the particles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Next in the code are two functions called ``exact_solution`` and
``get_circular_patch``. The former produces an exact solution for
comparison, the latter looks like:

.. code-block:: python

   def get_circular_patch(dx=0.025, **kwargs):
       """Create the circular patch of fluid."""
       name = 'fluid'
       x,y = mgrid[-1.05:1.05+1e-4:dx, -1.05:1.05+1e-4:dx]
       x = x.ravel()
       y = y.ravel()

       m = ones_like(x)*dx*dx
       h = ones_like(x)*hdx*dx
       rho = ones_like(x) * ro

       p = ones_like(x) * 1./7.0 * co**2
       cs = ones_like(x) * co

       u = -100*x
       v = 100*y

       # remove particles outside the circle
       indices = []
       for i in range(len(x)):
	   if sqrt(x[i]*x[i] + y[i]*y[i]) - 1 > 1e-10:
	       indices.append(i)

       pa = get_particle_array_wcsph(x=x, y=y, m=m, rho=rho, h=h, p=p, u=u, v=v,
				     cs=cs, name=name)

       la = LongArray(len(indices))
       la.set_data(array(indices))

       pa.remove_particles(la)

       print "Elliptical drop :: %d particles"%(pa.get_number_of_particles())

       # add requisite variables needed for this formulation
       for name in ('arho', 'au', 'av', 'aw', 'ax', 'ay', 'az', 'rho0', 'u0',
		    'v0', 'w0', 'x0', 'y0', 'z0'):
	   pa.add_property(name)

       return [pa,]


.. py:currentmodule:: pysph.base.particle_array

and is used to initialize the particles in Python. In PySPH, we use a
:py:class:`ParticleArray` object as a container for particles of a given
*species*. You can think of a particle species as any homogenous entity in a
simulation. For example, in a two-phase air water flow, a species could be
used to represent each phase. A :py:class:`ParticleArray` can be conveniently
created from the command line using NumPy arrays. For example

.. code-block:: python

    >>> from pysph.base.utils import get_particle_array
    >>> x, y = numpy.mgrid[0:1:0.01, 0:1:0.01]
    >>> x = x.ravel(); y = y.ravel()
    >>> pa = sph.get_particle_array(x=x, y=y)

would create a :py:class:`ParticleArray`, representing a uniform distribution
of particles on a Cartesian lattice in 2D using the helper function
:py:func:`get_particle_array` in the **base** subpackage.

.. note::

   **ParticleArrays** in PySPH use *flattened* or one-dimensional arrays.

The :py:class:`ParticleArray` is highly convenient, supporting methods for
insertions, deletions and concatenations. In the `get_circular_patch`
function, we use this convenience to remove a list of particles that fall
outside a circular region:

.. code-block:: python

   pa.remove_particles(la)

.. py:currentmodule:: pyzoltan.core.carray

where, a list of indices is provided in the form of a :py:class:`LongArray`
which, as the name suggests, is an array of 64 bit integers.

.. note::

   Any one-dimensional (NumPy) array is valid input for PySPH. You can
   generate this from an external program for solid modelling and load
   it.

.. note::

   PySPH works with multiple **ParticleArrays**. This is why we
   actually return a *list* in the last line of the
   `get_circular_patch` function above.

Setting up the PySPH framework
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As we move on, we encounter instantiations of the PySPH framework objects.
These are the :py:class:`pysph.solver.application.Application`,
:py:class:`pysph.sph.integrator.Integrator` and
:py:class:`pysph.solver.solver.Solver` objects:

.. code-block:: python

    # Create the application.
    app = Application()

    kernel = CubicSpline(dim=2)

    integrator = Integrator(fluid=WCSPHStep())

    # Create and setup a solver.
    solver = Solver(kernel=kernel, dim=2, integrator=integrator)

    # Setup default parameters.
    solver.set_time_step(1e-5)
    solver.set_final_time(0.0075)

.. py:currentmodule:: pysph.solver.application

The :py:class:`Application` makes it easy to pass command line arguments to
the solver. It is also important for the seamless parallel execution of the
same example. To appreciate the role of the :py:class:`Application` consider
for a moment how might we write a parallel version of the same example. At
some point, we would need some MPI imports and the particles should be created
in a distributed fashion. All this (and more) is handled through the
abstraction of the :py:class:`Application` which hides all this detail from
the user.

.. py:currentmodule:: pysph.sph.integrator

Intuitively, in an SPH simulation, the role of the :py:class:`Integrator`
should be obvious. In the code, we see that we ask for the "fluid" to be
stepped using a :py:class:`WCSPHStep` object. Taking a look at the
`get_circular_patch` function once more, we notice that the **ParticleArray**
representing the circular patch was named as `fluid`. So we're essentially
asking the PySPH framework to step or *integrate* the properties of the
**ParticleArray** fluid using :py:class:`WCSPHStep`. Safe to assume that the
framework takes the responsibility to call this integrator at the appropriate
time during a time-step.

.. py:currentmodule:: pysph.solver.solver

The :py:class:`Solver` is the main driver for the problem. It marshals a
simulation and takes the responsibility (through appropriate calls to the
integrator) to update the solution to the next time step. It also handles
input/output and computing global quantities (such as minimum time step) in
parallel.

Specifying the interactions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At this stage, we have the particles (represented by the fluid
**ParticleArray**) and the framework to integrate the solution and
marshall the simulation. What remains is to define how to actually go
about updating properties *within* a time step. That is, for each
particle we must "do something". This is where the *physics* for the
particular problem comes in.

For SPH, this would be the pairwise interactions between particles. In PySPH,
we provide a specific way to define the sequence of interactions which is a
*list* of **Equation** objects (see :doc:`../reference/equations`). For the
circular patch test, the sequence of interactions is relatively
straightforward:

    - Compute pressure from the EOS:  :math:`p = f(\rho)`
    - Compute the rate of change of density: :math:`\frac{d\rho}{dt}`
    - Compute the rate of change of velocity (accelerations): :math:`\frac{d\boldsymbol{v}}{dt}`
    - Compute corrections for the velocity (XSPH): :math:`\frac{d\boldsymbol{x}}{dt}`

We request this in PySPH like so:

.. code-block:: python

   # The equations of motion.
   equations = [
       # Equation of state: p = f(rho)
       TaitEOS(dest='fluid', sources=None, rho0=ro, c0=co, gamma=7.0),

       # Density rate: drho/dt
       ContinuityEquation(dest='fluid',  sources=['fluid',]),

       # Acceleration: du,v/dt
       MomentumEquation(dest='fluid', sources=['fluid'], alpha=1.0, beta=1.0),

       # XSPH velocity correction
       XSPHCorrection(dest='fluid', sources=['fluid']),

       ]

.. py:currentmodule:: pysph.sph.equation

Each *interaction* is specified through an :py:class:`Equation` object, which
is instantiated with the general syntax:

.. code-block:: python

   Equation(dest='array_name', sources, **kwargs)

The `dest` argument specifies the *target* or *destination*
**ParticleArray** on which this interaction is going to operate
on. Similarly, the `sources` argument specifies a *list* of
**ParticleArrays** from which the contributions are sought. For some
equations like the EOS, it doesn't make sense to define a list of
sources and a `None` suffices. The specification basically tells PySPH
that for one time step of the calculation:

    - Use the Tait's EOS to update the properties of the fluid array
    - Compute :math:`\frac{d\rho}{dt}` for the fluid from the fluid
    - Compute accelerations for the fluid from the fluid
    - Compute the XSPH corrections for the fluid, using fluid as the source

.. note::

   Notice the use of the **ParticleArray** name "fluid". It is the
   responsibility of the user to ensure that the equation
   specification is done in a manner consistent with the creation of
   the particles.

With the list of equations, our problem is completely defined. PySPH
now knows what to do with the particles within a time step. More
importantly, this information is enough to generate code to carry out
a complete SPH simulation.

Running the example
~~~~~~~~~~~~~~~~~~~

.. py:currentmodule:: pysph.solver.application

In the last two lines of the example, we use the :py:class:`Application`
to run the problem:

.. code-block:: python

   # Setup the application and solver.  This also generates the particles.
   app.setup(solver=solver, equations=equations,
             particle_factory=get_circular_patch)

   app.run()

We can see that the :py:meth:`Application.setup` method is where we tell PySPH
what we want it to do. We pass in the function to create the particles, the
list of equations defining the problem and the solver that will be used to
marshal the problem.

Many parameters can be configured via the command line, and these will
override any parameters setup before the ``app.setup`` call.  For
example one may do the following to find out the various options::

    $ python elliptical_drop.py -h

If we run the example without any arguments it will run until a final
time of 0.0075 seconds.  We can change this for example to 0.005 by
the following::

    $ python elliptical_drop.py --tf=0.005

When this is run, PySPH will generate Cython code from the equations and
integrators that have been provided, compiles that code and runs the
simulation.  This provides a great deal of convenience for the user without
sacrificing performance.  The generated code is available in
``~/.pysph/source``.  If the code/equations have not changed, then the code
will not be recompiled.  This is all handled automatically without user
intervention.

If we wish to run the code in parallel (and have compiled PySPH with Zoltan
and mpi4py) we can do::

    $ mpirun -np 4 /path/to/python elliptical_drop.py

This will automatically parallelize the run. In this example doing this will
only slow it down as the number of particles is extremely small.
