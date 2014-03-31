Using the PySPH framework
==========================

Overview
---------

The PySPH framework provides important functionality to simulate a variety of
problems using the SPH method.  We demonstrate the PySPH framework using a few
simple example problems.

The first example we'll look at is the ``examples/elliptical_drop.py``
example.  This example simulates the evolution of a 2D circular patch of fluid
under the influence of a velocity field given by,

.. math::
        u &= -100 x \\
        v &= 100 y

If one looks at the example, the first several lines are imports of various
modules which we will ignore for now.  There are two additional functions
called ``exact_solution`` and ``get_circular_patch``.  The former just
produces an exact solution for comparison.  The latter creates the initial
distribution of the particles.  The particles are created in the form of what
are called ``ParticleArray`` objects.  Each particle array has a name.  In
this case the array is called ``"fluid"``. We will look at the
``get_circular_patch`` code in  in a short while.  The main code is the
following::

    def exact_solution(...):
        pass
    def get_circular_patch(...):
        pass

    # Create the application.
    app = Application()

    kernel = CubicSpline(dim=2)

    integrator = Integrator(fluid=WCSPHStep())

    # Create and setup a solver.
    solver = Solver(kernel=kernel, dim=2, integrator=integrator)

    # Setup default parameters.
    solver.set_time_step(1e-5)
    solver.set_final_time(0.0075)

    # The equations of motion.
    equations = [
        TaitEOS(dest='fluid', sources=None, rho0=ro, c0=co, gamma=7.0),
        ContinuityEquation(dest='fluid',  sources=['fluid',]),
        MomentumEquation(dest='fluid', sources=['fluid'], alpha=1.0, beta=1.0),
        XSPHCorrection(dest='fluid', sources=['fluid']),
    ]

    # Setup the application and solver.  This also generates the particles.
    app.setup(solver=solver, equations=equations,
              particle_factory=get_circular_patch,
              name='fluid')

    app.run()

The following important classes are used in the above code.

 - The ``Application`` class manages the entire simulation including any
   command line options passed by the user.  The application creates the
   particles, sets up any parallel communications and runs the solver.

 - The ``Solver`` manages the actual simulation.  It manages the
   ``Integrator``.

 - The ``Integrator`` specifies how the variables are stepped in time.  For
   each particle array, we provide an instance of an ``IntegratorStep`` class.
   In this case we use the pre-defined ``WCSPHStep`` (subclass of
   ``IntegratorStep``). PySPH supports any predictor-corrector integrator.
   The ``IntegratorStep`` basically provides three methods:

     * ``initialize(...)``:  Intialize any variables before the step.

     * ``predictor(...)``: Perform the prediction step.

     * ``corrector(...)``:  Perform the corrector step.

   See the ``pysph.sph.integrator`` module to see several examples.

 - The most important part of the SPH simulation is the set of equations. This
   is specified as a list of ``Equation`` objects.  An equation can provide
   the following methods:

     * ``initialize(...)``:  Intialize any variables for the step.

     * ``loop(...)``: Perform a source - destintation computation.

     * ``post_loop(...)``:  Perform any post-loop calculations.

   The equations are described in greater detail in the next session.

 - As expected, ``kernel`` is the SPH kernel that should be used.  Kernels
   define ``kernel`` and ``gradient`` functions.

The ``app.setup(...)`` call sets up the application and the application is
then run.  The simulation completes until the desired final time.  Many
parameters can be configured via the command line, and these will override any
parameters setup before the ``app.setup`` call.  For example one may do the
following to find out the various options::

    $ python elliptical_drop.py -h

If we run the example without any arguments it will run until a final time of
0.0075 seconds.  We can change this to 0.005 by the following::

    $ python elliptical_drop.py --tf=0.005

In addition, if we wish to run the code in parallel (and have compiled PySPH
with Zoltan and mpi4py) we can do::

    $ mpirun -np 4 /path/to/python elliptical_drop.py

This will automatically parallelize the run.  In this example doing this will
only slow it down as the number of particles is extremely small.


Creating particles
-------------------

Details about particle arrays and how to create them.

Talk about the "empty" argument for parallel runs.


Writing the ``Equations``
--------------------------

As can be seen in the above, the equations are the crucial part of the
simulation.

General idea of an SPH loop.

Details about equations.

    - Pure Python.
    - Naming conventions.
    - Function signatures/default arguments.
    - Automatic declaration of types.
    - Special type declarations.
    - Predefined values.
    - Automatic code generation.
    - Example.

Writing the ``IntegratorSteps``
--------------------------------

General guidelines and a simple example.

Organization of the ``pysph`` package
--------------------------------------

Provide details of ``pysph.base``, ``pysph.sph``, ``pysph.solver``,
``pysph.parallel`` and ``pysph.tools``.
