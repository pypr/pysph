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

This will automatically parallelize the run.  In this example doing this will
only slow it down as the number of particles is extremely small.


Creating particles
-------------------

The SPH particles are managed in a ``ParticleArray`` instance.  For a
simulation involving a solid and fluid we would create two particle arrays.
One for the fluid and one for the solid.  The ``Application.setup`` method
expects a callable function which is called to create the particles.  The
function should return a list of particle arrays.  When the simulation is run
in parallel, the processor with rank 0 will have all of the particles and this
is then distributed evenly across the particles.  The other processors (with
rank > 0) will require an empty particle array the ``empty`` keyword argument
is used for this and if it is ``True`` one should create empty arrays with the
correct names.

Each particle array is created by the convenience function
``get_particle_array_wcsph`` which is available in the ``pysph.base.utils``
module.  This function sets up the ``ParticleArray``, one can pass a ``name``
keyword argument to set the name of the ``ParticleArray`` and any number of
numpy arrays which are set as particle properties.  By default the following
properties are made available for every particle array created with the
``get_particle_array_wcsph``: ``x, y, z, u, v, w, h, rho, m, p, cs, ax, ay,
az, au, av, aw, x0, y0, z0, u0, v0, w0, arho, rho0, div, gid, pid, tag``.

It is important to name each particle array with a reasonable name.  For
example in the elliptical drop example we use ``"fluid"``.


Writing the ``Equations``
--------------------------

It is important for users to be able to easily write out new SPH equations of
motion.  PySPH provides a very convenient way to write these equations.  The
PySPH framework allows the user to write these equations in pure Python.
These pure Python equations are then used to generate high-performance code
and then called appropriately to perform the simulations.

In general an SPH algorithm proceeds as the following pseudo-code
illustrates::

    for destination in particles:
        for equation in equations:
            equation.initialize(destination)

    # This is where bulk of the computation happens.
    for destination in particles:
        for source in particle.neighbors:
            for equation in equations:
                equation.loop(source, destination)

    for destination in particles:
        for equation in equations:
            equation.post_loop(destination)

The neighbors of a given particle are identified using a nearest neighbor
algorithm.  PySPH does this automatically for the user and internally uses a
link-list based algorithm to identify neighbors.

In PySPH we follow some simple conventions when writing equations.  Let us
look at a few equations first.  The equation is instantiated as::

        TaitEOS(dest='fluid', sources=None, rho0=ro, c0=co, gamma=7.0)

Each equation is provided a destination particle array name called ``dest``
and a list of source particle array names in ``sources``.  If ``sources=None``
this does not require any sources.  The other parameters are constants for the
particular equation.  Here is what the TaitEOS equation class looks like::

    class TaitEOS(Equation):
        def __init__(self, dest, sources=None,
                    rho0=1000.0, c0=1.0, gamma=7.0):
            self.rho0 = rho0
            self.rho01 = 1.0/rho0
            self.c0 = c0
            self.gamma = gamma
            self.gamma1 = 0.5*(gamma - 1.0)
            self.B = rho0*c0*c0/gamma
            super(TaitEOS, self).__init__(dest, sources)

        def loop(self, d_idx, d_rho, d_p, d_cs):
            ratio = d_rho[d_idx] * self.rho01
            tmp = pow(ratio, self.gamma)

            d_p[d_idx] = self.B * (tmp - 1.0)
            d_cs[d_idx] = self.c0 * pow( ratio, self.gamma1 )

Notice that it has only one ``loop`` method and this loop is applied for all
particles.  Since there are no sources, there is no need for to find the
neighbors.  There are a few important conventions that are to be followed when
writing the equations.

    - ``d_*`` indicates a destination array.

    - ``s_*`` indicates a source array.

    - ``d_idx`` and ``s_idx`` represent the destination and source index
      respectively.

    - Each function can take any number of arguments as required, these are
      automatically supplied internally when the application runs.

Let us look at the Continuity equation as another simple example.  It is
instantiated as ::

        ContinuityEquation(dest='fluid',  sources=['fluid',])

The class is defined as::

    class ContinuityEquation(Equation):
        def initialize(self, d_idx, d_arho):
            d_arho[d_idx] = 0.0

        def loop(self, d_idx, d_arho, s_idx, s_m, DWIJ=[0.0, 0.0, 0.0],
                VIJ=[0.0, 0.0, 0.0]):
            vijdotdwij = DWIJ[0]*VIJ[0] + DWIJ[1]*VIJ[1] + DWIJ[2]*VIJ[2]
            d_arho[d_idx] += s_m[s_idx]*vijdotdwij


Notice that the ``initialize`` method merely sets the value to zero.  The
``loop`` method also defines a few new quantities like ``DWIJ``, ``VIJ`` etc.
The method also prescribes default values to these quantities.  The defaults
are only set so that they may be declared appropriately in the
high-performance code that is generated from this Python code. These are
precomputed quantities and are automatically provided depending on the
equations needed for a particular source/destination pair.  The following
precomputed quantites are available:

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
    - ``DWI``: ``GRADIENT(XIJ, RIJ, s_h[s_idx], DWJ)``
    - ``DWI``: ``GRADIENT(XIJ, RIJ, d_h[d_idx], DWI)``

    - ``VIJ[0] = d_u[d_idx] - s_u[s_idx]``
      ``VIJ[1] = d_v[d_idx] - s_v[s_idx]``
      ``VIJ[2] = d_w[d_idx] - s_w[s_idx]``

    - ``DT_ADAPT``: is an array of three doubles that stores an adaptive
      time-step, the first element is the CFL based time-step limit, the
      second is the force-based limit and the third a viscosity based limit.
      See ``pysph.sph.wc.basic.MomentumEquation`` for an example of how this
      is used.

In an equation, any undeclared variables are automatically declared to be
doubles in the high-performance Cython code that is generated.  In addition
one may declare a temporary variable to be a ``matrix`` or a ``cPoint`` by
writing::

    mat = declare("matrix((3,3))")
    point = declare("cPoint")

When the Cython code is generated, this gets translated to::

    cdef double[3][3] mat
    cdef cPoint point

With this machinery, we are able to write complex equations.

If one wishes to write a new equation, one may simply do as above and
instantiate the equation in the list of equations.

Grouping equations
~~~~~~~~~~~~~~~~~~~

Often one wishes to compute a set of equations before running the remainder of
the equations.  For example, one may wish to run the Tait equation of state
first for all species of particles before any other computations are started.
In such a case one can simply group the equations and each group will be
completed before the next group is computed.  For example the ``dam_break.py``
example lists the following::

    equations = [
        # Equation of state
        Group(equations=[
                TaitEOS(dest='fluid', sources=None, rho0=ro, c0=co, gamma=gamma),
                TaitEOS(dest='boundary', sources=None, rho0=ro, c0=co, gamma=gamma),
                ]),

        Group(equations=[
                # Continuity equation
                ContinuityEquation(dest='fluid', sources=['fluid', 'boundary']),
                ContinuityEquation(dest='boundary', sources=['fluid']),
                # Momentum equation
                MomentumEquation(dest='fluid', sources=['fluid', 'boundary'],
                        alpha=alpha, beta=beta, gy=-9.81, c0=co),
                # Position step with XSPH
                XSPHCorrection(dest='fluid', sources=['fluid'])
                ]),
        ]

In this case, the TaitEOS is computed for all the fluid and boundary particles
first before the continuity, momentum and other equations are run.  This
ensures that the pressure and sound speed are correctly set before the other
equations are run.


Writing the ``IntegratorSteps``
--------------------------------

The integrator stepper code is similar to the equations in that they are all
written in pure Python and Cython code is automatically generated from it.
The simplest integrator is the Euler integrator which looks like this::

    class EulerStep(IntegratorStep):
        def initialize(self):
            pass
        def predictor(self):
            pass
        def corrector(self, d_idx, d_u, d_v, d_w, d_au, d_av, d_aw, d_x, d_y,
                      d_z, d_rho, d_arho, dt=0.0):
            d_u[d_idx] += dt*d_au[d_idx]
            d_v[d_idx] += dt*d_av[d_idx]
            d_w[d_idx] += dt*d_aw[d_idx]

            d_x[d_idx] += dt*d_u[d_idx]
            d_y[d_idx] += dt*d_v[d_idx]
            d_z[d_idx] += dt*d_w[d_idx]

            d_rho[d_idx] += dt*d_arho[d_idx]

As can be seen the general structure is very similar to how equations are
written in that the functions take an arbitrary number of arguments and are
set.  The value of ``dt`` is also provided automatically when the methods are
called.

It is important to note that if there are additional variables to be stepped
in addition to these standard ones, you must write your own stepper.
Currently, only predictor-corrector steppers are supported by the framework.
Take a look at the ``pysph.sph.integrator`` module for more examples.


Organization of the ``pysph`` package
--------------------------------------

PySPH is organized into several sub-packages.  These are:

  - ``pysph.base``:  This subpackage defines the ``ParticleArray``, ``CArray``
    (which are used by the particle arrays), Kernels, the nearest neighbor
    particle search (NNPS) code, and the Cython code generation utilities.

  - ``pysph.sph``: Contains the various ``Equation``, the ``Integrator``
    various integration steppers, and the code generation for the SPH looping.
    ``pysph.sph.wc`` contains the equations for the weakly compressible
    formulation.  ``pysph.sph.solid_mech`` contains the equations for solid
    mechanics and ``pysph.sph.misc`` has miscellaneous equations.

  - ``pysph.solver``: Provides the ``Solver``, the ``Application`` and a
    convenient way to interact with the solver as it is running.

  - ``pysph.parallel``: Provides the parallel functionality.

  - ``pysph.tools``: Provides some useful tools including the ``pysph_viewer``
    which is based on Mayavi.
