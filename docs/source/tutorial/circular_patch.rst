.. _tutorial:

========================
A more detailed tutorial
========================

In the previous tutorial (:doc:`circular_patch_simple`) we provided a high
level overview of the PySPH framework.  No details were provided on equations,
integrators and solvers.  This tutorial assumes that you have read the
previous one.

Recall that in the previous tutorial, a circular patch of fluid with a given
initial velocity field was simulated using a weaky-compressible SPH scheme.
In that example, a ``WCSPHScheme`` object was created in the ``create_scheme``
method.  The details of what exactly the scheme does was not discussed.  This
tutorial explains some of those details by solving the same problem using a
lower-level approach where the actual SPH equations, the integrator and the
solver are created manually.  This should help a user write their own schemes
or modify an existing scheme.  The full code for this example can be seen in `elliptical_drop_no_scheme.py
<https://bitbucket.org/pysph/pysph/src/master/pysph/examples/elliptical_drop_no_scheme.py>`_.


Imports
~~~~~~~~~~~~~

This example requires a few more imports than the previous case.

the first several lines are imports of various modules:

.. code-block:: python

    import os
    from numpy import array, ones_like, mgrid, sqrt

    # PySPH base and carray imports
    from pysph.base.utils import get_particle_array_wcsph
    from pysph.base.kernels import Gaussian

    # PySPH solver and integrator
    from pysph.solver.application import Application
    from pysph.solver.solver import Solver
    from pysph.sph.integrator import EPECIntegrator
    from pysph.sph.integrator_step import WCSPHStep

    # PySPH sph imports
    from pysph.sph.equation import Group
    from pysph.sph.basic_equations import XSPHCorrection, ContinuityEquation
    from pysph.sph.wc.basic import TaitEOS, MomentumEquation


.. note::

    This is common for all examples that do not use a scheme and it is worth
    noting the pattern of the PySPH imports. Fundamental SPH constructs like
    the kernel and particle containers are imported from the ``base``
    subpackage. The framework related objects like the solver and integrator
    are imported from the ``solver`` subpackage. Finally, we import from the
    ``sph`` subpackage, the physics related part for this problem.

The methods defined for creating the particles are the same as in the previous
tutorial with the exception of the call to
``self.scheme.setup_properties([pa])``.  In this example, we do not create a
scheme, we instead create all the required PySPH objects from the
application.  We do not override the ``create_scheme`` method but instead have
two other methods called ``create_solver`` and ``create_equations`` which
handle this.


Setting up the PySPH framework
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As we move on, we encounter instantiations of the PySPH framework objects.
These are the :py:class:`pysph.solver.application.Application`,
:py:class:`pysph.sph.integrator.TVDRK3Integrator` and
:py:class:`pysph.solver.solver.Solver` objects.  The ``create_solver`` method
constructs a ``Solver`` instance and returns it as seen below:

.. code-block:: python

        def create_solver(self):
            kernel = Gaussian(dim=2)

            integrator = EPECIntegrator( fluid=WCSPHStep() )

            dt = 5e-6; tf = 0.0076
            solver = Solver(kernel=kernel, dim=2, integrator=integrator,
                            dt=dt, tf=tf, adaptive_timestep=True,
                            cfl=0.05, n_damp=50,
                            output_at_times=[0.0008, 0.0038])

            return solver

As can be seen, various options are configured for the solver, including
initial damping etc.

.. py:currentmodule:: pysph.sph.integrator

Intuitively, in an SPH simulation, the role of the
:py:class:`EPECIntegrator` should be obvious. In the code, we see that we
ask for the "fluid" to be stepped using a :py:class:`WCSPHStep` object. Taking
a look at the ``create_particles`` method once more, we notice that the
**ParticleArray** representing the circular patch was named as `fluid`. So
we're essentially asking the PySPH framework to step or *integrate* the
properties of the **ParticleArray** fluid using :py:class:`WCSPHStep`. It is
safe to assume that the framework takes the responsibility to call this
integrator at the appropriate time during a time-step.

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

    - Compute pressure from the Equation of State (EOS):  :math:`p = f(\rho)`
    - Compute the rate of change of density: :math:`\frac{d\rho}{dt}`
    - Compute the rate of change of velocity (accelerations): :math:`\frac{d\boldsymbol{v}}{dt}`
    - Compute corrections for the velocity (XSPH): :math:`\frac{d\boldsymbol{x}}{dt}`

Care must be taken that the EOS equation should be evaluated for all the
particles before the other equations are evaluated.

.. py:currentmodule:: pysph.sph.equation


We request this in PySPH by creating a list of :py:class:`Equation` instances
in the ``create_equations`` method:

.. code-block:: python

    def create_equations(self):
        equations = [
            Group(equations=[
                TaitEOS(dest='fluid', sources=None, rho0=self.ro,
                        c0=self.co, gamma=7.0),
            ], real=False),

            Group(equations=[
                ContinuityEquation(dest='fluid',  sources=['fluid',]),

                MomentumEquation(dest='fluid', sources=['fluid'],
                                 alpha=self.alpha, beta=0.0, c0=self.co),

                XSPHCorrection(dest='fluid', sources=['fluid']),

            ]),
        ]
        return equations

Each ``Group`` instance is completed before the next is taken up.  Each group
contains a list of ``Equation`` objects.  Each *interaction* is specified
through an :py:class:`Equation` object, which is instantiated with the general
syntax:

.. code-block:: python

   Equation(dest='array_name', sources, **kwargs)

The ``dest`` argument specifies the *target* or *destination*
**ParticleArray** on which this interaction is going to operate
on. Similarly, the ``sources`` argument specifies a *list* of
**ParticleArrays** from which the contributions are sought. For some
equations like the EOS, it doesn't make sense to define a list of
sources and a ``None`` suffices. The specification basically tells PySPH
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
a complete SPH simulation.  For more details on how new equations can be
written please read :ref:`design_overview`.

The example may be run the same way as the previous example::

  $ pysph run elliptical_drop_no_scheme

The resulting output can be analyzed or viewed the same way as in the previous
example.

In the previous example (:doc:`circular_patch_simple`), the equations and
solver are created automatically by the ``WCSPHScheme``.  If the
``create_scheme`` is overwritten and returns a scheme, the
``create_equations`` and ``create_solver`` need not be implemented.
Implementing other schemes can be done by either implementing the equations
directly as done in this example or one could implement a new ``Scheme``.
