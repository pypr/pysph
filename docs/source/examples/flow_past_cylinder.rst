.. _flow_past_cylinder:

Flow past a circular cylinder using open boundary conditions
------------------------------------------------------------

This example demonstrates the API of inlet and outlet boundary conditions in
PySPH. The flow past a circular cylinder is an example which uses both
inlet and outlet boundary conditions. To run it one may do::

  $ pysph run flow_past_cylinder_2d

There are many command line options that this example provides, check them out with::

  $ pysph run flow_past_cylinder_2d -h

In this example, we have a wind tunnel with two bounding slip walls on the top
and bottom of the tunnel. The inlet is on the left and the outlet is on the
right. In order to perform the simulation five particle arrays, ``solid``,
``fluid``, ``wall``, ``inlet`` and ``outlet`` are generated. ``fluid``,
``solid`` and ``wall`` has to solved using ``edac`` scheme, whereas ``inlet``
and ``outlet`` are solved according to the equations provided by the Inlet
Outlet Manager (IOM). The example source can be seen at
`flow_past_cylinder_2d.py
<https://github.com/pypr/pysph/tree/master/pysph/examples/flow_past_cylinder_2d.py>`_.


This example demonstrates:

* Setting up a wind tunnel kind of simulation.
* Setting up inlet and outlet boundary condition
* Force evaluation on the solid body of interest

The IOM is created in the :py:class:`Application` instance however, it is passed
to a :py:class:`Scheme` instance and most of its methods are called in the
scheme only. We discuss the implementation in the EDAC :py:class:`Scheme` in
:ref:`iom`. The IOM has the following functions:

* Creation of ghost particle arrays
* Creation of inlet outlet stepper
* Creation of inlet outlet equations
* Creation of inlet outlet updater

The following are discussed in detail:

* Construction of IOM
* Passing IOM to the scheme
* Creating ghost particles
* Creating updater
* Overall setup
* Evaluating forces on solid

Construction of IOM
~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    def _get_io_info(self):
        from pysph.sph.bc.hybrid.inlet import Inlet
        from pysph.sph.bc.hybrid.outlet import Outlet
        from pysph.sph.bc.hybrid.simple_inlet_outlet import (
            SimpleInletOutlet)
        i_update_cls = Inlet
        o_update_cls = Outlet
        o_has_ghost = False
        manager = SimpleInletOutlet
        props_to_copy += ['uta', 'pta', 'u0', 'v0', 'w0', 'p0']

        inlet_info = InletInfo(
            pa_name='inlet', normal=[-1.0, 0.0, 0.0],
            refpoint=[0.0, 0.0, 0.0], equations=inleteqns,
            has_ghost=i_has_ghost, update_cls=i_update_cls,
            umax=umax
            )

        outlet_info = OutletInfo(
            pa_name='outlet', normal=[1.0, 0.0, 0.0],
            refpoint=[self.Lt, 0.0, 0.0], has_ghost=o_has_ghost,
            update_cls=o_update_cls, equations=None,
            props_to_copy=props_to_copy
        )

        return inlet_info, outlet_info, manager

    def _create_inlet_outlet_manager(self):
        inlet_info, outlet_info, manager = self._get_io_info()
        iom = manager(
            fluid_arrays=['fluid'], inletinfo=[inlet_info],
            outletinfo=[outlet_info]
        )
        return iom

In the function ``_get_io_info`` the ``inlet_info`` and ``outlet_info`` are
created, and manager class are returned. The ``inlet_info`` and ``outlet_info``
info contains specific information about inlet and outlet that enables IOM to
create equations, stepper and updater. In ``_create_inlet_outlet_manager``
the IOM is created using the info objects.

Note that the extra properties required by the equations are also passed by the IOM.

Passing IOM to scheme
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    def configure_scheme(self):
        scheme = self.scheme
        self.iom = self._create_inlet_outlet_manager()
        scheme.inlet_outlet_manager = self.iom
        pfreq = 100
        kernel = QuinticSpline(dim=2)
        self.iom.update_dx(self.dx)
        scheme.configure(h=self.h, nu=self.nu)

        scheme.configure_solver(kernel=kernel, tf=self.tf, dt=self.dt,
                                pfreq=pfreq, n_damp=0)


The IOM object of the application is initialized in the method
``configure_scheme`` of the ``Application`` class. All the post-initialization
method which require data from user could be called here e.g. ``update_dx``.

Creating ghost particles
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    def create_particles(self):
        fluid = self._create_fluid()
        solid = self._create_solid()
        outlet = self._create_outlet()
        inlet = self._create_inlet()
        wall = self._create_wall()

        ghost_inlet = self.iom.create_ghost(inlet, inlet=True)
        ghost_outlet = self.iom.create_ghost(outlet, inlet=False)

        particles = [fluid, inlet, outlet, solid, wall]
        if ghost_inlet:
            particles.append(ghost_inlet)
        if ghost_outlet:
            particles.append(ghost_outlet)

        self.scheme.setup_properties(particles)
        self._set_wall_normal(wall)

        if self.io_method == 'hybrid':
            fluid.uag[:] = umax
            fluid.uta[:] = umax
            outlet.uta[:] = umax

        return particles

The particle arrays ``ghost_inlet`` and ``ghost_outlet`` are generated by
the IOM depending upon the type of IOM subclass used. The properties
:math:`uag`, :math:`uta` are the time average and velocity array in :math:`x`
direction at t=0.

Creating updater
~~~~~~~~~~~~~~~~~

The purpose of the updater is to remove particle from ``inlet`` and add them to
``fluid`` whenever a particle crosses the inlet-outlet interface. Similarly, it
is done in case of the ``oulet``. It also adds new particle to ``inlet`` as
required and remove a particle from the simulation when they flow past
``outlet``.

.. code:: python

    def create_inlet_outlet(self, particle_arrays):
        iom = self.iom
        io = iom.get_inlet_outlet(particle_arrays)
        return io

the function ``create_inlet_outlet`` takes the updater ``io`` created by the
IOM and plugs it into the update routine of the application class automatically.

Overall setup
~~~~~~~~~~~~~

In order to run the simulation, the IOM object must be passed to the scheme.
In the scheme, the IOM object must be implemented in the manner as described in
:ref:`iom`.

A few points to note while dealing with inlet outlet boundary condition,

1. Construction of the IOM happens after the scheme is created with a
   ``void`` IOM.

    .. code:: python

        def create_scheme(self):
            h = nu = None
            s = EDACScheme(
                ['fluid'], ['solid'], dim=2, rho0=rho, c0=c0, h=h, pb=p0,
                nu=nu, inlet_outlet_manager=None,
                inviscid_solids=['wall']
            )
            return s


2. The IOM must be configured in the ``configure_scheme`` function.

3. In case you change the integrator of the function, make sure the updater
   ``io`` is updating in the appropriate stage. For example, in case of a
   ``PECIntegrator`` class of integrator, the particles integrated half step in
   stage 1 and finally advected in stage 2 then ``io`` updates the particle
   arrays after stage 2 is complete. In case one wants to do the update in stage
   1 (while using another integrator) the arguments must be passed to the updater appropriately.


Evaluating forces on solid
~~~~~~~~~~~~~~~~~~~~~~~~~~

The force on the fluid particles is evaluated using

.. math::
        a = \frac{-\nabla{p}}{\rho} + \nu \nabla^{2} \mathbf{u}

In order to evaluate the forces, the ``solid`` is considered as fluid and
force is evaluated by solving the following equations


.. code:: python

        equations = [
            Group(
                equations=[
                    SummationDensity(dest='fluid', sources=['fluid', 'solid']),
                    SummationDensity(dest='solid', sources=['fluid', 'solid']),
                    SetWallVelocity(dest='solid', sources=['fluid']),
                    ], real=False),
            Group(
                equations=[
                    # Pressure gradient terms
                    MomentumEquationPressureGradient(
                        dest='solid', sources=['fluid'], pb=p0),
                    SolidWallNoSlipBCReverse(
                        dest='solid', sources=['fluid'], nu=self.nu),
                    ], real=True),
        ]

The equations are solved on the output saved as *.npz files. In the
equation ``SolidWallNoSlipBCReverse`` we are just reversing the sign of the
velocity difference unlike the usual equation where :math:`u - u_g` is used.
The total force is evaluated by multiplying the acceleration with the mass of
the solid particles

.. code:: python

        fxp = sum(solid.m*solid.au)
        fyp = sum(solid.m*solid.av)
        fxf = sum(solid.m*solid.auf)
        fyf = sum(solid.m*solid.avf)
        fx = fxf + fxp
        fy = fyf + fyp

Here, the ``au`` is acceleration due to pressure and ``auf`` is due to shear
stress. The force ``fx`` provides the drag force and ``fy`` provides the lift
force.
