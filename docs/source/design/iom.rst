.. _iom:

===========================
Writing inlet oulet manager
===========================

This section discusses writing your own Inlet Outlet Manager (IOM). If you want
to use the existing IOM subclass present in PySPH see :ref:`flow_past_cylinder`.
The IOM manages all the inputs required to simulate the open boundaries in
PySPH. It has the following functions:

* Create ghost particles
* Create inlet/outlet stepper
* Creation of inlet/outlet equations
* Creation of inlet/outlet particle updater

Overview
~~~~~~~~

.. py:currentmodule:: pysph.sph.bc.inlet_outlet_manager

The brief overview of :py:class:`InletOutletManager` subclass::

    class MyIOM(InletOutletManager):
        def __init__(self, fluid_arrays, inletinfo, outletinfo,
                    extraeqns=None):
            # Create the object to manage inlet outlet boundary conditions.
            # Most of the variables are evaluated after the scheme and particles
            # are created after application.consume_user_options runs.

        def create_ghost(self, pa_arr, inlet=True):
            # Creates ghosts for the given inlet/outlet particles
            # return ghost_pa (the ghost particle array for the pa_arr)

        def update_dx(self, dx):
            # Update the discretization length

        def add_io_properties(self, pa, scheme=None):
            # Add properties to be used in inlet/outlet equations
            # return the list of properties

        def get_io_names(self, ghost=False):
            # Return all the names of inlets and outlets

        def get_stepper(self, scheme, integrator, **kw):
            # Returns the steppers for inlet/outlet

        def setup_iom(self, dim, kernel):
            # User data in application.consume_user_options are passed

        def get_equations(self, scheme, **kw):
            # Returns the equations for inlet/outlet

        def get_equations_post_compute_acceleration(self):
            # Returns the equations for inlet/outlet used post acceleration
            # computation

        def get_inlet_outlet(self, particle_array):
            # Returns list of `Inlet` and `Outlet` instances which
            # updates inlet particles to fluid and fluid particles to outlet.
            # This also creates new inlet particle and consume outlet particles.

.. py:currentmodule:: pysph.sph.wc.edac

- The `IOM` gets initialized in the ``configure_scheme`` method in the
  :py:class:`Application` instance.

- The IOM is initialized using the list of fluid particle array ``fluid_arrays``,
  and ``inlet_info`` and ``outlet_info`` instances of :py:class:`InletInfo` and
  :py:class:`OutletInfo`, respectively. These info class contains the
  information of inlet/outlet like direction, size etc.

To explain the inlet outlet manager in detail, let us consider the mirror
boundary implemented using IOM class in `simple_inlet_outlet.py
<https://github.com/pypr/pysph/blob/master/pysph/sph/bc/mirror/simple_inlet_outlet.py>`_
for `EDACScheme
<https://github.com/pypr/pysph/blob/master/pysph/sph/wc/edac.py>`_::


    class EDACScheme(Scheme):
        def __init__(self, fluids, solids, dim, c0, nu, rho0, pb=0.0,
                    gx=0.0, gy=0.0, gz=0.0, tdamp=0.0, eps=0.0, h=0.0,
                    edac_alpha=0.5, alpha=0.0, bql=True, clamp_p=False,
                    inlet_outlet_manager=None, inviscid_solids=None):
            ...
            self.inlet_outlet_manager = inlet_outlet_manager
            ...

        def configure_solver(self, kernel=None, integrator_cls=None,
                            extra_steppers=None, **kw):
            ...
            iom = self.inlet_outlet_manager
            if iom is not None:
                iom_stepper = iom.get_stepper(self, cls, self.use_tvf)
                for name in iom_stepper:
                    steppers[name] = iom_stepper[name]
            ...
            if iom is not None:
                iom.setup_iom(dim=self.dim, kernel=kernel)

        def setup_properties(self, particles, clean=True):
            ...
            iom = self.inlet_outlet_manager
            fluids_with_io = self.fluids
            if iom is not None:
                io_particles = iom.get_io_names(ghost=True)
                fluids_with_io = self.fluids + io_particles
            for fluid in fluids_with_io:
                ...
                if iom is not None:
                    iom.add_io_properties(pa, self)
            ...

        def create_equations(self):
            ...
            return self._get_internal_flow_equations()

        def _get_internal_flow_equations(self):
            ...
            iom = self.inlet_outlet_manager
            fluids_with_io = self.fluids
            if iom is not None:
                fluids_with_io = self.fluids + iom.get_io_names()

            equations = []
            if iom is not None:
                io_eqns = iom.get_equations(self, self.use_tvf)
                for grp in io_eqns:
                    equations.append(grp)
            ...
            if iom is not None:
                io_eqns = iom.get_equations_post_compute_acceleration()
                for grp in io_eqns:
                    equations.append(grp)

            return equations
.. py:currentmodule:: pysph.sph.wc.edac

- The additional properties can be added in the function ``add_io_properties``
  which is called in the function ``setup_properties`` of a :py:class:`Scheme`
  instance.

- The ``get_stepper`` function passes the appropriate stepper for the inlet and
  outlet in the ``configure_solver`` method of the :py:class:`Scheme` instance.

- The ``get_equations`` and ``get_equations_post_compute_acceleration`` provides
  the additional equations to be used to interpolate properties from fluid
  particle arrays. This is to be called in ``create_equations`` method of the
  :py:class:`Scheme` instance.

- Any additional data required from the :py:class:`Application` or
  :py:class:`Scheme` instance can be
  passed to the IOM using ``setup_iom`` method.

Additionally, in the :py:class:`Application` instance:

- The ``get_inlet_outlet`` methods provides the instances for the
  :py:class:`Inlet` and :py:class:`Outlet` which updates the particles
  when they cross the interface. This method is called in ``create_inlet_outlet``
  method of the :py:class:`Application` instance.

- In mirror type inlet-outlet a ghost layer of particles is required which is a
  mere reflection about the inlet/outlet-fluid interface. It is created in
  ``create_particles`` using ``create_ghost``.

The IOM enables the management of the above steps easy to handle. An example
showing the usage of IOM is the `flow_past_cylinder_2d.py
<https://github.com/pypr/pysph/tree/master/pysph/examples/flow_past_cylinder_2d.py>`_.

.. note::

   The IOM is a convenience to manage various attributes of inlet/outlet
   implementation in PySPH but all this is not automatic. The user has to take
   care of appropriate invocation of the methods in the IOM in
   :py:class:`Application` and :py:class:`Scheme` instances.