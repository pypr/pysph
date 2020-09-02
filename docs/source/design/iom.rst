.. _iom:

==================
Writing Inlet oulet manager (IOM)
==================

This section essentially talks about writing your own inlet outlet manager.
If you want to use the existing IOM subclass present in PySPH see :ref:`flow_past_cylinder`. The IOM manages all the inputs required to simulate open boundaries in PySPH. It has the following functions

* Create ghost particles
* Create Inlet outlet stepper
* Creation of Inlet outlet equations
* Creation of Inlet outlet updater

Overview
~~~~~~~~

.. py:currentmodule:: pysph.sph.bc.inlet_outlet_manager

The brief Overview of :py:class:`InletOutletManager` subclass:: 

    class MyIOM(InletOutletManager):
        def __init__(self, fluid_arrays, inletinfo, outletinfo,
                    extraeqns=None):
            # Create the object to manage inlet outlet boundary conditions.
            # Most of the variables are evaluated after the scheme and particles
            # are created.

        def create_ghost(self, pa_arr, inlet=True):
            # Creates ghosts for the given inlet/outlet particles
            # return ghost_pa ( the ghost particle array for the pa_arr)

        def update_dx(self, dx):
            # Update the discretisation length

        def add_io_properties(self, pa, scheme=None):
            # Add properties to be used in inlet/outlet equations tp pa
            # pass

        def get_io_names(self, ghost=False):
            # return all the names of inlets and outlets
            
        def get_stepper(self, scheme, integrator, **kw):
            # Returns the steppers for inlet/outlet
            raise NotImplementedError()

        def setup_iom(self, dim, kernel):
            # Essential data passed

        def get_equations(self, scheme, **kw):
            # Returns the equations for inlet/outlet

        def get_equations_post_compute_acceleration(self):
            # Returns the equations for inlet/outlet used post acceleration
            computation
            return []

        def get_inlet_outlet(self, particle_array):
            # Returns list of `Inlet` and `Outlet` instances which
            # exchanges inlet particles to fluid
            # particles creates new inlet particle.

.. py:currentmodule:: pysph.sph.wc.edac

To explain the inlet outlet manager in detail let us consider the Mirror boundary implemented using IOM class in `simple_inlet_outlet.py
<https://github.com/pypr/pysph/blob/master/pysph/sph/bc/mirror/simple_inlet_outlet.py>`_. for `EDACScheme
<https://github.com/pypr/pysph/blob/master/pysph/sph/wc/edac.py>`_. 


- The IOM is initialized using the list of fluid particle array ``fluid_arrays`` and ``inlet_info`` and ``outlet_info`` instance of :py:class:`InletInfo` and :py:class:`OutletInfo` containing the information like direction, size etc.

- The `IOM` gets initialized in the ``configure_scheme`` method.

- The additional properties can be added in the function ``add_io_properties`` which is called in the function ``setup_properties`` of a :py:class:`Scheme` instance.

- The ``get_stepper`` function passes the appropriate stepper for the inlet and outlet in the ``configure_solver`` method of the :py:class:`Scheme` instance.

- The ``get_equations`` and ``get_equations_post_compute_acceleration`` provides the additional equations to be used to interpolate properties from fluid particle arrays. This is to be called in ``create_equations`` method of the :py:class:`Scheme` instance. 

- The ``get_inlet_outlet`` methods provides the instances for the :py:class:`InletInfo` and :py:class:`OutletInfo` which updates the particles when they cross the interface. This method is called in ``create_inlet_outlet`` method of the :py:class:`Application` instance. 

- Any additional data required from the :py:class:`Application` instance can be passed using ``setup_iom`` method. 

- In mirror type inlet-outlet a ghost layer of particles is required which is a mere reflection about the inlet/outlet-fluid interface. It is created in ``create_particles`` using ``create_ghost``. 

The IOM enables the management of the above steps easy to handle.