.. _design_overview:

This document is an introduction to the design of PySPH. Read this if
you want to understand the automatic code generation and possibly
extend PySPH to solve problems other than those provided in the main
distribution.

=====================
PySPH code generation
=====================

To elucidate some of the internal details of PySPH, we will consider a
typical SPH problem and proceed to write the code that implements
it. Thereafter, we will describe auto-generated code that PySPH
generates. 

The dam-break problem
-------------------------

The problem that is used for the illustration is the Weakly
Compressible SPH (WCSPH) formulation for free surface flows, applied
to a breaking dam problem:

.. figure:: ../../Images/dam-break-schematic.png
   :align: center

A column of water is initially at rest (presumably held in place by
some membrane). The problem simulates a breaking dam in that the
membrane is instantly removed and the column is free to fall under
it's own weight and the effect of gravity. This and other variants of
the dam break problem can be found in the `examples` directory of
PySPH.

Equations
^^^^^^^^^^

The discrete equations for this formulation are given as

.. math::
   :label: eos 

   p_a = B\left( \left(\frac{\rho_a}{\rho_0}\right)^{\gamma} - 1 \right )

.. math::
   :label: continuity
 
   \frac{d\rho_a}{dt} = \sum_{b=1}^{N}m_b\,(\vec{v_b} - \vec{v_a})\cdot\,\nabla_a W_{ab}

.. math::
   :label: momentum
   
   \frac{d\vec{v_a}}{dt} = -\sum_{b=1}^Nm_b\left(\frac{p_a}{\rho_a^2} + \frac{p_b}{\rho_b^2}\right)\nabla W_{ab}

.. math::
   :label: position

   \frac{d\vec{x_a}}{dt} = \vec{v_a}

Boundary conditions
^^^^^^^^^^^^^^^^^^^^

The dam break problem involves two *types* of particles. Namely, the
*fluid* (water column) and *solid* (tank). The basic boundary
condition enforced on a solid wall is the no-penetration boundary
condition which can be stated as

.. math::

   \vec{v_f}\cdot \vec{n_b} = 0

Where :math:`\vec{n_b}` is the local normal vector for the
boundary. For this example, we use the *dynamic boundary conditions*.
For this boundary condition, the boundary particles are treated as
*fixed* fluid particles that evolve with the continuity
(:eq:`continuity`) and equation the of state (:eq:`eos`). In addition,
they contribute to the fluid acceleration via the momentum equation
(:eq:`momentum`). When fluid particles approach a solid wall, the
density of the fluids and the solids increase via the continuity
equation. With the increased density and consequently increased
pressure, the boundary particles express a repulsive force on the
fluid particles, thereby enforcing the no-penetration condition.

Time integration
^^^^^^^^^^^^^^^^^

For the time integration, we use a second order predictor-corrector
integrator. For the predictor stage, the following operations are
carried out:

.. math::
   :label: predictor

   \rho^{n + \frac{1}{2}} = \rho^n + \frac{\Delta t}{2}(a_\rho)^{n-\frac{1}{2}} \\

   \boldsymbol{v}^{n + \frac{1}{2}} = \boldsymbol{v}^n + \frac{\Delta t}{2}(\boldsymbol{a_v})^{n-\frac{1}{2}} \\

   \boldsymbol{x}^{n + \frac{1}{2}} = \boldsymbol{x}^n + \frac{\Delta t}{2}(\boldsymbol{u} + \boldsymbol{u}^{\text{XSPH}})^{n-\frac{1}{2}}

Once the variables are predicted to their half time step values, the
pairwise interactions are carried out to compute the
accelerations. Subsequently, the corrector is used to update the
particle positions:

.. math::
   :label: corrector
   
   \rho^{n + 1} = \rho^n + \Delta t(a_\rho)^{n+\frac{1}{2}} \\

   \boldsymbol{v}^{n + 1} = \boldsymbol{v}^n + \Delta t(\boldsymbol{a_v})^{n+\frac{1}{2}} \\

   \boldsymbol{x}^{n + 1} = \boldsymbol{x}^n + \Delta t(\boldsymbol{u} + \boldsymbol{u}^{\text{XSPH}})^{n+\frac{1}{2}}

.. note::
   
   The acceleration variables are *prefixed* like :math:`a_`. The
   boldface symbols in the above equations indicate vector
   quantities. Thus :math:`a_\boldsymbol{v}` represents :math:`a_u,\,
   a_v,\, \text{and}\, a_w` for the vector components of acceleration.


Required arrays and properties
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We will be using two **ParticleArrays**, one for the fluid and another
for the solid. Recall that for the dynamic boundary conditions, the
solid is treated like a fluid with the only difference being that the
velocity (:math:`a_\boldsymbol{v}`) and position accelerations
(:math:`a_\boldsymbol{x} = \boldsymbol{u} +
\boldsymbol{u}^{\text{XSPH}}`) are never calculated. The solid
particles therefore remain fixed for the duration of the simulation.

To carry out the integrations for the particles, we require the
following variables:

  - SPH properties: `x, y, z, u, v, w, h, m, rho, p, cs`
  - Acceleration variables: `au, av, aw, ax, ay, az, arho`
  - Properties at the beginning of a time step: `x0, y0, z0, u0, v0, w0, rho0`


A non-PySPH implementation
--------------------------

We first consider the pseudo-code for the non-PySPH implementation. We
assume we have been given two **ParticleArrays** `fluid` and `solid`
corresponding to the dam-break problem. We also assume that an
**NNPS** object `nps` is available and can be used for neighbor
queries:

.. code-block:: python

   from pysph.base import nnps
   fluid = get_particle_array_fluid(...)
   solid = get_particle_array_solid(...)
   particles = [fluid, solid]
   nps = nnps.LinkedListNNPS(dim=2, particles=particles, radius_scale=2.0)

The part of the code responsible for the interactions can be defined
as

.. code-block:: python

   class SPHCalc:
       def __init__(nnps, particles):
	   self.nnps = nnps
	   self.particles = particles

       def compute(self):
           self.eos()
           self.accelerations()

       def eos(self):
	   for array in self.particles:
	       num_particles = array.get_number_of_particles()
	       for i in range(num_particles):
		   array.p[i] =  # TAIT EOS function for pressure
		   array.cs[i] = # TAIT EOS function for sound speed

       def accelerations(self):
	   fluid, solid = self.particles[0], self.particles[1]
	   nps = self.nps
	   nbrs = UIntArray()

	   # continuity equation for the fluid
	   dst = fluid; dst_index = 0

	   # source is fluid
	   src = fluid; src_index = 0
	   num_particles = dst.get_number_of_particles()
	   for i in range(num_particles):

	       # get nearest fluid neigbors
	       nps.get_nearest_particles(src_index, dst_index, d_idx=i, nbrs)

	       for j in nbrs:
		   # pairwise quantities
		   xij = dst.x[i] - src.x[j]
		   yij = dst.y[i] - src.y[j]
		   ...

		   # kernel interaction terms
		   wij = kenrel.function(xi, ...)  # kernel function
		   dwij= kernel.gradient(xi, ...)  # kernel gradient

		   # compute the interaction and store the contribution
		   dst.arho[i] += # interaction term

	   # source is solid
	   src = solid; src_index = 1
	   num_particles = dst.get_number_of_particles()
	   for i in range(num_particles):

	       # get nearest fluid neigbors
	       nps.get_nearest_particles(src_index, dst_index, d_idx=i, nbrs)

	       for j in nbrs:
		   # pairwise quantities
		   xij = dst.x[i] - src.x[j]
		   yij = dst.y[i] - src.y[j]
		   ...

		   # kernel interaction terms
		   wij = kenrel.function(xi, ...)  # kernel function
		   dwij= kernel.gradient(xi, ...)  # kernel gradient

		   # compute the interaction and store the contribution
		   dst.arho[i] += # interaction term

	   # Destination is solid
	   dst = solid; dst_index = 1

	   # source is fluid
	   src = fluid; src_index = 0

	   num_particles = dst.get_number_of_particles()
	   for i in range(num_particles):

	       # get nearest fluid neigbors
	       nps.get_nearest_particles(src_index, dst_index, d_idx=i, nbrs)

	       for j in nbrs:
		   # pairwise quantities
		   xij = dst.x[i] - src.x[j]
		   yij = dst.y[i] - src.y[j]
		   ...

		   # kernel interaction terms
		   wij = kenrel.function(xi, ...)  # kernel function
		   dwij= kernel.gradient(xi, ...)  # kernel gradient

		   # compute the interaction and store the contribution
		   dst.arho[i] += # interaction term

We see that the use of multiple particle arrays has forced us to write
a fairly long piece of code for the accelerations. In fact, we have
only shown the part of the main loop that computes :math:`a_\rho` for
the continuity equation. Recall that our problem states that the
continuity equation should evaluated for all particles, taking
influences from all other particles into account. For two particle
arrays (*fluid*, *solid*), we have four such pairings (fluid-fluid,
fluid-solid, solid-fluid, solid-solid). The last one can be eliminated
when we consider the that the boundary has zero velocity and hence the
contribution will always be trivially zero.

The apparent complexity of the `SPHCalc.accelerations` method
notwithstanding, we notice that similar pieces of the code are being
repeated. In general, we can break down the computation for a general
source-destination pair like so:

.. code-block:: python

   # consider first destination particle array

   for all dst particles:
       get_neighbors_from_source()
       for all neighbors:
           compute_pairwise_terms()
           compute_inteactions_for_dst_particle()

   # consider next source for this destination particle array
   ...

   # consider the next destination particle array

.. note::

   The `SPHCalc.compute` method first calls the EOS before calling the
   main loop to compute the accelerations. This is because the EOS
   (which updates the pressure) must logically be completed for all
   particles before the accelerations (which uses the pressure) are
   computed.

The predictor-corrector integrator for this problem can be defined as

.. code-block:: python

   class Integrator:
       def __init__(self, particles, nps, calc):
           self.particles = particles
           self.nps = nps
           self.calc = calc

       def initialize(self):
           for array in self.particles:
               array.rho0[:] = array.rho[:]
	       ...
               array.w0[:] = array.w[:]    

      def predictor(self, dt):
	  dtb2 = 0.5 * dt
	  for array in self.particles:
	      array.rho = array.rho0[:] + dtb2*array.arho[:]

	      array.u = array.u0[:] + dtb2*array.au[:]
	      array.v = array.v0[:] + dtb2*array.av[:]
              ...
	      array.z = array.z0[:] + dtb2*array.az[:]

      def corrector(self, dt):
	  for array in self.particles:
	      array.rho = array.rho0[:] + dt*array.arho[:]

	      array.u = array.u0[:] + dt*array.au[:]
	      array.v = array.v0[:] + dt*array.av[:]
              ...
	      array.z = array.z0[:] + dt*array.az[:]

      def integrate(self, dt):
          self.initialize()
	  self.predictor(dt)   # predictor step
          self.nps.update()    # update NNPS structure
          self.calc.compute()  # compute the accelerations
          self.corrector(dt)   # corrector step

The `Integrator.integrate` method is responsible for updating the
solution the next time level. Before the predictor stage, the
`Integrator.initialize` method is called to store the values `x0,
y0...` at the beginning of a time-step. Given the positions of the
particles at the half time-step, the **NNPS** data structure is
updated before calling the `SPHCalc.compute` method. Finally, the
corrector step is called once we have the updated accelerations.

This hypothetical implementation can be integrated to the final time
by calling the `Integrator.integrate` method repeatedly. In the next
section, we will see how PySPH does this automatically.

.. Consider a typical SPH simulation described by the discrete equations:

.. .. math::

..    p_a = B\left( \left(\frac{\rho_a}{\rho_0}\right)^{\gamma} - 1 \right )
 
..    \frac{D\rho_a}{Dt} = \sum_{b=1}^{N}m_b\,(\vec{v_b} - \vec{v_a})\cdot\,\nabla_a W_{ab}
   
..    \frac{D\vec{v_a}}{Dt} = -\sum_{b=1}^Nm_b\left(\frac{p_a}{\rho_a^2} + \frac{p_b}{\rho_b^2}\right)\nabla W_{ab}

..    \frac{D\vec{x_a}}{Dt} = \vec{v_a}

.. One might expect an algorithm for SPH to proceed along the lines of
.. the following figure:

.. .. _sph-flowchart:
.. .. figure:: images/sph-flowchart.png
..    :align: center
..    :width: 300

.. --------------------
.. PySPH Design
.. --------------------

.. PySPH attempts to abstract out the operations represented in the
.. flowchart. To do this, PySPH is divided into four modules as shown in
.. the figure: `pysph_modules`_. The module breakup is implied by the
.. color scheme used.

.. .. _pysph_modules:
.. .. figure:: images/pysph-modules.png
..     :align: center
..     :width: 500

.. **********
.. pysph.base
.. **********

.. The :mod:`base` module defines the data structures to hold particle
.. information and index the particle positions for fast neighbor queries
.. and as such, is the building block for the particle framework that is
.. PySPH. 

.. As seen in the flowchart (`sph-flowchart`_), the *find neighbors*
.. process is within the inner loop, iterating over each particle. This
.. module can be thought of as the *base* over which all of the other
.. functionality of PySPH is built.

.. .. toctree::
..    :maxdepth: 2

..    base

.. **********
.. pysph.sph
.. **********

.. The :mod:`sph` module is where all the SPH functions are defined. It
.. also defines the **SPHCalc** object which is used to iterate over each
.. particle (colored gray in the flowchart: `sph-flowchart`_)

.. *************
.. pysph.solver
.. *************

.. The :mod:`solver` module is used to drive the simulation via the
.. **Solver** object and also the important function of integration
.. (represented as *step variables* in the flowchart:
.. `sph-flowchart`_). Other functions like computing the new time step
.. and saving output (not shown in the flowchart) are also under the
.. ambit of the :mod:`solver` module.

.. .. toctree::
..    :maxdepth: 2

..    solver_interfaces


