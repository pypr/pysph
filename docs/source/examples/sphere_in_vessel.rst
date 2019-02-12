.. _sphere_in_vessel:

A rigid sphere floating in an hydrostatic tank
----------------------------------------------

This example demonstrates the API of running a rigid fluid coupling problem in PySPH.
To run it one may do::

  $ cd ~/pysph/pysph/examples/rigid_body/
  $ python sphere_in_vessel_akinci.py

There are many command line options that this example provides, check them out with::

  $ python sphere_in_vessel.py -h

The example source can be seen at `sphere_in_vessel.py
<https://github.com/pypr/pysph/tree/master/pysph/examples/rigid_body/sphere_in_vessel_akinci.py>`_.


This example demonstrates:

* Setting up a simulation involving rigid bodies and fluid
* Discuss mainly about rigid fluid coupling

It is divided in to three parts:

* Create particles
* Create equations
* Run the application


Create particles
~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this example, we have a tank with a resting fluid and a sphere falling into
the tank. Create three particle arrays, ``tank``, ``fluid`` and ``cube``.
``tank`` and ``fluid`` has to obey ``wcsph`` scheme, where as ``cube`` has to obey
rigid body equations.

.. code:: python

    def create_particles(self):
        # elided
        fluid = get_particle_array_wcsph(x=xf, y=yf, h=h, m=m, rho=rho,
                                         name="fluid")

        # elided
        tank = get_particle_array_wcsph(x=xt, y=yt, h=h, m=m, rho=rho,
                                        rad_s=rad_s, V=V, name="tank")
        for name in ['fx', 'fy', 'fz']:
            tank.add_property(name)

        cube = get_particle_array_rigid_body(x=xc, y=yc, h=h, m=m, rho=rho,
                                             rad_s=rad_s, V=V, cs=cs,
                                             name="cube")

        return [fluid, tank, cube]

We will discuss the reason for adding the properties :math:`fx`, :math:`fy`, :math:`fz` to the
``tank`` particle array. The next step is to setup the equations.

Create equations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    def create_equations(self):
      equations = [
          Group(equations=[
              BodyForce(dest='cube', sources=None, gy=-9.81),
          ], real=False),
          Group(equations=[
              SummationDensity(
                  dest='fluid',
                  sources=['fluid'], ),
              SummationDensityBoundary(
                  dest='fluid', sources=['tank', 'cube'], fluid_rho=1000.0)
          ]),

          # Tait equation of state
          Group(equations=[
              TaitEOSHGCorrection(dest='fluid', sources=None, rho0=self.ro,
                                  c0=self.co, gamma=7.0),
          ], real=False),
          Group(equations=[
              MomentumEquation(dest='fluid', sources=['fluid'],
                               alpha=self.alpha, beta=0.0, c0=self.co,
                               gy=-9.81),
              AkinciRigidFluidCoupling(dest='fluid',
                                       sources=['cube', 'tank']),
              XSPHCorrection(dest='fluid', sources=['fluid', 'tank']),
          ]),
          Group(equations=[
              RigidBodyCollision(dest='cube', sources=['tank'], kn=1e5)
          ]),
          Group(equations=[RigidBodyMoments(dest='cube', sources=None)]),
          Group(equations=[RigidBodyMotion(dest='cube', sources=None)]),
      ]
      return equations


A few points to note while dealing with *Akinci* formulation,

1. As a first point, while computing the density of the ``fluid`` due to solid,
   make sure to use ``SummationDensityBoundary``, because usual
   ``SummationDensity`` computes density by considering the mass of the
   particle, where as ``SummationDensityBoundary`` will compute it by
   considering the volume of the particle. This makes a lot of difference
   while dealing with heavy density variation flows.

2. Apply ``TaitEOSHGCorrection`` so that there is no negative pressure.

3. The force from the boundary (here it is tank) on fluid is computed using
   ``AkinciRigidFluidCoupling`` equation, but in a usual case we do it using the
   momentum equation. There are a few advantages by doing this. If we are
   computing the boundary force using the momentum equation, then one should
   compute the density of the boundary, then compute the pressure. Using such
   pressure we will compute the force. But using ``AkinciRigidFluidCoupling`` we
   don't need to compute the pressure of the boundary because the force is
   dependent only on the fluid particle's pressure.

   .. code:: python

       def loop(self, d_idx, d_m, d_rho, d_au, d_av, d_aw,  d_p,
                s_idx, s_V, s_fx, s_fy, s_fz, DWIJ, s_m, s_p, s_rho):
           # elide
           d_au[d_idx] += -psi * _t1 * DWIJ[0]
           d_av[d_idx] += -psi * _t1 * DWIJ[1]
           d_aw[d_idx] += -psi * _t1 * DWIJ[2]

           s_fx[s_idx] += d_m[d_idx] * psi * _t1 * DWIJ[0]
           s_fy[s_idx] += d_m[d_idx] * psi * _t1 * DWIJ[1]
           s_fz[s_idx] += d_m[d_idx] * psi * _t1 * DWIJ[2]

   Since in ``AkinciRigidFluidCoupling`` (more in next point) we compute both
   force on fluid by solid particle and force on solid by fluid particle,
   which makes our sources to hold the properties ``fx``, ``fy`` and ``fz``.

4. Here first few equations deal with the simulation of fluid in hydrostatic
   tank. The equation dealing with rigid fluid coupling is
   ``AkinciRigidFluidCoupling`` . *Coupling* equation will deal with forces
   exerted by fluid on solid body, and forces exerted by solid on fluid. We
   find the force on fluid by solid and force on the solid by fluid in a singe
   equation.

   Usually in an SPH equation, we tend to change properties only of a destination
   particle array, but in this case, both destination and sources properties are
   manipulated.

5. The final equations deal with the dynamics of rigid bodies, which are
   discussed in other example files.

Run the application
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Finally run the application by

.. code:: python

    if __name__ == '__main__':
        app = RigidFluidCoupling()
        app.run()
