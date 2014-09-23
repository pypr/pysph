PySPH: a Python-based SPH framework
====================================


PySPH is an open source framework for Smoothed Particle Hydrodynamics (SPH)
simulations.  It is implemented in Python_ and the performance critical parts
are implemented in Cython_.

PySPH is implemented in a way that allows a user to specify the entire SPH
simulation in pure Python. High-performance code is generated from this
high-level Python code, compiled on the fly and executed.  PySPH also features
optional automatic parallelization using mpi4py_ and Zoltan_.  If you wish to
use the parallel capabilities you will need to have these installed.


.. _Python: http://www.python.org
.. _Cython: http://www.cython.org
.. _mpi4py: http://mpi4py.scipy.org
.. _Zoltan: http://www.cs.sandia.gov/zoltan/


Features
---------

  - User scripts and equations are written in pure Python.
  - Flexibility to define arbitrary SPH equations operating on particles.
  - Solver framework to add arbitrary collection of particles.
  - High-performance: our performance is comparable to hand-written solvers
    implemented in FORTRAN.
  - Seamless parallel integration using Zoltan_.

Solvers
--------

Currently, PySPH has numerous examples to solve the viscous, incompressible
Navier-Stokes equations using the weakly compressible (WCSPH) approach. The
following formulations are currently implemented:

- `Weakly Compressible SPH (WCSPH)`_ for free-surface flows (Gesteira et al. 2010, Journal of Hydraulic Research, 48, pp. 6--27)

.. figure:: ../Images/db3d.png
   :align: center

   3D dam-break past an obstacle SPHERIC benchmark `Test 2`_

- `Transport Velocity Formulation`_ for incompressilbe fluids (Adami et al. 2013, JCP, 241, pp. 292--307).

.. figure:: ../Images/ldc-streamlines.png
   :align: center

   Streamlines for a driven cavity

- `SPH for elastic dynamics`_ (Gray et al. 2001, CMAME, Vol. 190, pp 6641--6662)

.. figure:: ../Images/rings-collision.png
   :align: center

   Collision of two elastic rings.


- `Compressible SPH`_ (Puri et al. 2014, JCP, Vol. 256, pp 308--333)

.. _`Weakly Compressible SPH (WCSPH)`: http://www.tandfonline.com/doi/abs/10.1080/00221686.2010.9641250

.. _`Transport Velocity Formulation`: http://dx.doi.org/10.1016/j.jcp.2013.01.043

.. _`SPH for elastic dynamics`: http://dx.doi.org/10.1016/S0045-7825(01)00254-7

.. _`Compressible SPH`: http://dx.doi.org/10.1016/j.jcp.2013.08.060

.. _`Test 2`: https://wiki.manchester.ac.uk/spheric/index.php/Test2
