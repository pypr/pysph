PySPH: a Python-based SPH framework
====================================

.. image:: https://drone.io/bitbucket.org/pysph/pysph/status.png
    :alt: Build Status
    :target: https://drone.io/bitbucket.org/pysph/pysph/latest


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
  - High-performance.
  - Seamless parallel integration using Zoltan_.

Solvers
--------

Currently, PySPH has numerous examples to solve the viscous, incompressible
Navier-Stokes equations using the weakly compressible (WCSPH) approach. The
following formulations are currently implemented:

  - Classical WCSPH (Journal of Hydraulic Research Vol. 48 Extra Issue (2010),
    pp. 6–27)

  - Transport Velocity Formulation (Journal of Computational Physics Vol 241,
    May 2013, pp. 292-307)
