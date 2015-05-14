PySPH: a Python-based SPH framework
====================================


PySPH is an open source framework for Smoothed Particle Hydrodynamics (SPH)
simulations.  It is implemented in Python_ and the performance critical parts
are implemented in Cython_.

PySPH is implemented in a way that allows a user to specify the entire SPH
simulation in pure Python. High-performance code is generated from this
high-level Python code, compiled on the fly and executed.  PySPH can use OpenMP
to utilize multi-core CPUs effectively.  PySPH also features optional automatic
parallelization (multi-CPU) using mpi4py_ and Zoltan_.  If you wish to use the
parallel capabilities you will need to have these installed.

Here are videos of simulations made with PySPH.

.. raw:: html

    <div align="center">
    <iframe width="560" height="315"
        src="https://www.youtube.com/embed/videoseries?list=PLH8Y2KepC2_VPLrcTiWGaYYh88gGVAuVr"
        frameborder="0" allowfullscreen>
    </iframe>
    </div>


.. _Python: http://www.python.org
.. _Cython: http://www.cython.org
.. _mpi4py: http://mpi4py.scipy.org
.. _Zoltan: http://www.cs.sandia.gov/zoltan/


Features
---------

  - User scripts and equations are written in pure Python.
  - Flexibility to define arbitrary SPH equations operating on particles.
  - Ability to define your own multi-step integrators in pure Python.
  - High-performance: our performance is comparable to hand-written solvers
    implemented in FORTRAN.
  - Seamless multi-core support with OpenMP.
  - Seamless parallel integration using Zoltan_.

SPH formulations
-----------------

Currently, PySPH has numerous examples to solve the viscous, incompressible
Navier-Stokes equations using the weakly compressible (WCSPH) approach. The
following formulations are currently implemented:

- `Weakly Compressible SPH (WCSPH)`_ for free-surface flows (Gesteira et al. 2010, Journal of Hydraulic Research, 48, pp. 6--27)

.. figure:: ../Images/db3d.png
   :width: 500 px
   :align: center

   3D dam-break past an obstacle SPHERIC benchmark `Test 2`_

- `Transport Velocity Formulation`_ for incompressilbe fluids (Adami et al. 2013, JCP, 241, pp. 292--307).

.. figure:: ../Images/ldc-streamlines.png
   :width: 500 px
   :align: center

   Streamlines for a driven cavity

- `SPH for elastic dynamics`_ (Gray et al. 2001, CMAME, Vol. 190, pp 6641--6662)

.. figure:: ../Images/rings-collision.png
   :width: 500 px
   :align: center

   Collision of two elastic rings.


- `Compressible SPH`_ (Puri et al. 2014, JCP, Vol. 256, pp 308--333)

.. _`Weakly Compressible SPH (WCSPH)`: http://www.tandfonline.com/doi/abs/10.1080/00221686.2010.9641250

.. _`Transport Velocity Formulation`: http://dx.doi.org/10.1016/j.jcp.2013.01.043

.. _`SPH for elastic dynamics`: http://dx.doi.org/10.1016/S0045-7825(01)00254-7

.. _`Compressible SPH`: http://dx.doi.org/10.1016/j.jcp.2013.08.060

.. _`Test 2`: https://wiki.manchester.ac.uk/spheric/index.php/Test2
