PySPH: a Python-based SPH framework
-------------------------------------


[![Build Status](https://drone.io/bitbucket.org/pysph/pysph/status.png)](https://drone.io/bitbucket.org/pysph/pysph/latest)


PySPH is an open source framework for Smoothed Particle Hydrodynamics
(SPH) simulations.   It is implemented in
[Python](http://www.python.org) and the performance
critical parts are implemented in [Cython](http://www.cython.org).

This codebase is the new approach for PySPH that is heavily based on
code generation. It currently supports Cython and is configured to
work in parallel.

The latest documentation for PySPH is available at [pysph.readthedocs.org](http://pysph.readthedocs.org).

Features
---------

  - Solver framework to add arbitrary collection of particles
  - Flexibility to define arbitrary SPH equations operating on particles
  - Seamless parallel integration using [Zoltan](http://www.cs.sandia.gov/zoltan/)

Solvers:
---------

Currently, PySPH is capable and has numerous examples to solve the
viscous, incompressible Navier-Stokes equations using the weakly
compressible (WCSPH) approach. The features of the implementation are:

  - `Weakly Compressible SPH (WCSPH)`_ for free-surface flows (Gesteira et al. 2010, Journal of Hydraulic Research, 48, pp. 6--27)
  - `Transport Velocity Formulation`_ for incompressilbe fluids (Adami et al. 2013, JCP, 241, pp. 292--307)
  - `SPH for elastic dynamics`_ (Gray et al. 2001, CMAME, Vol. 190, pp 6641--6662)

.. _`Weakly Compressible SPH (WCSPH)`: http://www.tandfonline.com/doi/abs/10.1080/00221686.2010.9641250

.. _`Transport Velocity Formulation`: http://dx.doi.org/10.1016/j.jcp.2013.01.043

.. _`SPH for elastic dynamics`: http://dx.doi.org/10.1016/S0045-7825(01)00254-7

Installation
=============

To install PySPH, you need a working Python environment. We recommend
[Enthought Canopy](https://www.enthought.com/products/canopy/) if you
are new to Python. Additional dependencies to compile the code are:

  - Cython (ideally version 0.19 and above)
  - Mako

These can be installed from the command line using `easy_install Cython mako`

Optional dependencies:
-----------------------

If you want to use PySPH in parallel, you will need
[mpi4py](http://mpi4py.scipy.org/) and the
[Zoltan](http://www.cs.sandia.gov/zoltan/) data management library.

Building and linking PyZoltan
-------------------------------

 1. Build Zoltan with the following compile options:

 ```
 $ ../configure --with-cflags=-fPIC --enable-mpi -with-mpi-incdir=/usr/include/openmpi-x86_64 --with-mpi-libdir=/usr/lib64/openmpi/lib --prefix=/home/<username>/usr/local/Zoltan --with-mpi-compilers=/usr/lib64/openmpi/bin/
 ```
 Of course, you have to provide the appropriate MPI directories on your system.

 2. Declare the environment variables `ZOLTAN_INCLUDE` and `ZOLTAN_LIBRARY`

 3. Install PySPH. The PyZoltan wrappers will be compiled and available.

If you wish to see a working build/test script please see the continous
[build script](https://drone.io/bitbucket.org/pysph/pysph/admin) hosted at [drone.io](http://drone.io).


Running the examples
---------------------

You can verify the installation by exploring some examples:

    $ cd examples
    $ python dam_break3D.py

This runs the 3D dam-break problem which is also a SPHERIC benchmark `Test 2`_

![IMAGE](https://bitbucket.org/kunalp/pysph/raw/docs/docs/Images/db3d.png)

PySPH is more than a tool for wave-body interactions:

    $ cd examples/TransportVelocity
    $ python cavity.py

This runs the driven cavity problem using the transport velocity
formulation of Adami et al. You can verify the results for this
problem using the helper script
`examples/TransportVelocity/ldcavity_results.py` to plot, for example
the streamlines:

![IMAGE](https://bitbucket.org/kunalp/pysph/raw/docs/docs/Images/ldc-streamlines.png)

If you want to use PySPH for elastic dynamics, you can try some of the
examples from the directory `examples/solid_mech`

    $ cd examples/solid_mech
    $ python rings.py

Which runs the problem of the collision of two elastic rings:

![IMAGE](https://bitbucket.org/kunalp/pysph/raw/stress/docs/Images/rings-collision.png)

The auto-generated code for the example resides in the directory
`~/.pysph/source`. A note of caution however, it's not for the faint
hearted.

.. _`Test 2`: https://wiki.manchester.ac.uk/spheric/index.php/Test2