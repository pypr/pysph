PySPH: a Python-based SPH framework
-------------------------------------

[![Build Status](https://api.shippable.com/projects/53a87e47f989286000c78a06/badge/master)](https://www.shippable.com/projects/53a87e47f989286000c78a06)
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

  - Solver framework to add arbitrary collection of particles.
  - Flexibility to define arbitrary SPH equations operating on particles in pure Python.
  - Any kind of user-defined multi-step integrator.
  - Seamless parallel integration using [Zoltan](http://www.cs.sandia.gov/zoltan/).
  - High-performance: our performance is comparable to hand-written solvers implemented in FORTRAN.

Solvers
---------

Currently, PySPH is capable and has numerous examples to solve the
viscous, incompressible Navier-Stokes equations using the weakly
compressible (WCSPH) approach. The features of the implementation are:

  - [Weakly Compressible SPH (WCSPH)](http://www.tandfonline.com/doi/abs/10.1080/00221686.2010.9641250) for free-surface flows (Gesteira et al. 2010, Journal of Hydraulic Research, 48, pp. 6--27)
  - [Transport Velocity Formulation](http://dx.doi.org/10.1016/j.jcp.2013.01.043) for incompressilbe fluids (Adami et al. 2013, JCP, 241, pp. 292--307)
  - [SPH for elastic dynamics](http://dx.doi.org/10.1016/S0045-7825(01)00254-7) (Gray et al. 2001, CMAME, Vol. 190, pp 6641--6662)
  - [Compressible SPH](http://dx.doi.org/10.1016/j.jcp.2013.08.060) (Puri et al. 2014, JCP, Vol. 256, pp 308--333)

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

To use the `pysph_viewer` you will need to have
[Mayavi](http://code.enthought.com/projects/mayavi) installed.

Building and linking PyZoltan
-------------------------------

 1. We've provided a simple Zoltan build script in the repository.  This can
 be used as so:

 ```
 $ ./build_zoltan.sh  INSTALL_PREFIX
 ```
 where the `INSTALL_PREFIX` is where the library and includes will be
 installed.  You may edit and tweak the build to suit your installation.
 However, this script  what we use to build Zoltan on our continuous
 integration servers on drone and shippable.

 2. Declare the environment variables `ZOLTAN_INCLUDE` and `ZOLTAN_LIBRARY`.
 If you used the above script, this would be:

 ```
 $ export ZOLTAN_INCLUDE=$INSTALL_PREFIX/include
 $ export ZOLTAN_LIBRARY=$INSTALL_PREFIX/lib
 ```

 3. Install PySPH. The PyZoltan wrappers will be compiled and available.

If you wish to see a working build/test script please see the our
[shippable.yml](https://bitbucket.org/pysph/pysph/src/master/shippable.yml).
Or you could see the [build script](https://drone.io/bitbucket.org/pysph/pysph/admin)
hosted at [drone.io](http://drone.io).


Running the examples
---------------------

You can verify the installation by exploring some examples.  A fairly quick
running example (taking about 5-10 minutes) would be the following:

    $ cd examples
    $ python dam_break.py

The solution can be viewed live by running:

    $ pysph_viewer

This requires that Mayavi be installed.  The saved output data can be viewed
by running:

    $ pysph_viewer dam_break_output/*.npz

A 3D version of the dam-break problem is also available, and may be run as:

    $ python dam_break3D.py

This runs the 3D dam-break problem which is also a SPHERIC benchmark [Test 2](https://wiki.manchester.ac.uk/spheric/index.php/Test2)

![IMAGE](https://bitbucket.org/pysph/pysph/raw/master/docs/Images/db3d.png)

PySPH is more than a tool for wave-body interactions:

    $ cd examples/transport_velocity
    $ python cavity.py

This runs the driven cavity problem using the transport velocity
formulation of Adami et al. You can verify the results for this
problem using the helper script
`examples/transport_velocity/ldcavity_results.py` to plot, for example
the streamlines:

![IMAGE](https://bitbucket.org/pysph/pysph/raw/master/docs/Images/ldc-streamlines.png)

If you want to use PySPH for elastic dynamics, you can try some of the
examples from the directory `examples/solid_mech`

    $ cd examples/solid_mech
    $ python rings.py

Which runs the problem of the collision of two elastic rings:

![IMAGE](https://bitbucket.org/pysph/pysph/raw/master/docs/Images/rings-collision.png)

The auto-generated code for the example resides in the directory
`~/.pysph/source`. A note of caution however, it's not for the faint
hearted.
