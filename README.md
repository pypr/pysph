README
-------

This codebase represents a new approach for PySPH that is more heavily based
on code generation.  It currently only supports Cython but can be configured
to work well in parallel as well.

The weakly compressible SPH formulation is the one implemented.  Free surface
problems and the Transport Velocity formulation are implemented.

**Please note this code is still very experimental and we will be changing
  the API and basic pieces without warning.  Please bear with us.**

Requirements
-------------

The following Python packages are needed:
    
  - numpy
  - Cython (ideally version 0.19 and above)
  - Mako

Optional dependencies:

  - mpi4py
  - [PyZoltan](https://bitbucket.org/kunalp/pyzoltan)
  - Mayavi2 (for visualization)
  
PyZoltan is used for parallel support.  You can still use PySPH without these
optional dependencies.

Build
-----

Build/install these packages using `pip`.  Or from the source:

   $ python setup.py install

There are several examples to try inside the `examples` directory.
