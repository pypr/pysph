1.0b1
-----

Around 140 pull requests were merged. Thanks to all who contributed to this
release (in alphabetical order): Abhinav Muta, Aditya Bhosale, Amal Sebastian,
Ananyo Sen, Antonio Valentino, Dinesh Adepu, Jeffrey D. Daye, Navaneet, Miloni
Atal, Pawan Negi, Prabhu Ramachandran, Rohan Kaushik, Tetsuo Koyama, and Yash
Kothari.

* Release date: 1st March 2022.

* Enhancements:

  * Use github actions for tests and also test OpenCL support on CI.
  * Parallelize the build step of the octree NNPS on the CPU.
  * Support for packing initial particle distributions.
  * Add support for setting load balancing weights for particle arrays.
  * Use meshio to read data and convert them into particles.
  * Add support for conditional group of equations.
  * Add options to control loop limits in a Group.
  * Add ``pysph binder``, ``pysph cull``, and ``pysph cache``.
  * Use OpenMP for initialize, loop and post_loop.
  * Added many SPH schemes: CRKSPH, SISPH, basic ISPH, SWE, TSPH, PSPH.
  * Added a mirror boundary condition along coordinate axes.
  * Add support for much improved inlets and outlets.
  * Add option ``--reorder-freq`` to turn on spatial reordering of particles.
  * API: Integrators explicitly call update_domain.
  * Basic CUDA support.
  * Many important improvements to the pysph Mayavi viewer.
  * Many improvements to the 3D and 2D jupyter viewer.
  * ``Application.customize_output`` can be used to customize viewer.
  * Use ``~/.compyle/config.py`` for user customizations.
  * Remove pyzoltan, cyarray, and compyle into their own packages on pypi.

* Bug fixes:

  * Fix issue with update_nnps being called too many times when set for a
    group.
  * Many OpenCL related fixes and improvements.
  * Fix bugs in the parallel manager code and add profiling information.
  * Fix hdf5 compressed output.
  * Fix ``pysph dump_vtk``
  * Many fixes to various schemes.
  * Fix memory leak with the neighbor caching.
  * Fix issues with using PySPH on FreeBSD.


1.0a6
-----

90 pull requests were merged for this release. Thanks to the following who
contributed to this release (in alphabetical order): A Dinesh, Abhinav Muta,
Aditya Bhosale, Ananyo Sen, Deep Tavker, Prabhu Ramachandran, Vikas Kurapati,
nilsmeyerkit, Rahul Govind, Sanka Suraj.


* Release date: 26th November, 2018.

* Enhancements:

  * Initial support for transparently running PySPH on a GPU via OpenCL.
  * Changed the API for how adaptive DT is computed, this is now to be set in
    the particle array properties called ``dt_cfl, dt_force, dt_visc``.
  * Support for non-pairwise particle interactions via the ``loop_all``
    method. This is useful for MD simulations.
  * Add support for ``py_stage1, py_stage2 ...``, methods in the integrator.
  * Add support for ``py_initialize`` and ``initialize_pair`` in equations.
  * Support for using different sets of equations for different stages of the
    integration.
  * Support to call arbitrary Python code from a ``Group`` via the
    ``pre/post`` callback arguments.
  * Pass ``t, dt`` to the reduce method.
  * Allow particle array properties to have strides, this allows us to define
    properties with multiple components. For example if you need 3 values per
    particle, you can set the stride to 3.
  * Mayavi viewer can now show non-real particles also if saved in the output.
  * Some improvements to the simple remesher of particles.
  * Add simple STL importer to import geometries.
  * Allow user to specify openmp schedule.
  * Better documentation on equations and using a different compiler.
  * Print convenient warning when particles are diverging or if ``h, m`` are
    zero.
  * Abstract the code generation into a common core which supports Cython,
    OpenCL and CUDA. This will be pulled into a separate package in the next
    release.
  * New GPU NNPS algorithms including a very fast oct-tree.
  * Added several sphysics test cases to the examples.


* Schemes:

  * Add a working Implicit Incompressible SPH scheme (of Ihmsen et al., 2014)
  * Add GSPH scheme from SPH2D and all the approximate Riemann solvers from there.
  * Add code for Shepard and MLS-based density corrections.
  * Add kernel corrections proposed by Bonet and Lok (1999)
  * Add corrections from the CRKSPH paper (2017).
  * Add basic equations of Parshikov (2002) and Zhang, Hu, Adams (2017)

* Bug fixes:

  * Ensure that the order of equations is preserved.
  * Fix bug with dumping VTK files.
  * Fix bug in Adami, Hu, Adams scheme in the continuity equation.
  * Fix mistake in WCSPH scheme for solid bodies.
  * Fix bug with periodicity along the z-axis.


1.0a5
-----

* Release date:  17th September, 2017
* Mayavi viewer now supports empty particle arrays.
* Fix error in scheme chooser which caused problems with default scheme
  property values.
* Add starcluster support/documentation so PySPH can be easily used on EC2.
* Improve the particle array so it automatically ravel's the passed arrays and
  also accepts constant values without needing an array each time.
* Add a few new examples.
* Added 2D and 3D viewers for Jupyter notebooks.
* Add several new Wendland Quintic kernels.
* Add option to measure coverage of Cython code.
* Add EDAC scheme.
* Move project to github.
* Improve documentation and reference section.
* Fix various bugs.
* Switch to using pytest instead of nosetests.
* Add a convenient geometry creation module in ``pysph.tools.geometry``
* Add support to script the viewer with a Python file, see ``pysph view -h``.
* Add several new NNPS schemes like extended spatial hashing, SFC, oct-trees
  etc.
* Improve Mayavi viewer so one can view the velocity vectors and any other
  vectors.
* Viewer now has a button to edit the visualization properties easily.
* Add simple tests for all available kernels. Add ``SuperGaussian`` kernel.
* Add a basic dockerfile for pysph to help with the CI testing.
* Update build so pysph can be built with a system zoltan installation that is
  part of trilinos using the ``USE_TRILINOS`` environment variable.
* Wrapping the ``Zoltan_Comm_Resize`` function in ``pyzoltan``.


1.0a4
------

* Release date: 14th July, 2016.
* Improve many examples to make it easier to make comparisons.
* Many equation parameters no longer have defaults to prevent accidental
  errors from not specifying important parameters.
* Added support for ``Scheme`` classes that manage the generation of equations
  and solvers.  A user simply needs to create the particles and setup a scheme
  with the appropriate parameters to simulate a problem.
* Add support to easily handle multiple rigid bodies.
* Add support to dump HDF5 files if h5py_ is installed.
* Add support to directly dump VTK files using either Mayavi_ or PyVisfile_,
  see ``pysph dump_vtk``
* Improved the nearest neighbor code, which gives about 30% increase in
  performance in 3D.
* Remove the need for the ``windows_env.bat`` script on Windows.  This is
  automatically setup internally.
* Add test that checks if all examples run.
* Remove unused command line options and add a ``--max-steps`` option to allow
  a user to run a specified number of iterations.
* Added Ghia et al.'s results for lid-driven-cavity flow for easy comparison.
* Added some experimental results for the dam break problem.
* Use argparse instead of optparse as it is deprecated in Python 3.x.
* Add ``pysph.tools.automation`` to facilitate easier automation and
  reproducibility of PySPH simulations.
* Add spatial hash and extended spatial hash NNPS algorithms for comparison.
* Refactor and cleanup the NNPS related code.
* Add several gas-dynamics examples and the ``ADEKEScheme``.
* Work with mpi4py_ version 2.0.0 and older versions.
* Fixed major bug with TVF implementation and add support for 3D simulations
  with the TVF.
* Fix bug with uploaded tarballs that breaks ``pip install pysph`` on Windows.
* Fix the viewer UI to continue playing files when refresh is pushed.
* Fix bugs with the timestep values dumped in the outputs.
* Fix floating point issues with timesteps, where examples would run a final
  extremely tiny timestep in order to exactly hit the final time.

.. _h5py: http://www.h5py.org
.. _PyVisfile: http://github.com/inducer/pyvisfile
.. _Mayavi: http://code.enthought.com/projects/mayavi/

1.0a3
------

* Release date: 18th August, 2015.
* Fix bug with ``output_at_times`` specification for solver.
* Put generated sources and extensions into a platform specific directory in
  ``~/.pysph/sources/<platform-specific-dir>`` to avoid problems with multiple
  Python versions, operating systems etc.
* Use locking while creating extension modules to prevent problems when
  multiple processes generate the same extesion.
* Improve the ``Application`` class so users can subclass it to create
  examples. The users can also add their own command line arguments and add
  pre/post step/stage callbacks by creating appropriate methods.
* Moved examples into the ``pysph.examples``.  This makes the examples
  reusable and easier to run as installation of pysph will also make the
  examples available.  The examples also perform the post-processing to make
  them completely self-contained.
* Add support to write compressed output.
* Add support to set the kernel from the command line.
* Add a new ``pysph`` script that supports ``view``, ``run``, and ``test``
  sub-commands.  The ``pysph_viewer`` is now removed, use ``pysph view``
  instead.
* Add a simple remeshing tool in ``pysph.solver.tools.SimpleRemesher``.
* Cleanup the symmetric eigenvalue computing routines used for solid
  mechanics problems and allow them to be used with OpenMP.
* The viewer can now view the velocity magnitude (``vmag``) even if it
  is not present in the data.
* Port all examples to use new ``Application`` API.
* Do not display unnecessary compiler warnings when there are no errors but
  display verbose details when there is an error.

1.0a2
------

* Release date: 12th June, 2015
* Support for tox_, this makes it trivial to test PySPH on py26, py27 and py34
  (and potentially more if needed).
* Fix bug in code generator where it is unable to import pysph before it is
  installed.
* Support installation via ``pip`` by allowing ``egg_info`` to be run without
  cython or numpy.
* Added `Codeship CI build <https://codeship.com/projects/83729>`_ using tox
  for py27 and py34.
* CI builds for Python 2.7.x and 3.4.x.
* Support for Python-3.4.x.
* Support for Python-2.6.x.

.. _tox: https://pypi.python.org/pypi/tox

1.0a1
------

* Release date: 3rd June, 2015.
* First public release of the new PySPH code which uses code-generation and is
  hosted on `bitbucket <http://bitbucket.org/pysph/pysph>`_.
* OpenMP support.
* MPI support using `Zoltan <http://www.cs.sandia.gov/zoltan/>`_.
* Automatic code generation from high-level Python code.
* Support for various multi-step integrators.
* Added an interpolator utility module that interpolates the particle data
  onto a desired set of points (or grids).
* Support for inlets and outlets.
* Support for basic `Gmsh <http://geuz.org/gmsh/>`_ input/output.
* Plenty of examples for various SPH formulations.
* Improved documentation.
* Continuous integration builds on `Shippable
  <https://app.shippable.com/projects/540e849c3479c5ea8f9f030e/builds/latest>`_,
  `Drone.io <https://drone.io/bitbucket.org/pysph/pysph>`_, and `AppVeyor
  <https://ci.appveyor.com/project/prabhuramachandran/pysph>`_.
