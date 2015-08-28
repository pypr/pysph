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
