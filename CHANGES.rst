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

