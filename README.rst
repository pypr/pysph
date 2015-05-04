PySPH: a Python-based SPH framework
-----------------------------------

|Shippable Status| |Drone Status|

PySPH is an open source framework for Smoothed Particle Hydrodynamics
(SPH) simulations. It is implemented in
`Python <http://www.python.org>`__ and the performance critical parts
are implemented in `Cython <http://www.cython.org>`__.

PySPH allows users to write their high-level code in pure Python.  This Python
code is automatically converted to high-performance Cython which is compiled
and executed.  PySPH can also work with OpenMP and MPI for larger scale
computing.

The latest documentation for PySPH is available at
`pysph.readthedocs.org <http://pysph.readthedocs.org>`__.

.. |Shippable Status| image:: https://api.shippable.com/projects/540e849c3479c5ea8f9f030e/badge?branchName=master
   :target: https://app.shippable.com/projects/540e849c3479c5ea8f9f030e/builds/latest
.. |Drone Status| image:: https://drone.io/bitbucket.org/pysph/pysph/status.png
   :target: https://drone.io/bitbucket.org/pysph/pysph/latest

Features
--------

-  Solver framework to add arbitrary collection of particles.
-  Flexibility to define arbitrary SPH equations operating on particles
   in pure Python.
-  Any kind of user-defined multi-step integrator.
-  Seamless parallel integration using
   `Zoltan <http://www.cs.sandia.gov/zoltan/>`__.
-  High-performance: our performance is comparable to hand-written
   solvers implemented in FORTRAN.

Solvers
-------

Currently, PySPH has numerous examples to solve a variety of problems.  The
features of the implementation are:

-  `Weakly Compressible SPH
   (WCSPH) <http://www.tandfonline.com/doi/abs/10.1080/00221686.2010.9641250>`__
   for free-surface flows (Gesteira et al. 2010, Journal of Hydraulic
   Research, 48, pp. 6--27)
-  `Transport Velocity
   Formulation <http://dx.doi.org/10.1016/j.jcp.2013.01.043>`__ for
   incompressilbe fluids (Adami et al. 2013, JCP, 241, pp. 292--307)
-  `SPH for elastic
   dynamics <http://dx.doi.org/10.1016/S0045-7825(01)00254-7>`__ (Gray
   et al. 2001, CMAME, Vol. 190, pp 6641--6662)
-  `Compressible SPH <http://dx.doi.org/10.1016/j.jcp.2013.08.060>`__
   (Puri et al. 2014, JCP, Vol. 256, pp 308--333)

Installation
============

Up-to-date details on how to install PySPH on Linux/OS X and Windows are
available from
`here <http://pysph.readthedocs.org/en/latest/installation.html>`__.

If you wish to see a working build/test script please see our
`shippable.yml <https://bitbucket.org/pysph/pysph/src/master/shippable.yml>`__.
Or you could see the `build
script <https://drone.io/bitbucket.org/pysph/pysph/admin>`__ hosted at
`drone.io <http://drone.io>`__.

Running the examples
--------------------

You can verify the installation by exploring some examples. A fairly
quick running example (taking about 5-10 minutes) would be the
following::

    $ cd examples
    $ python dam_break.py

The solution can be viewed live by running::

    $ pysph_viewer

This requires that Mayavi be installed. The saved output data can be
viewed by running::

    $ pysph_viewer dam_break_output/

A 3D version of the dam-break problem is also available, and may be run
as::

    $ python dam_break3D.py

This runs the 3D dam-break problem which is also a SPHERIC benchmark
`Test 2 <https://wiki.manchester.ac.uk/spheric/index.php/Test2>`__

.. figure:: https://bitbucket.org/pysph/pysph/raw/master/docs/Images/db3d.png
   :alt: IMAGE

PySPH is more than a tool for wave-body interactions:::

    $ cd examples/transport_velocity
    $ python cavity.py

This runs the driven cavity problem using the transport velocity
formulation of Adami et al. You can verify the results for this problem
using the helper script
``examples/transport_velocity/ldcavity_results.py`` to plot, for example
the streamlines:

.. figure:: https://bitbucket.org/pysph/pysph/raw/master/docs/Images/ldc-streamlines.png
   :alt: IMAGE

If you want to use PySPH for elastic dynamics, you can try some of the
examples from the directory ``examples/solid_mech``::

    $ cd examples/solid_mech
    $ python rings.py

Which runs the problem of the collision of two elastic rings:

.. figure:: https://bitbucket.org/pysph/pysph/raw/master/docs/Images/rings-collision.png
   :alt: IMAGE

The auto-generated code for the example resides in the directory
``~/.pysph/source``. A note of caution however, it's not for the faint
hearted.

