.. _installation:

=================================
Installation and getting started
=================================

To install PySPH, you need a working Python environment. We recommend
`Enthought Canopy`_ if you are new to Python or any other Python distribution
of your choice.

------------------
Dependencies
------------------

^^^^^^^^^^^^^^^^^^
Core dependencies
^^^^^^^^^^^^^^^^^^

The core dependencies are:

  - NumPy_
  - Cython_ (ideally version 0.19 and above)
  - Mako_
  - nose_ for running the unit tests.

Cython_ and Mako_ can be installed from the command line using::

    $ easy_install Cython mako

or using pip_::

    $ pip install Cython mako

.. _NumPy: http://numpy.scipy.org
.. _Enthought Canopy: https://www.enthought.com/products/canopy/
.. _Cython: http://www.cython.org
.. _nose: https://pypi.python.org/pypi/nose
.. _Mako: https://pypi.python.org/pypi/Mako
.. _pip: http://www.pip-installer.org

^^^^^^^^^^^^^^^^^^^^^^
Optional dependencies
^^^^^^^^^^^^^^^^^^^^^^

PySPH provides a convenient viewer to view the output of results.  This viewer
is called ``pysph_viewer`` and requires Mayavi_ to be installed.  Since this
is only a viewer it is optional for use, however, it is highly recommended
that you have Mayavi installed.

If you want to use PySPH in parallel, you will need mpi4py_ and the Zoltan_
data management library.

.. _Mayavi: http://code.enthought.com/projects/mayavi
.. _mpi4py: http://mpi4py.scipy.org/
.. _Zoltan: http://www.cs.sandia.gov/zoltan/

-------------------------------
Building and linking PyZoltan
-------------------------------

We've provided a simple Zoltan build script in the repository.  This works on
Linux and OS X but not on Windows.  It can be used as so::

    $ ./build_zoltan.sh  INSTALL_PREFIX

where the ``INSTALL_PREFIX`` is where the library and includes will be
installed.  You may edit and tweak the build to suit your installation.
However, this script  what we use to build Zoltan on our continuous
integration servers on Drone_ and Shippable_.

Declare the environment variables ``ZOLTAN_INCLUDE`` and ``ZOLTAN_LIBRARY``.
If you used the above script, this would be::

    $ export ZOLTAN_INCLUDE=$INSTALL_PREFIX/include
    $ export ZOLTAN_LIBRARY=$INSTALL_PREFIX/lib

Install PySPH. The PyZoltan wrappers will be compiled and available.

If you wish to see a working build/test script please see our
`shippable.yml <https://bitbucket.org/pysph/pysph/src/master/shippable.yml>`_.
Or you could see the `build script <https://drone.io/bitbucket.org/pysph/pysph/admin>`_
hosted at `Drone.io <http://drone.io>`_.


.. _Drone: http://drone.io
.. _Shippable: http://shippable.com

-------------------------------------------------
Building and Installing PySPH on Linux and MacOS
-------------------------------------------------

^^^^^^^^^^^^^^
Getting PySPH
^^^^^^^^^^^^^^

The best way to currently get PySPH is via git_ ::

    $ git clone https://bitbucket.org/pysph/pysph.git

If you do not have git or do not wish to bother with it (a bad idea), you can
get a ZIP or tarball from the `pysph site
<https://bitbucket.org/pysph/pysph>`_. You can unzip/untar this and use the
sources.

.. _git: http://git-scm.com/

Once you have the dependencies installed you can install PySPH with::

    $ python setup.py install

You could also do::

    $ python setup.py develop

This is useful if you are tracking the latest version of PySPH via git.

^^^^^^^^^^^^^^^^^^^
Running the tests
^^^^^^^^^^^^^^^^^^^

To test PySPH from the source distribution you can do::

   $ python -m nose.core pysph

This should run all the tests that do not take a long while to complete. This
will not run some of the parallel tests that take a long while.  To run all
the tests you can run::

   $ python -m nose.core -A slow pysph


.. note::

    We use ``python -m nose.core`` instead of ``nosetests`` as this ensures
    that the right Python executable is used.  ``nostests`` is sometimes
    installed in the system in ``/usr/bin/nosetests`` and running that would
    pick up the system Python instead of the one in the virtualenv.  This
    results in incorrect test errors leading to confusion.


---------------------------------------------------------
Building and Installing PySPH on Windows using WinPython
---------------------------------------------------------

^^^^^^^^^^^^^^^^^^^^^^^^^^^
Getting Core Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^

To install PySPH you require to have python installed on your system. We
suggest you download WinPython_ 2.7.x.x. To obtain the core dependencies,
download the corresponding binaries from Christoph Gohlke's `Unofficial
Windows Binaries for Python Extension Packages
<http://www.lfd.uci.edu/~gohlke/pythonlibs/>`_. Mayavi is available through
the binary ETS.

You can now add these binaries to your WinPython installation by going to
WinPython Control Panel. The option to add packages is available under the
section Install/upgrade packages.

.. _WinPython: http://winpython.sourceforge.net/

^^^^^^^^^^^^^^
Getting PySPH
^^^^^^^^^^^^^^

Now, you can get a ZIP or tarball from the `pysph site
<https://bitbucket.org/pysph/pysph>`_. You can unzip/untar this and use the
sources. Make sure to set your system PATH variable pointing to the location
of the  scripts as required. If you have installed WinPython 2.7.6 64-bit,
make sure to set your system PATH variables to ``<path to installation
folder>/python-2.7.6.amd64`` and ``<path to installation
folder>/python-2.7.6.amd64/Scripts/``.

Open Command Prompt and change your directory to where PySPH is located.
Moving into the directory, you will see a file named setup.py To install
PySPH, one simply needs to::

    $ python setup.py install

This should install the package PySPH. A common error message you may
encounter is "unable to find vcvarsall.bat". Please follow this post_ to sort
out your problem. If you don't have any sort of C++ compiler, we recommend you
to download `VS2010 Express Edition
<http://www.visualstudio.com/en-us/downloads#d-2010-express>`_. To test your
PySPH installation, you can do the tests as given above.

.. _post: http://stackoverflow.com/questions/2817869/error-unable-to-find-vcvarsall-bat

-------------------------------
Using a virtualenv for PySPH
-------------------------------

A virtualenv_ allows you to create an isolated environment for PySPH and its
related packages.  This is useful in a variety of situations.

    - Your OS does not provide a recent enough Cython_ version (say you are
      running Debian stable).
    - You do not have root access to install any packages PySPH requires.
    - You do not want to mess up your system.
    - You wish to use other packages with conflicting requirements.
    - You want PySPH and its related packages to be in an "isolated" environment.

You can either install virtualenv_ (or ask your system administrator to) or
just download the `virtualenv.py
<http://github.com/pypa/virtualenv/tree/master/virtualenv.py>`_ script and use
it.

.. _virtualenv: http://www.virtualenv.org

Create a virtualenv like so::

    $ virtualenv pysph_env

This creates a directory called ``pysph_env`` which contains all the relevant
files for your virtualenv, this includes any new packages you wish to install
into it.  You can delete this directory if you don't want it anymore for some
reason.  If you want this virtualenv to also "inherit" packages from your
system you can create the virtualenv like so::

    $ virtualenv --system-site-packages pysph_env

Once you create a virtualenv you can activate it as follows (on a bash shell)::

    $ source pysph_env/bin/activate

On Windows you run a bat file as follows::

    $ pysph_env/bin/activate

This sets up the PATH to point to your virtualenv's Python.  You may now run
any normal Python commands and it will use your virtualenv's Python.  For
example you can do the following::

    $ virtualenv myenv
    $ source myenv/bin/activate
    (myenv) $ pip install Cython mako nose
    (myenv) $ cd pysph
    (myenv) $ python setup.py install

Now PySPH will be installed into ``myenv``.  You may deactivate your
virtualenv using the ``deactivate`` command::

    (myenv) $ deactivate
    $

On Windows, use ``myenv\Scripts\activate.bat`` and
``myenv\Scripts\deactivate.bat``.

If for whatever reason you wish to delete ``myenv`` just remove the entire
directory::

    $ rm -rf myenv

.. note::

    With a virtualenv, one should be careful while running things like
    ``ipython`` or ``nosetests`` as these are sometimes also installed on the
    system in ``/usr/bin``.  If you suspect that you are not running the
    correct Python, you could simply run (on *nix/OS X)::

        $ python `which ipython`

    to be absolutely sure.


^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Using Virtualenv on Canopy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you are using `Enthought Canopy`_, it already bundles virtualenv for you but
you should use the ``venv`` script.  For example::

    $ venv --help
    $ venv --system-site-packages myenv
    $ source myenv/bin/activate

The rest of the steps are the same as above.


---------------------
Running the examples
---------------------

You can verify the installation by exploring some examples::

    $ cd examples
    $ python elliptical_drop.py

Try this::

    $ python elliptical_drop.py -h

to see the different options supported by each example.  You can view the data
generated by the simulation (after the simulation is complete or during the
simulation) by running the ``pysph_viewer`` application.  To view the
simulated data you may do::

    $ pysph_viewer elliptical_drop_output/*.npz

If you have Mayavi_ installed this should show a UI that looks like:

.. image:: ../Images/pysph_viewer.png
    :width: 800px
    :alt: PySPH viewer

There are other examples like those in the ``transport_velocity`` directory::

    $ cd transport_velocity
    $ python cavity.py

This runs the driven cavity problem using the transport velocity formulation
of Adami et al. You can verify the results for this problem using the helper
script ``examples/transport_velocity/ldcavity_results.py`` to plot, for example
the streamlines:

.. image:: ../Images/ldc-streamlines.png

If you want to use PySPH for elastic dynamics, you can try some of the
examples from Gray et al., Comput. Methods Appl. Mech. Engrg. 190
(2001), 6641-6662::

    $ cd examples/solid_mech
    $ python rings.py

Which runs the problem of the collision of two elastic rings:

.. image:: ../Images/rings-collision.png

The auto-generated code for the example resides in the directory
``~/.pysph/source``. A note of caution however, it's not for the faint hearted.

--------------------------------------
Organization of the ``pysph`` package
--------------------------------------

PySPH is organized into several sub-packages.  These are:

  - ``pysph.base``:  This subpackage defines the
    :py:class:`pysph.base.particle_array.ParticleArray`,
    :py:class:`pysph.base.carray.CArray` (which are used by the particle
    arrays), the various :doc:`reference/kernels`, the nearest neighbor
    particle search (NNPS) code, and the Cython code generation utilities.

  - ``pysph.sph``: Contains the various :doc:`reference/equations`, the
    :doc:`reference/integrator` and associated integration steppers, and the
    code generation for the SPH looping. ``pysph.sph.wc`` contains the
    equations for the weakly compressible formulation.
    ``pysph.sph.solid_mech`` contains the equations for solid mechanics and
    ``pysph.sph.misc`` has miscellaneous equations.

  - ``pysph.solver``: Provides the :py:class:`pysph.solver.solver.Solver`, the
    :py:class:`pysph.solver.application.Application` and a convenient way to
    interact with the solver as it is running.

  - ``pysph.parallel``: Provides the parallel functionality.

  - ``pysph.tools``: Provides some useful tools including the ``pysph_viewer``
    which is based on Mayavi_.
