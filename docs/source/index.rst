.. PySPH documentation master file, created by
   sphinx-quickstart on Mon Mar 31 01:01:41 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the PySPH documentation!
====================================

PySPH is an open source framework for Smoothed Particle Hydrodynamics (SPH)
simulations.  Users can implement an SPH formulation in pure Python_ and still
obtain excellent performance.  PySPH can make use of multiple cores via OpenMP
or be run seamlessly in parallel using MPI.

Here are some videos of simulations made with PySPH.

.. raw:: html

    <div align="center">
    <iframe width="560" height="315"
        src="https://www.youtube.com/embed/videoseries?list=PLH8Y2KepC2_VPLrcTiWGaYYh88gGVAuVr"
        frameborder="0" allowfullscreen>
    </iframe>
    </div>


PySPH is hosted on `github <http://github.com/pypr/pysph>`_.  Please see
the `github <http://github.com/pypr/pysph>`_ site for development
details.

.. _Python: http://www.python.org


**********
Overview
**********

.. toctree::
   :maxdepth: 2

   overview.rst

*********************************
Installation and getting started
*********************************

.. toctree::
   :maxdepth: 2

   installation.rst
   tutorial/circular_patch_simple.rst
   tutorial/circular_patch.rst


***************************
The framework and library
***************************

.. toctree::
   :maxdepth: 2

   design/overview.rst
   design/equations.rst
   design/iom.rst
   starcluster/overview
   using_pysph.rst
   contribution/how_to_write_docs.rst

**************************
Gallery of PySPH examples
**************************

.. toctree::
   :maxdepth: 2

   examples/index.rst

************************
Reference documentation
************************

Autogenerated from doc strings using sphinx's autodoc feature.

.. toctree::
   :maxdepth: 2

   reference/index
   design/solver_interfaces


==================
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
