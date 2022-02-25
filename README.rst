PySPH: a Python-based SPH framework
------------------------------------

|CI Status|  |MPI Build status|  |Documentation Status|


PySPH is an open source framework for Smoothed Particle Hydrodynamics
(SPH) simulations. It is implemented in
`Python <http://www.python.org>`_ and the performance critical parts
are implemented in `Cython <http://www.cython.org>`_ and PyOpenCL_.

PySPH allows users to write their high-level code in pure Python. This Python
code is automatically converted to high-performance Cython or OpenCL which is
compiled and executed. PySPH can also be configured to work seamlessly with
OpenMP, OpenCL, and MPI.

The latest documentation for PySPH is available at
`pysph.readthedocs.org <http://pysph.readthedocs.org>`_.

.. |CI Status| image:: https://github.com/pypr/pysph/actions/workflows/tests.yml/badge.svg
    :target: https://github.com/pypr/pysph/actions/workflows/tests.yml
.. |MPI Build Status| image:: https://github.com/pypr/pysph/actions/workflows/zoltan-tests.yml/badge.svg
    :target: https://github.com/pypr/pysph/actions/workflows/zoltan-tests.yml
.. |Documentation Status| image:: https://readthedocs.org/projects/pysph/badge/?version=latest
    :target: https://pysph.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status


Here are `videos
<https://www.youtube.com/playlist?list=PLH8Y2KepC2_VPLrcTiWGaYYh88gGVAuVr>`_
of some example problems solved using PySPH.


.. _PyOpenCL: https://documen.tician.de/pyopencl/
.. _PyZoltan: https://github.com/pypr/pyzoltan

Features
--------

- Flexibility to define arbitrary SPH equations operating on particles
  in pure Python.
- Define your own multi-step integrators in pure Python.
- High-performance: our performance is comparable to hand-written
  solvers implemented in FORTRAN.
- Seamless multi-core support with OpenMP.
- Seamless GPU support with PyOpenCL_.
- Seamless parallel support using
  `Zoltan <http://www.cs.sandia.gov/zoltan/>`_ and PyZoltan_.

SPH formulations
-----------------

PySPH ships with a variety of standard SPH formulations along with
basic examples.  Some of the formulations available are:

-  `Weakly Compressible SPH
   (WCSPH) <http://www.tandfonline.com/doi/abs/10.1080/00221686.2010.9641250>`_
   for free-surface flows (Gesteira et al. 2010, Journal of Hydraulic
   Research, 48, pp. 6--27)
-  `Transport Velocity
   Formulation <http://dx.doi.org/10.1016/j.jcp.2013.01.043>`_ for
   incompressilbe fluids (Adami et al. 2013, JCP, 241, pp. 292--307)
-  `SPH for elastic
   dynamics <http://dx.doi.org/10.1016/S0045-7825(01)00254-7>`_ (Gray
   et al. 2001, CMAME, Vol. 190, pp 6641--6662)
-  `Compressible SPH <http://dx.doi.org/10.1016/j.jcp.2013.08.060>`_
   (Puri et al. 2014, JCP, Vol. 256, pp 308--333)
-  `Generalized Transport Velocity Formulation (GTVF)
   <https://doi.org/10.1016/j.jcp.2017.02.016>`_ (Zhang et al. 2017, JCP, 337,
   pp. 216--232)
-  `Entropically Damped Artificial Compressibility (EDAC)
   <http://dx.doi.org/10.1016/j.compfluid.2018.11.023>`_ (Ramachandran et
   al. 2019, Computers and Fluids, 179, pp. 579--594)
-  `delta-SPH <http://dx.doi.org/10.1016/j.cma.2010.12.016>`_ (Marrone et
   al. CMAME, 2011, 200, pp. 1526--1542)
-  `Dual Time SPH (DTSPH) <https://arxiv.org/abs/1904.00861>`_ (Ramachandran et
   al. arXiv preprint)
-  `Incompressible (ISPH) <https://doi.org/10.1006/jcph.1999.6246>`_ (Cummins et
   al. JCP, 1999, 152, pp. 584--607)
-  `Simple Iterative SPH (SISPH) <https://arxiv.org/abs/1908.01762>`_ (Muta et
   al. arXiv preprint)
-  `Implicit Incompressibel SPH (IISPH)
   <https://doi.org/10.1109/TVCG.2013.105>`_ (Ihmsen et al. 2014, IEEE
   Trans. Vis. Comput. Graph., 20, pp 426--435)
-  `Gudnov SPH (GSPH) <https://doi.org/10.1006/jcph.2002.7053>`_ (Inutsuka et
   al. JCP, 2002, 179, pp. 238--267)
-  `Conservative Reproducible Kernel SPH (CRKSPH)
   <http://dx.doi.org/10.1016/j.jcp.2016.12.004>`_ (Frontiere et al. JCP, 2017,
   332, pp. 160--209)
-  `Approximate Gudnov SPH (AGSPH) <https://doi.org/10.1016/j.jcp.2014.03.055>`_
   (Puri et al. JCP, 2014, pp. 432--458)
-  `Adaptive Density Kernel Estimate (ADKE)
   <https://doi.org/10.1016/j.jcp.2005.06.016>`_ (Sigalotti et al. JCP, 2006,
   pp. 124--149)
-  `Akinci <http://doi.acm.org/10.1145/2185520.2185558>`_ (Akinci et al. ACM
   Trans. Graph., 2012, pp. 62:1--62:8)

Boundary conditions from the following papers are implemented:

-  `Generalized Wall BCs
   <http://dx.doi.org/10.1016/j.jcp.2012.05.005>`_ (Adami et al. JCP,
   2012, pp. 7057--7075)
-  `Do nothing type outlet BC
   <https://doi.org/10.1016/j.euromechflu.2012.02.002>`_ (Federico
   et al. European Journal of Mechanics - B/Fluids, 2012, pp. 35--46)
-  `Outlet Mirror BC
   <http://dx.doi.org/10.1016/j.cma.2018.08.004>`_ (Tafuni et al. CMAME,
   2018, pp. 604--624)
-  `Method of Characteristics BC
   <http://dx.doi.org/10.1002/fld.1971>`_ (Lastiwaka
   et al. International Journal for Numerical Methods in Fluids, 2012,
   pp. 35--46)
-  `Hybrid  BC <https://arxiv.org/abs/1907.04034>`_ (Negi et
   al. arXiv preprint)

Corrections proposed in the following papers are also the part for PySPH:

-  `Corrected SPH <http://dx.doi.org/10.1016/S0045-7825(99)00051-1>`_ (Bonet et
   al. CMAME, 1999, pp. 97--115)
-  `hg-correction <https://doi.org/10.1080/00221686.2010.9641251>`_ (Hughes et
   al. Journal of Hydraulic Research, pp. 105--117)
-  `Tensile instability correction' <https://doi.org/10.1006/jcph.2000.6439>`_
   (Monaghan J. J. JCP, 2000, pp. 2990--311)
-  Particle shift algorithms
   (`Xu et al <http://dx.doi.org/10.1016/j.jcp.2009.05.032>`_. JCP, 2009, pp. 6703--6725),
   (`Skillen et al <http://dx.doi.org/10.1016/j.cma.2013.05.017>`_. CMAME, 2013, pp. 163--173)

Surface tension models are implemented from:

-  `Morris surface tension`_ (Morris et al. Internaltional Journal for Numerical
   Methods in Fluids, 2000, pp. 333--353)
-  `Adami Surface tension formulation
   <https://doi.org/10.1016/j.jcp.2010.03.022>`_ (Adami et al. JCP, 2010,
   pp. 5011--5021)

.. _Morris surface tension:
   https://dx.doi.org/10.1002/1097-0363(20000615)33:3<333::AID-FLD11>3.0.CO;2-7

Installation
-------------

Up-to-date details on how to install PySPH on Linux/OS X and Windows are
available from
`here <http://pysph.readthedocs.org/en/latest/installation.html>`_.

If you wish to see a working build/test script please see our `shippable.yml
<https://github.com/pypr/pysph/blob/master/shippable.yml>`_. For
Windows platforms see the `appveyor.yml
<https://github.com/pypr/pysph/blob/master/appveyor.yml>`_.

Running the examples
--------------------

You can verify the installation by exploring some examples. A fairly
quick running example (taking about 20 seconds) would be the
following::

    $ pysph run elliptical_drop

This requires that Mayavi be installed. The saved output data can be
viewed by running::

    $ pysph view elliptical_drop_output/

A more interesting example would be a 2D dam-break example (this takes about 30
minutes in total to run)::

    $ pysph run dam_break_2d

The solution can be viewed live by running (on another shell)::

    $ pysph view

The generated output can also be viewed and the newly generated output files
can be refreshed on the viewer UI.

A 3D version of the dam-break problem is also available, and may be run
as::

    $ pysph run dam_break_3d

This runs the 3D dam-break problem which is also a SPHERIC benchmark
`Test 2 <https://wiki.manchester.ac.uk/spheric/index.php/Test2>`_

.. figure:: https://github.com/pypr/pysph/raw/master/docs/Images/db3d.png
   :width: 550px
   :alt: Three-dimensional dam-break example

PySPH is more than a tool for wave-body interactions:::

    $ pysph run cavity

This runs the driven cavity problem using the transport velocity formulation of
Adami et al. The output directory ``cavity_output`` will also contain
streamlines and other post-processed results after the simulation completes.
For example the streamlines look like the following image:

.. figure:: https://github.com/pypr/pysph/raw/master/docs/Images/ldc-streamlines.png
   :width: 550px
   :alt: Lid-driven-cavity example

If you want to use PySPH for elastic dynamics, you can try some of the
examples from the ``pysph.examples.solid_mech`` package::

    $ pysph run solid_mech.rings

Which runs the problem of the collision of two elastic rings:

.. figure:: https://github.com/pypr/pysph/raw/master/docs/Images/rings-collision.png
   :width: 550px
   :alt: Collision of two steel rings

The auto-generated code for the example resides in the directory
``~/.pysph/source``. A note of caution however, it's not for the faint
hearted.

There are many more examples, they can be listed by simply running::

    $ pysph run


Research papers using PySPH
----------------------------

The following are some of the works that use PySPH,

- Adaptive SPH method: https://gitlab.com/pypr/adaptive_sph
- Adaptive SPH method applied to moving bodies: https://gitlab.com/pypr/asph_motion
- Convergence of the SPH method: https://gitlab.com/pypr/convergence_sph
- Corrected transport velocity formulation: https://gitlab.com/pypr/ctvf
- Dual-Time SPH method: https://gitlab.com/pypr/dtsph
- Entropically damped artificial compressibility SPH formulation: https://gitlab.com/pypr/edac_sph
- Generalized inlet and outlet boundary conditions for SPH: https://gitlab.com/pypr/inlet_outlet
- Method of manufactured solutions for SPH: https://gitlab.com/pypr/mms_sph
- A demonstration of the binder support provided by PySPH: https://gitlab.com/pypr/pysph_demo
- Manuscript and code for a paper on PySPH: https://gitlab.com/pypr/pysph_paper
- Simple Iterative Incompressible SPH scheme: https://gitlab.com/pypr/sisph
- Geometry generation and preprocessing for SPH simulations: https://gitlab.com/pypr/sph_geom


Credits
--------

PySPH is primarily developed at the `Department of Aerospace
Engineering, IIT Bombay <http://www.aero.iitb.ac.in>`_. We are grateful
to IIT Bombay for their support.  Our primary goal is to build a
powerful SPH based tool for both application and research. We hope that
this makes it easy to perform reproducible computational research.

To see the list of contributors the see `github contributors page
<https://github.com/pypr/pysph/graphs/contributors>`_


Some earlier developers not listed on the above are:

- Pankaj Pandey (stress solver and improved load balancing, 2011)
- Chandrashekhar Kaushik (original parallel and serial implementation in 2009)


Citing PySPH
-------------

You may use the following article to formally refer to PySPH,
a freely-available arXiv copy of the below paper is at
https://arxiv.org/abs/1909.04504,

 - Prabhu Ramachandran, Aditya Bhosale, Kunal Puri, Pawan Negi, Abhinav
   Muta, A. Dinesh, Dileep Menon, Rahul Govind, Suraj Sanka, Amal
   S Sebastian, Ananyo Sen, Rohan Kaushik, Anshuman Kumar,  Vikas
   Kurapati, Mrinalgouda Patil, Deep Tavker, Pankaj Pandey,
   Chandrashekhar Kaushik, Arkopal Dutt, Arpit Agarwal. "PySPH:
   A Python-Based Framework for Smoothed Particle Hydrodynamics". ACM
   Transactions on Mathematical Software 47, no. 4 (31 December 2021):
   1--38. DOI: https://doi.org/10.1145/3460773.

The bibtex entry is:::

    @article{ramachandran2021a,
        title = {{{PySPH}}: {{A Python-based Framework}} for {{Smoothed Particle Hydrodynamics}}},
        shorttitle = {{{PySPH}}},
        author = {Ramachandran, Prabhu and Bhosale, Aditya and Puri,
        Kunal and Negi, Pawan and Muta, Abhinav and Dinesh,
        A. and Menon, Dileep and Govind, Rahul and Sanka, Suraj and Sebastian,
        Amal S. and Sen, Ananyo and Kaushik, Rohan and Kumar,
        Anshuman and Kurapati, Vikas and Patil, Mrinalgouda and Tavker,
        Deep and Pandey, Pankaj and Kaushik, Chandrashekhar and Dutt,
        Arkopal and Agarwal, Arpit},
        year = {2021},
        month = dec,
        journal = {ACM Transactions on Mathematical Software},
        volume = {47},
        number = {4},
        pages = {1--38},
        issn = {0098-3500, 1557-7295},
        doi = {10.1145/3460773},
        langid = {english}
    }


Support
-------

If you have any questions or are running into any difficulties with PySPH, you
can use the `PySPH discussions <https://github.com/pypr/pysph/discussions>`_.

Please also take a look at the `PySPH issue tracker
<https://github.com/pypr/pysph/issues>`_ if you have bugs or issues to report.

You could also email or post your questions on the pysph-users mailing list here:
https://groups.google.com/d/forum/pysph-users
