===========
Overview
===========


PySPH is an open source framework for Smoothed Particle Hydrodynamics (SPH)
simulations.  It is implemented in Python_ and the performance critical parts
are implemented in Cython_ and PyOpenCL_.

PySPH is implemented in a way that allows a user to specify the entire SPH
simulation in pure Python. High-performance code is generated from this
high-level Python code, compiled on the fly and executed. PySPH can use OpenMP
to utilize multi-core CPUs effectively. PySPH can work with OpenCL and use
your GPGPUs. PySPH also features optional automatic parallelization
(multi-CPU) using mpi4py_ and Zoltan_. If you wish to use the parallel
capabilities you will need to have these installed.

Here are videos of simulations made with PySPH.

.. raw:: html

    <div align="center">
    <iframe width="560" height="315"
        src="https://www.youtube.com/embed/videoseries?list=PLH8Y2KepC2_VPLrcTiWGaYYh88gGVAuVr"
        frameborder="0" allowfullscreen>
    </iframe>
    </div>


PySPH is hosted on `github <https://github.com/pypr/pysph>`_. Please see the
site for development details.

.. _Python: http://www.python.org
.. _Cython: http://www.cython.org
.. _PyOpenCL: https://documen.tician.de/pyopencl/
.. _mpi4py: http://mpi4py.scipy.org
.. _Zoltan: http://www.cs.sandia.gov/zoltan/


---------
Features
---------

  - User scripts and equations are written in pure Python.
  - Flexibility to define arbitrary SPH equations operating on particles.
  - Ability to define your own multi-step integrators in pure Python.
  - High-performance: our performance is comparable to hand-written solvers
    implemented in FORTRAN.
  - Seamless multi-core support with OpenMP.
  - Seamless GPU support with PyOpenCL_.
  - Seamless parallel integration using Zoltan_.
  - `BSD license <https://github.com/pypr/pysph/tree/master/LICENSE.txt>`_.

-----------------
SPH formulations
-----------------

Currently, PySPH has numerous examples to solve the viscous, incompressible
Navier-Stokes equations using the weakly compressible (WCSPH) approach. The
following formulations are currently implemented:

- `Weakly Compressible SPH (WCSPH)`_ for free-surface flows (Gesteira et al. 2010, Journal of Hydraulic Research, 48, pp. 6--27)

.. figure:: ../Images/db3d.png
   :width: 500 px
   :align: center

   3D dam-break past an obstacle SPHERIC benchmark `Test 2`_

- `Transport Velocity Formulation`_ for incompressilbe fluids (Adami et al. 2013, JCP, 241, pp. 292--307).

.. figure:: ../Images/ldc-streamlines.png
   :width: 500 px
   :align: center

   Streamlines for a driven cavity

- `SPH for elastic dynamics`_ (Gray et al. 2001, CMAME, Vol. 190, pp 6641--6662)

.. figure:: ../Images/rings-collision.png
   :width: 500 px
   :align: center

   Collision of two elastic rings.


- `Compressible SPH`_ (Puri et al. 2014, JCP, Vol. 256, pp 308--333)

.. _`Weakly Compressible SPH (WCSPH)`: http://www.tandfonline.com/doi/abs/10.1080/00221686.2010.9641250

.. _`Transport Velocity Formulation`: http://dx.doi.org/10.1016/j.jcp.2013.01.043

.. _`SPH for elastic dynamics`: http://dx.doi.org/10.1016/S0045-7825(01)00254-7

.. _`Compressible SPH`: http://dx.doi.org/10.1016/j.jcp.2013.08.060

.. _`Test 2`: https://wiki.manchester.ac.uk/spheric/index.php/Test2


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
-  `Hybrid  BC <arXiv preprint arXiv:https://arxiv.org/abs/1907.04034>`_ (Negi et
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


--------
Credits
--------

PySPH is primarily developed at the `Department of Aerospace
Engineering, IIT Bombay <http://www.aero.iitb.ac.in>`__. We are grateful
to IIT Bombay for the support.  Our primary goal is to build a
powerful SPH-based tool for both application and research. We hope that
this makes it easy to perform reproducible computational research.

To see the list of contributors the see `github contributors page
<https://github.com/pypr/pysph/graphs/contributors>`_


Some earlier developers not listed on the above are:

- Pankaj Pandey (stress solver and improved load balancing, 2011)
- Chandrashekhar Kaushik (original parallel and serial implementation in 2009)


---------------------------
Research papers using PySPH
---------------------------


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


-------------
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

The following are older presentations:

 - Prabhu Ramachandran, *PySPH: a reproducible and high-performance framework
   for smoothed particle hydrodynamics*, In Proceedings of the 15th Python in
   Science Conference, pages 127--135, July 11th to 17th, 2016. `Link to paper
   <http://conference.scipy.org/proceedings/scipy2016/prabhu_ramachandran_pysph.html>`_.

 - Prabhu Ramachandran and Kunal Puri, *PySPH: A framework for parallel
   particle simulations*, In proceedings of the 3rd International
   Conference on Particle-Based Methods (Particles 2013), Stuttgart,
   Germany, 18th September 2013.


--------
History
--------

- 2009: PySPH started with a simple Cython based 1D implementation written by
  Prabhu.

- 2009-2010: Chandrashekhar Kaushik worked on a full 3D SPH implementation with
  a more general purpose design.  The implementation was in a mix of Cython and
  Python.

- 2010-2012: The previous implementation was a little too complex and was
  largely overhauled by Kunal and Pankaj.  This became the PySPH 0.9beta
  release.  The difficulty with this version was that it was almost entirely
  written in Cython, making it hard to extend or add new formulations without
  writing more Cython code.  Doing this was difficult and not too pleasant.  In
  addition it was not as fast as we would have liked it. It ended up feeling
  like we might as well have implemented it all in C++ and exposed a Python
  interface to that.

- 2011-2012: Kunal also implemented SPH2D_ and another internal version called
  ZSPH in Cython which included Zoltan_ based parallelization using PyZoltan_.
  This was specific to his PhD research and again required writing Cython
  making it difficult for the average user to extend.

- 2013-present In early 2013, Prabhu reimplemented the core of PySPH to be
  almost entirely auto-generated from pure Python.  The resulting code was
  faster than previous implementations and very easy to extend entirely from
  pure Python.  Kunal and Prabhu integrated PyZoltan into PySPH and the current
  version of PySPH was born.  Subsequently, OpenMP support was also added in
  2015.


.. _SPH2D: https://bitbucket.org/kunalp/sph2d
.. _PyZoltan: https://github.com/pypr/pyzoltan
.. _Zoltan: http://www.cs.sandia.gov/zoltan/


-------
Support
-------

If you have any questions or are running into any difficulties with PySPH you
can use the `PySPH discussions <https://github.com/pypr/pysph/discussions>`_
to ask questions or look for answers.

Please also take a look at the `PySPH issue tracker
<https://github.com/pypr/pysph/issues>`_ if you have bugs or issues to report.

You could also email or post your questions on the pysph-users mailing list here:
https://groups.google.com/d/forum/pysph-users


----------
Changelog
----------

.. include:: ../../CHANGES.rst
