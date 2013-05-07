"""
PySPH
=====

A general purpose Smoothed Particle Hydrodynamics framework.

This package provides a general purpose framework for SPH simulations
in Python.  The framework emphasizes flexibility and efficiency while
allowing most of the user code to be written in pure Python.  See here:

    http://pysph.googlecode.com

for more information.
"""

import numpy
from setuptools import find_packages, setup

from numpy.distutils.extension import Extension
from Cython.Distutils import build_ext

#import commands
#import os
#import mpi4py
#zoltan_include_dirs = [ os.environ['ZOLTAN_INCLUDE'] ]
#zoltan_library_dirs = [ os.environ['ZOLTAN_LIBRARY'] ]
#mpic = 'mpicc'
#mpi_include_dirs = [ commands.getoutput( mpic + ' --showme:incdirs' ) ]
#mpi_include_dirs.append(mpi4py.get_include())
#mpi_library_dirs = [ commands.getoutput( mpic + ' --showme:link' ) ]
#sph2d_include_dirs = ['/home/pysph/sph2d/src/sph2d/']
#include_dirs = zoltan_include_dirs + mpi_include_dirs + sph2d_include_dirs
#library_dirs = zoltan_library_dirs + mpi_library_dirs

include_dirs = [numpy.get_include()]

cmdclass = {'build_ext': build_ext}

ext_modules = [
    # base module
    Extension( name="pysph.base.carray",
               sources=["pysph/base/carray.pyx"]),

    Extension( name="pysph.base.particle_array",
               sources=["pysph/base/particle_array.pyx"]),

    Extension( name="pysph.base.point",
               sources=["pysph/base/point.pyx"]),

    Extension( name="pysph.base.nnps",
               sources=["pysph/base/nnps.pyx"]),
    
    # sph module
    Extension( name="pysph.sph.integrator",
               sources=["pysph/sph/integrator.pyx"])
    ]
    
for ext in ext_modules:
    ext.include_dirs = include_dirs

setup(name='PySPH',
      version = '1.0alpha',
      author = 'PySPH Developers',
      author_email = 'pysph-dev@googlegroups.com',
      description = "A general purpose Smoothed Particle Hydrodynamics framework",
      long_description = __doc__,
      url = 'http://pysph.googlecode.com',
      license = "BSD",
      keywords = "SPH simulation computational fluid dynamics",
      test_suite = "nose.collector",
      packages = find_packages(),

      ext_modules = ext_modules,
      
      include_package_data = True,
      cmdclass=cmdclass,
      #install_requires=['mpi4py>=1.2', 'numpy>=1.0.3', 'Cython>=0.14'],
      #setup_requires=['Cython>=0.14', 'setuptools>=0.6c1'],
      #extras_require={'3D': 'Mayavi>=3.0'},
      zip_safe = False,
      #entry_points = """
      #    [console_scripts]
      #    pysph_viewer = pysph.tools.mayavi_viewer:main
      #    """,
      platforms=['Linux', 'Mac OS-X', 'Unix', 'Windows'],
      classifiers = [c.strip() for c in """\
        Development Status :: 3 - Alpha
        Environment :: Console
        Intended Audience :: Developers
        Intended Audience :: Science/Research
        License :: OSI Approved :: BSD License
        Natural Language :: English
        Operating System :: MacOS :: MacOS X
        Operating System :: Microsoft :: Windows
        Operating System :: POSIX
        Operating System :: Unix
        Programming Language :: Python
        Topic :: Scientific/Engineering
        Topic :: Scientific/Engineering :: Physics
        Topic :: Software Development :: Libraries
        """.splitlines() if len(c.split()) > 0],
      )

