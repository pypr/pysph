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
import commands
import os

from setuptools import find_packages, setup
from numpy.distutils.extension import Extension
from Cython.Distutils import build_ext

Have_MPI = True
try:
    import mpi4py
except ImportError:
    Have_MPI = False

Have_Zoltan=True
try:
    import pyzoltan
except ImportError:
    Have_Zoltan=False

mpi_inc_dirs = []
mpi_compile_args = []
mpi_link_args = []

mpic = 'mpicc'
mpi_link_args.append(commands.getoutput(mpic + ' --showme:link'))
mpi_compile_args.append(commands.getoutput(mpic +' --showme:compile'))
mpi_inc_dirs.append(mpi4py.get_include())

# Zoltan Headers
zoltan_include_dirs = [ os.environ['ZOLTAN_INCLUDE'] ]
zoltan_library_dirs = [ os.environ['ZOLTAN_LIBRARY'] ]

# PyZoltan includes
pyzoltan_include = []
if Have_Zoltan:
    pyzoltan_include = [pyzoltan.get_include()]

include_dirs = [numpy.get_include()]

cmdclass = {'build_ext': build_ext}

ext_modules = [

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

# add the include dirs for the extension modules
for ext in ext_modules:
    ext.include_dirs = include_dirs

parallel_modules = [
    Extension( name="pysph.parallel.parallel_manager",
               sources=["pysph/parallel/parallel_manager.pyx"],
               include_dirs = include_dirs + mpi_inc_dirs + zoltan_include_dirs + pyzoltan_include,
               library_dirs = zoltan_library_dirs,
               libraries = ['zoltan', 'mpi'] ),

    Extension( name="pysph.parallel._kernels",
               sources=["pysph/parallel/_kernels.pyx"]
               ),
    ]

# currently we depend on PyZoltan
if Have_MPI and Have_Zoltan:
    ext_modules += parallel_modules

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
      entry_points = """
          [console_scripts]
          pysph_viewer = pysph.tools.mayavi_viewer:main
          """,
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

