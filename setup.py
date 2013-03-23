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

from setuptools import find_packages, setup

from distutils.core import setup
from distutils.extension import Extension
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

cmdclass = {'build_ext': build_ext}

ext_modules = [

    Extension( name="integrator",
               sources=["integrator.pyx"])
    ]

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
      packages = find_packages('pysph'),
      
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

