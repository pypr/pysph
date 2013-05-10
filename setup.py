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

import sys
import os
import numpy
from setuptools import find_packages, setup
from numpy.distutils.extension import Extension
from Cython.Distutils import build_ext

from os import path

mpi_inc_dirs = []
mpi_compile_args = []
mpi_link_args = []
import mpi4py
import commands
mpic = 'mpicc'
mpi_link_args.append(commands.getoutput(mpic + ' --showme:link'))
mpi_compile_args.append(commands.getoutput(mpic +' --showme:compile'))
mpi_inc_dirs.append(mpi4py.get_include())

include_dirs = [numpy.get_include()]

zoltan_include_dirs = [ os.environ['ZOLTAN_INCLUDE'] ]
zoltan_library_dirs = [ os.environ['ZOLTAN_LIBRARY'] ]

zoltan_cython_include = [ os.path.abspath('./pyzoltan/czoltan') ]
zoltan_include_dirs += zoltan_cython_include

cmdclass = {'build_ext': build_ext}

ext_modules = [
    # core modules
    Extension( name="pyzoltan.core.carray",
               sources=["pyzoltan/core/carray.pyx"],
               include_dirs = include_dirs),

    Extension( name="pyzoltan.core.zoltan",
               sources=["pyzoltan/core/zoltan.pyx"],
               include_dirs = include_dirs+mpi_inc_dirs+zoltan_include_dirs,
               library_dirs = zoltan_library_dirs,
               libraries=['zoltan', 'mpi']),

    Extension( name="pyzoltan.core.zoltan_dd",
               sources=["pyzoltan/core/zoltan_dd.pyx"],
               include_dirs = include_dirs + mpi_inc_dirs + zoltan_include_dirs,
               library_dirs = zoltan_library_dirs,
               libraries=['zoltan', 'mpi']),

    # sph modules
    Extension( name="pyzoltan.sph.point",
               sources=["pyzoltan/sph/point.pyx"],
               include_dirs = include_dirs),
    
    Extension( name="pyzoltan.sph.kernels",
               sources=["pyzoltan/sph/kernels.pyx"],
               include_dirs = include_dirs),
    
    Extension( name="pyzoltan.sph.particle_array",
               sources=["pyzoltan/sph/particle_array.pyx"],
               include_dirs = include_dirs),

    Extension( name="pyzoltan.sph.nnps",
               sources=["pyzoltan/sph/nnps.pyx"],
               include_dirs = include_dirs + mpi_inc_dirs + zoltan_include_dirs,
               library_dirs = zoltan_library_dirs,
               libraries = ['zoltan', 'mpi'])
    ]

if 'build_ext' in sys.argv or 'develop' in sys.argv or 'install' in sys.argv:
    generator = path.join( path.abspath('.'), 'pyzoltan/core/generator.py' )
    d = {'__file__': generator }
    execfile(generator, d)
    d['main'](None)

setup(name='PyZoltan',
      version = '1.0',
      author = 'PySPH Developers',
      author_email = 'pysph-dev@googlegroups.com',
      description = "Wrapper for the Zotlan data management library",
      long_description = __doc__,
      url = 'http://ww.bitbucket.org/kunalp/pyzoltan',
      license = "BSD",
      keywords = "Dynamic load balancing SPH AMR",
      test_suite = "nose.collector",
      packages = find_packages(),

      # include Cython headers in the install directory
      package_data={'' : ['*.pxd']},

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
      platforms=['Linux', 'Mac OS-X', 'Unix'],
      classifiers = [c.strip() for c in """\
        Development Status :: 3 - Alpha
        Environment :: Console
        Intended Audience :: Developers
        Intended Audience :: Science/Research
        License :: OSI Approved :: BSD License
        Natural Language :: English
        Operating System :: MacOS :: MacOS X
        Operating System :: POSIX
        Operating System :: Unix
        Programming Language :: Python
        Topic :: Scientific/Engineering
        Topic :: Scientific/Engineering :: Physics
        Topic :: Software Development :: Libraries
        """.splitlines() if len(c.split()) > 0],
      )

