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
import sys
from os import path

from setuptools import find_packages, setup
from numpy.distutils.extension import Extension
from Cython.Distutils import build_ext

Have_MPI = True
try:
    import mpi4py
except ImportError:
    Have_MPI = False

compiler = 'gcc'
#compiler = 'intel'
if compiler == 'intel':
    extra_compile_args = ['-O3']
else:
    extra_compile_args = []


mpi_inc_dirs = []
mpi_compile_args = []
mpi_link_args = []

def get_zoltan_directory(varname):
    d = os.environ.get(varname, '')
    if len(d) == 0:
        print "*"*80
        print "Environment variable", varname, \
              "not set, not using ZOLTAN!"
        print "*"*80
        return ''
    if not path.exists(d):
        print "*"*80
        print varname, "incorrectly set, not using ZOLTAN!"
        print "*"*80
        return ''
    return d

if Have_MPI:
    mpic = 'mpicc'
    if compiler == 'intel':
        link_args = commands.getoutput(mpic + ' -cc=icc -link_info')
        link_args = link_args[3:]
        compile_args = commands.getoutput(mpic +' -cc=icc -compile_info')
        compile_args = compile_args[3:]
    else:
        link_args = commands.getoutput(mpic + ' --showme:link')
        compile_args = commands.getoutput(mpic +' --showme:compile')
    mpi_link_args.append(link_args)
    mpi_compile_args.append(compile_args)
    mpi_inc_dirs.append(mpi4py.get_include())

    inc = get_zoltan_directory('ZOLTAN_INCLUDE')
    lib = get_zoltan_directory('ZOLTAN_LIBRARY')
    if len(inc) == 0 or len(lib) == 0:
        Have_MPI = False
    else:
        zoltan_include_dirs = [ inc ]
        zoltan_library_dirs = [ lib ]

        # PyZoltan includes
        zoltan_cython_include = [ os.path.abspath('./pyzoltan/czoltan') ]
        zoltan_include_dirs += zoltan_cython_include

include_dirs = [numpy.get_include()]

cmdclass = {'build_ext': build_ext}

ext_modules = [
    Extension( name="pyzoltan.core.carray",
               sources=["pyzoltan/core/carray.pyx"],
               include_dirs = include_dirs,
               extra_compile_args=extra_compile_args),

    Extension( name="pysph.base.particle_array",
               sources=["pysph/base/particle_array.pyx"],
               extra_compile_args=extra_compile_args),

    Extension( name="pysph.base.point",
               sources=["pysph/base/point.pyx"],
               extra_compile_args=extra_compile_args),

    Extension( name="pysph.base.nnps",
               sources=["pysph/base/nnps.pyx"],
               extra_compile_args=extra_compile_args),

    # kernels used for tests
    Extension( name="pysph.base.c_kernels",
               sources=["pysph/base/c_kernels.pyx"],
               include_dirs=include_dirs,
               extra_compile_args=extra_compile_args),

    # Eigen decomposition code
    Extension( name="pysph.sph.solid_mech.linalg",
               sources=["pysph/sph/solid_mech/linalg.pyx"],
               include_dirs=include_dirs,
               extra_compile_args=extra_compile_args,
               ),
    ]

# add the include dirs for the extension modules
for ext in ext_modules:
    ext.include_dirs = include_dirs

if Have_MPI:
    zoltan_modules = [
        Extension( name="pyzoltan.core.zoltan",
                   sources=["pyzoltan/core/zoltan.pyx"],
                   include_dirs = include_dirs+zoltan_include_dirs+mpi_inc_dirs,
                   library_dirs = zoltan_library_dirs,
                   libraries=['zoltan', 'mpi'],
                   extra_link_args=mpi_link_args,
                   extra_compile_args=mpi_compile_args+extra_compile_args),

        Extension( name="pyzoltan.core.zoltan_dd",
                   sources=["pyzoltan/core/zoltan_dd.pyx"],
                   include_dirs = include_dirs + zoltan_include_dirs + mpi_inc_dirs,
                   library_dirs = zoltan_library_dirs,
                   libraries=['zoltan', 'mpi'],
                   extra_link_args=mpi_link_args,
                   extra_compile_args=mpi_compile_args+extra_compile_args),

        Extension( name="pyzoltan.core.zoltan_comm",
                   sources=["pyzoltan/core/zoltan_comm.pyx"],
                   include_dirs = include_dirs + zoltan_include_dirs + mpi_inc_dirs,
                   library_dirs = zoltan_library_dirs,
                   libraries=['zoltan', 'mpi'],
                   extra_link_args=mpi_link_args,
                   extra_compile_args=mpi_compile_args+extra_compile_args),
        ]

    parallel_modules = [

        Extension( name="pysph.parallel.parallel_manager",
                   sources=["pysph/parallel/parallel_manager.pyx"],
                   include_dirs = include_dirs + mpi_inc_dirs + zoltan_include_dirs,
                   library_dirs = zoltan_library_dirs,
                   libraries = ['zoltan', 'mpi'],
                   extra_link_args=mpi_link_args,
                   extra_compile_args=mpi_compile_args+extra_compile_args),
        ]

    ext_modules += zoltan_modules + parallel_modules

if 'build_ext' in sys.argv or 'develop' in sys.argv or 'install' in sys.argv:
    for pth in (path.join('pyzoltan', 'core'), path.join('pysph', 'base')):
        generator = path.join( path.abspath('.'), path.join(pth, 'generator.py'))
        d = {'__file__': generator }
        execfile(generator, d)
        d['main'](None)


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
      # include Cython headers in the install directory
      package_data={'' : ['*.pxd', '*.mako']},

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
