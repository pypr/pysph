from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import commands
import os

import mpi4py

#zoltan_include_dirs = [ os.environ['ZOLTAN_INCLUDE'] ]
#zoltan_library_dirs = [ os.environ['ZOLTAN_LIBRARY'] ]

#mpic = 'mpicc'
#mpi_include_dirs = [ commands.getoutput( mpic + ' --showme:incdirs' ) ]
#mpi_include_dirs.append(mpi4py.get_include())

#mpi_library_dirs = [ commands.getoutput( mpic + ' --showme:link' ) ]

#sph2d_include_dirs = ['/home/pysph/sph2d/src/sph2d/']

#include_dirs = zoltan_include_dirs + mpi_include_dirs + sph2d_include_dirs
#library_dirs = zoltan_library_dirs + mpi_library_dirs

ext_modules = [

    Extension( name="integrator",
               sources=["integrator.pyx"])
    ]
                   
setup(
    name="PySPH",
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules,
    )
