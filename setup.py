'''Setup script for PySPH.

You can use some environment variables to control the build. Setting CC/CXX
will let you choose a custom compiler (these can also be set in the config file
discussed below). You can also export ZOLTAN or ZOLTAN_INCLUDE/ZOLTAN_LIBRARY.

These are more configuration file options that trump everything. The file is in
~/.compyle/config.py.  The options are:

OMP_CFLAGS = ['-fopenmp']
OMP_LINK = ['-fopenmp']

# MPI options: handy on clusters like a Cray.
MPI_CFLAGS = ['...']  # must be a list.
MPI_LINK = ['...']

# Zoltan options
USE_TRILINOS = 1  # When set to anything, use "-ltrilinos_zoltan".
ZOLTAN = '/path/to_zoltan'  # looks inside this for $ZOLTAN/include/, lib/

# Not needed if using ZOLTAN
ZOLTAN_INCLUDE = 'path/include'  # path to zoltan.h
ZOLTAN_LIBRARY = 'path/lib'  # path to libzoltan.a

'''

import os
from subprocess import check_output
import sys


# This is taken from compyle.ext_module.
def get_config_file_opts():
    '''A global configuration file is used to configure build options
    for compyle and other packages.  This is located in:

    ~/.compyle/config.py

    The file can contain arbitrary Python that is exec'd. The variables defined
    here specify the compile and link args. For example, one may set:

    OMP_CFLAGS = ['-fopenmp']
    OMP_LINK = ['-fopenmp']

    Will use these instead of the defaults that are automatically determined.
    These must be lists.

    '''
    fname = os.path.expanduser(os.path.join('~', '.compyle', 'config.py'))
    opts = {}
    if os.path.exists(fname):
        print('Reading configuration options from %s.' % fname)
        with open(fname) as fp:
            exec(compile(fp.read(), fname, 'exec'), opts)
        opts.pop('__builtins__', None)
    return opts


# NOTE: the configuration options in the file trump everything else!
# These are options from the .compyle/config.py
CONFIG_OPTS = get_config_file_opts()


if len(os.environ.get('COVERAGE', '')) > 0:
    MACROS = [("CYTHON_TRACE", "1"), ("CYTHON_TRACE_NOGIL", "1")]
    COMPILER_DIRECTIVES = {"linetrace": True}
    print("-" * 80)
    print("Enabling linetracing for cython and setting CYTHON_TRACE = 1")
    print("-" * 80)
else:
    MACROS = []
    COMPILER_DIRECTIVES = {}

MODE = 'normal'
if len(sys.argv) >= 2 and \
   ('--help' in sys.argv[1:] or
    sys.argv[1] in ('--help-commands', 'egg_info', '--version',
                    'clean', 'sdist')):
    MODE = 'info'

HAVE_MPI = True
try:
    import mpi4py
except ImportError:
    HAVE_MPI = False

USE_ZOLTAN = True
HAVE_ZOLTAN = True
try:
    import pyzoltan  # noqa: F401
except ImportError:
    HAVE_ZOLTAN = False

base_includes = [] if sys.platform == 'win32' else ['/usr/local/include/']

compiler = 'gcc'
# compiler = 'intel'
if compiler == 'intel':
    extra_compile_args = ['-O3']
else:
    extra_compile_args = []


def get_deps(*args):
    """Given a list of basenames, this checks if a .pyx or .pxd exists
    and returns the list.
    """
    result = []
    for basename in args:
        for ext in ('.pyx', '.pxd'):
            f = basename + ext
            if os.path.exists(f):
                result.append(f)
    return result


def _get_openmp_flags():
    """Return the OpenMP flags for the platform.

    This returns two lists, [extra_compile_args], [extra_link_args]
    """
    # Copied from compyle.ext_module
    if 'OMP_CFLAGS' in CONFIG_OPTS or 'OMP_LINK' in CONFIG_OPTS:
        return CONFIG_OPTS['OMP_CFLAGS'], CONFIG_OPTS['OMP_LINK']

    if sys.platform == 'win32':
        return ['/openmp'], []
    elif sys.platform == 'darwin':
        if (os.environ.get('CC') is not None and
           os.environ.get('CXX') is not None):
            return ['-fopenmp'], ['-fopenmp']
        else:
            return ['-Xpreprocessor', '-fopenmp'], ['-lomp']
    else:
        return ['-fopenmp'], ['-fopenmp']


def get_openmp_flags():
    """Returns any OpenMP related flags if OpenMP is avaiable on the system.
    """
    omp_compile_flags, omp_link_flags = _get_openmp_flags()

    from textwrap import dedent
    try:
        from Cython.Distutils import Extension
        from pyximport import pyxbuild
    except ImportError:
        print("Unable to import Cython, disabling OpenMP for now.")
        return [], [], False

    from distutils.errors import CompileError, LinkError
    import shutil
    import tempfile
    test_code = dedent("""
    from cython.parallel import parallel, prange, threadid
    cimport openmp
    def n_threads():
        with nogil, parallel():
            openmp.omp_get_num_threads()
    """)
    tmp_dir = tempfile.mkdtemp()
    fname = os.path.join(tmp_dir, 'check_omp.pyx')
    with open(fname, 'w') as fp:
        fp.write(test_code)
    extension = Extension(
        name='check_omp', sources=[fname],
        include_dirs=base_includes,
        extra_compile_args=omp_compile_flags,
        extra_link_args=omp_link_flags,
    )
    has_omp = True
    try:
        pyxbuild.pyx_to_dll(fname, extension, pyxbuild_dir=tmp_dir)
        print("-" * 70)
        print("Using OpenMP.")
        print("-" * 70)
    except CompileError:
        print("*" * 70)
        print("Unable to compile OpenMP code. Not using OpenMP.")
        print("*" * 70)
        has_omp = False
    except LinkError:
        print("*" * 70)
        print("Unable to link OpenMP code. Not using OpenMP.")
        print("*" * 70)
        has_omp = False
    finally:
        shutil.rmtree(tmp_dir)

    if has_omp:
        return omp_compile_flags, omp_link_flags, True
    else:
        return [], [], False


def get_zoltan_directory(varname):
    global USE_ZOLTAN
    if varname in CONFIG_OPTS:
        return os.path.expanduser(CONFIG_OPTS[varname])

    d = os.environ.get(varname, '')
    if len(d) == 0:
        USE_ZOLTAN = False
        return ''
    else:
        USE_ZOLTAN = True
    if not os.path.exists(d):
        print("*" * 80)
        print("%s incorrectly set to %s, not using ZOLTAN!" % (varname, d))
        print("*" * 80)
        USE_ZOLTAN = False
        return ''
    return d


def get_mpi_flags():
    """Returns mpi_inc_dirs, mpi_compile_args, mpi_link_args.
    """
    global HAVE_MPI
    mpi_inc_dirs = []
    mpi_compile_args = []
    mpi_link_args = []
    if not HAVE_MPI:
        return mpi_inc_dirs, mpi_compile_args, mpi_link_args
    elif 'MPI_CFLAGS' in CONFIG_OPTS:
        mpi_compile_args = CONFIG_OPTS['MPI_CFLAGS']
        mpi_link_args = CONFIG_OPTS['MPI_LINK']
        mpi_inc_dirs.append(mpi4py.get_include())
    else:
        try:
            mpic = 'mpic++'
            if compiler == 'intel':
                link_args = check_output(
                    [mpic, '-cc=icc', '-link_info'],
                    universal_newlines=True
                ).strip()
                link_args = link_args[3:]
                compile_args = check_output(
                    [mpic, '-cc=icc', '-compile_info'],
                    universal_newlines=True
                ).strip()
                compile_args = compile_args[3:]
            else:
                link_args = check_output(
                    [mpic, '--showme:link'],
                    universal_newlines=True
                ).strip()
                compile_args = check_output(
                    [mpic, '--showme:compile'],
                    universal_newlines=True
                ).strip()
        except:  # noqa: E722
            print('-' * 80)
            print("Unable to run mpic++ correctly, skipping parallel build")
            print('-' * 80)
            HAVE_MPI = False
        else:
            mpi_link_args.extend(link_args.split())
            mpi_compile_args.extend(compile_args.split())
            mpi_inc_dirs.append(mpi4py.get_include())

    return mpi_inc_dirs, mpi_compile_args, mpi_link_args


def get_zoltan_args():
    """Returns zoltan_include_dirs, zoltan_library_dirs
    """
    global HAVE_MPI, USE_ZOLTAN
    zoltan_include_dirs, zoltan_library_dirs = [], []
    if not HAVE_MPI or not HAVE_ZOLTAN:
        return zoltan_include_dirs, zoltan_library_dirs
    # First try with the environment variable 'ZOLTAN'
    zoltan_base = get_zoltan_directory('ZOLTAN')
    inc = lib = ''
    if len(zoltan_base) > 0:
        inc = os.path.join(zoltan_base, 'include')
        lib = os.path.join(zoltan_base, 'lib')
        if not os.path.exists(inc) or not os.path.exists(lib):
            inc = lib = ''

    # try with the older ZOLTAN include directories
    if len(inc) == 0 or len(lib) == 0:
        inc = get_zoltan_directory('ZOLTAN_INCLUDE')
        lib = get_zoltan_directory('ZOLTAN_LIBRARY')

    if HAVE_ZOLTAN and not USE_ZOLTAN:
        # Try with default in sys.prefix/{include,lib}, this is what is done
        # by any conda installs of zoltan.
        inc = os.path.join(sys.prefix, 'include')
        lib = os.path.join(sys.prefix, 'lib')
        if os.path.exists(os.path.join(inc, 'zoltan.h')):
            USE_ZOLTAN = True

    if (not USE_ZOLTAN):
        print("*" * 80)
        print("Zoltan Environment variable not set, not using ZOLTAN!")
        print("*" * 80)
        HAVE_MPI = False
    else:
        print('-' * 70)
        print("Using Zoltan from:\n%s\n%s" % (inc, lib))
        print('-' * 70)
        zoltan_include_dirs = [inc]
        zoltan_library_dirs = [lib]

        # PyZoltan includes
        zoltan_cython_include = [
            os.path.abspath(
                os.path.join(os.path.dirname(pyzoltan.__file__), 'czoltan')
            )
        ]
        zoltan_include_dirs += zoltan_cython_include

    return zoltan_include_dirs, zoltan_library_dirs


def get_basic_extensions():
    if MODE == 'info':
        try:
            from Cython.Distutils import Extension
        except ImportError:
            from distutils.core import Extension
        try:
            import numpy
        except ImportError:
            include_dirs = []
        else:
            include_dirs = [numpy.get_include()]
    else:
        from Cython.Distutils import Extension
        import numpy
        include_dirs = [numpy.get_include()]

    include_dirs += base_includes
    openmp_compile_args, openmp_link_args, openmp_env = get_openmp_flags()

    ext_modules = [
        Extension(
            name="pysph.base.particle_array",
            sources=["pysph/base/particle_array.pyx"],
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
            language="c++",
            define_macros=MACROS,
        ),

        Extension(
            name="pysph.base.point",
            sources=["pysph/base/point.pyx"],
            extra_compile_args=extra_compile_args,
            include_dirs=include_dirs,
            language="c++",
            define_macros=MACROS,
        ),

        Extension(
            name="pysph.base.nnps_base",
            sources=["pysph/base/nnps_base.pyx"],
            depends=get_deps(
                "pysph/base/point",
                "pysph/base/particle_array",
            ),
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args + openmp_compile_args,
            extra_link_args=openmp_link_args,
            cython_compile_time_env={'OPENMP': openmp_env},
            language="c++",
            define_macros=MACROS,
        ),


        Extension(
            name="pysph.base.linked_list_nnps",
            sources=["pysph/base/linked_list_nnps.pyx"],
            depends=get_deps(
                "pysph/base/nnps_base"
            ),
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args + openmp_compile_args,
            extra_link_args=openmp_link_args,
            cython_compile_time_env={'OPENMP': openmp_env},
            language="c++",
            define_macros=MACROS,
        ),

        Extension(
            name="pysph.base.box_sort_nnps",
            sources=["pysph/base/box_sort_nnps.pyx"],
            depends=get_deps(
                "pysph/base/nnps_base", "pysph/base/linked_list_nnps"
            ),
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args + openmp_compile_args,
            extra_link_args=openmp_link_args,
            cython_compile_time_env={'OPENMP': openmp_env},
            language="c++",
            define_macros=MACROS,
        ),

        Extension(
            name="pysph.base.spatial_hash_nnps",
            sources=["pysph/base/spatial_hash_nnps.pyx"],
            depends=get_deps(
                "pysph/base/nnps_base"
            ),
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args + openmp_compile_args,
            extra_link_args=openmp_link_args,
            cython_compile_time_env={'OPENMP': openmp_env},
            language="c++",
            define_macros=MACROS,
        ),

        Extension(
            name="pysph.base.cell_indexing_nnps",
            sources=["pysph/base/cell_indexing_nnps.pyx"],
            depends=get_deps(
                "pysph/base/nnps_base"
            ),
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args + openmp_compile_args,
            extra_link_args=openmp_link_args,
            cython_compile_time_env={'OPENMP': openmp_env},
            language="c++",
            define_macros=MACROS,
        ),


        Extension(
            name="pysph.base.z_order_nnps",
            sources=["pysph/base/z_order_nnps.pyx"],
            depends=get_deps(
                "pysph/base/nnps_base"
            ),
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args + openmp_compile_args,
            extra_link_args=openmp_link_args,
            cython_compile_time_env={'OPENMP': openmp_env},
            language="c++",
            define_macros=MACROS,
        ),

        Extension(
            name="pysph.base.stratified_hash_nnps",
            sources=["pysph/base/stratified_hash_nnps.pyx"],
            depends=get_deps(
                "pysph/base/nnps_base"
            ),
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args + openmp_compile_args,
            extra_link_args=openmp_link_args,
            cython_compile_time_env={'OPENMP': openmp_env},
            language="c++",
            define_macros=MACROS,
        ),

        Extension(
            name="pysph.base.stratified_sfc_nnps",
            sources=["pysph/base/stratified_sfc_nnps.pyx"],
            depends=get_deps(
                "pysph/base/nnps_base", "pysph/base/z_order_nnps"
            ),
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args + openmp_compile_args,
            extra_link_args=openmp_link_args,
            cython_compile_time_env={'OPENMP': openmp_env},
            language="c++",
            define_macros=MACROS,
        ),

        Extension(
            name="pysph.base.octree",
            sources=["pysph/base/octree.pyx"],
            depends=get_deps(
                "pysph/base/nnps_base"
            ),
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args + openmp_compile_args,
            extra_link_args=openmp_link_args,
            cython_compile_time_env={'OPENMP': openmp_env},
            language="c++",
            define_macros=MACROS,
        ),

        Extension(
            name="pysph.base.octree_nnps",
            sources=["pysph/base/octree_nnps.pyx"],
            depends=get_deps(
                "pysph/base/nnps_base", "pysph/base/octree"
            ),
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args + openmp_compile_args,
            extra_link_args=openmp_link_args,
            cython_compile_time_env={'OPENMP': openmp_env},
            language="c++",
            define_macros=MACROS,
        ),

        # kernels used for tests
        Extension(
            name="pysph.base.c_kernels",
            sources=["pysph/base/c_kernels.pyx"],
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
            language="c++",
            define_macros=MACROS,
        ),

        # Eigen decomposition code
        Extension(
            name="pysph.base.linalg3",
            sources=["pysph/base/linalg3.pyx"],
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
            language="c++",
            define_macros=MACROS,
        ),

        # STL tool
        Extension(
            name="pysph.tools.mesh_tools",
            sources=["pysph/tools/mesh_tools.pyx"],
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
            language="c++",
            define_macros=MACROS,
        ),

    ]

    # OpenCL related modules
    ext_modules.extend((
        Extension(
            name="pysph.base.gpu_nnps_base",
            sources=["pysph/base/gpu_nnps_base.pyx"],
            depends=get_deps(
                "pysph/base/point",
                "pysph/base/particle_array", "pysph/base/nnps_base"
            ),
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args + openmp_compile_args,
            extra_link_args=openmp_link_args,
            cython_compile_time_env={'OPENMP': openmp_env},
            language="c++",
            define_macros=MACROS,
        ),

        Extension(
            name="pysph.base.z_order_gpu_nnps",
            sources=["pysph/base/z_order_gpu_nnps.pyx"],
            depends=get_deps(
                "pysph/base/nnps_base"
            ),
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args + openmp_compile_args,
            extra_link_args=openmp_link_args,
            cython_compile_time_env={'OPENMP': openmp_env},
            language="c++",
            define_macros=MACROS,
        ),
        Extension(
            name="pysph.base.stratified_sfc_gpu_nnps",
            sources=["pysph/base/stratified_sfc_gpu_nnps.pyx"],
            depends=get_deps(
                "pysph/base/nnps_base"
            ),
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args + openmp_compile_args,
            extra_link_args=openmp_link_args,
            cython_compile_time_env={'OPENMP': openmp_env},
            language="c++",
            define_macros=MACROS,
        ),
        Extension(
            name="pysph.base.octree_gpu_nnps",
            sources=["pysph/base/octree_gpu_nnps.pyx"],
            depends=get_deps(
                "pysph/base/nnps_base"
            ),
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args + openmp_compile_args,
            extra_link_args=openmp_link_args,
            cython_compile_time_env={'OPENMP': openmp_env},
            language="c++",
            define_macros=MACROS,
        ))
    )

    return ext_modules


def get_parallel_extensions():
    if not HAVE_MPI:
        return []

    if MODE == 'info':
        from distutils.core import Extension
        include_dirs = []
        mpi_inc_dirs, mpi_compile_args, mpi_link_args = [], [], []
        zoltan_include_dirs, zoltan_library_dirs = [], []
    else:
        from Cython.Distutils import Extension
        import numpy
        include_dirs = [numpy.get_include()]
        mpi_inc_dirs, mpi_compile_args, mpi_link_args = get_mpi_flags()
        zoltan_include_dirs, zoltan_library_dirs = get_zoltan_args()

    # We should check again here as HAVE_MPI may be set to False when we try to
    # get the MPI flags and are not successful.
    if not HAVE_MPI:
        return []

    include_dirs += base_includes

    MPI4PY_V2 = False if mpi4py.__version__.startswith('1.') else True
    cython_compile_time_env = {'MPI4PY_V2': MPI4PY_V2}

    zoltan_lib = 'zoltan'
    if 'USE_TRILINOS' in CONFIG_OPTS:
        zoltan_lib = 'trilinos_zoltan'
    elif os.environ.get('USE_TRILINOS', None) is not None:
        zoltan_lib = 'trilinos_zoltan'

    parallel_modules = [
        Extension(
            name="pysph.parallel.parallel_manager",
            sources=["pysph/parallel/parallel_manager.pyx"],
            depends=get_deps(
                "pysph/base/point", "pysph/base/particle_array",
                "pysph/base/nnps_base"
            ),
            include_dirs=include_dirs + mpi_inc_dirs + zoltan_include_dirs,
            library_dirs=zoltan_library_dirs,
            libraries=[zoltan_lib, 'mpi'],
            extra_link_args=mpi_link_args,
            extra_compile_args=mpi_compile_args + extra_compile_args,
            cython_compile_time_env=cython_compile_time_env,
            language="c++",
            define_macros=MACROS,
        ),
    ]
    return parallel_modules


def create_sources():
    argv = sys.argv
    if 'build_ext' in argv or 'develop' in sys.argv or 'install' in argv:
        pth = os.path.join('pysph', 'base')
        cmd = [sys.executable, '-m', 'cyarray.generator', os.path.abspath(pth)]
        print(check_output(cmd).decode())


def _is_cythonize_default():
    import warnings
    result = True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            # old_build_ext was introduced in Cython 0.25 and this is when
            # cythonize was made the default.
            from Cython.Distutils import old_build_ext  # noqa: F401
        except ImportError:
            result = False
    return result


def setup_package():
    from setuptools import find_packages, setup
    if MODE == 'info':
        cmdclass = {}
    else:
        from Cython.Distutils import build_ext
        cmdclass = {'build_ext': build_ext}

        create_sources()

    # Extract the version information from pysph/__init__.py
    info = {}
    module = os.path.join('pysph', '__init__.py')
    exec(compile(open(module).read(), module, 'exec'), info)

    # The requirements.
    install_requires = [
        'numpy', 'mako', 'cyarray', 'compyle>=0.8', 'Cython>=0.20',
        'setuptools>=42.0.0', 'pytools', 'Beaker'
    ]
    tests_require = ['pytest>=3.0', 'h5py', 'vtk']
    if sys.version_info[:2] == (2, 6):
        install_requires += [
            'ordereddict', 'importlib'
        ]
        tests_require += ['unittest2']
    if sys.version_info[0] < 3:
        tests_require += [
            'mock>=1.0'
        ]
    docs_require = ["sphinx"]

    extras_require = dict(
        mpi=['mpi4py>=1.2', 'pyzoltan'],
        opencl=['pyopencl'],
        ui=['mayavi>=4.0', 'pyside2', 'h5py'],
        tests=tests_require,
        docs=docs_require,
        dev=tests_require + docs_require,
    )

    everything = set()
    for dep in extras_require.values():
        everything.update(dep)
    extras_require['all'] = everything

    ext_modules = get_basic_extensions() + get_parallel_extensions()
    if MODE != 'info' and _is_cythonize_default():
        # Cython >= 0.25 uses cythonize to compile the extensions. This
        # requires the compile_time_env to be set explicitly to work.
        compile_env = {}
        include_path = set()
        for mod in ext_modules:
            compile_env.update(mod.cython_compile_time_env or {})
            include_path.update(mod.include_dirs)
        from Cython.Build import cythonize
        ext_modules = cythonize(
            ext_modules, compile_time_env=compile_env,
            include_path=list(include_path),
            language="c++",
            compiler_directives=COMPILER_DIRECTIVES,
        )

    setup(name='PySPH',
          version=info['__version__'],
          author='PySPH Developers',
          author_email='pysph-dev@googlegroups.com',
          description="A general purpose Smoothed Particle Hydrodynamics "
          "framework",
          long_description=open('README.rst').read(),
          url='http://github.com/pypr/pysph',
          license="BSD",
          keywords="SPH simulation computational fluid dynamics",
          packages=find_packages(),
          package_data={
              '': ['*.pxd', '*.mako', '*.txt.gz', '*.txt', '*.txt.bz2',
                   '*.vtk.gz', '*.gz', '*.csv',
                   '*.rst', 'ndspmhd-sedov-initial-conditions.npz']
          },
          # exclude package data in installation.
          exclude_package_data={
              '': ['Makefile', '*.bat', '*.cfg', '*.rst', '*.sh', '*.yml'],
          },
          ext_modules=ext_modules,
          include_package_data=True,
          cmdclass=cmdclass,
          install_requires=install_requires,
          extras_require=extras_require,
          zip_safe=False,
          entry_points="""
              [console_scripts]
              pysph = pysph.tools.cli:main
              """,
          platforms=['Linux', 'Mac OS-X', 'Unix', 'Windows'],
          classifiers=[c.strip() for c in """\
            Development Status :: 4 - Beta
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
            Programming Language :: Python :: 3
            Topic :: Scientific/Engineering
            Topic :: Scientific/Engineering :: Physics
            Topic :: Software Development :: Libraries
            """.splitlines() if len(c.split()) > 0],
          )


if __name__ == '__main__':
    setup_package()
