import os
from os import path
try:
    from subprocess import check_output
except ImportError:
    # check_output does not exist in Python-2.6
    from subprocess import Popen, PIPE, CalledProcessError
    def check_output(*popenargs, **kwargs):
        if 'stdout' in kwargs:
            raise ValueError('stdout argument not allowed, it will be overridden.')
        process = Popen(stdout=PIPE, *popenargs, **kwargs)
        output, unused_err = process.communicate()
        retcode = process.poll()
        if retcode:
            cmd = kwargs.get("args")
            if cmd is None:
                cmd = popenargs[0]
            raise CalledProcessError(retcode, cmd, output=output)
        return output

import sys

MODE = 'normal'
if len(sys.argv) >= 2 and ('--help' in sys.argv[1:] or
        sys.argv[1] in ('--help-commands', 'egg_info', '--version',
                        'clean', 'sdist')):
    MODE = 'info'

HAVE_MPI = True
try:
    import mpi4py
except ImportError:
    HAVE_MPI = False

USE_ZOLTAN=True

compiler = 'gcc'
#compiler = 'intel'
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
            if path.exists(f):
                result.append(f)
    return result

def get_openmp_flags():
    """Returns any OpenMP related flags if OpenMP is avaiable on the system.
    """
    if sys.platform == 'win32':
        omp_compile_flags, omp_link_flags = ['/openmp'], []
    else:
        omp_compile_flags, omp_link_flags = ['-fopenmp'], ['-fopenmp']

    env_var = os.environ.get('USE_OPENMP', '')
    if env_var.lower() in ("0", 'false', 'n'):
        print("-"*70)
        print("OpenMP disabled by environment variable (USE_OPENMP).")
        return [], [], False

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
    fname = path.join(tmp_dir, 'check_omp.pyx')
    with open(fname, 'w') as fp:
        fp.write(test_code)
    extension = Extension(
        name='check_omp', sources=[fname],
        extra_compile_args=omp_compile_flags,
        extra_link_args=omp_link_flags,
    )
    has_omp = True
    try:
        mod = pyxbuild.pyx_to_dll(fname, extension, pyxbuild_dir=tmp_dir)
        print("-"*70)
        print("Using OpenMP.")
        print("-"*70)
    except CompileError:
        print("*"*70)
        print("Unable to compile OpenMP code. Not using OpenMP.")
        print("*"*70)
        has_omp = False
    except LinkError:
        print("*"*70)
        print("Unable to link OpenMP code. Not using OpenMP.")
        print("*"*70)
        has_omp = False
    finally:
        shutil.rmtree(tmp_dir)

    if has_omp:
        return omp_compile_flags, omp_link_flags, True
    else:
        return [], [], False


def get_zoltan_directory(varname):
    global USE_ZOLTAN
    d = os.environ.get(varname, '')
    if ( len(d) == 0 ):
        USE_ZOLTAN=False
        return ''
    if not path.exists(d):
        print("*"*80)
        print("%s incorrectly set to %s, not using ZOLTAN!"%(varname, d))
        print("*"*80)
        USE_ZOLTAN=False
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
    try:
        mpic = 'mpic++'
        if compiler == 'intel':
            link_args = check_output(
                [mpic, '-cc=icc', '-link_info'], universal_newlines=True
            ).strip()
            link_args = link_args[3:]
            compile_args = check_output(
                [mpic, '-cc=icc', '-compile_info'], universal_newlines=True
            ).strip()
            compile_args = compile_args[3:]
        else:
            link_args = check_output(
                [mpic, '--showme:link'], universal_newlines=True
            ).strip()
            compile_args = check_output(
                [mpic, '--showme:compile'], universal_newlines=True
            ).strip()
    except:
        print('-'*80)
        print("Unable to run mpic++ correctly, skipping parallel build")
        print('-'*80)
        HAVE_MPI = False
    else:
        mpi_link_args.extend(link_args.split())
        mpi_compile_args.extend(compile_args.split())
        mpi_inc_dirs.append(mpi4py.get_include())

    return mpi_inc_dirs, mpi_compile_args, mpi_link_args


def get_zoltan_args():
    """Returns zoltan_include_dirs, zoltan_library_dirs
    """
    global HAVE_MPI
    zoltan_include_dirs, zoltan_library_dirs = [], []
    if not HAVE_MPI:
        return zoltan_include_dirs, zoltan_library_dirs
    # First try with the environment variable 'ZOLTAN'
    zoltan_base = get_zoltan_directory('ZOLTAN')
    inc = lib = ''
    if len(zoltan_base) > 0:
        inc = path.join(zoltan_base, 'include')
        lib = path.join(zoltan_base, 'lib')
        if not path.exists(inc) or not path.exists(lib):
            inc = lib = ''

    # try with the older ZOLTAN include directories
    if len(inc) == 0 or len(lib) == 0:
        inc = get_zoltan_directory('ZOLTAN_INCLUDE')
        lib = get_zoltan_directory('ZOLTAN_LIBRARY')

    if (not USE_ZOLTAN):
        print("*"*80)
        print("Zoltan Environment variable not set, not using ZOLTAN!")
        print("*"*80)
        HAVE_MPI = False
    else:
        print('-'*70)
        print("Using Zoltan from:\n%s\n%s"%(inc, lib))
        print('-'*70)
        zoltan_include_dirs = [ inc ]
        zoltan_library_dirs = [ lib ]

        # PyZoltan includes
        zoltan_cython_include = [ os.path.abspath('./pyzoltan/czoltan') ]
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

    openmp_compile_args, openmp_link_args, openmp_env = get_openmp_flags()

    ext_modules = [
        Extension(
            name="pyzoltan.core.carray",
            sources=["pyzoltan/core/carray.pyx"],
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
            language="c++"
        ),

        Extension(
            name="pysph.base.particle_array",
            sources=["pysph/base/particle_array.pyx"],
            depends=get_deps("pyzoltan/core/carray"),
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
            language="c++"
        ),

        Extension(
            name="pysph.base.point",
            sources=["pysph/base/point.pyx"],
            extra_compile_args=extra_compile_args,
            include_dirs=include_dirs,
            language="c++"
        ),

        Extension(
            name="pysph.base.nnps_base",
            sources=["pysph/base/nnps_base.pyx"],
            depends=get_deps(
                "pyzoltan/core/carray", "pysph/base/point",
                "pysph/base/particle_array",
            ),
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args + openmp_compile_args,
            extra_link_args=openmp_link_args,
            cython_compile_time_env={'OPENMP': openmp_env},
            language="c++"
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
            language="c++"
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
            language="c++"
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
            language="c++"
        ),

        # kernels used for tests
        Extension(
            name="pysph.base.c_kernels",
            sources=["pysph/base/c_kernels.pyx"],
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
            language="c++"
        ),

        # Eigen decomposition code
        Extension(
            name="pysph.base.linalg3",
            sources=["pysph/base/linalg3.pyx"],
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
            language="c++"
        ),
    ]
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

    MPI4PY_V2 = False if mpi4py.__version__.startswith('1.') else True
    cython_compile_time_env = {'MPI4PY_V2': MPI4PY_V2}

    zoltan_modules = [
        Extension(
            name="pyzoltan.core.zoltan",
            sources=["pyzoltan/core/zoltan.pyx"],
            depends=get_deps(
                "pyzoltan/core/carray",
                "pyzoltan/czoltan/czoltan",
                "pyzoltan/czoltan/czoltan_types",
            ),
            include_dirs = include_dirs+zoltan_include_dirs+mpi_inc_dirs,
            library_dirs = zoltan_library_dirs,
            libraries=['zoltan', 'mpi'],
            extra_link_args=mpi_link_args,
            extra_compile_args=mpi_compile_args+extra_compile_args,
            cython_compile_time_env=cython_compile_time_env,
            language="c++"
        ),

        Extension(
            name="pyzoltan.core.zoltan_dd",
            sources=["pyzoltan/core/zoltan_dd.pyx"],
            depends=get_deps(
                "pyzoltan/core/carray",
                "pyzoltan/czoltan/czoltan_dd",
                "pyzoltan/czoltan/czoltan_types"
            ),
            include_dirs = include_dirs + zoltan_include_dirs + mpi_inc_dirs,
            library_dirs = zoltan_library_dirs,
            libraries=['zoltan', 'mpi'],
            extra_link_args=mpi_link_args,
            extra_compile_args=mpi_compile_args+extra_compile_args,
            cython_compile_time_env=cython_compile_time_env,
            language="c++"
        ),

        Extension(
            name="pyzoltan.core.zoltan_comm",
            sources=["pyzoltan/core/zoltan_comm.pyx"],
            depends=get_deps(
                "pyzoltan/core/carray",
                "pyzoltan/czoltan/zoltan_comm"
            ),
            include_dirs = include_dirs + zoltan_include_dirs + mpi_inc_dirs,
            library_dirs = zoltan_library_dirs,
            libraries=['zoltan', 'mpi'],
            extra_link_args=mpi_link_args,
            extra_compile_args=mpi_compile_args+extra_compile_args,
            cython_compile_time_env=cython_compile_time_env,
            language="c++"
        ),
    ]

    parallel_modules = [

        Extension(
            name="pysph.parallel.parallel_manager",
            sources=["pysph/parallel/parallel_manager.pyx"],
            depends=get_deps("pyzoltan/core/carray",
                "pyzoltan/core/zoltan", "pyzoltan/core/zoltan_comm",
                "pysph/base/point", "pysph/base/particle_array",
                "pysph/base/nnps_base"
            ),
            include_dirs = include_dirs + mpi_inc_dirs + zoltan_include_dirs,
            library_dirs = zoltan_library_dirs,
            libraries = ['zoltan', 'mpi'],
            extra_link_args=mpi_link_args,
            extra_compile_args=mpi_compile_args+extra_compile_args,
            cython_compile_time_env=cython_compile_time_env,
            language="c++"
        ),
    ]
    return zoltan_modules + parallel_modules


def create_sources():
    if 'build_ext' in sys.argv or 'develop' in sys.argv or 'install' in sys.argv:
        generator = path.join('pyzoltan', 'core', 'generator.py')
        for pth in (path.join('pyzoltan', 'core'), path.join('pysph', 'base')):
            cmd = [sys.executable, generator, path.abspath(pth)]
            print(check_output(cmd).decode())


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
    module = path.join('pysph', '__init__.py')
    exec(compile(open(module).read(), module, 'exec'), info)

    # The requirements.
    install_requires = ['numpy', 'mako', 'Cython>=0.20', 'setuptools>=3.2']
    if sys.version_info[:2] == (2, 6):
        install_requires += [
            'ordereddict', 'importlib', 'unittest2'
        ]

    extras_require = dict(
        mpi=['mpi4py>=1.2'],
        ui=['mayavi>=4.0'],
        test=['nose>=1.0.0', 'mock>=1.0']
    )

    everything = set()
    for dep in extras_require.values():
        everything.update(dep)
    extras_require['all'] = everything

    ext_modules = get_basic_extensions() + get_parallel_extensions()


    setup(name='PySPH',
          version = info['__version__'],
          author = 'PySPH Developers',
          author_email = 'pysph-dev@googlegroups.com',
          description = "A general purpose Smoothed Particle Hydrodynamics framework",
          long_description = open('README.rst').read(),
          url = 'http://pysph.bitbucket.org',
          license = "BSD",
          keywords = "SPH simulation computational fluid dynamics",
          test_suite = "nose.collector",
          packages = find_packages(),
          package_data = {
              '': ['*.pxd', '*.mako', '*.txt.gz', '*.txt', '*.vtk.gz', '*.gz',
                  '*.rst', 'ndspmhd-sedov-initial-conditions.npz']
          },
          # exclude package data in installation.
          exclude_package_data = {
              '' : ['Makefile', '*.bat', '*.cfg', '*.rst', '*.sh', '*.yml'],
          },
          ext_modules = ext_modules,
          include_package_data = True,
          cmdclass = cmdclass,
          install_requires = install_requires,
          extras_require = extras_require,
          zip_safe = False,
          entry_points = """
              [console_scripts]
              pysph = pysph.tools.cli:main
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
            Programming Language :: Python :: 3
            Topic :: Scientific/Engineering
            Topic :: Scientific/Engineering :: Physics
            Topic :: Software Development :: Libraries
            """.splitlines() if len(c.split()) > 0],
          )

if __name__ == '__main__':
    setup_package()
