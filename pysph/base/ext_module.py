# Standard library imports
from contextlib import contextmanager
from distutils.sysconfig import get_config_var
from distutils.util import get_platform
from distutils.errors import CompileError, LinkError
import hashlib
import imp
import importlib
import numpy
import os
from os.path import dirname, exists, expanduser, isdir, join
from pyximport import pyxbuild
import shutil
import sys
import time

# Conditional/Optional imports.
if sys.platform == 'win32':
    from setuptools.extension import Extension
else:
    from distutils.extension import Extension

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

# Package imports.
import pysph
from pysph.base.config import get_config
from pysph.base.capture_stream import CaptureMultipleStreams


def get_platform_dir():
    return 'py{version}-{platform_dir}'.format(
        version=sys.version[:3], platform_dir=get_platform()
    )

def get_md5(data):
    """Return the MD5 sum of the given data.
    """
    return hashlib.md5(data.encode()).hexdigest()


class ExtModule(object):
    """Encapsulates the generated code, extension module etc.
    """
    def __init__(self, src, extension='pyx', root=None, verbose=False,
                 depends=None):
        """Initialize ExtModule.

        Parameters
        -----------

        src : str : source code.

        ext : str : extension for source code file.
            Do not specify the '.' (defaults to 'pyx').

        root : str: root of directory to store code and modules in.
            If not set it defaults to "~/.pysph/source/<platform-directory>".
            where <platform-directory> is platform specific.

        verbose : Bool : Print messages for convenience.

        depends : list : a list of modules that this extension depends on
            if any of these have an m_time greater than the compiled extension
            module, the extension will be recompiled.

        """
        self._setup_root(root)
        self.code = src
        self.hash = get_md5(src)
        self.extension = extension
        self.name = 'm_{0}'.format(self.hash)
        self._setup_filenames()
        self.verbose = verbose
        self.depends = depends

        if MPI is not None:
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.num_procs = self.comm.Get_size()
        else:
            self.rank = 0
            self.num_procs = 1

        self.shared_filesystem = False
        self._create_source()

    def _setup_filenames(self):
        base = self.name
        self.src_path = join(self.root, base + '.' + self.extension)
        self.ext_path = join(self.root, base + get_config_var('SO'))
        self.lock_path = join(self.root, base + '.lock')

    @contextmanager
    def _lock(self, timeout=90):
        t1 = time.time()
        def _is_timed_out():
            if timeout is None:
                return False
            else:
                return (time.time() - t1) > timeout
        def _try_to_lock():
            if not exists(self.lock_path):
                try:
                    os.mkdir(self.lock_path)
                except OSError:
                    return False
                else:
                    return True
            return False

        while not _try_to_lock():
            time.sleep(0.1)
            if _is_timed_out():
                break
        try:
            yield
        finally:
            os.rmdir(self.lock_path)

    def _create_source(self):
        # Create the source.
        if self.rank == 0:
            with self._lock():
                self._write_source(self.src_path)
        if self.num_procs > 1:
            self.comm.barrier()
            if not exists(self.src_path):
                # Not a shared filesystem so append rank to filename.
                # This is needed since there may be other nodes using the same
                # filesystem (multi-core CPUs) whose rank is non-zero.
                self.name = 'm_{0}_{1}'.format(self.hash, self.rank)
                self._setup_filenames()
                with self._lock():
                    self._write_source(self.src_path)
            else:
                self.shared_filesystem = True

    def _write_source(self, path):
        if not exists(path):
            with open(path, 'w') as f:
                f.write(self.code)

    def _setup_root(self, root):
        if root is None:
            plat_dir = get_platform_dir()
            self.root = expanduser(join('~', '.pysph', 'source', plat_dir))
        else:
            self.root = root

        self.build_dir = join(self.root, 'build')

        if not isdir(self.build_dir):
            try:
                os.makedirs(self.build_dir)
            except OSError:
                # The directory was created at the same time by another process.
                pass

    def _dependencies_have_changed(self):
        depends = self.depends
        if not depends:
            return False
        else:
            ext_mtime = os.stat(self.ext_path).st_mtime
            for name in depends:
                try:
                    mod = importlib.import_module(name)
                    mod_mtime = os.stat(mod.__file__).st_mtime
                    if ext_mtime < mod_mtime:
                        return True
                except ImportError:
                    pass
            return False

    def should_recompile(self):
        if not exists(self.ext_path):
            return True
        elif self._dependencies_have_changed():
            return True
        else:
            return False

    def build(self, force=False):
        """Build source into an extension module.  If force is False
        previously compiled module is returned.
        """
        if not self.shared_filesystem or self.rank == 0:
            with self._lock():
                if force or self.should_recompile():
                    self._message("Compiling code at:", self.src_path)
                    inc_dirs = [numpy.get_include()]
                    # Add pysph/base directory to inc_dirs for including spatial_hash.h
                    # for SpatialHashNNPS
                    inc_dirs.append(os.path.dirname(os.path.realpath(__file__)))
                    extra_compile_args, extra_link_args = self._get_extra_args()

                    extension = Extension(
                        name=self.name, sources=[self.src_path],
                        include_dirs=inc_dirs,
                        extra_compile_args=extra_compile_args,
                        extra_link_args=extra_link_args,
                        language="c++"
                    )

                    if not hasattr(sys.stdout, 'errors'):
                        # FIXME: This happens when nosetests replaces the
                        # stdout with the a Tee instance.  This Tee instance
                        # does not have errors which breaks the tests so we
                        # disable verbose reporting.
                        script_args = []
                    else:
                        script_args = ['--verbose']
                    try:
                        with CaptureMultipleStreams() as stream:
                            mod = pyxbuild.pyx_to_dll(self.src_path, extension,
                                pyxbuild_dir=self.build_dir, force_rebuild=True,
                                setup_args={'script_args': script_args}
                            )
                    except (CompileError, LinkError):
                        hline = "*"*80
                        print(hline + "\nERROR")
                        print(stream.get_output()[0])
                        print(stream.get_output()[1])
                        msg = "Compilation of code failed, please check "\
                                "error messages above."
                        print(hline + "\n" + msg)
                        sys.exit(1)
                    shutil.copy(mod, self.ext_path)
                else:
                    self._message("Precompiled code from:", self.src_path)
        if MPI is not None:
            self.comm.barrier()

    def load(self):
        """Build and load the built extension module.

        Returns
        """
        self.build()
        file, path, desc = imp.find_module(self.name, [dirname(self.ext_path)])
        return imp.load_module(self.name, file, path, desc)

    def _get_extra_args(self):
        if get_config().use_openmp:
            if sys.platform == 'win32':
                return ['/openmp'], []
            else:
                return ['-fopenmp'], ['-fopenmp']
        else:
            return [], []

    def _message(self, *args):
        if self.verbose:
            print(' '.join(args))
