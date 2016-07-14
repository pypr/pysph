# See PEP 440 for more on suitable version numbers.
__version__ = '1.0a4'

# Utility functions to determine if Zoltan/MPI are available.
_has_mpi = None
_has_zoltan = None
_in_parallel = None


def has_mpi():
    """Return True if mpi4py is available.
    """
    global _has_mpi
    if _has_mpi is None:
        _has_mpi = True
        try:
            import mpi4py
        except ImportError:
            _has_mpi = False
    return _has_mpi

def has_zoltan():
    """Return True if zoltan is available.
    """
    global _has_zoltan
    if _has_zoltan is None:
        _has_zoltan = True
        try:
            from pyzoltan.core import zoltan
        except ImportError:
            _has_zoltan = False
    return _has_zoltan

def in_parallel():
    """Return true if we're running with MPI and Zoltan support
    """
    global _in_parallel
    if _in_parallel is None:
        _in_parallel = has_mpi() and has_zoltan()

    return _in_parallel

# Utility function to determine the possible output files

_has_h5py = None
_has_pyvisfile = None
_has_tvtk = None


def has_h5py():
    """Return True if h5py is available.
    """
    global _has_h5py
    if _has_h5py is None:
        _has_h5py = True
        try:
            import h5py
        except ImportError:
            _has_h5py = False
    return _has_h5py

def has_tvtk():
    """Return True if tvtk is available.
    """
    global _has_tvtk
    if _has_tvtk is None:
        _has_tvtk = True
        try:
            import tvtk
        except ImportError:
            _has_tvtk = False
    return _has_tvtk

def has_pyvisfile():
    """Return True if pyvisfile is available.
    """
    global _has_pyvisfile
    if _has_pyvisfile is None:
        _has_pyvisfile = True
        try:
            import pyvisfile
        except ImportError:
            _has_pyvisfile = False
    return _has_pyvisfile
