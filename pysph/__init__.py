# See PEP 440 for more on suitable version numbers.
__version__ = '1.0a3'

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
