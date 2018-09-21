def get_include():
    """Return the directory in the package that contains header files."""
    from os.path import dirname, abspath, join
    return abspath(join(dirname(__file__), '../'))


_has_mpi = None


def has_mpi():
    """Return True if mpi4py is available.
    """
    global _has_mpi
    if _has_mpi is None:
        _has_mpi = True
        try:
            import mpi4py  # noqa: 401
        except ImportError:
            _has_mpi = False
        else:
            mpi4py.rc.initialize = False
            mpi4py.rc.finalize = True
    return _has_mpi


# Call this to disable mpi4py from initializing MPI on import.
has_mpi()
