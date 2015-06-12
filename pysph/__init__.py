# See PEP 440 for more on suitable version numbers.
__version__ = '1.0a2'


# Conditional Imports for Parallel stuff

# Check for MPI
Has_MPI=True
try:
    import mpi4py
except ImportError:
    Has_MPI=False

# Check for Zoltan
Has_Zoltan=True
try:
    from pyzoltan.core import zoltan
except ImportError:
    # We switch off parallel mode
    Has_Zoltan=False
    Has_MPI=False
