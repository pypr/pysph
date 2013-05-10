# Conditional Imports for Parallel stuff

Has_MPI=True
try:
    import mpi4py
except ImportError:
    Has_MPI=False

Has_Zoltan=True
try:
    import pyzoltan
except ImportError:
    Has_Zoltan=False
