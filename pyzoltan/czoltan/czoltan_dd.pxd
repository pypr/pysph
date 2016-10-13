"""Cython wrapper for the Zoltan Distributed Directory"""

if MPI4PY_V2:
   from mpi4py.libmpi cimport MPI_Comm
else:
   from mpi4py.mpi_c cimport MPI_Comm

from czoltan_types cimport ZOLTAN_ID_TYPE, ZOLTAN_ID_PTR

cdef extern from "zoltan_dd.h":

    struct Zoltan_DD_Struct:
        pass

    ctypedef Zoltan_DD_Struct Zoltan_DD_Directory


    #/*********** Distributed Directory Function Prototypes ************/
    int Zoltan_DD_Create( Zoltan_DD_Directory** dd, MPI_Comm comm,
                          int num_gid, int num_lid, int user_length,
                          int table_length, int debug_level)

    void Zoltan_DD_Destroy( Zoltan_DD_Directory** dd)

    int Zoltan_DD_Update (Zoltan_DD_Directory *dd, ZOLTAN_ID_PTR gid,
                          ZOLTAN_ID_PTR lid, char *user, int *partition, int count)

    int Zoltan_DD_Find (Zoltan_DD_Directory *dd, ZOLTAN_ID_PTR gid,
                        ZOLTAN_ID_PTR lid, char *data, int *partition, int count,
                        int *owner)

    int Zoltan_DD_GetLocalKeys( Zoltan_DD_Directory* dd, ZOLTAN_ID_PTR* gid,
                                int* size )

    int Zoltan_DD_Remove (Zoltan_DD_Directory *dd, ZOLTAN_ID_PTR gid,
                          int count)

    void Zoltan_DD_Stats (Zoltan_DD_Directory *dd)

    int Zoltan_DD_Print (Zoltan_DD_Directory *dd)
