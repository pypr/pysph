"""Cython wrapper for the Zoltan unstructured communication package"""

if MPI4PY_V2:
   from mpi4py.libmpi cimport MPI_Comm
else:
   from mpi4py.mpi_c cimport MPI_Comm


cdef extern from "zoltan_comm.h":

    struct Zoltan_Comm_Obj:
        pass

    ctypedef Zoltan_Comm_Obj ZOLTAN_COMM_OBJ

    #/* function prototypes */

    int Zoltan_Comm_Create(ZOLTAN_COMM_OBJ**, int, int*, MPI_Comm, int, int*)

    int Zoltan_Comm_Destroy(ZOLTAN_COMM_OBJ**)

    int Zoltan_Comm_Do     (ZOLTAN_COMM_OBJ*, int, char*, int, char*)
    int Zoltan_Comm_Do_Post(ZOLTAN_COMM_OBJ*, int, char*, int, char*)
    int Zoltan_Comm_Do_Wait(ZOLTAN_COMM_OBJ*, int, char*, int, char*)
    int Zoltan_Comm_Do_AlltoAll(ZOLTAN_COMM_OBJ*, char*, int, char*)

    int Zoltan_Comm_Do_Reverse     (ZOLTAN_COMM_OBJ*, int, char*, int, int*, char*)
    int Zoltan_Comm_Do_Reverse_Post(ZOLTAN_COMM_OBJ*, int, char*, int, int*, char*)
    int Zoltan_Comm_Do_Reverse_Wait(ZOLTAN_COMM_OBJ*, int, char*, int, int*, char*)

    int Zoltan_Comm_Info(ZOLTAN_COMM_OBJ*, int*, int*, int*, int*, int*, int*, int*,
                         int*, int*, int*, int*, int*, int*)

    int Zoltan_Comm_Invert_Plan(ZOLTAN_COMM_OBJ**)
