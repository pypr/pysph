"""Cython Wrapper for Zoltan. """

if MPI4PY_V2:
   from mpi4py.libmpi cimport MPI_Comm
else:
   from mpi4py.mpi_c cimport MPI_Comm

from czoltan_types cimport ZOLTAN_ID_PTR

cdef extern from "zoltan.h":

    # Zoltan version number
    float ZOLTAN_VERSION_NUMBER

    # ****************************************************************** #
    # *  Enumerated type used to indicate which function is to be set by
    # *  ZOLTAN_Set_Fn.
    enum Zoltan_Fn_Type:
        ZOLTAN_NUM_EDGES_FN_TYPE
        ZOLTAN_NUM_EDGES_MULTI_FN_TYPE
        ZOLTAN_EDGE_LIST_FN_TYPE
        ZOLTAN_EDGE_LIST_MULTI_FN_TYPE
        ZOLTAN_NUM_GEOM_FN_TYPE
        ZOLTAN_GEOM_MULTI_FN_TYPE
        ZOLTAN_GEOM_FN_TYPE
        ZOLTAN_NUM_OBJ_FN_TYPE
        ZOLTAN_OBJ_LIST_FN_TYPE
        ZOLTAN_FIRST_OBJ_FN_TYPE
        ZOLTAN_NEXT_OBJ_FN_TYPE
        ZOLTAN_NUM_BORDER_OBJ_FN_TYPE
        ZOLTAN_BORDER_OBJ_LIST_FN_TYPE
        ZOLTAN_FIRST_BORDER_OBJ_FN_TYPE
        ZOLTAN_NEXT_BORDER_OBJ_FN_TYPE
        ZOLTAN_PRE_MIGRATE_PP_FN_TYPE
        ZOLTAN_MID_MIGRATE_PP_FN_TYPE
        ZOLTAN_POST_MIGRATE_PP_FN_TYPE
        ZOLTAN_PRE_MIGRATE_FN_TYPE
        ZOLTAN_MID_MIGRATE_FN_TYPE
        ZOLTAN_POST_MIGRATE_FN_TYPE
        ZOLTAN_OBJ_SIZE_FN_TYPE
        ZOLTAN_PACK_OBJ_FN_TYPE
        ZOLTAN_UNPACK_OBJ_FN_TYPE
        ZOLTAN_NUM_COARSE_OBJ_FN_TYPE
        ZOLTAN_COARSE_OBJ_LIST_FN_TYPE
        ZOLTAN_FIRST_COARSE_OBJ_FN_TYPE
        ZOLTAN_NEXT_COARSE_OBJ_FN_TYPE
        ZOLTAN_NUM_CHILD_FN_TYPE
        ZOLTAN_CHILD_LIST_FN_TYPE
        ZOLTAN_CHILD_WEIGHT_FN_TYPE
        ZOLTAN_OBJ_SIZE_MULTI_FN_TYPE
        ZOLTAN_PACK_OBJ_MULTI_FN_TYPE
        ZOLTAN_UNPACK_OBJ_MULTI_FN_TYPE
        ZOLTAN_PART_FN_TYPE
        ZOLTAN_PART_MULTI_FN_TYPE
        ZOLTAN_PROC_NAME_FN_TYPE
        ZOLTAN_HG_SIZE_CS_FN_TYPE
        ZOLTAN_HG_CS_FN_TYPE
        ZOLTAN_HG_SIZE_EDGE_WTS_FN_TYPE
        ZOLTAN_HG_EDGE_WTS_FN_TYPE
        ZOLTAN_NUM_FIXED_OBJ_FN_TYPE
        ZOLTAN_FIXED_OBJ_LIST_FN_TYPE
        ZOLTAN_HIER_NUM_LEVELS_FN_TYPE
        ZOLTAN_HIER_PART_FN_TYPE
        ZOLTAN_HIER_METHOD_FN_TYPE
        ZOLTAN_MAX_FN_TYPES

    ctypedef Zoltan_Fn_Type ZOLTAN_FN_TYPE

    # ************************************************************************
    # * Enumerated type used to indicate what type of refinement was used when
    # * building a refinement tree.
    # */
    enum Zoltan_Ref_Type:
        ZOLTAN_OTHER_REF
        ZOLTAN_IN_ORDER
        ZOLTAN_TRI_BISECT
        ZOLTAN_QUAD_QUAD
        ZOLTAN_HEX3D_OCT

    ctypedef Zoltan_Ref_Type ZOLTAN_REF_TYPE

    # /***********************************************************************
    # /***********************************************************************
    # /*******  Functions to set-up Zoltan load-balancing data structure  ****
    # /***********************************************************************
    # /***********************************************************************

    # /*
    #  *  Function to initialize values needed in load balancing tools, and
    #  *  returns which version of the library this is. If the application
    #  *  uses MPI, call this function after calling MPI_Init. If the
    #  *  application does not use MPI, this function calls MPI_Init for
    #  *  use by Zoltan. This function returns the version of
    #  *  the Zoltan library.
    #  *  Input:
    #  *    argc                --  Argument count from main()
    #  *    argv                --  Argument list from main()
    #  *  Output:
    #  *    ver                 --  Version of Zoltan library
    #  *  Returned value:       --  Error code
    #  */
    extern int Zoltan_Initialize(int, char**, float* ver)

    # Zoltan structure
    struct Zoltan_Struct:
        pass

    # function to create a zoltan structure
    extern Zoltan_Struct* Zoltan_Create(MPI_Comm)

    # /*****************************************************************************/
    # /*
    # *  Function to free the space associated with a Zoltan structure.
    # *  The input pointer is set to NULL when the routine returns.
    # *  Input/Output:
    # *    zz                  --  Pointer to a Zoltan structure.
    # */

    extern void Zoltan_Destroy(
        Zoltan_Struct **zz
        )

    # /*****************************************************************************/
    # /*
    #  *  Function to change a parameter value within Zoltan.
    #  *  Default values will be used for all parameters not explicitly altered
    #  *  by a call to this routine.
    #  *
    #  *  Input
    #  *    zz                  --  The Zoltan structure to which this
    #  *                            parameter alteration applies.
    #  *    name                --  The name of the parameter to have its
    #  *                            value changed.
    #  *    val                 --  The new value of the parameter.
    #  *
    #  *  Returned value:       --  Error code
    #  */
    extern int Zoltan_Set_Param(
        Zoltan_Struct *zz, char *name, char *val )

    #     /*****************************************************************************/
    # /*
    #  *  Function to return, for the calling processor, the number of objects
    #  *  located in that processor's memory.
    #  *  Input:
    #  *    data                --  pointer to user defined data structure
    #  *  Output:
    #  *    ierr                --  error code
    #  *  Returned value:       --  the number of local objects.
    #  */

    ctypedef int ZOLTAN_NUM_OBJ_FN(
        void *data,
        int *ierr
        )

    extern int Zoltan_Set_Num_Obj_Fn(
        Zoltan_Struct *zz, ZOLTAN_NUM_OBJ_FN *fn_ptr, void *data_ptr)

    # /*****************************************************************************/
    # /*
    #  *  Function to return a list of all local objects on a processor.
    #  *  Input:
    #  *    data                --  pointer to user defined data structure
    #  *    num_gid_entries     --  number of array entries of type ZOLTAN_ID_TYPE
    #  *                            in a global ID
    #  *    num_lid_entries     --  number of array entries of type ZOLTAN_ID_TYPE
    #  *                            in a local ID
    #  *    wdim                --  dimension of object weights, or 0 if
    #  *                            object weights are not sought.
    #  *  Output:
    #  *    global_ids          --  array of Global IDs of all objects on the
    #  *                            processor.
    #  *    local_ids           --  array of Local IDs of all objects on the
    #  *                            processor.
    #  *    objwgts             --  objwgts[i*wdim:(i+1)*wdim-1] correponds
    #  *                            to the weight of object i
    #  *    ierr                --  error code
    #  */

    ctypedef void ZOLTAN_OBJ_LIST_FN(
        void *data,
        int num_gid_entries,
        int num_lid_entries,
        ZOLTAN_ID_PTR global_ids,
        ZOLTAN_ID_PTR local_ids,
        int wdim,
        float *objwgts,
        int *ierr
        )

    extern int Zoltan_Set_Obj_List_Fn(
        Zoltan_Struct *zz,
        ZOLTAN_OBJ_LIST_FN *fn_ptr,
        void *data_ptr
        )

    # /*****************************************************************************/
    # /*
    #  *  Function to return
    #  *  the number of geometry fields per object (e.g., the number of values
    #  *  used to express the coordinates of the object).
    #  *  Input:
    #  *    data                --  pointer to user defined data structure
    #  *  Output:
    #  *    ierr                --  error code
    #  *  Returned value:       --  the number of geometry fields.
    #  */

    # A ZOLTAN_NUM_GEOM_FN query function returns the number of values
    # needed to express the geometry of an object. For example, for a
    # two-dimensional mesh-based application, (x,y) coordinates are needed
    # to describe an object's geometry; thus the ZOLTAN_NUM_GEOM_FN query
    # function should return the value of two. For a similar
    # three-dimensional application, the return value should be three.

    ctypedef int ZOLTAN_NUM_GEOM_FN(
        void *data,
        int *ierr
        )

    extern int Zoltan_Set_Num_Geom_Fn(
        Zoltan_Struct *zz,
        ZOLTAN_NUM_GEOM_FN *fn_ptr,
        void *data_ptr
        )

    # /*****************************************************************************/

    # A ZOLTAN_GEOM_MULTI FN query function returns a vector of geometry
    # values for a list of given objects. The geometry vector is
    # allocated by Zoltan to be of size num_obj * num_dim; its format is
    # described below

    # /*
    # *  Function to return the geometry information (e.g., coordinates) for
    # *  all objects.
    # *  Input:
    #     *    data                --  pointer to user defined data structure
    #     *    num_gid_entries     --  number of array entries of type ZOLTAN_ID_TYPE
    #     *                            in a global ID
    #     *    num_lid_entries     --  number of array entries of type ZOLTAN_ID_TYPE
    #     *                            in a local ID
    #     *    num_obj             --  number of objects whose coordinates are needed.
    #     *    global_id           --  array of Global IDs for the objects
    #     *    local_id            --  array of Local IDs for the objects;
    #     *                            NULL if num_lid_entries == 0.
    #     *    num_dim             --  dimension of coordinates for each object.
    #     *  Output:
    #     *    geom_vec            --  the geometry info for the objects;
    #     *                            (e.g., coordinates)
    #     *                            If num_dim == n, geom_vec[i*n+j] is the
    #     *                            jth coordinate for object i.
    #     *    ierr                --  error code
    #     */

    ctypedef void ZOLTAN_GEOM_MULTI_FN(
        void *data,
        int num_gid_entries,
        int num_lid_entries,
        int num_obj,
        ZOLTAN_ID_PTR global_id,
        ZOLTAN_ID_PTR local_id,
        int num_dim,
        double *geom_vec,
        int *ierr
        )

    extern int Zoltan_Set_Geom_Multi_Fn(
        Zoltan_Struct *zz,
        ZOLTAN_GEOM_MULTI_FN *fn_ptr,
        void *data_ptr
        )

    # /*****************************************************************************/
    # /*
    #  *  Function to invoke the partitioner.
    #  *
    #  *  Input:
    #  *    zz                  --  The Zoltan structure returned by Zoltan_Create.
    #  *  Output:
    #  *    changes             --  This value tells whether the new
    #  *                            decomposition computed by Zoltan differs
    #  *                            from the one given as input to Zoltan.
    #  *                            It can be either a one or a zero:
    #  *                            zero - No changes to the decomposition
    #  *                                   were made by the partitioning
    #  *                                   algorithm; migration isn't needed.
    #  *                            one  - A new decomposition is suggested
    #  *                                   by the partitioner; migration
    #  *                                   is needed to establish the new
    #  *                                   decomposition.
    #  *    num_gid_entries     --  number of entries of type ZOLTAN_ID_TYPE
    #  *                            in a global ID
    #  *    num_lid_entries     --  number of entries of type ZOLTAN_ID_TYPE
    #  *                            in a local ID
    #  *    num_import          --  The number of non-local objects in the
    #  *                            processor's new decomposition (i.e.,
    #  *                            number of objects to be imported).
    #  *    import_global_ids   --  Pointer to array of Global IDs for the
    #  *                            objects to be imported.
    #  *    import_local_ids    --  Pointer to array of Local IDs for the
    #  *                            objects to be imported (local to the
    #  *                            exporting processor).
    #  *    import_procs        --  Pointer to array of Processor IDs for the
    #  *                            objects to be imported (processor IDs of
    #  *                            source processor).
    #  *    import_to_part      --  Pointer to array of partition numbers to
    #  *                            which the imported objects should be assigned.
    #  *    num_export          --  The number of local objects that must be
    #  *                            exported from the processor to establish
    #  *                            the new decomposition.
    #  *    export_global_ids   --  Pointer to array of Global IDs for the
    #  *                            objects to be exported from the current
    #  *                            processor.
    #  *    export_local_ids    --  Pointer to array of Local IDs for the
    #  *                            objects to be exported (local to the
    #  *                            current processor).
    #  *    export_procs        --  Pointer to array of Processor IDs for the
    #  *                            objects to be exported (processor IDs of
    #  *                            destination processors).
    #  *    export_to_part      --  Pointer to array of partition numbers to
    #  *                            which the exported objects should be assigned.
    #  *  Returned value:       --  Error code
    #  */
    extern int Zoltan_LB_Partition(
        Zoltan_Struct *zz,
        int *changes,
        int *num_gid_entries,
        int *num_lid_entries,
        int *num_import,
        ZOLTAN_ID_PTR *import_global_ids,
        ZOLTAN_ID_PTR *import_local_ids,
        int **import_procs,
        int **import_to_part,
        int *num_export,
        ZOLTAN_ID_PTR *export_global_ids,
        ZOLTAN_ID_PTR *export_local_ids,
        int **export_procs,
        int **export_to_part
        )

    # /*****************************************************************************/
    # /*
    #  *  Function to invoke the load-balancer.
    #  *  Appropriate only when the number of requested partitions is equal to the
    #  *  number of processors.
    #  *
    #  *  Input and output:
    #  *    Arguments are analogous to Zoltan_LB_Partition.  Arrays import_to_part
    #  *    and export_to_part are not included, as Zoltan_LB_Balance assumes
    #  *    partitions and processors are equivalent.
    #  *  Returned value:       --  Error code
    #  */
    extern int Zoltan_LB_Balance(
        Zoltan_Struct *zz,
        int *changes,
        int *num_gid_entries,
        int *num_lid_entries,
        int *num_import,
        ZOLTAN_ID_PTR *import_global_ids,
        ZOLTAN_ID_PTR *import_local_ids,
        int **import_procs,
        int *num_export,
        ZOLTAN_ID_PTR *export_global_ids,
        ZOLTAN_ID_PTR *export_local_ids,
        int **export_procs
        )

    # /*****************************************************************************/
    # /*
    #  *  Routine to free the data arrays returned by Zoltan_Balance.  The arrays
    #  *  are freed and the pointers are set to NULL.
    #  *
    #  *  Input:
    #  *    import_global_ids   --  Pointer to array of global IDs for
    #  *                            imported objects.
    #  *    import_local_ids    --  Pointer to array of local IDs for
    #  *                            imported objects.
    #  *    import_procs        --  Pointer to array of processor IDs of
    #  *                            imported objects.
    #  *    export_global_ids   --  Pointer to array of global IDs for
    #  *                            exported objects.
    #  *    export_local_ids    --  Pointer to array of local IDs for
    #  *                            exported objects.
    #  *    export_procs        --  Pointer to array of destination processor
    #  *                            IDs of exported objects.
    #  *  Returned value:       --  Error code
    #  */

    extern int Zoltan_LB_Free_Data(
        ZOLTAN_ID_PTR *import_global_ids,
        ZOLTAN_ID_PTR *import_local_ids,
        int **import_procs,
        ZOLTAN_ID_PTR *export_global_ids,
        ZOLTAN_ID_PTR *export_local_ids,
        int **export_procs
        )

    # /*****************************************************************************/
    # /*
    #  *  Function to return the bounding box of a partition generated by RCB.
    #  *  Input:
    #  *    zz                  --  The Zoltan structure returned by Zoltan_Create.
    #  *    part                --  The partition number whose bounding box is to
    #  *                            be returned.
    #  *  Output:
    #  *    ndim                --  Number of dimensions in the bounding box.
    #  *    xmin                --  lower x extent of box
    #  *    ymin                --  lower y extent of box
    #  *    zmin                --  lower z extent of box
    #  *    xmax                --  upper x extent of box
    #  *    ymax                --  upper y extent of box
    #  *    zmax                --  upper z extent of box
    #  *  Returned value:       --  Error code
    #  */

    int Zoltan_RCB_Box(
        Zoltan_Struct *zz,
        int     part,
        int    *ndim,
        double *xmin,
        double *ymin,
        double *zmin,
        double *xmax,
        double *ymax,
        double *zmax
        )

    # /*****************************************************************************/
    # /*
    #  * Routine to determine which partitions and processors
    #  * a bounding box intersects.
    #  * Note that this only works of the current partition was produced via a
    #  * geometric algorithm - currently RCB, RIB, HSFC.
    #  *
    #  * Input:
    #  *   zz                   -- pointer to Zoltan structure
    #  *   xmin, ymin, zmin     -- lower left corner of bounding box
    #  *   xmax, ymax, zmax     -- upper right corner of bounding box
    #  *
    #  * Output:
    #  *   procs                -- list of processors that box intersects.
    #  *                           Note: application is
    #  *                               responsible for ensuring sufficient memory.
    #  *   numprocs             -- number of processors box intersects
    #  *   parts                -- list of partitions that box intersects.
    #  *                           Note: application is
    #  *                               responsible for ensuring sufficient memory.
    #  *   numparts             -- number of partitions box intersects (may differ
    #  *                           from numprocs).
    #  *
    #  * Returned value:       --  Error code
    #  */

    extern int Zoltan_LB_Box_PP_Assign(
        Zoltan_Struct *zz,
        double xmin,
        double ymin,
        double zmin,
        double xmax,
        double ymax,
        double zmax,
        int *procs,
        int *numprocs,
        int *parts,
        int *numparts
        )

    # /*****************************************************************************/
    # /*
    #  * Routine to determine which processor and partition a new point should be
    #  * assigned to.
    #  * Note that this only works of the current partition was produced via a
    #  * geometric algorithm - currently RCB, RIB, HSFC.
    #  *
    #  * Input:
    #  *   zz                   -- pointer to Zoltan structure
    #  *   coords               -- vector of coordinates of new point
    #  *
    #  * Output:
    #  *   proc                 -- processor that point should be assigned to
    #  *   part                 -- partition that point should be assigned to
    #  *
    #  *  Returned value:       --  Error code
    #  */

    extern int Zoltan_LB_Point_PP_Assign(
        Zoltan_Struct *zz,
        double *coords,
        int *proc,
        int *part
        )

    extern int Zoltan_LB_Point_Assign(
        Zoltan_Struct *zz,
        double *coords,
        int *proc
        )

    # /*****************************************************************************/
    # /*
    #  *  Routine to compute the inverse map:  Given, for each processor, a list
    #  *  of objects to be received by a processor, compute the list of objects
    #  *  that processor needs to send to other processors to complete a
    #  *  remapping.  Conversely, given a list of objects to be sent to other
    #  *  processors, compute the list of objects to be received.
    #  *
    #  *  Input:
    #  *    zz                  --  Zoltan structure for current
    #  *                            balance.
    #  *    num_input           --  Number of objects known to be
    #  *                            sent/received.
    #  *    input_global_ids    --  Array of global IDs for known objects.
    #  *    input_local_ids     --  Array of local IDs for known objects.
    #  *    input_procs         --  Array of IDs of processors to/from whom the
    #  *                            known objects will be sent/received.
    #  *    input_to_part       --  Array of partition numbers to
    #  *                            which the known objects should be assigned.
    #  *  Output:
    #  *    num_output          --  The number of objects will be received/sent.
    #  *    output_global_ids   --  Pointer to array of Global IDs for the
    #  *                            objects to be received/sent.
    #  *    output_local_ids    --  Pointer to array of Local IDs for the
    #  *                            objects to be received/sent.
    #  *    output_procs        --  Pointer to array of Processor IDs
    #  *                            from/to which the output_global_ids will be
    #  *                            received/sent.
    #  *    output_to_part      --  Pointer to array of partition numbers to
    #  *                            which the output_global_ids should be assigned.
    #  *  Returned value:       --  Error code
    #  */

    extern int Zoltan_Invert_Lists(
        Zoltan_Struct *zz,
        int num_input,
        ZOLTAN_ID_PTR input_global_ids,
        ZOLTAN_ID_PTR input_local_ids,
        int *input_procs,
        int *input_to_part,
        int *num_output,
        ZOLTAN_ID_PTR *output_global_ids,
        ZOLTAN_ID_PTR *output_local_ids,
        int **output_procs,
        int **output_to_part
        )

    # /*****************************************************************************/
    # /*
    # *  Wrapper around Zoltan_Invert_Lists, appropriate only when
    # *  number of partitions == number of processors (or when partition information
    #                                                  *  is not desired).
    # *
    # *  Input and Output:
    # *    Arguments are analogous to Zoltan_Invert_Lists.  Arrays import_to_part
    # *    and export_to_part are not included, as Zoltan_Compute_Destinations
    # *    assumes partitions and processors are equivalent.
    # *  Returned value:       --  Error code
    # */
    extern int Zoltan_Compute_Destinations(
        Zoltan_Struct *zz,
        int num_input,
        ZOLTAN_ID_PTR input_global_ids,
        ZOLTAN_ID_PTR input_local_ids,
        int *input_procs,
        int *num_output,
        ZOLTAN_ID_PTR *output_global_ids,
        ZOLTAN_ID_PTR *output_local_ids,
        int **output_procs
        )

    # /*****************************************************************************/
    # /*
    #  *  Routine to free the data arrays returned by Zoltan_LB_Partition,
    #  *  Zoltan_LB_Balance, Zoltan_Invert_Lists, and
    #  *  Zoltan_Compute_Destinations.  The arrays
    #  *  are freed and the pointers are set to NULL.
    #  *
    #  *  Input:
    #  *    global_ids   --  Pointer to array of global IDs
    #  *    local_ids    --  Pointer to array of local IDs
    #  *    procs        --  Pointer to array of processor IDs
    #  *    to_proc      --  Pointer to array of partition assignments
    #  *  Returned value:       --  Error code
    #  */
    extern int Zoltan_LB_Free_Part(
        ZOLTAN_ID_PTR *global_ids,
        ZOLTAN_ID_PTR *local_ids,
        int **procs,
        int **to_part
        )
