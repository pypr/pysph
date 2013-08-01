cdef extern from "zoltan_types.h":

    # basic type used by all of Zoltan
    ctypedef unsigned int ZOLTAN_ID_TYPE

    # MPI data type
    cdef unsigned int ZOLTAN_ID_MPI_TYPE

    # pointer to the basic type
    ctypedef ZOLTAN_ID_TYPE* ZOLTAN_ID_PTR

    # /*****************************************************************************/
    # /*
    #  * Error codes for Zoltan library
    #  *   ZOLTAN_OK     - no errors
    #  *   ZOLTAN_WARN   - some warning occurred in Zoltan library;
    #  *                   application should be able to continue running
    #  *   ZOLTAN_FATAL  - a fatal error occurred
    #  *   ZOLTAN_MEMERR - memory allocation failed; with this error, it could be
    #  *                   possible to try a different, more memory-friendly,
    #  *                   algorithm.
    #  */
    # /*****************************************************************************/
    cdef int ZOLTAN_OK
    cdef int ZOLTAN_WARN
    cdef int ZOLTAN_FATAL
    cdef int ZOLTAN_MEMERR

    # /*****************************************************************************/
    # /* Hypergraph query function types
    #  */
    # /*****************************************************************************/
    cdef int _ZOLTAN_COMPRESSED_EDGE
    cdef int _ZOLTAN_COMPRESSED_VERTEX
