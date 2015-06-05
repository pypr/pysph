"""NNPS utility functions to work with Zoltan lists"""
import numpy

from pyzoltan.core.zoltan import get_zoltan_id_type_max
from pysph.base.particle_array import ParticleArray

UINT_MAX = get_zoltan_id_type_max()

def invert_export_lists(comm, exportProcs, recv_count):
    """Invert a given set of export indices.

    Parameters:
    ------------

    comm : mpi4py.MPI.Comm
        A valid MPI communicator

    exportProcs : IntArray
        A list of processors to send objects to

    recv_count : np.ndarray (out)
        Return array of length size which upon output, gives the number of
        objects to be received from a given processor.

    Given a list of objects that need to be exported to remote processors,
    the job of invert lists is to inform each processor the number of
    objects it will receive from other processors. This situation arises
    for example in the cell based partitioning in PySPH. From the cell
    export lists, we have a list of particle indices that need to be
    exported to remote neighbors.

    """
    # reset the recv_counts to 0
    recv_count[:] = 0

    # get the rank and size for the communicator
    size = comm.Get_size()
    rank = comm.Get_rank()

    # count the number of objects we need to send to each processor
    send_count = np.zeros(shape=size, dtype=np.uint32)
    numExport = exportProcs.length

    for i in range(numExport):
        pid = exportProcs[i]
        send_count[pid] += 1

    # receive buffer for all gather
    recvbuf = np.zeros(shape=size*size, dtype=np.uint32)

    # do an all gather to receive the data
    comm.Allgather(sendbuf=send_count, recvbuf=recvbuf)

    # store the number of objects to be received from each processor
    for i in range(size):
        proc_send_count = recvbuf[i*size:(i+1)*size]
        recv_count[i] = proc_send_count[rank]

def count_recv_data(
    comm, recv, numImport, importProcs):
    """Count the data to be received from different processors.

    Parameters:
    -----------

    comm : mpi.Comm
        MPI communicator

    recv : dict
        Upon output, will contain keys corresponding to processors and
        values indicating number of objects to receive from that proc.

    numImport : int
        Zoltan generated total number of objects to be imported
        to the calling proc

    importProcs : DoubleArray
        Zoltan generated list for processors from where objects are
        to be received.

    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    recv.clear()
    for processor in range(size):
        recv[processor] = 0

    for i in range(numImport):
        processor = importProcs[i]
        recv[processor] += 1

    for processor in recv.keys():
        if recv[processor] == 0:
            del recv[processor]

def get_send_data(
    comm, pa, lb_props, _exportIndices, _exportProcs):
    """Collect the data to send in a dictionary.

    Parameters:
    -----------

    comm : mpi.Comm
        MPI communicator

    pa : ParticleArray
        Reference to the particle array from where send data is gathered

    lb_props : list
        A list of prop names to collect data

    _exportIndices : UIntArray
        Zoltan generated list of local indices to export

    _exportProcs : IntArray
        Zoltan generated list of processors to export to

    Returns a dictionary of dictionaries 'send' which is keyed on
    processor id and with values a dictionary of prop names and
    corresponding particle data.

    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    procs = _exportProcs.get_npy_array()
    exportIndices = _exportIndices.get_npy_array()

    props = {}
    for prop in lb_props:
        props[prop] = pa.get_carray(prop).get_npy_array()

    send = {}
    for pid in range(size):
        indices = numpy.where( procs == pid )[0]
        #if len(indices) > 0:
        send[pid] = {}
        for prop, prop_array in props.items():
            send[pid][prop] = prop_array[ exportIndices[indices] ]

        # save the local ids exported to each processor
        send[pid]['lid'] = exportIndices[indices]
        send[pid]['msglength'] = exportIndices[indices].size

    return send

def Recv(comm, localbuf, recvbuf, source, localbufsize=0, tag=0):
    """MPI Receive operation

    Parameters:
    -----------

    comm : mpi.Comm
        The mpi communcator

    localbuf : CArray
        The local buffer to which the data is received in

    recvbuf : CArray
        the buffer in which to receive data from comm.Recv

    source : int
        processor from where the data originates

    localbufsize : int
        Current length index for the local buffer. Defaults to 0

    tag : int
        optional message tag

    For the situation in which we receive data from multiple
    processors to be stored in a single array (localbuf), we receive
    the data in 'recvbuf' and then add it to the correct indices using
    a pointer to the current index (localbufsize) and the message
    length (recvbuf.length)

    """
    # get the message length. we assume this is known before actually
    # doing the receive.
    msglength = recvbuf.length

    # get the Numpy buffer for the C-arrays
    _localbuf = localbuf.get_npy_array()
    _recvbuf = recvbuf.get_npy_array()

    # Receive the Numpy buffer from source
    comm.Recv( buf=_recvbuf, source=source, tag=tag )

    # add the contents to the local buffer. If localbufsize is 0, then
    # the two arrays are the same.
    _localbuf[localbufsize:localbufsize+msglength] = _recvbuf[:]

def get_particle_array(name="", **props):
    """Return a particle array"""
    nprops = len(props)
    np = 0

    prop_dict = {}
    for prop in props.keys():
        data = numpy.asarray(props[prop])
        np = data.size

        if prop in ['pid', 'type', 'tag']:
            prop_dict[prop] = {'data':data,
                               'type':'int',
                               'name':prop}

        elif prop in ['gid']:
            prop_dict[prop] = {'data':data.astype(numpy.uint32),
                               'type': 'unsigned int',
                               'name':prop}
        else:
            prop_dict[prop] = {'data':data,
                               'type':'double',
                               'name':prop}

    default_props = ['x', 'y', 'z', 'h', 'rho', 'gid', 'tag', 'type', 'pid']

    for prop in default_props:
        if not prop in prop_dict:
            if prop in ["type", "tag", "pid"]:
                prop_dict[prop] = {'name':prop, 'type':'int',
                                   'default':0}

            elif prop in ['gid']:
                data = numpy.ones(shape=np, dtype=numpy.uint32)
                data[:] = UINT_MAX

                prop_dict[prop] = {'name':prop, 'type':'unsigned int',
                                   'data':data}

            else:
                prop_dict[prop] = {'name':prop, 'type':'double',
                                   'default':0}
    # create the particle array
    pa = ParticleArray(name="",**prop_dict)

    return pa
