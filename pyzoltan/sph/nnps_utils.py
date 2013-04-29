"""NNPS utility functions to work with Zoltan lists"""
import numpy
from pyzoltan.sph.particle_array import ParticleArray
from pyzoltan.core.zoltan import get_zoltan_id_type_max

UINT_MAX = get_zoltan_id_type_max()

class ParticleTAGS:
    Local = 0           # Particles local to this processor
    Remote = 1          # Remote particles for computations
    Ghost = 2           # Ghost particles for periodicity

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
        if len(indices) > 0:
            send[pid] = {}
            for prop, prop_array in props.iteritems():
                send[pid][prop] = prop_array[ exportIndices[indices] ]

            # save the local ids exported to each processor
            send[pid]['lid'] = exportIndices[indices]

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
    comm.Recv( _recvbuf, source, tag )

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
