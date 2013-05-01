# Cython
cimport cython

# Numpy
import numpy as np
cimport numpy as np

# PyZoltan
from pyzoltan.czoltan.czoltan cimport Zoltan_Struct
from pyzoltan.core import zoltan_utils

# local imports
import nnps_utils
from nnps_utils import ParticleTAGS

cdef extern from 'math.h':
    int abs(int)
    cdef double ceil(double)
    cdef double floor(double)
    cdef double fabs(double)

cdef extern from 'limits.h':
    cdef int INT_MAX
    cdef unsigned int UINT_MAX

cdef inline int real_to_int(double real_val, double step):
    """ Return the bin index to which the given position belongs.

    Parameters:
    -----------
    val -- The coordinate location to bin
    step -- the bin size    

    Example:
    --------
    real_val = 1.5, step = 1.0 --> ret_val = 1

    real_val = -0.5, step = 1.0 --> real_val = -1
    
    """
    cdef int ret_val = <int>floor( real_val/step )

    return ret_val

cdef inline cIntPoint find_cell_id(cPoint pnt, double cell_size):
    """ Find the cell index for the corresponding point 

    Parameters:
    -----------
    pnt -- the point for which the index is sought
    cell_size -- the cell size to use
    id -- output parameter holding the cell index 

    Algorithm:
    ----------
    performs a box sort based on the point and cell size

    Notes:
    ------
    Uses the function  `real_to_int`
    
    """
    cdef cIntPoint p = cIntPoint(real_to_int(pnt.x, cell_size),
                                 real_to_int(pnt.y, cell_size),
                                 0)
    return p

cdef inline cPoint _get_centroid(double cell_size, cIntPoint cid):
        """ Get the centroid of the cell.

        Parameters:
        -----------

        cell_size : double (input)
            Cell size used for binning

        cid : cPoint (input)
            Spatial index for a cell

        Returns:
        ---------
        
        centroid : cPoint 
             
        Notes:
        ------
        The centroid in any coordinate direction is defined to be the
        origin plus half the cell size in that direction

        """
        centroid = cPoint_new(0.0, 0.0, 0.0)
        centroid.x = (<double>cid.x + 0.5)*cell_size
        centroid.y = (<double>cid.y + 0.5)*cell_size

        return centroid

def get_centroid(double cell_size, IntPoint cid):
    """ Get the centroid of the cell.
    
    Parameters:
    -----------
    
    cell_size : double (input)
        Cell size used for binning
    
    cid : IntPoint (input)
        Spatial index for a cell
    
    Returns:
    ---------
    
    centroid : Point 
    
    Notes:
    ------
    The centroid in any coordinate direction is defined to be the
    origin plus half the cell size in that direction
    
    """
    cdef cPoint _centroid = _get_centroid(cell_size, cid.data)
    centroid = Point_new(0.0, 0.0, 0.0)

    centroid.data = _centroid
    return centroid

cpdef UIntArray arange_uint(int start, int stop=-1):
    """Utility function to return a numpy.arange for a UIntArray"""
    cdef int size
    cdef UIntArray arange
    cdef int i = 0

    if stop == -1:
        arange = UIntArray(start)
        for i in range(start):
            arange.data[i] = <unsigned int>i
    else:
        size = stop-start
        arange = UIntArray(size)
        for i in range(size):
            arange.data[i] = <unsigned int>(start + i)

    return arange

################################################################
# ParticleArrayWrapper
################################################################
cdef class ParticleArrayWrapper:
    def __init__(self, ParticleArray pa):
        self.pa = pa
        self.name = pa.name

        self.x = pa.get_carray('x')
        self.y = pa.get_carray('y')
        self.z = pa.get_carray('z')
        self.h = pa.get_carray('h')
        self.gid = pa.get_carray('gid')
        self.tag = pa.get_carray('tag')

#################################################################
# NNPS extension classes
#################################################################
cdef class DomainLimits:
    """This class determines the limits of the solution domain.

    We expect all simulations to have well defined domain limits
    beyond which we are either not interested or the solution is
    invalid to begin with. Thus, if a particle leaves the domain,
    the simulation should be considered invalid.

    The initial domain limits could be given explicitly or asked to be
    computed from the particle arrays. The domain could be periodic.

    """
    def __init__(self, int dim=1, double xmin=-1000, double xmax=1000, double ymin=0,
                 double ymax=0, double zmin=0, double zmax=0, is_periodic=False):
        """Constructor"""
        self._check_limits(dim, xmin, xmax, ymin, ymax, zmin, zmax)
        
        self.xmin = xmin; self.xmax = xmax
        self.ymin = ymin; self.ymax = ymax
        self.zmin = zmin; self.zmax = zmax

        # Indicates if the domain is periodic
        self.is_periodic = is_periodic

        # get the translates in each coordinate direction
        self.xtranslate = xmax - xmin
        self.ytranslate = ymax - ymin
        self.ztranslate = zmax - zmin

        # store the dimension
        self.dim = dim

    def _check_limits(self, dim, xmin, xmax, ymin, ymax, zmin, zmax):
        """Sanity check on the limits"""
        if ( (xmax < xmin) or (ymax < ymin) or (zmax < zmin) ):
            raise ValueError("Invalid domain limits!")

cdef class Cell:
    """Basic indexing structure.

    For a spatial indexing based on the box-sort algorithm, this class
    defines the spatial data structure used to hold particle indices
    (local and global) that are within this cell. 

    """
    def __init__(self, IntPoint cid, double cell_size, int layers=2):
        """Constructor

        Parameters:
        -----------

        cid : IntPoint
            Spatial index (unflattened) for the cell

        cell_size : double
            Spatial extent of the cell in each dimension

        layers : int default (2)
            Factor to compute the bounding box        

        """
        self._cid = cIntPoint_new(cid.x, cid.y, cid.z)
        self.cell_size = cell_size

        self.lindices = lindices = UIntArray()
        self.gindices = gindices = UIntArray()
        self.nparticles = lindices.length
        self.is_boundary = False

        # compute the centroid for the cell
        self.centroid = _get_centroid(cell_size, cid.data)

        # cell bounding box
        self.layers = layers
        self._compute_bounding_box(cell_size, layers)

        # list of neighboring processors
        self.nbrprocs = IntArray(0)

    cpdef set_indices(self, UIntArray lindices, UIntArray gindices):
        """Set the global and local indices for the cell"""
        self.lindices = lindices
        self.gindices = gindices
        self.nparticles = lindices.length

    def get_centroid(self, Point pnt):
        """Utility function to get the centroid of the cell.

        Parameters:
        -----------

        pnt : Point (input/output)
            The centroid is cmoputed and stored in this object.

        The centroid is defined as the origin plus half the cell size
        in each dimension.
        
        """
        cdef cPoint centroid = self.centroid
        pnt.data = centroid

    def get_bounding_box(self, Point boxmin, Point boxmax, int layers=2,
                         cell_size=None):
        """Compute the bounding box for the cell.

        Parameters:
        ------------

        boxmin : Point (output)
            The bounding box min coordinates are stored here

        boxmax : Point (output)
            The bounding box max coordinates are stored here

        layers : int (input) default (2)
            Number of offset layers to define the bounding box

        cell_size : double (input) default (None)
            Optional cell size to use to compute the bounding box.
            If not provided, the cell's size will be used.        

        """
        if cell_size is None:
            cell_size = self.cell_size

        self._compute_bounding_box(cell_size, layers)
        boxmin.data = self.boxmin
        boxmax.data = self.boxmax

    cdef _compute_bounding_box(self, double cell_size,
                               int layers):
        self.layers = layers
        cdef cPoint centroid = self.centroid
        cdef cPoint boxmin = cPoint_new(0., 0., 0.)
        cdef cPoint boxmax = cPoint_new(0., 0., 0.)

        boxmin.x = centroid.x - (layers+0.5) * cell_size
        boxmax.x = centroid.x + (layers+0.5) * cell_size

        boxmin.y = centroid.y - (layers+0.5) * cell_size
        boxmax.y = centroid.y + (layers+0.5) * cell_size

        boxmin.z = 0.
        boxmax.z = 0.

        self.boxmin = boxmin
        self.boxmax = boxmax

cdef class NNPSParticleGeometric(ZoltanGeometricPartitioner):
    """Nearest neighbor query class using the box-sort algorithm.

    In this class, the particles are treated as the objects to be
    partitioned. Each particle is assigned a unique global index which
    is given to Zoltan for the load balancing.

    NNPS bins all local particles using the box sort algorithm in
    Cells. The cells are stored in a dictionary 'cells' which is keyed
    on the spatial index (IntPoint) of the cell.

    """
    def __init__(self, int dim, ParticleArray pa, object comm,
                 double radius_scale=2.0,
                 int ghost_layers=2, domain=None,
                 lb_props=None):
        """Constructor for NNPS

        Parameters:
        -----------

        dim : int
            Dimension (Not sure if this is really needed)

        pa : ParticleArray
            Particle data

        comm : mpi4py.MPI.COMM, default (None)
            MPI communicator for parallel invocations

        radius_scale : double, default (2)
            Optional kernel radius scale. Defaults to 2

        ghost_layers : int, default (2)
            Optional factor for computing bounding boxes for cells.

        domain : DomainLimits, default (None)
            Optional limits for the domain            

        """
        self.pa = pa
        self.pa_wrapper = pa_wrapper = ParticleArrayWrapper(pa)
        super(NNPSParticleGeometric, self).__init__(dim, comm,
                                                    pa_wrapper.x,
                                                    pa_wrapper.y,
                                                    pa_wrapper.z,
                                                    pa_wrapper.gid)

        self.num_particles = pa.get_number_of_particles()
        self.num_remote = 0
        self.num_global = 0
        self.num_ghost = 0

        self.radius_scale = radius_scale
        self.dim = dim

        self.hmax_sendbuf = np.ones(1)

        rank = self.rank
        size = self.size
        if size == 1:
            self.in_parallel = False
            self.procs = np.ones(1)
            self.parts = np.ones(1)
            self.hmax_recvbuf = np.ones(1)
        else:
            self.in_parallel = True
            self.procs = np.ones(size, dtype=np.int32)
            self.parts = np.ones(size, dtype=np.int32)
            self.hmax_recvbuf = np.zeros(size, dtype=np.float64)

        # load balancing props
        if lb_props is None:
            self.lb_props = ['x','y','h','gid']
        else:
            self.lb_props = lb_props

        self.ghost_layers = ghost_layers

        self.domain = domain
        if domain is None:
            self.domain = DomainLimits(dim)

        # periodicity
        self.is_periodic = self.domain.is_periodic            

        # initialize the cells dict
        self.cells = {}

        # compute the cell size
        self.compute_cell_size()
        self.local_bin()

        # setup the initial global particle ids
        if self.in_parallel:
            self.update_particle_gid()

    def update(self, initial=False):
        """Perform one step of a parallel update.

        There are three phases to a parallel update:

          (1) Exchange particles so that the cell map is unique.

          (2) Call a load balancing function and exchange particles

          (3) Compute remote particles given the new partition.

        """
        # remove ghost particles from a previous step
        self.remove_remote_particles()

        # bin locally
        self.local_bin()

        if self.in_parallel:
            # set the local/global number of particles

            # call a load balancing function and exchange data
            self.load_balance()
            self.lb_exchange_data()

            # compute remote particles and exchange data
            self.compute_remote_particles()
            self.remote_exchange_data()

            # align particles. I hate this but it must be done!
            self.pa.align_particles()

    cpdef local_bin(self):
        """Bin the particles by deleting any previous cells and
        re-computing the indexing structure. This corresponds to a
        local binning process that is called when each processor has
        a given list of particles to deal with.

        """
        cdef dict cells = self.cells
        cdef int num_particles

        # clear the cells dict
        cells.clear()

        # compute the cell size
        self.compute_cell_size()

        # deal with ghosts
        if self.is_periodic:
            # remove-ghost-particles
            self._remove_ghost_particles()

            # adjust local particles
            self._adjust_particles()

            # create new ghosts
            self._create_ghosts()

        # indices on which to bin
        num_particles = self.num_particles
        cdef UIntArray indices = arange_uint(num_particles)

        # bin the particles
        self._bin(indices)

    cpdef remote_bin(self):
        """Bin the remote particles.

        This function assumes that a previously created cell structure
        exists for local particles, prefrebaly after a load balancing
        step and a subsequent call to 'update_data'

        """
        cdef int num_particles = self.num_particles
        cdef int num_remote = self.num_remote
        cdef UIntArray indices = arange_uint(num_particles, num_remote+num_particles)

        # bin the remote particles
        self._bin(indices)

    cdef _bin(self, UIntArray indices):
        """Bin particles given by indices.

        Parameters:
        -----------
        indices : UIntArray
            Subset of particles to bin

        """
        cdef DoubleArray x = self.x
        cdef DoubleArray y = self.y
        cdef UIntArray gid = self.gid

        cdef dict cells = self.cells
        cdef double cell_size = self.cell_size
        cdef int layers = self.ghost_layers

        cdef UIntArray lindices, gindices
        cdef size_t num_particles, indexi, i

        cdef cIntPoint _cid
        cdef IntPoint cid = IntPoint()

        cdef Cell cell

        # now bin the particles
        num_particles = indices.length
        for indexi in range(num_particles):
            i = indices.data[indexi]

            globalid = gid.data[i]

            pnt = cPoint_new( x.data[i], y.data[i], 0.0 )
            _cid = find_cell_id( pnt, cell_size )

            cid = IntPoint_from_cIntPoint(_cid)

            if not cells.has_key( cid ):
                cell = Cell(cid, cell_size, layers)
                lindices = UIntArray()
                gindices = UIntArray()

                cell.set_indices( lindices, gindices )
                cells[ cid ] = cell

            # add this particle to the list of indicies
            cell = cells[ cid ]
            lindices = cell.lindices
            gindices = cell.gindices

            lindices.append( <ZOLTAN_ID_TYPE> i )
            gindices.append( gid.data[i] )        

    cpdef update_data(self):
        """Re-bin after a load balancing step"""
        # set the number of particles
        self.num_particles = self.x.length
        self.set_num_local_objects( self.num_particles )

        # re-bin given the new arrays
        self.local_bin()

    cpdef update_remote_particle_data(self):
        """Bin remote particles.

        The number remote particles is set after a call to this
        function. Only remote particles are binned using this
        function.
        
        """
        cdef DoubleArray x = self.x

        # set the number of remote particles
        self.num_remote = x.length - self.num_particles

        # bin the remote particles
        self.remote_bin()

    cpdef compute_remote_particles(self):
        """Compute remote particles we need to export.

        Particles to be exported are determined by flagging individual
        cells and where they need to be shared to meet neighbor
        requirements.
        
        """
        cdef Zoltan_Struct* zz = self._zstruct.zz
        cdef UIntArray gid = self.gid
        cdef UIntArray exportGlobalids, exportLocalids
        cdef IntArray exportProcs

        cdef Cell cell
        cdef cPoint boxmin, boxmax

        cdef object comm = self.comm
        cdef int rank = self.rank
        cdef int size = self.size
        
        cdef dict cells = self.cells
        cdef IntPoint cid
        cdef UIntArray lindices, gindices

        cdef np.ndarray nbrprocs
        cdef np.ndarray[ndim=1, dtype=np.int32_t] procs, parts
        
        cdef int nbrproc, num_particles
        cdef ZOLTAN_ID_TYPE i

        cdef int *_procs, *_parts
        cdef int numprocs = 0
        cdef int numparts = 0

        # reset the Zoltan lists
        self.reset_Zoltan_lists()
        exportGlobalids = self.exportGlobalids
        exportLocalids = self.exportLocalids
        exportProcs = self.exportProcs

        # initialize the procs and parts
        procs = self.procs
        parts = self.parts
        
        for cid in cells:
            cell = cells[ cid ]

            # reset the procs and parts to -1
            procs[:] = -1; parts[:] = -1

            _procs = <int*>procs.data
            _parts = <int*>parts.data

            # get the bounding box for this cell
            boxmin = cell.boxmin; boxmax = cell.boxmax

            czoltan.Zoltan_LB_Box_PP_Assign(
                zz,
                boxmin.x, boxmin.y, boxmin.z,
                boxmax.x, boxmax.y, boxmax.z,
                _procs, &numprocs,
                _parts, &numparts
                )                

            # array of neighboring processors
            nbrprocs = procs[np.where( (procs != -1) * (procs != rank) )[0]]

            if nbrprocs.size > 0:
                cell.is_boundary = True

                lindices = cell.lindices
                cell.nbrprocs.resize( nbrprocs.size )
                cell.nbrprocs.set_data( nbrprocs )
                
                num_particles = lindices.length
                for nbrproc in nbrprocs:
                    for indexi in range( num_particles ):
                        i = lindices.data[indexi]

                        exportGlobalids.append( gid.data[i] )
                        exportLocalids.append( i )
                        exportProcs.append( nbrproc )

        # set the numImport and numExport
        self.numImport = 0
        self.numExport = exportProcs.length

        # Invert the lists...
        self.Zoltan_Invert_Lists()

        # copy the lists to the particle lists
        self.numParticleExport = self.numExport
        self.numParticleImport = self.numImport

        self.exportParticleGlobalids.resize( self.numExport )
        self.exportParticleGlobalids.copy_subset( self.exportGlobalids )

        self.exportParticleLocalids.resize( self.numExport )
        self.exportParticleLocalids.copy_subset( self.exportLocalids )
        
        self.exportParticleProcs.resize( self.numExport )
        self.exportParticleProcs.copy_subset( self.exportProcs )

        self.importParticleGlobalids.resize( self.numImport )
        self.importParticleGlobalids.copy_subset( self.importGlobalids )

        self.importParticleLocalids.resize( self.numImport )
        self.importParticleLocalids.copy_subset( self.importLocalids )

        self.importParticleProcs.resize( self.numImport )
        self.importParticleProcs.copy_subset( self.importProcs )

    cpdef compute_cell_size(self):
        cdef DoubleArray h = self.pa_wrapper.h
        cdef double cell_size
        cdef object comm = self.comm
        cdef np.ndarray[ndim=1,dtype=np.float64_t] recvbuf = self.hmax_recvbuf
        cdef np.ndarray[ndim=1,dtype=np.float64_t] sendbuf = self.hmax_sendbuf

        h.update_min_max()
        cdef double maxh = h.maximum

        # compute the global max h if in parallel
        if self.in_parallel:
            recvbuf[:] = 0.0
            sendbuf[0] = maxh

            comm.Gather(sendbuf=sendbuf, recvbuf=recvbuf, root=0)
            recvbuf[:] = np.max(recvbuf)

            comm.Scatter(sendbuf=recvbuf, recvbuf=recvbuf, root=0)
            maxh = recvbuf[0]

        cell_size = maxh * self.radius_scale
        if cell_size < 1e-6:
            msg = """Cell size too small %g. Perhaps h = 0?
            Setting cell size to 1"""%(cell_size)
            print msg
            cell_size = 1.0

        self.cell_size = cell_size
        #print self.cell_size, h.maximum

    def set_num_global_particles(self, size_t num_particles):
        self.num_global = num_particles

    def load_balance(self):
        """Do the load balancing and copy the arrays."""
        self.Zoltan_LB_Balance()

        # resize and copy to particle export lists
        self.numParticleExport = self.numExport
        self.exportParticleGlobalids.resize( self.numExport )
        self.exportParticleGlobalids.copy_subset( self.exportGlobalids )

        self.exportParticleLocalids.resize( self.numExport )
        self.exportParticleLocalids.copy_subset( self.exportLocalids )

        self.exportParticleProcs.resize( self.numExport )
        self.exportParticleProcs.copy_subset( self.exportProcs )

        # resize and copy to particle import lists
        self.numParticleImport = self.numImport
        self.importParticleGlobalids.resize( self.numImport )
        self.importParticleGlobalids.copy_subset( self.importGlobalids )

        self.importParticleLocalids.resize( self.numImport )
        self.importParticleLocalids.copy_subset( self.importLocalids )

        self.importParticleProcs.resize( self.numImport )
        self.importParticleProcs.copy_subset( self.importProcs )
        
    def lb_exchange_data(self):
        """Share particle info after Zoltan_LB_Balance

        After an initial call to 'Zoltan_LB_Balance', the new size of
        the arrays should be (num_particles - numExport + numImport)
        to reflect particles that should be imported and exported.

        This function should be called after the load balancing lists
        a defined. The particles to be exported are removed and the
        arrays re-sized. MPI is then used to send and receive the data

        """
        # data arrays
        cdef ParticleArray pa = self.pa
        
        # Zoltan generated lists
        cdef UIntArray exportGlobalids = self.exportParticleGlobalids
        cdef UIntArray exportLocalids = self.exportParticleLocalids
        cdef IntArray exportProcs = self.exportParticleProcs
        cdef int numExport = self.numParticleExport

        cdef UIntArray importGlobalids = self.importParticleGlobalids
        cdef UIntArray importLocalids = self.importParticleLocalids
        cdef IntArray importProcs = self.importParticleProcs
        cdef int numImport = self.numParticleImport

        # dicts to send/recv data 
        cdef dict recv = {}

        # collect the data to send
        cdef dict send = nnps_utils.get_send_data(self.comm, pa, self.lb_props, exportLocalids,
                                                  exportProcs)

        # current number of particles
        cdef int current_size = self.num_particles
        cdef int count, new_size

        # MPI communicator
        cdef object comm = self.comm

        # count the number of objects to receive from remote procesors
        zoltan_utils.count_recv_data(comm, recv, numImport, importProcs)

        # Remove particles to be exported
        pa.remove_particles( exportLocalids )

        # resize the arrays
        newsize = current_size - numExport + numImport
        count = current_size - numExport

        pa.resize( newsize )
        self.set_tag(count, newsize, 0)
        
        # exchange the data
        self._exchange_data(count, send, recv)

        # update the data
        self.update_data()

    def remote_exchange_data(self):
        """Share particle info after computing remote particles.

        After a load balancing and the corresponding exhange of
        particles, additional particles are needed as remote
        (remote). Calls to 'compute_remote_particles' and
        'Zoltan_Invert_Lists' builds the lists required.

        The arrays must now be re-sized to (num_particles + numImport)
        to reflect the new particles that come in as remote particles.

        """
        # data arrays
        cdef ParticleArray pa = self.pa
        
        # Import/Export lists
        cdef UIntArray exportGlobalids = self.exportParticleGlobalids
        cdef UIntArray exportLocalids = self.exportParticleLocalids
        cdef IntArray exportProcs = self.exportParticleProcs

        cdef UIntArray importGlobalids = self.importParticleGlobalids
        cdef UIntArray importLocalids = self.importParticleLocalids
        cdef IntArray importProcs = self.importParticleProcs
        cdef int numImport = self.numParticleImport

        # collect the data to send
        cdef dict recv = {}
        cdef dict send = nnps_utils.get_send_data(
            self.comm, pa, self.lb_props, exportLocalids,exportProcs)

        # current number of particles
        cdef int current_size = self.num_particles
        cdef int count, new_size

        # MPI communicator
        cdef object comm = self.comm

        # count the number of objects to receive from remote procesors
        zoltan_utils.count_recv_data(comm, recv, numImport, importProcs)

        # resize the arrays
        newsize = current_size + numImport
        count = current_size

        pa.resize( newsize )
        self.set_tag(count, newsize, 0)

        # share the data
        self._exchange_data(count, send, recv)

        # update NNPS with the remote particle data
        self.update_remote_particle_data()

    cdef _exchange_data(self, int count, dict send, dict recv):
        """New send and receive."""
        # data arrays
        cdef ParticleArray pa = self.pa

        # MPI communicator
        cdef object comm = self.comm

        # temp buffers to store info
        cdef DoubleArray doublebuf = self.doublebuf
        cdef UIntArray uintbuf = self.idbuf
        cdef LongArray longbuf = self.longbuf
        cdef IntArray intbuf = self.intbuf

        cdef int rank = self.rank
        cdef int size = self.size
        cdef int i, j

        ctype_map = {'unsigned int':uintbuf,
                     'double':doublebuf,
                     'long':longbuf,
                     'int':intbuf}

        props = {}
        for prop in self.lb_props:
            prop_array = pa.get_carray(prop)
            props[ prop ] = ctype_map[ prop_array.get_c_type() ]
            
        ############### SEND AND RECEIVE START ########################
        # Receive from processors with a lower rank
        for i in recv.keys():
            if i < rank:
                for prop, recvbuf in props.iteritems():
                    recvbuf.resize( recv[i] )

                    nnps_utils.Recv(comm=comm,
                                    localbuf=pa.get_carray(prop),
                                    localbufsize=count,
                                    recvbuf=recvbuf,
                                    source=i)
                
                # update the local buffer size
                count = count + recv[i]

        # Send the data across
        for pid in send.keys():
            for prop in props:
                sendbuf = send[pid][prop]
                comm.Send( sendbuf, dest=pid )
                    
        # recv from procs with higher rank i value as set in first loop
        for i in recv.keys():
            if i > rank:
                for prop, recvbuf in props.iteritems():
                    recvbuf.resize( recv[i] )

                    nnps_utils.Recv(comm=comm,
                                    localbuf=pa.get_carray(prop),
                                    localbufsize=count,
                                    recvbuf=recvbuf,
                                    source=i)

                # update to the new length
                count = count + recv[i]

        ############### SEND AND RECEIVE STOP ########################                

    def remove_remote_particles(self):
        cdef int num_particles = self.num_particles

        # resize the particle array
        self.pa.resize( num_particles )
        self.pa.align_particles()
        
        # reset the number of remote particles
        self.num_remote = 0

    def set_tag(self, int start, int end, int value):
        """Reset the annoying tag value after particles are resized."""
        cdef int i
        cdef IntArray tag = self.pa_wrapper.tag

        for i in range(start, end):
            tag[i] = value

    #######################################################
    # Functions for periodicity
    #######################################################
    cpdef adjust_particles(self):
        cdef DomainLimits domain = self.domain
        cdef ParticleArray pa = self.pa

        cdef DoubleArray x = self.x
        cdef DoubleArray y = self.y

        cdef size_t np = pa.get_number_of_particles()
        cdef size_t i

        cdef double xmin = domain.xmin
        cdef double xmax = domain.xmax
        cdef double ymin = domain.ymin
        cdef double ymax = domain.ymax

        cdef double xtranslate = domain.xtranslate
        cdef double ytranslate = domain.ytranslate

        cdef int dim = domain.dim

        cdef double xi, yi

        for i in range(np):
            xi = x.data[i]; yi = y.data[i]

            if xi < xmin:
                x.data[i] = x.data[i] + xtranslate
            if xi > xmax:
                x.data[i] = x.data[i] - xtranslate

            if dim == 2:
                if yi < ymin:
                    y.data[i] = y.data[i] + ytranslate
                if yi > ymax:
                    y.data[i] = y.data[i] - ytranslate

    cpdef _create_ghosts(self):
        """Create ghost particles.

        This function creates ghost particles by identifying regions
        within the vicinity of a domain boundary. The particles are
        created and added as local particles. We tag these particles
        so that they can be removed at a later stage.

        In parallel, after particles are created and added locally,
        the global indices are invalid. These must be re-created to
        reflect the added ghosts. The same must be done when removing
        these ghost particles. This creates a new number of particles
        locally and new global indices should be created.

        """
        cdef DomainLimits domain = self.domain
        cdef ParticleArray pa = self.pa
        cdef double cell_size = self.cell_size
        cdef int nghost = 0

        cdef ParticleArray copy
        cdef size_t np, i
        cdef double xmin = domain.xmin
        cdef double xmax = domain.xmax
        cdef double ymin = domain.ymin
        cdef double ymax = domain.ymax

        cdef double xtranslate = domain.xtranslate
        cdef double ytranslate = domain.ytranslate

        cdef int dim = domain.dim

        cdef double xi, yi

        cdef DoubleArray x = self.x
        cdef DoubleArray y = self.y

        cdef LongArray left = LongArray()
        cdef LongArray right = LongArray()
        cdef LongArray top = LongArray()
        cdef LongArray bottom = LongArray()
        cdef LongArray lt = LongArray()
        cdef LongArray rt = LongArray()
        cdef LongArray lb = LongArray()
        cdef LongArray rb = LongArray()

        np = pa.get_number_of_particles()
        for i in range(np):
            xi = x.data[i]; yi = y.data[i]

            # particle near the left wall
            if ( (xi - xmin) <= cell_size ):
                left.append(i)

                if dim == 2:
                    # left top corner
                    if ( (ymax - yi) < cell_size ):
                        lt.append( i )
                    # left bottom corner
                    if ( (yi - ymin) < cell_size ):
                        lb.append( i )

            # particle near the right wall
            if ( (xmax - xi) < cell_size ):
                right.append(i)

                if dim == 2:
                    #right top corner
                    if ( (ymax - yi) < cell_size ):
                        rt.append(i)
                    if ( (yi - ymin) < cell_size ):
                        rb.append(i)

            # particle near the top
            if ( ((ymax - yi) < cell_size) and dim == 2 ):
                top.append(i)
            # particle near the bottom
            if ( ((yi - ymin) < cell_size) and dim == 2 ):
                bottom.append(i)

        # now treat each case separately and count the number of ghosts
        nghost = 0

        # left
        copy = pa.extract_particles( left )
        copy.x += xtranslate
        copy.tag[:] = ParticleTAGS.Ghost
        pa.append_parray(copy)

        nghost += copy.get_number_of_particles()

        # right
        copy = pa.extract_particles( right )
        copy.x -= xtranslate
        copy.tag[:] = ParticleTAGS.Ghost
        pa.append_parray(copy)

        nghost += copy.get_number_of_particles()

        # top
        copy = pa.extract_particles( top )
        copy.y -= ytranslate
        copy.tag[:] = ParticleTAGS.Ghost
        pa.append_parray(copy)

        nghost += copy.get_number_of_particles()

        # bottom
        copy = pa.extract_particles( bottom )
        copy.y += ytranslate
        copy.tag[:] = ParticleTAGS.Ghost
        pa.append_parray(copy)

        nghost += copy.get_number_of_particles()

        # left top
        copy = pa.extract_particles( lt )
        copy.x += xtranslate
        copy.y -= ytranslate
        copy.tag[:] = ParticleTAGS.Ghost
        pa.append_parray(copy)

        nghost += copy.get_number_of_particles()

        # left bottom
        copy = pa.extract_particles( lb )
        copy.x += xtranslate
        copy.y += ytranslate
        copy.tag[:] = ParticleTAGS.Ghost
        pa.append_parray(copy)

        nghost += copy.get_number_of_particles()

        # right top
        copy = pa.extract_particles( rt )
        copy.x -= xtranslate
        copy.y -= ytranslate
        copy.tag[:] = ParticleTAGS.Ghost
        pa.append_parray(copy)

        nghost += copy.get_number_of_particles()

        # right bottom
        copy = pa.extract_particles( rb )
        copy.x -= xtranslate
        copy.y += ytranslate
        copy.tag[:] = ParticleTAGS.Ghost
        pa.append_parray(copy)

        # save the total number of ghost particles
        nghost += copy.get_number_of_particles()
        self.num_ghost = nghost

        # update the total number of particles
        self.num_particles = pa.get_number_of_particles()

        # update gid if in parallel
        if self.in_parallel:
            self.update_particle_gid()

    def _remove_ghost_particles(self):
        """Remove ghost particles.

        Ghost particles have the tag set as ParticleTAGS.Ghost and we
        use the ParticleArray function 'remove_tagged_particles' to
        get rid of them.

        Since particles have been removed, we should update the number
        of particles and number of global particles. The gid could
        also be re-computed but this seems unnecessary since the
        remaining ids will be unique.

        """
        self.pa.remove_tagged_particles( ParticleTAGS.Ghost )

        # call update gid. This will compute the new number of
        # particles as well setting the new unique global indices.
        self.update_particle_gid()

    ######################################################################
    # Neighbor location routines
    ######################################################################
    cpdef get_nearest_particles(self, size_t i, UIntArray nbrs):
        """Utility function to get near-neighbors for a particle.

        Parameters:
        -----------

        i : int (input)
            Particle for which neighbors are sought.

        nbrs : UIntArray (output)
            Neighbors for the requested particle are stored here.

        """
        cdef Cell cell
        cdef dict cells = self.cells

        # data arrays
        cdef DoubleArray x = self.pa_wrapper.x
        cdef DoubleArray y = self.pa_wrapper.y
        cdef DoubleArray h = self.pa_wrapper.h
        
        cdef double cell_size = self.cell_size
        cdef double radius_scale = self.radius_scale
        cdef UIntArray lindices
        cdef size_t indexj
        cdef ZOLTAN_ID_TYPE j

        cdef cPoint xi = cPoint_new(x.data[i], y.data[i], 0.0)
        cdef cIntPoint _cid = find_cell_id( xi, cell_size )
        cdef IntPoint cid = IntPoint_from_cIntPoint( _cid )
        cdef IntPoint cellid = IntPoint(0, 0, 0)

        cdef cPoint xj
        cdef double xij

        cdef double hi, hj
        hi = radius_scale * h.data[i]

        cdef int nnbrs = 0

        cdef int ix, iy
        for ix in [cid.data.x -1, cid.data.x, cid.data.x + 1]:
            for iy in [cid.data.y - 1, cid.data.y, cid.data.y + 1]:
                cellid.data.x = ix; cellid.data.y = iy

                if cells.has_key( cellid ):
                    cell = cells[ cellid ]
                    lindices = cell.lindices
                    for indexj in range( lindices.length ):
                        j = lindices.data[indexj]

                        xj = cPoint_new( x.data[j], y.data[j], 0.0 )
                        xij = cPoint_distance( xi, xj )

                        hj = radius_scale * h.data[j]

                        if ( (xij < hi) or (xij < hj) ):
                            if nnbrs == nbrs.length:
                                nbrs.resize( nbrs.length + 50 )
                                print """Warning: Extending the neighbor list to %d"""%(nbrs.length)

                            nbrs.data[ nnbrs ] = j
                            nnbrs = nnbrs + 1
                            #nbrs.append( j )

        # update the _length for nbrs to indicate the number of neighbors
        nbrs._length = nnbrs

    cpdef brute_force_neighbors(self, size_t i, UIntArray nbrs):
        cdef DoubleArray x = self.pa_wrapper.x
        cdef DoubleArray y = self.pa_wrapper.y
        cdef DoubleArray h = self.pa_wrapper.h

        cdef double radius_scale = self.radius_scale
        cdef double cell_size = self.cell_size

        cdef size_t num_particles, j

        num_particles = x.length
        cdef double xi = x.data[i]
        cdef double yi = y.data[i]

        cdef double hi = h.data[i] * radius_scale
        cdef double xj, yj, hj, hj2

        cdef double xij2
        cdef double hi2 = hi*hi

        # reset the neighbors
        nbrs.reset()

        for j in range(num_particles):
            xj = x.data[j]; yj = y.data[j]; hj = h.data[j]
            xij2 = (xi - xj)*(xi - xj) + (yi - yj)*(yi - yj)

            hj2 = hj*hj
            if ( (xij2 < hi2) or (xij2 < hj2) ):
                nbrs.append( <ZOLTAN_ID_TYPE> j )

        return nbrs

    def _setup_zoltan_arrays(self):
        super( NNPSParticleGeometric, self )._setup_zoltan_arrays()

        # Import/Export lists for particles
        self.exportParticleGlobalids = UIntArray()
        self.exportParticleLocalids = UIntArray()
        self.exportParticleProcs = IntArray()

        self.importParticleGlobalids = UIntArray()
        self.importParticleLocalids = UIntArray()
        self.importParticleProcs = IntArray()

        self.numParticleExport = 0
        self.numParticleImport = 0

cdef class NNPSCellGeometric(NNPSParticleGeometric):
    """ Zoltan enabled NNPS which uses the cells for load balancing."""

    def __init__(self, int dim, ParticleArray pa, object comm,
                 double radius_scale=2.0,
                 int ghost_layers=2, domain=None,
                 lb_props=None):
        """Constructor for NNPS

        Parameters:
        -----------

        dim : int
            Dimension (Not sure if this is really needed)

        pa : ParticleArray
            Particle data

        comm : mpi4py.MPI.COMM, default (None)
            MPI communicator for parallel invocations

        radius_scale : double, default (2)
            Optional kernel radius scale. Defaults to 2

        ghost_layers : int, default (2)
            Optional factor for computing bounding boxes for cells.

        domain : DomainLimits, default (None)
            Optional limits for the domain            

        """
        super(NNPSCellGeometric, self).__init__(
            dim, pa, comm, radius_scale, ghost_layers, domain)

        # set up the cell_gid array
        self.cell_gid = UIntArray()

        # cell coordinate values
        self.cx = DoubleArray()
        self.cy = DoubleArray()
    
    def update_cell_gid(self):
        """Update the local and global indices for the cell map.

        The objects to be partitioned in this class are the cells and
        we need to number them uniquely across processors. The
        numbering is done sequentially starting from the processor
        with rank 0 to the processor with rank size-1

        """
        self.num_local_objects = len(self.cells)
        super(NNPSCellGeometric, self)._update_gid( self.cell_gid )

    def update(self, initial=False):
        """Update the partition."""

        # remove ghost particles from a previous step
        self.remove_remote_particles()

        # bin locally
        self.local_bin()

        # update the global indices for the cells
        self.update_cell_gid()

        if self.in_parallel:

            # call a load balancing function and exchange data
            self.load_balance()
            self.create_particle_lists()
            self.lb_exchange_data()

            # compute remote particles and exchange data
            self.compute_remote_particles()
            self.remote_exchange_data()

            # align particles. I hate this but it must be done!
            self.pa.align_particles()

    def create_particle_lists(self):
        """Create export/import lists based on particles

        For the cell based partitioner, Zoltan generated import and
        export lists apply to the cells. From this data, we can
        generate local export indices for the particles which must
        then be inverted to get the requisite import lists.

        We first construct the particle export lists in local arrays
        using the information from Zoltan_LB_Balance and subsequently
        call Zoltan_Invert_Lists to get the import lists. These arrays
        are then copied to the class arrays.

        """
        # these are the Zoltan generated lists that correspond to cells
        cdef UIntArray exportCellLocalids = self.exportCellLocalids
        cdef UIntArray exportCellGlobalids = self.exportCellGlobalids
        cdef IntArray exportCellProcs = self.exportCellProcs
        cdef int numCellExport = self.numCellExport

        # the local cell map and cellids
        cdef dict cell_map = self.cells
        cdef list cellids = cell_map.keys()
        cdef list cells = cell_map.values()
        cdef IntPoint cellid
        cdef Cell cell

        # temp buffers to create the particle lists. these will be
        # copied over to the main lists
        cdef UIntArray _exportLocalids = UIntArray()
        cdef UIntArray _exportGlobalids = UIntArray()
        cdef IntArray _exportProcs = IntArray()
        cdef int _numExport = 0

        cdef int indexi, i, j, export_proc, nindices
        cdef UIntArray lindices, gindices
        cdef int ierr

        # iterate over the Zoltan generated export lists
        for indexi in range(numCellExport):
            i = exportCellLocalids[indexi]             # local index for the cell to export
            cell = cells[ i ]                      # local cell to export
            export_proc = exportCellProcs.data[indexi] # processor to export cell to
            lindices = cell.lindices               # local particle  indices in cell
            gindices = cell.gindices               # global particle indices in cell
            nindices = lindices.length

            for j in range( nindices ):
                _exportLocalids.append( lindices.data[j] )
                _exportGlobalids.append( gindices.data[j] )
                _exportProcs.append( export_proc )

                _numExport = _numExport + 1

            # remove the cell from the local cell map
            cellid = cellids[ i ]
            cell_map.pop( cellid )

        # resize the export lists and copy
        self.numExport =  _numExport
        self.exportLocalids.resize( _numExport )
        self.exportGlobalids.resize( _numExport )
        self.exportProcs.resize( _numExport )

        self.exportLocalids.copy_subset( _exportLocalids )
        self.exportGlobalids.copy_subset( _exportGlobalids )
        self.exportProcs.copy_subset( _exportProcs )

        # Given the export particle indices, we invert the lists to
        # get the import lists from remote processors...
        self.Zoltan_Invert_Lists()

        # now copy over to the particle lists
        self.numParticleExport = self.numExport
        self.exportParticleGlobalids.resize( self.numExport )
        self.exportParticleGlobalids.copy_subset( self.exportGlobalids )

        self.exportParticleLocalids.resize( self.numExport )
        self.exportParticleLocalids.copy_subset( self.exportLocalids )

        self.exportParticleProcs.resize( self.numExport )
        self.exportParticleProcs.copy_subset( self.exportProcs )

        self.numParticleImport = self.numImport
        self.importParticleGlobalids.resize( self.numImport )
        self.importParticleGlobalids.copy_subset( self.importGlobalids )

        self.importParticleLocalids.resize( self.numImport )
        self.importParticleLocalids.copy_subset( self.importLocalids )

        self.importParticleProcs.resize( self.numImport )
        self.importParticleProcs.copy_subset( self.importProcs )

    def load_balance(self):
        self.Zoltan_LB_Balance()

        # copy the Zoltan export lists to the cell lists
        self.numCellExport = self.numExport
        self.exportCellGlobalids.resize( self.numExport )
        self.exportCellGlobalids.copy_subset( self.exportGlobalids )

        self.exportCellLocalids.resize( self.numExport )
        self.exportCellLocalids.copy_subset( self.exportLocalids )

        self.exportCellProcs.resize( self.numExport )
        self.exportCellProcs.copy_subset( self.exportProcs )

        # copy the Zoltan import lists to the cell lists
        self.numCellImport = self.numImport
        self.importCellGlobalids.resize( self.numImport )
        self.importCellGlobalids.copy_subset( self.importGlobalids )

        self.importCellLocalids.resize( self.numImport )
        self.importCellLocalids.copy_subset( self.importLocalids )

        self.importCellProcs.resize( self.numImport )
        self.importCellProcs.copy_subset( self.importProcs )

    #######################################################################
    # Private interface
    #######################################################################
    def _set_data(self):
        """Set the user defined particle data structure for Zoltan."""

        cdef dict cell_map = self.cells
        cdef list cells = cell_map.values()
        cdef int num_local_objects = self.num_local_objects
        cdef int num_global_objects = self.num_global_objects

        cdef UIntArray gid = self.cell_gid
        cdef DoubleArray x = self.cx
        cdef DoubleArray y = self.cy

        cdef int i
        cdef Cell cell
        cdef cPoint centroid

        # resize the coordinate arrays
        x.resize( num_local_objects )
        y.resize( num_local_objects )

        # populate the arrays
        for i in range( num_local_objects ):
            cell = cells[ i ]
            centroid = cell.centroid

            x.data[i] = centroid.x
            y.data[i] = centroid.y        
        
        self._cdata.numGlobalPoints = <ZOLTAN_ID_TYPE>num_global_objects
        self._cdata.numMyPoints = <ZOLTAN_ID_TYPE>num_local_objects
        
        self._cdata.myGlobalIDs = gid.data
        self._cdata.x = x.data
        self._cdata.y = y.data

    def _setup_zoltan_arrays(self):
        super( NNPSCellGeometric, self )._setup_zoltan_arrays()

        # Import/Export lists for cells
        self.exportCellGlobalids = UIntArray()
        self.exportCellLocalids = UIntArray()
        self.exportCellProcs = IntArray()

        self.importCellGlobalids = UIntArray()
        self.importCellLocalids = UIntArray()
        self.importCellProcs = IntArray()

        self.numCellImport = 0
        self.numCellExport = 0
