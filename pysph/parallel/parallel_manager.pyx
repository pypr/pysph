#cython: embedsignature=True

# Numpy
import numpy as np
cimport numpy as np

# MPI4PY
import mpi4py.MPI as mpi

from cpython.list cimport PyList_Append, PyList_GET_SIZE

# PyZoltan
from pyzoltan.czoltan cimport czoltan
from pyzoltan.czoltan.czoltan cimport Zoltan_Struct
from pyzoltan.core import zoltan_utils

# PySPH imports
from pysph.base.nnps_base cimport DomainManager, Cell, find_cell_id, arange_uint
from pysph.base.utils import ParticleTAGS

cdef int Local = ParticleTAGS.Local
cdef int Remote = ParticleTAGS.Remote

cdef extern from 'math.h':
    cdef double ceil(double) nogil
    cdef double floor(double) nogil
    cdef double fabs(double) nogil

cdef extern from 'limits.h':
    cdef int INT_MAX
    cdef unsigned int UINT_MAX

################################################################
# ParticleArrayExchange
################################################################w
cdef class ParticleArrayExchange:
    def __init__(self, int pa_index, ParticleArray pa, object comm):
        self.pa_index = pa_index
        self.pa = pa
        self.pa_wrapper = NNPSParticleArrayWrapper( pa )

        # unique data and message length tags for MPI communications
        name = pa.name
        self.msglength_tag_remote = sum( [ord(c) for c in name + '_msglength_remote'] )
        self.data_tag_remote = sum( [ord(c) for c in name + '_data_remote'] )

        self.msglength_tag_lb = sum( [ord(c) for c in name + '_msglength_lb'] )
        self.data_tag_lb = sum( [ord(c) for c in name + '_data_lb'] )

        # MPI comm rank and size
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()

        # num particles
        self.num_local = pa.get_number_of_particles()
        self.num_remote = 0
        self.num_global = 0
        self.num_ghost = 0

        # Particle Import/Export lists
        self.exportParticleGlobalids = UIntArray()
        self.exportParticleLocalids = UIntArray()
        self.exportParticleProcs = IntArray()
        self.numParticleExport = 0

        self.importParticleGlobalids = UIntArray()
        self.importParticleLocalids = UIntArray()
        self.importParticleProcs = IntArray()
        self.numParticleImport = 0

        # load balancing props are taken from the particle array
        lb_props = pa.get_lb_props()
        lb_props.sort()

        self.lb_props = lb_props
        self.nprops = len( lb_props )

        # exchange flags
        self.lb_exchange = True
        self.remote_exchange = True

    def lb_exchange_data(self):
        """Share particle info after Zoltan_LB_Balance

        After an initial call to 'Zoltan_LB_Balance', the new size of
        the arrays should be (num_particles - numExport + numImport)
        to reflect particles that should be imported and exported.

        This function should be called after the load balancing lists
        a defined. The particles to be exported are removed and the
        arrays re-sized. MPI is then used to send and receive the data

        """
        # data array
        cdef ParticleArray pa = self.pa

        # Export lists for the particles
        cdef UIntArray exportGlobalids = self.exportParticleGlobalids
        cdef UIntArray exportLocalids = self.exportParticleLocalids
        cdef IntArray exportProcs = self.exportParticleProcs
        cdef int numImport, numExport = self.numParticleExport

        # MPI communicator
        cdef object comm = self.comm

        # message tags
        cdef int dtag = self.data_tag_lb

        # create the zoltan communicator and get number of objects to import
        cdef ZComm zcomm = ZComm(comm, dtag, numExport, exportProcs.get_npy_array())
        numImport = zcomm.nreturn

        # extract particles to be exported
        cdef dict sendbufs = self.get_sendbufs( exportLocalids )

        # remove particles to be exported
        pa.remove_particles(exportLocalids)
        pa.align_particles()

        # old and new sizes
        cdef int current_size = self.num_local
        cdef int count = current_size - numExport
        cdef int newsize = count + numImport

        # resize the particle array to accomodate the receive
        pa.resize( newsize )

        # exchange data
        self.exchange_data(zcomm, sendbufs, count)

        # set all particle tags to local
        self.set_tag(count, newsize, Local)
        pa.align_particles()

        # update the number of particles
        self.num_local = newsize

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

        # Export lists
        cdef UIntArray exportGlobalids = self.exportParticleGlobalids
        cdef UIntArray exportLocalids = self.exportParticleLocalids
        cdef IntArray exportProcs = self.exportParticleProcs
        cdef int numImport, numExport = self.numParticleExport

        # current number of particles
        cdef int current_size = self.num_local
        cdef int count, new_size

        # MPI communicator
        cdef object comm = self.comm

        # zoltan communicator
        cdef int dtag = self.data_tag_remote
        cdef ZComm zcomm = ZComm(comm, dtag, numExport, exportProcs.get_npy_array())
        numImport = zcomm.nreturn

        # old and new sizes
        count = current_size
        newsize = current_size + numImport

        # copy particles to be exported
        cdef dict sendbufs = self.get_sendbufs( exportLocalids )

        # update the size of the array
        pa.resize( newsize )
        self.exchange_data(zcomm, sendbufs, count)

        # set tags for all received particles as Remote
        self.set_tag(count, newsize, Remote)
        pa.align_particles()

        # store the number of remote particles
        self.num_remote = newsize - current_size

    cdef exchange_data(self, ZComm zcomm, dict sendbufs, int count):
        cdef ParticleArray pa = self.pa
        cdef str prop
        cdef int prop_tag, nbytes, i, nprops = self.nprops

        cdef np.ndarray prop_arr, sendbuf, recvbuf

        cdef list props = self.lb_props

        for i in range(nprops):
            prop = props[i]

            prop_arr = pa.properties[prop].get_npy_array()
            nbytes = prop_arr.dtype.itemsize

            # the send and receive buffers
            sendbuf = sendbufs[prop]
            recvbuf = prop_arr[count:]

            # tag for this property
            prop_tag = i

            # set the nbytes and tag for the zoltan communicator
            zcomm.set_nbytes( nbytes )
            zcomm.set_tag( prop_tag )

            # exchange the data
            zcomm.Comm_Do( sendbuf, recvbuf )

    def remove_remote_particles(self):
        self.num_local = self.pa.get_number_of_particles(real=True)
        cdef int num_local = self.num_local

        # resize the particle array
        self.pa.resize( num_local )
        self.pa.align_particles()

        # reset the number of remote particles
        self.num_remote = 0

    def align_particles(self):
        self.pa.align_particles()

    def set_tag(self, int start, int end, int value):
        """Reset the annoying tag value after particles are resized."""
        cdef int i
        cdef IntArray tag = self.pa_wrapper.tag

        for i in range(start, end):
            tag[i] = value

    def update_particle_gids(self):
        """Update the global indices.

        We call a utility function to get the new number of particles
        across the processors and then linearly assign indices to the
        objects.

        """
        cdef int num_global_objects, num_local_objects, _sum, i
        cdef np.ndarray[ndim=1, dtype=np.int32_t] num_objects_data

        # the global indices array
        cdef UIntArray gid = self.pa_wrapper.gid

        cdef object comm = self.comm
        cdef int rank = self.rank
        cdef int size = self.size

        num_objects_data = zoltan_utils.get_num_objects_per_proc(
            comm, self.num_local)

        num_local_objects = num_objects_data[ rank ]
        num_global_objects = np.sum( num_objects_data )

        _sum = np.sum( num_objects_data[:rank] )

        gid.resize( num_local_objects )
        for i in range( num_local_objects ):
            gid.data[i] = <ZOLTAN_ID_TYPE> ( _sum + i )

        # set the number of local and global objects
        self.num_global = num_global_objects

    def reset_lists(self):
        """Reset the particle lists"""
        self.numParticleExport = 0
        self.numParticleImport = 0

        self.exportParticleGlobalids.reset()
        self.exportParticleLocalids.reset()
        self.exportParticleProcs.reset()

        self.importParticleGlobalids.reset()
        self.importParticleLocalids.reset()
        self.importParticleProcs.reset()

    def extend(self, int currentsize, int newsize):
        self.pa.resize( newsize )

    def get_sendbufs(self, UIntArray exportIndices):
        cdef ParticleArray pa = self.pa
        cdef dict sendbufs = {}
        cdef str prop

        cdef np.ndarray indices = exportIndices.get_npy_array()

        for prop in pa.properties:
            prop_arr = pa.properties[prop].get_npy_array()
            sendbufs[prop] = prop_arr[ indices ]

        return sendbufs

# #################################################################
# # ParallelManager extension classes
# #################################################################
cdef class ParallelManager:
    """Base class for all parallel managers."""
    def __init__(self, int dim, list particles, object comm,
                 double radius_scale=2.0,
                 int ghost_layers=2, domain=None,
                 bint update_cell_sizes=True):
        """Constructor.

        Parameters
        ----------

        dim : int
            Dimension

        particles : list
            list of particle arrays to be managed.

        comm : mpi4py.MPI.COMM, default
            MPI communicator for parallel invocations

        radius_scale : double, default (2)
            Optional kernel radius scale. Defaults to 2

        ghost_layers : int, default (2)
            Optional factor for computing bounding boxes for cells.

        domain : DomainManager, default (None)
            Optional limits for the domain

        """
        # number of arrays and a reference to the particle list
        self.narrays = len(particles)
        self.particles = particles

        # particle array exchange instances
        self.pa_exchanges = [ParticleArrayExchange(i, pa, comm) \
                             for i, pa in enumerate(particles)]

        # particle array wrappers
        self.pa_wrappers = [exchange.pa_wrapper for exchange in self.pa_exchanges]

        # number of local/global/remote particles
        self.num_local = [exchange.num_local for exchange in self.pa_exchanges]
        self.num_global = [0] * self.narrays
        self.num_remote = [0] * self.narrays

        # global indices used for load balancing
        self.cell_gid = UIntArray()

        # cell coordinate values used for load balancing
        self.cx = DoubleArray()
        self.cy = DoubleArray()
        self.cz = DoubleArray()

        # minimum and maximum arrays for MPI reduce operations. These
        # are used to find global bounds across processors.
        self.minx = np.zeros( shape=1, dtype=np.float64 )
        self.miny = np.zeros( shape=1, dtype=np.float64 )
        self.minz = np.zeros( shape=1, dtype=np.float64 )

        # global minimum values for x, y and z
        self.mx = 0.0; self.my = 0.0; self.mz = 0.0

        self.maxx = np.zeros( shape=1, dtype=np.float64 )
        self.maxy = np.zeros( shape=1, dtype=np.float64 )
        self.maxz = np.zeros( shape=1, dtype=np.float64 )
        self.maxh = np.zeros( shape=1, dtype=np.float64 )

        # global max values for x ,y, z & h
        self.Mx = 0.0; self.My = 0.0; self.Mz = 0.0; self.Mh = 0.0

        # MPI comm rank and size
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()

        self.in_parallel = True
        if self.size == 1: self.in_parallel = False

        # The cell_map dictionary, radius scale for binning and ghost
        # layers for remote neighbors.
        self.cell_map = {}
        self.cell_list = []

        # number of loca/remote cells
        self.ncells_local = 0
        self.ncells_remote = 0
        self.ncells_total = 0

        # radius scale and ghost layers
        self.radius_scale = radius_scale
        self.ghost_layers = ghost_layers

        # setup cell import/export lists
        self._setup_arrays()

        # flags to re-compute cell sizes
        self.initial_update = True
        self.update_cell_sizes = update_cell_sizes

        # update the particle global ids at startup
        self.update_particle_gids()
        self.local_bin()

        # load balancing frequency and counter
        self.lb_count = 0
        self.lb_freq = 1

        # array for global reduction of time steps
        self.dt_sendbuf = np.array( [1.0], dtype=np.float64 )

    def update_time_steps(self, double local_dt):
        """Peform a reduction to compute the globally stable time steps"""
        cdef np.ndarray dt_sendbuf = self.dt_sendbuf
        cdef np.ndarray dt_recvbuf = np.zeros_like(dt_sendbuf)

        comm = self.comm

        # set the local time step and peform the global reduction
        dt_sendbuf[0] = local_dt
        comm.Allreduce( sendbuf=dt_sendbuf, recvbuf=dt_recvbuf, op=mpi.MIN )

        return dt_recvbuf[0]

    cpdef compute_cell_size(self):
        """Compute the cell size for the binning.

        The cell size is chosen as the kernel radius scale times the
        maximum smoothing length in the local processor. For parallel
        runs, we would need to communicate the maximum 'h' on all
        processors to decide on the appropriate binning size.

        """
        # compute global bounds for the solution
        self._compute_bounds()
        cdef double hmax = self.Mh

        cell_size = self.radius_scale * hmax
        if cell_size < 1e-6:
            msg = """Cell size too small %g. Perhaps h = 0?
            Setting cell size to 1"""%(cell_size)
            print msg
            cell_size = 1.0
        self.cell_size = cell_size

    def update_cell_gids(self):
        """Update global indices for the cell_map dictionary.

        The objects to be partitioned in this class are the cells and
        we need to number them uniquely across processors. The
        numbering is done sequentially starting from the processor
        with rank 0 to the processor with rank size-1

        """
        # update the cell gids
        cdef PyZoltan pz = self.pz

        pz.num_local_objects = PyList_GET_SIZE( self.cell_list )
        pz._update_gid( self.cell_gid )

    def update_particle_gids(self):
        """Update individual particle global indices"""
        for i in range(self.narrays):
            pa_exchange = self.pa_exchanges[i]
            pa_exchange.update_particle_gids()

            self.num_local[i] = pa_exchange.num_local
            self.num_global[i] = pa_exchange.num_global

    def update(self):
        cdef int lb_freq = self.lb_freq
        cdef int lb_count = self.lb_count

        lb_count += 1

        # remove remote particles from a previous step
        self.remove_remote_particles()

        if ( lb_count == lb_freq ):
            self.update_partition()
            self.lb_count = 0
        else:
            self.migrate_partition()
            self.lb_count = lb_count

    def update_partition(self):
        """Update the partition.

        This is the main entry point for the parallel manager. Given
        particles distributed across processors, we bin them, assign
        uniqe global indices for the cells and particles and
        subsequently perform the following steps:

        (a) Call a load balancing function to get import and export
            lists for the cell indices.

        for each particle array:

            (b) Create particle import/export lists using the cell
                import/export lists

            (c) Call ParticleArrayExchange's lb_exchange data with these particle
                import/export lists to effect the data movement for particles.

        (d) Update the local cell map after the data has been exchanged.

        now that we have the updated map, for each particle array:

            (e) Compute remote particle import/export lists from the cell map

            (f) Call ParticleArrayExchange's remote_exchange_data with these lists
                to effect the data movement.

        now that the data movement is done,

        (g) Update the local cell map to accomodate remote particles.

        Notes
        -----

        Although not intended to be a 'parallel' NNPS, the
        ParallelManager can be used for this purpose. I don't think
        this is advisable in variable smoothing length simulations
        where we should be using a very coarse grained partitioning to
        ensure safe local computations.

        For two step integrators commonly employed in SPH and with a
        sufficient number of ghost layers for remote particles, we
        should call 'update' only at the end of a time step. The idea
        of multiple ghost layers is to ensure that local particles
        have a sufficiently large halo region around them.

        """
        # update particle gids, bin particles and update cell gids
        #self.update_particle_gids()
        self.local_bin()
        self.update_cell_gids()

        if self.in_parallel:
            # use Zoltan to get the cell import/export lists
            self.load_balance()

            # move the data to the new partitions
            if self.changes == 1:
                self.create_particle_lists()

                # exchange the data for each array
                for i in range(self.narrays):
                    self.pa_exchanges[i].lb_exchange_data()

                # update the local cell map after data movement
                self.update_local_data()

            # compute remote particles and exchange data
            self.compute_remote_particles()

            for i in range(self.narrays):
                self.pa_exchanges[i].remote_exchange_data()

            # update the local cell map to accomodate remote particles
            self.update_remote_data()

            # set the particle pids now that we have the partitions
            for i in range(self.narrays):
                pa = self.particles[i]
                pa.set_pid( self.rank )

    def migrate_partition(self):
        # migrate particles
        self.migrate_particles()

        # update local data
        self.update_local_data()

        # compute remote particles and exchange data
        self.compute_remote_particles()
        for i in range(self.narrays):
            self.pa_exchanges[i].remote_exchange_data()

        # update the local cell map to accomodate remotes
        self.update_remote_data()

        # set the particle pids
        for i in range(self.narrays):
            self.particles[i].set_pid( self.rank )

    def remove_remote_particles(self):
        """Remove remote particles"""
        cdef int narrays = self.narrays
        cdef ParticleArrayExchange pa_exchange

        for i in range(narrays):
            pa_exchange = self.pa_exchanges[i]
            pa_exchange.remove_remote_particles()

            self.num_local[i] = pa_exchange.num_local
            self.num_remote[i] = pa_exchange.num_remote

    def local_bin(self):
        """Create the local cell map.

        Bin the particles by deleting any previous cells and
        re-computing the indexing structure. This corresponds to a
        local binning process that is called when each processor has
        a given list of particles to deal with.

        """
        cdef dict cell_map = self.cell_map
        cdef int num_particles
        cdef UIntArray indices

        # clear the indices for the cell_map.
        self.cell_map.clear()
        self.cell_list = []
        self.ncells_total = 0

        # compute the cell size
        if self.initial_update or self.update_cell_sizes:
            self.compute_cell_size()

        # # deal with ghosts
        # if self.is_periodic:
        #     # remove-ghost-particles
        #     self._remove_ghost_particles()

        #     # adjust local particles
        #     self._adjust_particles()

        #     # create new ghosts
        #     self._create_ghosts()

        # bin each particle array separately
        for i in range(self.narrays):
            num_particles = self.num_local[i]

            # bin the particles
            indices = arange_uint(num_particles)
            self._bin( i, indices )

        # local number of cells at this point are the total number of cells
        self.ncells_local = self.ncells_total

    cdef _bin(self, int pa_index, UIntArray indices):
        """Bin a given particle array with indices.

        Parameters
        ----------

        pa_index : int
            Index of the particle array corresponding to the particles list

        indices : UIntArray
            Subset of particles to bin

        """
        cdef NNPSParticleArrayWrapper pa_wrapper = self.pa_wrappers[ pa_index ]
        cdef DoubleArray x = pa_wrapper.x
        cdef DoubleArray y = pa_wrapper.y
        cdef DoubleArray z = pa_wrapper.z
        cdef UIntArray gid = pa_wrapper.gid

        cdef dict cell_map = self.cell_map
        cdef list cell_list = self.cell_list
        cdef double cell_size = self.cell_size

        cdef UIntArray lindices, gindices
        cdef size_t num_particles, indexi
        cdef ZOLTAN_ID_TYPE i

        cdef Cell cell
        cdef cIntPoint _cid
        cdef IntPoint cid = IntPoint()
        cdef cPoint pnt

        cdef int ncells_total = 0
        cdef int narrays = self.narrays
        cdef int layers = self.ghost_layers

        # now bin the particles
        num_particles = indices.length
        for indexi in range(num_particles):
            i = indices.data[indexi]

            pnt = cPoint_new( x.data[i], y.data[i], z.data[i] )
            _cid = find_cell_id( pnt, cell_size )

            cid = IntPoint_from_cIntPoint(_cid)

            if not cell_map.has_key( cid ):
                cell = Cell(cid=cid, cell_size=cell_size,
                            narrays=self.narrays, layers=layers)

                # store this cell in the cells list and cell map
                PyList_Append( cell_list, cell )
                cell_map[ cid ] = cell

                ncells_total = ncells_total + 1

            # add this particle to the list of indicies
            cell = cell_map[ cid ]

            lindices = cell.lindices[pa_index]
            gindices = cell.gindices[pa_index]

            lindices.append( i )
            gindices.append( gid.data[i] )

            # increment the size of the cells to reflect the added
            # particle. The size will be used to set the weights.
            cell.size += 1

        # update the total number of cells
        self.ncells_total = self.ncells_total + ncells_total

    def update_local_data(self):
        """Update the cell map after load balance.

        After the load balancing step, each processor has a new set of
        local particles which need to be indexed. This new cell
        structure is then used to compute remote particles.

        """
        cdef ParticleArrayExchange pa_exchange
        cdef int num_particles, i
        cdef UIntArray indices

        # clear the local cell_map dict
        self.cell_map.clear()
        self.cell_list = []
        self.ncells_total = 0

        for i in range(self.narrays):
            pa_exchange = self.pa_exchanges[i]
            num_particles = pa_exchange.num_local

            # set the number of local and global particles
            self.num_local[i] = pa_exchange.num_local
            self.num_global[i] = pa_exchange.num_global

            # bin the particles
            indices = arange_uint( num_particles )
            self._bin(i, indices)

        self.ncells_local = self.ncells_total

    def update_remote_data(self):
        """Update the cell structure after sharing remote particles.

        After exchanging remote particles, we need to index the new
        remote particles for a possible neighbor query.

        """
        cdef int narrays = self.narrays
        cdef int num_local, num_remote, i
        cdef ParticleArrayExchange pa_exchange
        cdef UIntArray indices

        for i in range(narrays):
            pa_exchange = self.pa_exchanges[i]

            # get the number of local and remote particles
            num_local = pa_exchange.num_local
            num_remote = pa_exchange.num_remote

            # update the number of local/remote particles
            self.num_local[i] = num_local
            self.num_remote[i] = num_remote

            indices = arange_uint( num_local, num_local + num_remote )
            self._bin( i, indices )

        # compute the number of remote cells added
        self.ncells_remote = self.ncells_total - self.ncells_local

    def save_partition(self, fname, count=0):
        """Collect cell data from processors and save"""
        # get the global number of cells
        cdef cPoint centroid
        cdef int ncells_total = self.ncells_total
        cdef int ncells_local = self.ncells_local

        # cell centroid arrays
        cdef double[:] x = np.zeros( shape=ncells_total, dtype=np.float64 )
        cdef double[:] y = np.zeros( shape=ncells_total, dtype=np.float64 )
        cdef int[:] lid = np.zeros( shape=ncells_total, dtype=np.int32 )

        for i in range( ncells_total ):
            cell = self.cell_list[i]; centroid = cell.centroid
            x[i] = centroid.x; y[i] = centroid.y
            lid[i] = i

        cdef int[:] tag = np.zeros(shape=(ncells_total,), dtype=np.int32)
        tag[ncells_local:] = 1

        cdef double cell_size = self.cell_size

        # save the partition locally
        fname = fname + '/partition%03d.%d'%(count, self.rank)
        np.savez(
            fname, x=x, y=y, lid=lid, tag=tag, cell_size=cell_size,
            ncells_local=self.ncells_local, ncells_total=self.ncells_total
        )

    def set_lb_freq(self, int lb_freq):
        self.lb_freq = lb_freq

    def load_balance(self):
        raise NotImplementedError("ParallelManager::load_balance")

    def create_particle_lists(self, pa_index):
        raise NotImplementedError("ParallelManager::create_particle_lists")

    def compute_remote_particles(self, pa_index):
        raise NotImplementedError("ParallelManager::compute_remote_particles")

    #######################################################################
    # Private interface
    #######################################################################
    def set_data(self):
        """Compute the user defined data for use with Zoltan"""
        raise NotImplementedError("ZoltanParallelManager::_set_data should not be called!")

    def _setup_arrays(self):
        # Import/Export lists for cells
        self.exportCellGlobalids = UIntArray()
        self.exportCellLocalids = UIntArray()
        self.exportCellProcs = IntArray()

        self.importCellGlobalids = UIntArray()
        self.importCellLocalids = UIntArray()
        self.importCellProcs = IntArray()

        self.numCellImport = 0
        self.numCellExport = 0

    cdef _compute_bounds(self):
        """Compute the domain bounds for indexing."""

        cdef double mx, my, mz
        cdef double Mx, My, Mz, Mh

        cdef list pa_wrappers = self.pa_wrappers
        cdef NNPSParticleArrayWrapper pa
        cdef DoubleArray x, y, z, h

        # set some high and low values
        cdef double high = -1e20, low = 1e20
        mx = low; my = low; mz = low
        Mx = high; My = high; Mz = high; Mh = high

        # find the local min and max for all arrays on this proc
        for pa in pa_wrappers:
            x = pa.x; y = pa.y; z = pa.z; h = pa.h
            if x.length > 0:
                x.update_min_max(); y.update_min_max()
                z.update_min_max(); h.update_min_max()

                if x.minimum < mx: mx = x.minimum
                if x.maximum > Mx: Mx = x.maximum

                if y.minimum < my: my = y.minimum
                if y.maximum > My: My = y.maximum

                if z.minimum < mz: mz = z.minimum
                if z.maximum > Mz: Mz = z.maximum

                if h.maximum > Mh: Mh = h.maximum

        self.minx[0] = mx; self.miny[0] = my; self.minz[0] = mz
        self.maxx[0] = Mx; self.maxy[0] = My; self.maxz[0] = Mz
        self.maxh[0] = Mh

        # now compute global min and max if in parallel
        comm = self.comm
        if self.in_parallel:

            # revc buffers for all reduce
            _minx = np.zeros_like(self.minx)
            _miny = np.zeros_like(self.miny)
            _minz = np.zeros_like(self.minz)

            _maxx = np.zeros_like(self.maxx)
            _maxy = np.zeros_like(self.maxy)
            _maxz = np.zeros_like(self.maxz)
            _maxh = np.zeros_like(self.maxh)

            # global reduction for minimum values
            comm.Allreduce(sendbuf=self.minx, recvbuf=_minx, op=mpi.MIN)
            comm.Allreduce(sendbuf=self.miny, recvbuf=_miny, op=mpi.MIN)
            comm.Allreduce(sendbuf=self.minz, recvbuf=_minz, op=mpi.MIN)

            # global reduction for maximum values
            comm.Allreduce(sendbuf=self.maxx, recvbuf=_maxx, op=mpi.MAX)
            comm.Allreduce(sendbuf=self.maxy, recvbuf=_maxy, op=mpi.MAX)
            comm.Allreduce(sendbuf=self.maxz, recvbuf=_maxz, op=mpi.MAX)
            comm.Allreduce(sendbuf=self.maxh, recvbuf=_maxh, op=mpi.MAX)

        self.mx = _minx[0]; self.my = _miny[0]; self.mz = _minz[0]
        self.Mx = _maxx[0]; self.My = _maxy[0]; self.Mz = _maxz[0]
        self.Mh = _maxh[0]

    ######################################################################
    # Neighbor location routines
    ######################################################################
    cpdef get_nearest_particles(self, int src_index, int dst_index,
                                size_t d_idx, UIntArray nbrs):
        """Utility function to get near-neighbors for a particle.

        Parameters
        ----------

        src_index : int
            Index of the source particle array in the particles list

        dst_index : int
            Index of the destination particle array in the particles list

        d_idx : int (input)
            Particle for which neighbors are sought.

        nbrs : UIntArray (output)
            Neighbors for the requested particle are stored here.

        """
        cdef dict cell_map = self.cell_map
        cdef Cell cell

        cdef NNPSParticleArrayWrapper src = self.pa_wrappers[ src_index ]
        cdef NNPSParticleArrayWrapper dst = self.pa_wrappers[ dst_index ]

        # Source data arrays
        cdef DoubleArray s_x = src.x
        cdef DoubleArray s_y = src.y
        cdef DoubleArray s_z = src.z
        cdef DoubleArray s_h = src.h

        # Destination particle arrays
        cdef DoubleArray d_x = dst.x
        cdef DoubleArray d_y = dst.y
        cdef DoubleArray d_z = dst.z
        cdef DoubleArray d_h = dst.h

        cdef double radius_scale = self.radius_scale
        cdef double cell_size = self.cell_size
        cdef UIntArray lindices
        cdef size_t indexj
        cdef ZOLTAN_ID_TYPE j

        cdef cPoint xi = cPoint_new(d_x.data[d_idx], d_y.data[d_idx], d_z.data[d_idx])
        cdef cIntPoint _cid = find_cell_id( xi, cell_size )
        cdef IntPoint cid = IntPoint_from_cIntPoint( _cid )
        cdef IntPoint cellid = IntPoint(0, 0, 0)

        cdef cPoint xj
        cdef double xij

        cdef double hi, hj
        hi = radius_scale * d_h.data[d_idx]

        cdef int nnbrs = 0

        cdef int ix, iy, iz
        for ix in [cid.data.x -1, cid.data.x, cid.data.x + 1]:
            for iy in [cid.data.y - 1, cid.data.y, cid.data.y + 1]:
                for iz in [cid.data.z -1, cid.data.z, cid.data.z + 1]:
                    cellid.data.x = ix; cellid.data.y = iy
                    cellid.data.z = iz

                    if cell_map.has_key( cellid ):
                        cell = cell_map[ cellid ]
                        lindices = cell.lindices[src_index]
                        for indexj in range( lindices.length ):
                            j = lindices.data[indexj]

                            xj = cPoint_new( s_x.data[j], s_y.data[j], s_z.data[j] )
                            xij = cPoint_distance( xi, xj )

                            hj = radius_scale * s_h.data[j]

                            if ( (xij < hi) or (xij < hj) ):
                                if nnbrs == nbrs.length:
                                    nbrs.resize( nbrs.length + 50 )
                                    print """Neighbor search :: Extending the neighbor list to %d"""%(nbrs.length)

                                nbrs.data[ nnbrs ] = j
                                nnbrs = nnbrs + 1

        # update the length for nbrs to indicate the number of neighbors
        nbrs.length = nnbrs

cdef class ZoltanParallelManager(ParallelManager):
    """Base class for Zoltan enabled parallel cell managers.

    To partition a list of arrays, we do an NNPS like box sort on all
    arrays to create a global spatial indexing structure. The cell_map
    are then used as 'objects' to be partitioned by Zoltan. The cell_map
    dictionary (cell-map) need not be unique across processors. We are
    responsible for assignning unique global ids for the cells.

    The Zoltan generated (cell) import/export lists are then used to
    construct particle import/export lists which are used to perform
    the data movement of the particles. A ParticleArrayExchange object
    (lb_exchange_data and remote_exchange_data) is used for each
    particle array in turn to effect this movement.

    """
    def __init__(self, int dim, list particles, object comm,
                 double radius_scale=2.0,
                 int ghost_layers=2, domain=None,
                 bint update_cell_sizes=True):
        """Constructor.

        Parameters
        ----------

        dim : int
            Dimension

        particles : list
            list of particle arrays to be managed.

        comm : mpi4py.MPI.COMM, default
            MPI communicator for parallel invocations

        radius_scale : double, default (2)
            Optional kernel radius scale. Defaults to 2

        ghost_layers : int, default (2)
            Optional factor for computing bounding boxes for cells.

        domain : DomainManager, default (None)
            Optional limits for the domain

        """
        super(ZoltanParallelManager, self).__init__(
            dim, particles, comm, radius_scale, ghost_layers, domain, update_cell_sizes)

        # Initialize the base PyZoltan class.
        self.pz = PyZoltan(comm)

    def create_particle_lists(self):
        """Create particle export/import lists

        For the cell based partitioner, Zoltan generated import and
        export lists apply to the cells. From this data, we can
        generate local export indices for the particles which must
        then be inverted to get the requisite import lists.

        We first construct the particle export lists in local arrays
        using the information from Zoltan_LB_Balance and subsequently
        call Zoltan_Invert_Lists to get the import lists. These arrays
        are then copied to the ParticleArrayExchange import/export
        lists.

        """
        # these are the Zoltan generated lists that correspond to cells
        cdef UIntArray exportCellLocalids = self.exportCellLocalids
        cdef UIntArray exportCellGlobalids = self.exportCellGlobalids
        cdef IntArray exportCellProcs = self.exportCellProcs
        cdef int numCellExport = self.numCellExport

        # the local cell map and cellids
        cdef int narrays = self.narrays
        cdef list cell_list = self.cell_list
        cdef ParticleArrayExchange pa_exchange

        cdef UIntArray exportLocalids, exportGlobalids
        cdef IntArray exportProcs

        cdef int indexi, i, j, export_proc, nindices, pa_index
        cdef UIntArray lindices, gindices
        cdef IntPoint cellid
        cdef Cell cell

        # initialize the particle export lists
        for pa_index in range(self.narrays):
            pa_exchange = self.pa_exchanges[pa_index]
            pa_exchange.reset_lists()

        # Iterate over the cells to be exported
        for indexi in range(numCellExport):
            i = <int>exportCellLocalids[indexi]            # local index for the cell to export
            cell = cell_list[i]                            # local cell to export
            export_proc = exportCellProcs.data[indexi]     # processor to export cell to

            # populate the export lists for each array in this cell
            for pa_index in range(self.narrays):
                # the ParticleArrayExchange object for this array
                pa_exchange = self.pa_exchanges[pa_index]

                exportLocalids = pa_exchange.exportParticleLocalids
                exportGlobalids = pa_exchange.exportParticleGlobalids
                exportProcs = pa_exchange.exportParticleProcs

                # local and global indices in this cell
                lindices = cell.lindices[pa_index]
                gindices = cell.gindices[pa_index]
                nindices = lindices.length

                for j in range( nindices ):
                    exportLocalids.append( lindices.data[j] )
                    exportGlobalids.append( gindices.data[j] )
                    exportProcs.append( export_proc )

        # save the number of particles
        for pa_index in range(narrays):
            pa_exchange = self.pa_exchanges[pa_index]
            pa_exchange.numParticleExport = pa_exchange.exportParticleProcs.length

    def compute_remote_particles(self):
        """Compute remote particles.

        Particles to be exported are determined by flagging individual
        cells and where they need to be shared to meet neighbor
        requirements.

        """
        # the PyZoltan object used to find intersections
        cdef PyZoltan pz = self.pz

        cdef Cell cell
        cdef cPoint boxmin, boxmax

        cdef object comm = self.comm
        cdef int rank = self.rank, narrays = self.narrays

        cdef list cell_list = self.cell_list
        cdef UIntArray lindices, gindices

        cdef np.ndarray nbrprocs
        cdef np.ndarray[ndim=1, dtype=np.int32_t] procs

        cdef int nbrproc, num_particles, indexi, pa_index
        cdef ZOLTAN_ID_TYPE i

        cdef int numprocs = 0

        cdef ParticleArrayExchange pa_exchange
        cdef UIntArray exportGlobalids, exportLocalids
        cdef IntArray exportProcs

        # reset the export lists
        for pa_index in range(narrays):
            pa_exchange = self.pa_exchanges[pa_index]
            pa_exchange.reset_lists()

        # Chaeck for each cell
        for cell in cell_list:

            # get the bounding box for this cell
            boxmin = cell.boxmin; boxmax = cell.boxmax

            numprocs = pz.Zoltan_Box_PP_Assign(
                boxmin.x, boxmin.y, boxmin.z,
                boxmax.x, boxmax.y, boxmax.z)

            # the array of processors that this box intersects with
            procs = pz.procs

            # array of neighboring processors
            nbrprocs = procs[np.where( (procs != -1) * (procs != rank) )[0]]

            if nbrprocs.size > 0:
                cell.is_boundary = True

                # store the neighboring processors for this cell
                cell.nbrprocs.resize( nbrprocs.size )
                cell.nbrprocs.set_data( nbrprocs )

                # populate the particle export lists for each array
                for pa_index in range(narrays):
                    pa_exchange = self.pa_exchanges[pa_index]

                    exportGlobalids = pa_exchange.exportParticleGlobalids
                    exportLocalids = pa_exchange.exportParticleLocalids
                    exportProcs = pa_exchange.exportParticleProcs

                    # local and global indices in this cell
                    lindices = cell.lindices[pa_index]
                    gindices = cell.gindices[pa_index]
                    num_particles = lindices.length

                    for nbrproc in nbrprocs:
                        for indexi in range( num_particles ):
                            i = lindices.data[indexi]

                            exportLocalids.append( i )
                            exportGlobalids.append( gindices.data[indexi] )
                            exportProcs.append( nbrproc )

        # set the numParticleExport for each array
        for pa_index in range(narrays):
            pa_exchange = self.pa_exchanges[pa_index]
            pa_exchange.numParticleExport = pa_exchange.exportParticleProcs.length

    def load_balance(self):
        """Use Zoltan to generate import/export lists for the cells.

        For the Zoltan interface, we require to populate a user
        defined struct with appropriate data for Zoltan to deal
        with. For the geometric based partitioners for example, we
        require the unique cell global indices and arrays for the cell
        centroid coordinates. Computation of this data must be done
        prior to calling 'pz.Zoltan_LB_Balance' through a call to
        'set_data'

        """
        cdef PyZoltan pz = self.pz

        # set the data which will be used by the Zoltan wrapper
        self.set_data()

        # call Zoltan to get the cell import/export lists
        self.changes = pz.Zoltan_LB_Balance()

        # copy the Zoltan export lists to the cell lists
        self.numCellExport = pz.numExport
        self.exportCellGlobalids.resize( pz.numExport )
        self.exportCellGlobalids.copy_subset( pz.exportGlobalids )

        self.exportCellLocalids.resize( pz.numExport )
        self.exportCellLocalids.copy_subset( pz.exportLocalids )

        self.exportCellProcs.resize( pz.numExport )
        self.exportCellProcs.copy_subset( pz.exportProcs )

        # copy the Zoltan import lists to the cell lists
        self.numCellImport = pz.numImport
        self.importCellGlobalids.resize( pz.numImport )
        self.importCellGlobalids.copy_subset( pz.importGlobalids )

        self.importCellLocalids.resize( pz.numImport )
        self.importCellLocalids.copy_subset( pz.importLocalids )

        self.importCellProcs.resize( pz.numImport )
        self.importCellProcs.copy_subset( pz.importProcs )

    def migrate_particles(self):
        raise NotImplementedError("ZoltanParallelManager::migrate_particles called")

cdef class ZoltanParallelManagerGeometric(ZoltanParallelManager):
    """Zoltan enabled parallel manager for use with geometric load balancing.

    Use this class for the Zoltan RCB, RIB and HSFC load balancing
    algorithms.

    """
    def __init__(self, int dim, list particles, object comm,
                 double radius_scale=2.0,
                 int ghost_layers=2, domain=None,
                 str lb_method='RCB',
                 str keep_cuts="1",
                 str obj_weight_dim="1",
                 bint update_cell_sizes=True
                 ):

        # initialize the base class
        super(ZoltanParallelManagerGeometric, self).__init__(
            dim, particles, comm, radius_scale, ghost_layers, domain,
            update_cell_sizes)

        # concrete implementation of a PyZoltan class
        self.pz = ZoltanGeometricPartitioner(
            dim, self.comm, self.cx, self.cy, self.cz, self.cell_gid,
            obj_weight_dim=obj_weight_dim,
            keep_cuts=keep_cuts,
            lb_method=lb_method)

    def set_zoltan_rcb_lock_directions(self):
        """Set the Zoltan flag to fix the directions of the RCB cuts"""
        self.pz.set_rcb_lock_directions('1')

    def set_zoltan_rcb_reuse(self):
        """Use previous cuts as guesses for the current RCB cut"""
        self.pz.set_rcb_reuse('1')

    def set_zoltan_rcb_rectilinear_blocks(self):
        """Flag to control the shape of the RCB regions """
        self.pz.set_rcb_rectilinear_blocks('1')

    def set_zoltan_rcb_directions(self, value):
        """Option to group the cuts along a given direction

        Legal values (refer to the Zoltan User Guide):

        '0' = don't order cuts;
        '1' = xyz
        '2' = xzy
        '3' = yzx
        '4' = yxz
        '5' = zxy
        '6' = zyx

        """
        self.pz.set_rcb_directions(value)

    def set_data(self):
        """Set the user defined particle data structure for Zoltan.

        For the geometric based partitioners, Zoltan requires as input
        the number of local/global objects, unique global indices for
        the local objects and coordinate values for the local
        objects.

        """
        cdef ZoltanGeometricPartitioner pz = self.pz
        cdef list cell_list = self.cell_list

        cdef int num_local_objects = pz.num_local_objects
        cdef double num_global_objects1 = 1.0/pz.num_global_objects
        cdef double weight

        cdef DoubleArray x = self.cx
        cdef DoubleArray y = self.cy
        cdef DoubleArray z = self.cz

        # the weights array for PyZoltan
        cdef DoubleArray weights = pz.weights

        cdef int i
        cdef Cell cell
        cdef cPoint centroid

        # resize the coordinate and PyZoltan weight arrays
        x.resize( num_local_objects )
        y.resize( num_local_objects )
        z.resize( num_local_objects )

        weights.resize( num_local_objects )

        # populate the arrays
        for i in range( num_local_objects ):
            cell = cell_list[ i ]
            centroid = cell.centroid

            # weights are defined as cell.size/num_total
            weights.data[i] = num_global_objects1 * cell.size

            x.data[i] = centroid.x
            y.data[i] = centroid.y
            z.data[i] = centroid.z

    def migrate_particles(self):
        """Update an existing partition"""
        cdef PyZoltan pz = self.pz
        cdef int pa_index, narrays=self.narrays
        cdef double xi, yi, zi

        cdef NNPSParticleArrayWrapper pa_wrapper
        cdef ParticleArrayExchange pa_exchange

        cdef UIntArray exportGlobalids, exportLocalids
        cdef IntArray exportParticleProcs

        cdef DoubleArray x, y, z
        cdef UIntArray gid

        cdef ZOLTAN_ID_TYPE local_id, global_id

        cdef int num_particles, i, proc
        cdef int rank = self.rank

        # iterate over all arrays
        for pa_index in range(narrays):
            pa_exchange = self.pa_exchanges[ pa_index ]
            pa_wrapper = self.pa_wrappers[ pa_index ]

            # reset the export lists
            pa_exchange.reset_lists()
            exportGlobalids = pa_exchange.exportParticleGlobalids
            exportLocalids = pa_exchange.exportParticleLocalids
            exportProcs = pa_exchange.exportParticleProcs

            # particle arrays
            x = pa_wrapper.x; y = pa_wrapper.y; z = pa_wrapper.z
            gid = pa_wrapper.gid

            num_particles = self.num_local[pa_index]
            for i in range(num_particles):
                xi = x.data[i]; yi = y.data[i]; zi = z.data[i]

                # find the processor to which this point belongs
                proc = pz.Zoltan_Point_PP_Assign(xi, yi, zi)
                if proc != rank:
                    exportGlobalids.append( gid.data[i] )
                    exportLocalids.append( <ZOLTAN_ID_TYPE>i )
                    exportProcs.append(proc)

            # exchange the data given the export lists
            pa_exchange.numParticleExport = exportProcs.length
            pa_exchange.lb_exchange_data()
