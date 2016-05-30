#cython: embedsignature=True
# Library imports.
import numpy as np
cimport numpy as np

# Cython imports
from cython.operator cimport dereference as deref, preincrement as inc
from cython.parallel import parallel, prange, threadid

# malloc and friends
from libc.stdlib cimport malloc, free
from libcpp.map cimport map
from libcpp.pair cimport pair
from libcpp.vector cimport vector

cdef extern from "<algorithm>" namespace "std" nogil:
    void sort[Iter, Compare](Iter first, Iter last, Compare comp)
    void sort[Iter](Iter first, Iter last)

# cpython
from cpython.dict cimport PyDict_Clear, PyDict_Contains, PyDict_GetItem
from cpython.list cimport PyList_GetItem, PyList_SetItem, PyList_GET_ITEM

# Cython for compiler directives
cimport cython

IF OPENMP:
    cimport openmp
    cpdef int get_number_of_threads():
        cdef int i, n
        with nogil, parallel():
            for i in prange(1):
                n = openmp.omp_get_num_threads()
        return n
    cpdef set_number_of_threads(int n):
        openmp.omp_set_num_threads(n)
ELSE:
    cpdef int get_number_of_threads():
        return 1
    cpdef set_number_of_threads(int n):
        print "OpenMP not available, cannot set number of threads."


IF UNAME_SYSNAME == "Windows":
    cdef inline double fmin(double x, double y) nogil:
        return x if x < y else y
    cdef inline double fmax(double x, double y) nogil:
        return x if x > y else y

# Particle Tag information
from pyzoltan.core.carray cimport BaseArray, aligned_malloc, aligned_free
from utils import ParticleTAGS

cdef int Local = ParticleTAGS.Local
cdef int Remote = ParticleTAGS.Remote
cdef int Ghost = ParticleTAGS.Ghost

ctypedef pair[unsigned int, unsigned int] id_gid_pair_t


cdef inline bint _compare_gids(id_gid_pair_t x, id_gid_pair_t y) nogil:
    return y.second > x.second

def py_flatten(IntPoint cid, IntArray ncells_per_dim, int dim):
    """Python wrapper"""
    cdef cIntPoint _cid = cid.data
    cdef int flattened_index = flatten( _cid, ncells_per_dim, dim )
    return flattened_index

def py_get_valid_cell_index(IntPoint cid, IntArray ncells_per_dim, int dim,
                            int n_cells):
    """Return the flattened cell index for a valid cell"""
    return get_valid_cell_index(cid.data.x, cid.data.y, cid.data.z,
            ncells_per_dim.data, dim, n_cells)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline cIntPoint unflatten(long cell_index, IntArray ncells_per_dim,
                                int dim):
    """Un-flatten a linear cell index"""
    cdef int ncx = ncells_per_dim.data[0]
    cdef int ncy = ncells_per_dim.data[1]

    cdef cIntPoint cid
    cdef int ix=0, iy=0, iz=0, tmp=0

    if dim > 1:
        if dim > 2:
            tmp = ncx * ncy

            iz = cell_index/tmp
            cell_index = cell_index - iz * tmp


        iy = cell_index/ncx
        ix = cell_index - (iy * ncx)

    else:
        ix = cell_index

    # return the tuple cell index
    cid = cIntPoint_new( ix, iy, iz )
    return cid

def py_unflatten(long cell_index, IntArray ncells_per_dim, int dim):
    """Python wrapper"""
    cdef cIntPoint _cid = unflatten( cell_index, ncells_per_dim, dim )
    cdef IntPoint cid = IntPoint_from_cIntPoint(_cid)
    return cid

cdef cIntPoint find_cell_id(cPoint pnt, double cell_size):
    """ Find the cell index for the corresponding point

    Parameters
    ----------
    pnt -- the point for which the index is sought
    cell_size -- the cell size to use
    id -- output parameter holding the cell index


    Notes
    -----
    Performs a box sort based on the point and cell size

    Uses the function  `real_to_int`

    """
    cdef cIntPoint p = cIntPoint(real_to_int(pnt.x, cell_size),
                                 real_to_int(pnt.y, cell_size),
                                 real_to_int(pnt.z, cell_size))
    return p

cdef inline cPoint _get_centroid(double cell_size, cIntPoint cid):
    """ Get the centroid of the cell.

    Parameters
    ----------

    cell_size : double (input)
    Cell size used for binning

    cid : cPoint (input)
    Spatial index for a cell

    Returns
    -------

    centroid : cPoint

    Notes
    -----
    The centroid in any coordinate direction is defined to be the
    origin plus half the cell size in that direction

    """
    centroid = cPoint_new(0.0, 0.0, 0.0)
    centroid.x = (<double>cid.x + 0.5)*cell_size
    centroid.y = (<double>cid.y + 0.5)*cell_size
    centroid.z = (<double>cid.z + 0.5)*cell_size

    return centroid

def get_centroid(double cell_size, IntPoint cid):
    """ Get the centroid of the cell.

    Parameters
    ----------

    cell_size : double (input)
        Cell size used for binning

    cid : IntPoint (input)
        Spatial index for a cell

    Returns
    -------

    centroid : Point

    Notes
    -----
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

##############################################################################
cdef class NNPSParticleArrayWrapper:
    def __init__(self, ParticleArray pa):
        self.pa = pa
        self.name = pa.name

        self.x = pa.get_carray('x')
        self.y = pa.get_carray('y')
        self.z = pa.get_carray('z')
        self.h = pa.get_carray('h')

        self.gid = pa.get_carray('gid')
        self.tag = pa.get_carray('tag')

        self.np = pa.get_number_of_particles()

    cdef int get_number_of_particles(self):
        cdef ParticleArray pa = self.pa
        return pa.get_number_of_particles()

    def remove_tagged_particles(self, int tag):
        cdef ParticleArray pa = self.pa
        pa.remove_tagged_particles(tag)


##############################################################################
cdef class DomainManager:
    """This class determines the limits of the solution domain.

    We expect all simulations to have well defined domain limits
    beyond which we are either not interested or the solution is
    invalid to begin with. Thus, if a particle leaves the domain,
    the solution should be considered invalid (at least locally).

    The initial domain limits could be given explicitly or asked to be
    computed from the particle arrays. The domain could be periodic.

    """
    def __init__(self, double xmin=-1000, double xmax=1000, double ymin=0,
                 double ymax=0, double zmin=0, double zmax=0,
                 periodic_in_x=False, periodic_in_y=False, periodic_in_z=False):
        """Constructor"""
        self._check_limits(xmin, xmax, ymin, ymax, zmin, zmax)

        self.xmin = xmin; self.xmax = xmax
        self.ymin = ymin; self.ymax = ymax
        self.zmin = zmin; self.zmax = zmax

        # Indicates if the domain is periodic
        self.periodic_in_x = periodic_in_x
        self.periodic_in_y = periodic_in_y
        self.periodic_in_z = periodic_in_z
        self.is_periodic = periodic_in_x or periodic_in_y or periodic_in_z

        # get the translates in each coordinate direction
        self.xtranslate = xmax - xmin
        self.ytranslate = ymax - ymin
        self.ztranslate = zmax - zmin

        # empty list of particle array wrappers for now
        self.pa_wrappers = []
        self.narrays = 0

        # default value for the cell size
        self.cell_size = 1.0

        # default DomainManager in_parallel is set to False
        self.in_parallel = False

    #### Public protocol ################################################
    def set_pa_wrappers(self, wrappers):
        self.pa_wrappers = wrappers
        self.narrays = len(wrappers)

    def set_cell_size(self, cell_size):
        self.cell_size = cell_size

    def set_in_parallel(self, bint in_parallel):
        self.in_parallel = in_parallel

    def set_radius_scale(self, double radius_scale):
        self.radius_scale = radius_scale

    def compute_cell_size_for_binning(self):
        self._compute_cell_size_for_binning()

    def update(self, *args, **kwargs):
        """General method that is called before NNPS can bin particles.

        This method is responsible for the computation of cell sizes
        and creation of any ghost particles for periodic or wall
        boundary conditions.

        """
        # compute the cell sizes
        self._compute_cell_size_for_binning()

        # Periodicity is handled by adjusting particles according to a
        # given cubic domain box. In parallel, it is expected that the
        # appropriate parallel NNPS is responsible for the creation of
        # ghost particles.
        if self.is_periodic and not self.in_parallel:
            # remove periodic ghost particles from a previous step
            self._remove_ghosts()

            # box-wrap current particles for periodicity
            self._box_wrap_periodic()

            # create new periodic ghosts
            self._create_ghosts_periodic()

    #### Private protocol ###############################################
    cdef _add_to_array(self, DoubleArray arr, double disp):
        cdef int i
        for i in range(arr.length):
            arr.data[i] += disp

    cdef _box_wrap_periodic(self):
        """Box-wrap particles for periodicity

        The periodic domain is a rectangular box defined by minimum
        and maximum values in each coordinate direction. These values
        are used in turn to define translation values used to box-wrap
        particles that cross a periodic boundary.

        The periodic domain is specified using the DomainManager object

        """
        # minimum and maximum values of the domain
        cdef double xmin = self.xmin, xmax = self.xmax
        cdef double ymin = self.ymin, ymax = self.ymax,
        cdef double zmin = self.zmin, zmax = self.zmax

        # translations along each coordinate direction
        cdef double xtranslate = self.xtranslate
        cdef double ytranslate = self.ytranslate
        cdef double ztranslate = self.ztranslate

        # periodicity flags for NNPS
        cdef bint periodic_in_x = self.periodic_in_x
        cdef bint periodic_in_y = self.periodic_in_y
        cdef bint periodic_in_z = self.periodic_in_z

        # locals
        cdef NNPSParticleArrayWrapper pa_wrapper
        cdef DoubleArray x, y, z
        cdef double xi, yi, zi
        cdef int i, np

        # iterate over each array and mark for translation
        for pa_wrapper in self.pa_wrappers:
            x = pa_wrapper.x; y = pa_wrapper.y; z = pa_wrapper.z
            np = x.length

            # iterate over particles and box-wrap
            for i in range(np):

                if periodic_in_x:
                    if x.data[i] < xmin : x.data[i] = x.data[i] + xtranslate
                    if x.data[i] > xmax : x.data[i] = x.data[i] - xtranslate

                if periodic_in_y:
                    if y.data[i] < ymin : y.data[i] = y.data[i] + ytranslate
                    if y.data[i] > ymax : y.data[i] = y.data[i] - ytranslate

                if periodic_in_z:
                    if z.data[i] < zmin : z.data[i] = z.data[i] + ztranslate
                    if z.data[i] > zmax : z.data[i] = z.data[i] - ztranslate

    def _check_limits(self, xmin, xmax, ymin, ymax, zmin, zmax):
        """Sanity check on the limits"""
        if ( (xmax < xmin) or (ymax < ymin) or (zmax < zmin) ):
            raise ValueError("Invalid domain limits!")

    cdef _create_ghosts_periodic(self):
        """Identify boundary particles and create images.

        We need to find all particles that are within a specified
        distance from the boundaries and place image copies on the
        other side of the boundary. Corner reflections need to be
        accounted for when using domains with multiple periodicity.

        The periodic domain is specified using the DomainManager object

        """
        cdef list pa_wrappers = self.pa_wrappers
        cdef int narrays = self.narrays

        # cell size used to check for periodic ghosts. For summation
        # density like operations, we need to create two layers of
        # ghost images.
        cdef double cell_size = 2.0 * self.cell_size

        # periodic domain values
        cdef double xmin = self.xmin, xmax = self.xmax
        cdef double ymin = self.ymin, ymax = self.ymax,
        cdef double zmin = self.zmin, zmax = self.zmax

        cdef double xtranslate = self.xtranslate
        cdef double ytranslate = self.ytranslate
        cdef double ztranslate = self.ztranslate

        # periodicity flags
        cdef bint periodic_in_x = self.periodic_in_x
        cdef bint periodic_in_y = self.periodic_in_y
        cdef bint periodic_in_z = self.periodic_in_z

        # locals
        cdef NNPSParticleArrayWrapper pa_wrapper
        cdef ParticleArray pa, added
        cdef DoubleArray x, y, z, xt, yt, zt
        cdef double xi, yi, zi
        cdef int array_index, i, np

        # temporary indices for particles to be replicated
        cdef LongArray x_low, x_high, y_high, y_low, z_high, z_low, low, high

        x_low = LongArray(); x_high = LongArray()
        y_high = LongArray(); y_low = LongArray()
        z_high = LongArray(); z_low = LongArray()
        low = LongArray(); high = LongArray()

        for array_index in range(narrays):
            pa_wrapper = pa_wrappers[ array_index ]
            pa = pa_wrapper.pa
            x = pa_wrapper.x; y = pa_wrapper.y; z = pa_wrapper.z

            # reset the length of the arrays
            x_low.reset(); x_high.reset(); y_high.reset(); y_low.reset()

            np = x.length
            for i in range(np):
                xi = x.data[i]; yi = y.data[i]; zi = z.data[i]

                if periodic_in_x:
                    if ( (xi - xmin) <= cell_size ): x_low.append(i)
                    if ( (xmax - xi) <= cell_size ): x_high.append(i)

                if periodic_in_y:
                    if ( (yi - ymin) <= cell_size ): y_low.append(i)
                    if ( (ymax - yi) <= cell_size ): y_high.append(i)

                if periodic_in_z:
                    if ( (zi - zmin) <= cell_size ): z_low.append(i)
                    if ( (zmax - zi) <= cell_size ): z_high.append(i)


            # now treat each case separately and append to the main array
            added = ParticleArray(x=None, y=None, z=None)
            x = added.get_carray('x')
            y = added.get_carray('y')
            z = added.get_carray('z')
            if periodic_in_x:
                # x_low
                copy = pa.extract_particles( x_low )
                self._add_to_array(copy.get_carray('x'), xtranslate)
                added.append_parray(copy)

                # x_high
                copy = pa.extract_particles( x_high )
                self._add_to_array(copy.get_carray('x'), -xtranslate)
                added.append_parray(copy)

            if periodic_in_y:
                # Now do the corners from the previous.
                low.reset(); high.reset()
                np = x.length
                for i in range(np):
                    yi = y.data[i]
                    if ( (yi - ymin) <= cell_size ): low.append(i)
                    if ( (ymax - yi) <= cell_size ): high.append(i)

                copy = added.extract_particles(low)
                self._add_to_array(copy.get_carray('y'), ytranslate)
                added.append_parray(copy)

                copy = added.extract_particles(high)
                self._add_to_array(copy.get_carray('y'), -ytranslate)
                added.append_parray(copy)

                # Add the actual y_high and y_low now.
                # y_high
                copy = pa.extract_particles( y_high )
                self._add_to_array(copy.get_carray('y'), -ytranslate)
                added.append_parray(copy)

                # y_low
                copy = pa.extract_particles( y_low )
                self._add_to_array(copy.get_carray('y'), ytranslate)
                added.append_parray(copy)

            if periodic_in_z:
                # Now do the corners from the previous.
                low.reset(); high.reset()
                np = x.length
                for i in range(np):
                    zi = z.data[i]
                    if ( (zi - zmin) <= cell_size ): low.append(i)
                    if ( (zmax - zi) <= cell_size ): high.append(i)

                copy = added.extract_particles(low)
                self._add_to_array(copy.get_carray('z'), ztranslate)
                added.append_parray(copy)

                copy = added.extract_particles(high)
                self._add_to_array(copy.get_carray('z'), -ztranslate)
                added.append_parray(copy)

                # Add the actual z_high and z_low now.
                # z_high
                copy = pa.extract_particles( z_high )
                self._add_to_array(copy.get_carray('z'), -ztranslate)
                added.append_parray(copy)

                # z_low
                copy = pa.extract_particles( z_low )
                self._add_to_array(copy.get_carray('z'), ztranslate)
                added.append_parray(copy)


            added.tag[:] = Ghost
            pa.append_parray(added)

    cdef _compute_cell_size_for_binning(self):
        """Compute the cell size for the binning.

        The cell size is chosen as the kernel radius scale times the
        maximum smoothing length in the local processor. For parallel
        runs, we would need to communicate the maximum 'h' on all
        processors to decide on the appropriate binning size.

        """
        cdef list pa_wrappers = self.pa_wrappers
        cdef NNPSParticleArrayWrapper pa_wrapper
        cdef DoubleArray h
        cdef double cell_size
        cdef double _hmax, hmax = -1.0

        for pa_wrapper in pa_wrappers:
            h = pa_wrapper.h
            h.update_min_max()

            _hmax = h.maximum
            if _hmax > hmax:
                hmax = _hmax

        cell_size = self.radius_scale * hmax

        if cell_size < 1e-6:
            cell_size = 1.0

        self.cell_size = cell_size

        # set the cell size for the DomainManager
        self.set_cell_size(cell_size)

    cdef _remove_ghosts(self):
        """Remove all ghost particles from a previous step

        While creating periodic neighbors, we create new particles and
        give them the tag utils.ParticleTAGS.Ghost. Before repeating
        this step in the next iteration, all current particles with
        this tag are removed.

        """
        cdef list pa_wrappers = self.pa_wrappers
        cdef int narrays = self.narrays

        cdef int array_index
        cdef NNPSParticleArrayWrapper pa_wrapper

        for array_index in range( narrays ):
            pa_wrapper = <NNPSParticleArrayWrapper>PyList_GetItem( pa_wrappers, array_index )
            pa_wrapper.remove_tagged_particles(Ghost)


##############################################################################
cdef class Cell:
    """Basic indexing structure for the box-sort NNPS.

    For a spatial indexing based on the box-sort algorithm, this class
    defines the spatial data structure used to hold particle indices
    (local and global) that are within this cell.

    """
    def __init__(self, IntPoint cid, double cell_size, int narrays,
                 int layers=2):
        """Constructor

        Parameters
        ----------

        cid : IntPoint
            Spatial index (unflattened) for the cell

        cell_size : double
            Spatial extent of the cell in each dimension

        narrays : int
            Number of arrays being binned

        layers : int
            Factor to compute the bounding box

        """
        self._cid = cIntPoint_new(cid.x, cid.y, cid.z)
        self.cell_size = cell_size

        self.narrays = narrays

        self.lindices = [UIntArray() for i in range(narrays)]
        self.gindices = [UIntArray() for i in range(narrays)]

        self.nparticles = [lindices.length for lindices in self.lindices]
        self.is_boundary = False

        # compute the centroid for the cell
        self.centroid = _get_centroid(cell_size, cid.data)

        # cell bounding box
        self.layers = layers
        self._compute_bounding_box(cell_size, layers)

        # list of neighboring processors
        self.nbrprocs = IntArray(0)

        # current size of the cell
        self.size = 0

    #### Public protocol ################################################
    def get_centroid(self, Point pnt):
        """Utility function to get the centroid of the cell.

        Parameters
        ----------

        pnt : Point (input/output)
            The centroid is cmoputed and stored in this object.

        The centroid is defined as the origin plus half the cell size
        in each dimension.

        """
        cdef cPoint centroid = self.centroid
        pnt.data = centroid

    def get_bounding_box(self, Point boxmin, Point boxmax, int layers = 1,
                         cell_size=None):
        """Compute the bounding box for the cell.

        Parameters
        ----------

        boxmin : Point (output)
            The bounding box min coordinates are stored here

        boxmax : Point (output)
            The bounding box max coordinates are stored here

        layers : int (input) default (1)
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

    cpdef set_indices(self, int index, UIntArray lindices, UIntArray gindices):
        """Set the global and local indices for the cell"""
        self.lindices[index] = lindices
        self.gindices[index] = gindices
        self.nparticles[index] = lindices.length

    #### Private protocol ###############################################

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

        boxmin.z = centroid.z - (layers + 0.5) * cell_size
        boxmax.z = centroid.z + (layers + 0.5) * cell_size

        self.boxmin = boxmin
        self.boxmax = boxmax


###############################################################################
cdef class NeighborCache:
    def __init__(self, NNPS nnps, int dst_index, int src_index):
        self._dst_index = dst_index
        self._src_index = src_index
        self._nnps = nnps
        self._particles = nnps.particles
        self._narrays = nnps.narrays
        cdef long n_p = self._particles[dst_index].get_number_of_particles()
        cdef int nnbr = 10
        cdef size_t i
        if self._nnps.dim == 1:
            nnbr = 10
        elif self._nnps.dim == 2:
            nnbr = 60
        elif self._nnps.dim == 3:
            nnbr = 120

        self._n_threads = get_number_of_threads()

        self._cached = IntArray(n_p)
        for i in range(n_p):
            self._cached.data[i] = 0

        self._last_avg_nbr_size = nnbr
        self._start_stop = UIntArray()
        self._pid_to_tid = UIntArray()
        self._neighbor_arrays = []
        self._neighbors = <void**>aligned_malloc(
            sizeof(void*)*self._n_threads
        )

        cdef UIntArray _arr
        for i in range(self._n_threads):
            _arr = UIntArray()
            self._neighbor_arrays.append(_arr)
            self._neighbors[i] = <void*>_arr

    def __dealloc__(self):
        aligned_free(self._neighbors)

    #### Public protocol ################################################

    cpdef get_neighbors(self, int src_index, size_t d_idx, UIntArray nbrs):
        self.get_neighbors_raw(d_idx, nbrs)

    cdef void get_neighbors_raw(self, size_t d_idx, UIntArray nbrs) nogil:
        if self._cached.data[d_idx] == 0:
            self._find_neighbors(d_idx)
        cdef size_t start, end, tid
        start = self._start_stop.data[2*d_idx]
        end = self._start_stop.data[2*d_idx + 1]
        tid = self._pid_to_tid.data[d_idx]
        nbrs.c_set_view(
            &(<UIntArray>self._neighbors[tid]).data[start], end - start
        )

    cpdef find_all_neighbors(self):
        cdef long d_idx
        cdef long np = \
                self._particles[self._dst_index].get_number_of_particles()

        with nogil, parallel():
            for d_idx in prange(np):
                if self._cached.data[d_idx] == 0:
                    self._find_neighbors(d_idx)

    cpdef update(self):
        self._update_last_avg_nbr_size()
        cdef int n_threads = self._n_threads
        cdef int dst_index = self._dst_index
        cdef size_t i
        cdef long np = self._particles[dst_index].get_number_of_particles()
        self._start_stop.resize(np*2)
        self._pid_to_tid.resize(np)
        self._cached.resize(np)
        for i in range(np):
            self._cached.data[i] = 0
            self._start_stop.data[2*i] = 0
            self._start_stop.data[2*i+1] = 0
        # This is an upper limit for the number of neighbors in a worst
        # case scenario.
        cdef size_t safety = 1024
        for i in range(n_threads):
            (<UIntArray>self._neighbors[i]).c_reserve(
                self._last_avg_nbr_size*np/n_threads + safety
            )

    #### Private protocol ################################################

    cdef void _update_last_avg_nbr_size(self):
        cdef size_t i
        cdef size_t np = self._pid_to_tid.length
        cdef UIntArray start_stop = self._start_stop
        cdef long total = 0
        for i in range(np):
            total += start_stop.data[2*i + 1] - start_stop.data[2*i]
        if total > 0 and np > 0:
            self._last_avg_nbr_size = int(total/np) + 1

    cdef void _find_neighbors(self, long d_idx) nogil:
        cdef int thread_id = threadid()
        self._pid_to_tid.data[d_idx] = thread_id
        self._start_stop.data[d_idx*2] = \
            (<UIntArray>self._neighbors[thread_id]).length
        self._nnps.find_nearest_neighbors(
            d_idx, <UIntArray>self._neighbors[thread_id]
        )
        self._start_stop.data[d_idx*2+1] = \
            (<UIntArray>self._neighbors[thread_id]).length
        self._cached.data[d_idx] = 1


##############################################################################
cdef class NNPS:
    """Nearest neighbor query class using the box-sort algorithm.

    NNPS bins all local particles using the box sort algorithm in
    Cells. The cells are stored in a dictionary 'cells' which is keyed
    on the spatial index (IntPoint) of the cell.

    """
    def __init__(self, int dim, list particles, double radius_scale=2.0,
                 int ghost_layers=1, domain=None, bint cache=False,
                 bint sort_gids=False):
        """Constructor for NNPS

        Parameters
        ----------

        dim : int
            Dimension (fixme: Not sure if this is really needed)

        particles : list
            The list of particle arrays we are working on.

        radius_scale : double, default (2)
            Optional kernel radius scale. Defaults to 2

        domain : DomainManager, default (None)
            Optional limits for the domain

        cache : bint
            Flag to set if we want to cache neighbor calls. This costs
            storage but speeds up neighbor calculations.

        sort_gids : bint, default (False)
            Flag to sort neighbors using gids (if they are available).
            This is useful when comparing parallel results with those
            from a serial run.
        """
        # store the list of particles and number of arrays
        self.particles = particles
        self.narrays = len( particles )

        # create the particle array wrappers
        self.pa_wrappers = [NNPSParticleArrayWrapper(pa) for pa in particles]

        # radius scale and problem dimensionality.
        self.radius_scale = radius_scale
        self.dim = dim

        self.domain = domain
        if domain is None:
            self.domain = DomainManager()

        # set the particle array wrappers for the domain manager
        self.domain.set_pa_wrappers(self.pa_wrappers)

        # set the radius scale to determine the cell size
        self.domain.set_radius_scale(self.radius_scale)

        # periodicity
        self.is_periodic = self.domain.is_periodic

        # The total number of cells.
        self.n_cells = 0

        # cell shifts. Indicates the shifting value for cell indices
        # in each co-ordinate direction.
        self.cell_shifts = IntArray(3)
        self.cell_shifts.data[0] = -1
        self.cell_shifts.data[1] = 0
        self.cell_shifts.data[2] = 1

        # min and max coordinate values
        self.xmin = DoubleArray(3)
        self.xmax = DoubleArray(3)

        # The cache.
        self.use_cache = cache
        _cache = []
        for d_idx in range(len(particles)):
            for s_idx in range(len(particles)):
                _cache.append(NeighborCache(self, d_idx, s_idx))
        self.cache = _cache

    #### Public protocol #################################################

    cpdef brute_force_neighbors(self, int src_index, int dst_index,
                                size_t d_idx, UIntArray nbrs):
        cdef NNPSParticleArrayWrapper src = self.pa_wrappers[src_index]
        cdef NNPSParticleArrayWrapper dst = self.pa_wrappers[dst_index]

        cdef DoubleArray s_x = src.x
        cdef DoubleArray s_y = src.y
        cdef DoubleArray s_z = src.z
        cdef DoubleArray s_h = src.h

        cdef DoubleArray d_x = dst.x
        cdef DoubleArray d_y = dst.y
        cdef DoubleArray d_z = dst.z
        cdef DoubleArray d_h = dst.h

        cdef double cell_size = self.cell_size
        cdef double radius_scale = self.radius_scale

        cdef size_t num_particles, j

        num_particles = s_x.length
        cdef double xi = d_x.data[d_idx]
        cdef double yi = d_y.data[d_idx]
        cdef double zi = d_z.data[d_idx]

        cdef double hi = d_h.data[d_idx] * radius_scale # gather radius
        cdef double xj, yj, hj, xij2, xij

        # reset the neighbors
        nbrs.reset()

        for j in range(num_particles):
            xj = s_x.data[j]; yj = s_y.data[j]; zj = s_z.data[j];
            hj = radius_scale * s_h.data[j] # scatter radius

            xij2 = (xi - xj)*(xi - xj) + \
                   (yi - yj)*(yi - yj) + \
                   (zi - zj)*(zi - zj)
            xij = sqrt(xij2)

            if ( (xij < hi) or (xij < hj) ):
                nbrs.append( <ZOLTAN_ID_TYPE> j )

    cdef void find_nearest_neighbors(self, size_t d_idx, UIntArray nbrs) nogil:
        # Implement this in the subclass to actually do something useful.
        pass

    cdef void get_nearest_neighbors(self, size_t d_idx, UIntArray nbrs) nogil:
        if self.use_cache:
            self.current_cache.get_neighbors_raw(d_idx, nbrs)
        else:
            nbrs.c_reset()
            self.find_nearest_neighbors(d_idx, nbrs)

    cpdef get_nearest_particles(self, int src_index, int dst_index,
                                size_t d_idx, UIntArray nbrs):
        cdef int idx = dst_index*self.narrays + src_index
        if self.use_cache:
            if self.src_index != src_index \
                or self.dst_index != dst_index:
                self.set_context(src_index, dst_index)
            return self.cache[idx].get_neighbors(src_index, d_idx, nbrs)
        else:
            return self.get_nearest_particles_no_cache(
                src_index, dst_index, d_idx, nbrs, False
            )

    cpdef get_nearest_particles_no_cache(self, int src_index, int dst_index,
                            size_t d_idx, UIntArray nbrs, bint prealloc):
        raise NotImplementedError("NNPS :: get_nearest_particles_no_cache called")

    cpdef get_spatially_ordered_indices(self, int pa_index, LongArray indices):
        raise NotImplementedError("NNPS :: get_spatially_ordered_indices called")

    cpdef set_context(self, int src_index, int dst_index):
        """Setup the context before asking for neighbors.  The `dst_index`
        represents the particles for whom the neighbors are to be determined
        from the particle array with index `src_index`.

        Parameters
        ----------

         src_index: int: the source index of the particle array.
         dst_index: int: the destination index of the particle array.
        """
        cdef int idx = dst_index*self.narrays + src_index
        self.src_index = src_index
        self.dst_index = dst_index
        self.current_cache = self.cache[idx]

    def set_in_parallel(self, bint in_parallel):
        self.domain.in_parallel = in_parallel

    cpdef spatially_order_particles(self, int pa_index):
        """Spatially order particles such that nearby particles have indices
        nearer each other.  This may improve pre-fetching on the CPU.
        """
        cdef LongArray indices = LongArray()
        cdef ParticleArray pa = self.pa_wrappers[pa_index].pa
        self.get_spatially_ordered_indices(pa_index, indices)
        cdef BaseArray arr

        for arr in pa.properties.values():
            arr.c_align_array(indices)

    def update_domain(self, *args, **kwargs):
        self.domain.update()

    cpdef update(self):
        """Update the local data after particles have moved.

        For parallel runs, we want the NNPS to be independent of the
        ParallelManager which is solely responsible for distributing
        particles across available processors. We assume therefore
        that after a parallel update, each processor has all the local
        particle information it needs and this operation is carried
        out locally.

        For serial runs, this method should be called when the
        particles have moved.

        """
        cdef int i, num_particles
        cdef ParticleArray pa
        cdef UIntArray indices

        cdef DomainManager domain = self.domain

        # use cell sizes computed by the domain.
        self.cell_size = domain.cell_size

        # compute bounds and refresh the data structure
        self._compute_bounds()
        self._refresh()

        # indices on which to bin. We bin all local particles
        for i in range(self.narrays):
            pa = self.particles[i]
            num_particles = pa.get_number_of_particles()
            indices = arange_uint(num_particles)

            # bin the particles
            self._bin( pa_index=i, indices=indices )

        if self.use_cache:
            for cache in self.cache:
                cache.update()

    #### Private protocol ################################################

    cpdef _bin(self, int pa_index, UIntArray indices):
        raise NotImplementedError("NNPS :: _bin called")

    cpdef _refresh(self):
        raise NotImplementedError("NNPS :: _refresh called")

    cdef void _sort_neighbors(self, unsigned int* nbrs, size_t length,
                              unsigned int *gids) nogil:
        if length == 0:
            return
        cdef id_gid_pair_t _entry
        cdef vector[id_gid_pair_t] _data
        cdef vector[unsigned int] _ids
        cdef int i
        cdef unsigned int _id

        if gids[0] == UINT_MAX:
            # Serial runs will have invalid gids so just compare the ids.
            _ids.resize(length)
            for i in range(length):
                _ids[i] = nbrs[i]
            sort(_ids.begin(), _ids.end())
            for i in range(length):
                nbrs[i] = _ids[i]
        else:
            # Copy the neighbor id and gid data.
            _data.resize(length)
            for i in range(length):
                _id = nbrs[i]
                _entry.first = _id
                _entry.second = gids[_id]
                _data[i] = _entry
            # Sort it.
            sort(_data.begin(), _data.end(), _compare_gids)
            # Set the sorted neighbors.
            for i in range(length):
                nbrs[i] = _data[i].first

    cdef _compute_bounds(self):
        """Compute coordinate bounds for the particles"""
        cdef list pa_wrappers = self.pa_wrappers
        cdef NNPSParticleArrayWrapper pa_wrapper
        cdef DoubleArray x, y, z
        cdef double xmax = -1e100, ymax = -1e100, zmax = -1e100
        cdef double xmin = 1e100, ymin = 1e100, zmin = 1e100
        cdef double lx, ly, lz

        for pa_wrapper in pa_wrappers:
            x = pa_wrapper.x
            y = pa_wrapper.y
            z = pa_wrapper.z

            # find min and max of variables
            x.update_min_max()
            y.update_min_max()
            z.update_min_max()

            xmax = fmax(x.maximum, xmax)
            ymax = fmax(y.maximum, ymax)
            zmax = fmax(z.maximum, zmax)

            xmin = fmin(x.minimum, xmin)
            ymin = fmin(y.minimum, ymin)
            zmin = fmin(z.minimum, zmin)

        # Add a small offset to the limits.
        lx, ly, lz = xmax - xmin, ymax - ymin, zmax - zmin
        xmin -= lx*0.01; ymin -= ly*0.01; zmin -= lz*0.01
        xmax += lx*0.01; ymax += ly*0.01; zmax += lz*0.01

        # If all of the dimensions have very small extent give it a unit size.
        cdef double _eps = 1e-12
        if (fabs(xmax - xmin) < _eps) and (fabs(ymax - ymin) < _eps) \
            and (fabs(zmax - zmin) < _eps):
            xmin -= 0.5; xmax += 0.5
            ymin -= 0.5; ymax += 0.5
            zmin -= 0.5; zmax += 0.5

        # store the minimum and maximum of physical coordinates
        self.xmin.set_data(np.asarray([xmin, ymin, zmin]))
        self.xmax.set_data(np.asarray([xmax, ymax, zmax]))


