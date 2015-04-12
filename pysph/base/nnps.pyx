# numpy import
import numpy as np
cimport numpy as np

# malloc and friends
from libc.stdlib cimport malloc, free

# cpython
from cpython.dict cimport PyDict_Clear, PyDict_Contains, PyDict_GetItem
from cpython.list cimport PyList_GetItem

# Cython for compiler directives
cimport cython

cdef extern from 'math.h':
    int abs(int)
    double ceil(double)
    double floor(double)
    double fabs(double)
    double fmax(double, double)
    double fmin(double, double)

IF UNAME_SYSNAME == "Windows":
    cdef inline double fmin(double x, double y):
        return x if x < y else y
    cdef inline double fmax(double x, double y):
        return x if x > y else y


cdef extern from 'limits.h':
    cdef unsigned int UINT_MAX
    cdef int INT_MAX

# ZOLTAN ID TYPE AND PTR
ctypedef unsigned int ZOLTAN_ID_TYPE
ctypedef unsigned int* ZOLTAN_ID_PTR

# Particle Tag information
from utils import ParticleTAGS

cdef int Local = ParticleTAGS.Local
cdef int Remote = ParticleTAGS.Remote
cdef int Ghost = ParticleTAGS.Ghost

cdef inline double norm2(double x, double y, double z):
    return x*x + y*y + z*z

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int flatten(cIntPoint cid, IntArray ncells_per_dim, int dim):
    """Return a flattened index for a cell

    The flattening is determined using the row-order indexing commonly
    employed in SPH. This would need to be changed for hash functions
    based on alternate orderings.

    """
    cdef int ncx = ncells_per_dim.data[0]
    cdef int ncy = ncells_per_dim.data[1]

    return <int>( cid.x + ncx * cid.y + ncx*ncy * cid.z )

def py_flatten(IntPoint cid, IntArray ncells_per_dim, int dim):
    """Python wrapper"""
    cdef cIntPoint _cid = cid.data
    cdef int flattened_index = flatten( _cid, ncells_per_dim, dim )
    return flattened_index

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int get_valid_cell_index(
    cIntPoint cid, IntArray ncells_per_dim, int dim, int n_cells):
    """Return the flattened index for a valid cell"""
    cdef int ncy = ncells_per_dim.data[1]
    cdef int ncz = ncells_per_dim.data[2]

    cdef int cell_index = -1

    # basic test for valid indices. Since we bin the particles with
    # respect to the origin, negative indices can never occur.
    cdef bint is_valid = (cid.x > -1) and (cid.y > -1) and (cid.z > -1)

    # additional check for 1D. This is because we search in all 26
    # neighboring cells for neighbors. In 1D this can be problematic
    # since (ncy = ncz = 0) which means (ncy=1 or ncz=1) will also
    # result in a valid cell with a flattened index < ncells
    if dim == 1:
        if ( (cid.y > ncy) or (cid.z > ncz) ):
            is_valid = False

    # Given the validity of the cells, return the flattened cell index
    if is_valid:
        cell_index = flatten(cid, ncells_per_dim, dim)

        if not (-1 < cell_index < n_cells):
            cell_index = -1

    return cell_index

def py_get_valid_cell_index(IntPoint cid, IntArray ncells_per_dim, int dim, int n_cells):
    """Return the flattened cell index for a valid cell"""
    return get_valid_cell_index(cid.data, ncells_per_dim, dim, n_cells)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline cIntPoint unflatten(int cell_index, IntArray ncells_per_dim, int dim):
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

def py_unflatten(int cell_index, IntArray ncells_per_dim, int dim):
    """Python wrapper"""
    cdef cIntPoint _cid = unflatten( cell_index, ncells_per_dim, dim )
    cdef IntPoint cid = IntPoint_from_cIntPoint(_cid)
    return cid

@cython.cdivision(True)
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

cdef cIntPoint find_cell_id(cPoint pnt, double cell_size):
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
                                 real_to_int(pnt.z, cell_size))
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
    centroid.z = (<double>cid.z + 0.5)*cell_size

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

#################################################################
# NNPS extension classes
#################################################################
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

    def _check_limits(self, xmin, xmax, ymin, ymax, zmin, zmax):
        """Sanity check on the limits"""
        if ( (xmax < xmin) or (ymax < ymin) or (zmax < zmin) ):
            raise ValueError("Invalid domain limits!")

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

    ####################################################################
    # Functions for periodicity
    ####################################################################
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

    cdef _add_to_array(self, DoubleArray arr, double disp):
        cdef int i
        for i in range(arr.length):
            arr.data[i] += disp

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

cdef class Cell:
    """Basic indexing structure for the box-sort NNPS.

    For a spatial indexing based on the box-sort algorithm, this class
    defines the spatial data structure used to hold particle indices
    (local and global) that are within this cell.

    """
    def __init__(self, IntPoint cid, double cell_size, int narrays,
                 int layers=2):
        """Constructor

        Parameters:
        -----------

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

    cpdef set_indices(self, int index, UIntArray lindices, UIntArray gindices):
        """Set the global and local indices for the cell"""
        self.lindices[index] = lindices
        self.gindices[index] = gindices
        self.nparticles[index] = lindices.length

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

    def get_bounding_box(self, Point boxmin, Point boxmax, int layers = 1,
                         cell_size=None):
        """Compute the bounding box for the cell.

        Parameters:
        ------------

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

cdef class NeighborCache:
    def __init__(self, NNPS nnps, int dst_index):
        self._dst_index = dst_index
        self._nnps = nnps
        self._particles = nnps.particles
        cdef int nnbr = 10
        if self._nnps.dim == 1:
            nnbr = 10
        elif self._nnps.dim == 2:
            nnbr = 60
        elif self._nnps.dim == 3:
            nnbr = 120

        self._dirty = np.ones(nnps.narrays, dtype=bool)
        self._last_avg_nbr_size = [nnbr for i in range(nnps.narrays)]
        self._start_stop = [UIntArray() for i in range(nnps.narrays)]
        self._neighbors = [UIntArray() for i in range(nnps.narrays)]

    cpdef update(self):
        self._dirty = np.ones(self._nnps.narrays, dtype=bool)

    cdef _find_all_neighbors(self, int src_idx):
        cdef size_t count, d_idx, avg_nnbr
        cdef UIntArray nbrs = UIntArray()
        cdef UIntArray start_stop, neighbors
        cdef int dst_index = self._dst_index
        # This is an upper limit for the number of neighbors in a worst
        # case scenario.
        cdef size_t safety = 1024
        cdef size_t np = self._particles[dst_index].get_number_of_particles()

        count = 0
        start_stop = self._start_stop[src_idx]
        start_stop.resize(np*2)
        neighbors = self._neighbors[src_idx]
        neighbors.resize(self._last_avg_nbr_size[src_idx]*np + safety)

        for d_idx in range(np):

            if neighbors.length < count + safety:
                avg_nnbr = int(count/d_idx) + 1
                neighbors.resize(int(avg_nnbr*np*1.1) + safety)

            nbrs.set_view(neighbors, count, neighbors.length)

            self._nnps.get_nearest_particles_no_cache(
                src_idx, dst_index, d_idx, nbrs, True
            )
            start_stop.data[d_idx*2] = count
            count += nbrs.length
            start_stop.data[d_idx*2+1] = count

        neighbors.length = count
        neighbors.squeeze()

        self._last_avg_nbr_size[src_idx] = int(neighbors.length/np) + 1
        self._dirty[src_idx] = False

    cpdef get_neighbors(self, int src_index, size_t d_idx, UIntArray nbrs):
        if self._dirty[src_index]:
            self._find_all_neighbors(src_index)
        cdef size_t start, end
        cdef UIntArray start_stop = self._start_stop[src_index]
        cdef UIntArray neighbors = self._neighbors[src_index]
        start = start_stop.data[2*d_idx]
        end = start_stop.data[2*d_idx + 1]
        nbrs.set_view(neighbors, start, end)

cdef class NNPS:
    """Nearest neighbor query class using the box-sort algorithm.

    NNPS bins all local particles using the box sort algorithm in
    Cells. The cells are stored in a dictionary 'cells' which is keyed
    on the spatial index (IntPoint) of the cell.

    """
    def __init__(self, int dim, list particles, double radius_scale=2.0,
                 int ghost_layers=1, domain=None, bint warn=True,
                 cache=False):
        """Constructor for NNPS

        Parameters:
        -----------

        dim : int
            Dimension (fixme: Not sure if this is really needed)

        particles : list
            The list of particle arrays we are working on.

        radius_scale : double, default (2)
            Optional kernel radius scale. Defaults to 2

        domain : DomainManager, default (None)
            Optional limits for the domain

        warn : bint
            Flag to warn when extending particle lists

        cache : bint
            Flag to set if we want to cache neighbor calls. This costs
            storage but speeds up neighbor calculations.
        """
        # store the list of particles and number of arrays
        self.particles = particles
        self.narrays = len( particles )

        # create the particle array wrappers
        self.pa_wrappers = [NNPSParticleArrayWrapper(pa) for pa in particles]

        # radius scale and problem dimensionality.
        self.radius_scale = radius_scale * 1.101
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

        # warn
        self.warn = warn

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

        # number of particles per cell
        self.n_part_per_cell = [IntArray() for pa in particles]

        # The cache.
        self.use_cache = cache
        self.cache = [NeighborCache(self, i) for i in range(len(particles))]

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

            self.cache[i].update()

    cdef _bin(self, int pa_index, UIntArray indices):
        raise NotImplementedError("NNPS :: _bin called")

    cpdef _refresh(self):
        raise NotImplementedError("NNPS :: _refresh called")

    cpdef get_number_of_cells(self):
        return self.n_cells

    cpdef get_particles_in_cell(self, int cell_index, int pa_index,
                                UIntArray indices):
        raise NotImplementedError()

    # return the indices for the particles in neighboring cells
    cpdef get_particles_in_neighboring_cells(self, int cell_index,
        int pa_index, UIntArray nbrs):
        raise NotImplementedError()

    cdef _compute_bounds(self):
        """Compute coordinate bounds for the particles"""
        cdef list pa_wrappers = self.pa_wrappers
        cdef NNPSParticleArrayWrapper pa_wrapper
        cdef DoubleArray x, y, z
        cdef double xmax = -1e100, ymax = -1e100, zmax = -1e100
        cdef double xmin = 1e100, ymin = 1e100, zmin = 1e100

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

        # store the minimum and maximum of physical coordinates
        self.xmin.data[0]=xmin; self.xmin.data[1]=ymin; self.xmin.data[2]=zmin
        self.xmax.data[0]=xmax; self.xmax.data[1]=ymax; self.xmax.data[2]=zmax

    cpdef count_n_part_per_cell(self):
        """Count the number of particles in each cell"""
        raise NotImplementedError("NNPS :: count_n_part_per_cell called")

    def set_in_parallel(self, bint in_parallel):
        self.domain.in_parallel = in_parallel

    ######################################################################
    # Neighbor location routines
    ######################################################################
    cpdef get_nearest_particles(self, int src_index, int dst_index,
                                size_t d_idx, UIntArray nbrs):
        if self.use_cache:
            return self.cache[dst_index].get_neighbors(src_index, d_idx, nbrs)
        else:
            return self.get_nearest_particles_no_cache(
                src_index, dst_index, d_idx, nbrs, False
            )

    cpdef get_nearest_particles_no_cache(self, int src_index, int dst_index,
                            size_t d_idx, UIntArray nbrs, bint prealloc):
        raise NotImplementedError("NNPS :: get_nearest_particles_no_cache called")

    cpdef get_nearest_particles_filtered(
        self,int src_index, int dst_index, int d_idx, UIntArray potential_nbrs,
        UIntArray nbrs):
        """Filter the nearest neighbors from a list of potential neighbors

        For cell based iteration, the nearest neighbors for a given
        particle, using the cell iteration (which is faster) is
        obtained in the following way:

        (a) For each cell (cell_index), we first get the indices for
        the destination particle array within that cell using
        'nnps.get_particles_in_cell(cell_index, ...)

        (b) The list of potential neighbors from the source array,
        with respect to this cell is then sought using
        'nnps.get_particles_in_neighboring_cells(cell_index, ...)'

        (c) Now we iterate over each destination particle (obtained
        from step (a) above) and filter the neighbors based on a
        gather and scatter approach using
        'nnps.get_nearest_particles_filtered(...)'

        """
        # src and dst particle arrays
        cdef NNPSParticleArrayWrapper src = self.pa_wrappers[ src_index ]
        cdef NNPSParticleArrayWrapper dst = self.pa_wrappers[ dst_index ]

        # Source data arrays
        cdef DoubleArray s_x = src.x
        cdef DoubleArray s_y = src.y
        cdef DoubleArray s_z = src.z
        cdef DoubleArray s_h = src.h
        cdef UIntArray s_gid = src.gid

        # Destination data arrays
        cdef DoubleArray d_x = dst.x
        cdef DoubleArray d_y = dst.y
        cdef DoubleArray d_z = dst.z
        cdef DoubleArray d_h = dst.h
        cdef UIntArray d_gid = dst.gid

        # cell size and radius_scale
        cdef double cell_size = self.cell_size
        cdef double radius_scale = self.radius_scale

        # locals
        cdef cPoint xi, xj
        cdef double hi, hj, xij
        cdef int npotential_nbrs = potential_nbrs.length
        cdef int indexj, j
        cdef int nnbrs

        # gather search radius for particle 'i'
        xi = cPoint_new(d_x.data[d_idx], d_y.data[d_idx],  d_z.data[d_idx])
        hi = radius_scale * d_h.data[d_idx]

        # reset the nbr array length
        nbrs.reset()

        # search for true neighbors
        for indexj in range( npotential_nbrs ):
            j = potential_nbrs.data[indexj]

            xj = cPoint_new( s_x.data[j], s_y.data[j], s_z.data[j] )
            hj = radius_scale * s_h.data[j]

            xij = cPoint_distance(xi, xj)

            # select neighbor
            if ( (xij < hi) or (xij < hj) ):
                nbrs.append( j )

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

cdef class BoxSortNNPS(NNPS):
    """Nearest neighbor query class using the box-sort algorithm.

    NNPS bins all local particles using the box sort algorithm in
    Cells. The cells are stored in a dictionary 'cells' which is keyed
    on the spatial index (IntPoint) of the cell.

    """
    def __init__(self, int dim, list particles, double radius_scale=2.0,
                 int ghost_layers=1, domain=None, warn=True, cache=False):
        """Constructor for NNPS

        Parameters:
        -----------

        dim : int
            Number of dimensions.

        particles : list
            The list of particle arrays we are working on.

        radius_scale : double, default (2)
            Optional kernel radius scale. Defaults to 2

        domain : DomainManager, default (None)
            Optional limits for the domain

        warn : bint
            Flag to warn when extending particle lists

        cache : bint
            Flag to set if we want to cache neighbor calls. This costs
            storage but speeds up neighbor calculations.
        """
        # initialize the base class
        NNPS.__init__(
            self, dim, particles, radius_scale, ghost_layers, domain, warn,
            cache
        )

        # initialize the cells dict
        self.cells = {}

        # compute the intial box sort. First, the Domain Manager's
        # update method is called to comptue the maximum smoothing
        # length for particle binning.
        self.domain.update()
        self.update()

    cpdef _refresh(self):
        "Clear the cells dict"
        cdef dict cells = self.cells
        PyDict_Clear( cells )

    cdef _bin(self, int pa_index, UIntArray indices):
        """Bin a given particle array with indices.

        Parameters:
        -----------

        pa_index : int
            Index of the particle array corresponding to the particles list

        indices : UIntArray
            Subset of particles to bin

        """
        cdef NNPSParticleArrayWrapper pa_wrapper = self.pa_wrappers[ pa_index ]
        cdef DoubleArray x = pa_wrapper.x
        cdef DoubleArray y = pa_wrapper.y
        cdef DoubleArray z = pa_wrapper.z

        cdef dict cells = self.cells
        cdef double cell_size = self.cell_size

        cdef UIntArray lindices, gindices
        cdef size_t num_particles, indexi, i

        cdef cIntPoint _cid
        cdef IntPoint cid

        cdef Cell cell
        cdef int ierr, narrays = self.narrays

        # now begin binning the particles
        num_particles = indices.length
        for indexi in range(num_particles):
            i = indices.data[indexi]

            pnt = cPoint_new( x.data[i], y.data[i], z.data[i] )
            _cid = find_cell_id( pnt, cell_size )

            cid = IntPoint_from_cIntPoint(_cid)

            ierr = PyDict_Contains(cells, cid)
            if ierr == 0:
                cell = Cell(cid, cell_size, narrays)
                cells[ cid ] = cell

            # add this particle to the list of indicies
            cell = <Cell>PyDict_GetItem( cells, cid )
            lindices = <UIntArray>PyList_GetItem( cell.lindices, pa_index )

            lindices.append( <ZOLTAN_ID_TYPE> i )
            #gindices.append( gid.data[i] )

        self.n_cells = <int>len(cells)
        self._cell_keys = cells.keys()

    ######################################################################
    # Neighbor location routines
    ######################################################################
    cpdef get_particles_in_cell(
        self, int cell_index, int pa_index, UIntArray indices):
        """Return the indices for the particles within this cell"""
        cdef dict cells = self.cells

        # locals
        cdef int i
        cdef Cell cell = <Cell>PyDict_GetItem(cells,
                                              self._cell_keys[cell_index])
        cdef UIntArray cell_indices = cell.lindices[ pa_index ]
        cdef int nindices = cell_indices.length

        # reset the indices to 0
        indices.reset()

        # now add indices from this cell
        for i in range( nindices ):
            indices.append( cell_indices.data[i] )

    cpdef get_particles_in_neighboring_cells(
        self, int cell_index, int pa_index, UIntArray nbrs):
        """Return indices for particles in neighboring cells.

        Parameters:
        -----------

        cell_index : int
            Cell index in the range [0,ncells_tot]

        pa_index : int
            Index of the particle array in the particles list

        nbrs : UIntArray (output)
            Neighbors for the requested particle are stored here.

        """
        cdef dict cells = self.cells
        cdef IntArray shifts = self.cell_shifts

        # locals
        cdef int ierr, i, j, k
        cdef Cell cell
        cdef IntPoint _cid = IntPoint_new(0, 0, 0)
        cdef UIntArray cell_indices
        cdef int num_indices, local_index

        cdef IntPoint cell_id = <IntPoint>self._cell_keys[cell_index]
        cdef int cx = cell_id.data.x
        cdef int cy = cell_id.data.y
        cdef int cz = cell_id.data.z

        # reset the nbr array length to 0
        nbrs.reset()

        # search for potential neighbors
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    _cid.data.x = cx + shifts.data[i]
                    _cid.data.y = cy + shifts.data[j]
                    _cid.data.z = cz + shifts.data[k]

                    # see if the cell exists
                    ierr = PyDict_Contains( cells, _cid )

                    if ierr == 1:
                        cell  = <Cell>PyDict_GetItem(cells, _cid)
                        cell_indices = <UIntArray>PyList_GetItem(cell.lindices, pa_index)
                        num_indices = cell_indices.length

                        # add the potential neighbors
                        for local_index in range(num_indices):
                            nbrs.append( cell_indices.data[local_index] )

    cpdef get_nearest_particles_no_cache(self, int src_index, int dst_index,
                            size_t d_idx, UIntArray nbrs, bint prealloc):
        """Utility function to get near-neighbors for a particle.

        Parameters:
        -----------

        src_index : int
            Index of the source particle array in the particles list

        dst_index : int
            Index of the destination particle array in the particles list

        d_idx : int (input)
            Destination particle for which neighbors are sought.

        nbrs : UIntArray (output)
            Neighbors for the requested particle are stored here.

        prealloc : bool
            Specifies if the neighbor array already has pre-allocated space
            for the neighbor list.  In this case the neighbors are directly set
            in the given array without resetting or appending to the array.
            This improves performance when the neighbors are cached.
        """
        cdef dict cells = self.cells
        cdef Cell cell

        cdef NNPSParticleArrayWrapper src = self.pa_wrappers[ src_index ]
        cdef NNPSParticleArrayWrapper dst = self.pa_wrappers[ dst_index ]

        # Source data arrays
        cdef DoubleArray s_x = src.x
        cdef DoubleArray s_y = src.y
        cdef DoubleArray s_z = src.z
        cdef DoubleArray s_h = src.h
        cdef UIntArray s_gid = src.gid

        # Destination particle arrays
        cdef DoubleArray d_x = dst.x
        cdef DoubleArray d_y = dst.y
        cdef DoubleArray d_z = dst.z
        cdef DoubleArray d_h = dst.h
        cdef UIntArray d_gid = dst.gid

        cdef double radius_scale = self.radius_scale
        cdef double cell_size = self.cell_size
        cdef UIntArray lindices
        cdef size_t indexj, count
        cdef ZOLTAN_ID_TYPE j

        cdef cPoint xi = cPoint_new(d_x.data[d_idx], d_y.data[d_idx], d_z.data[d_idx])
        cdef cIntPoint _cid = find_cell_id( xi, cell_size )
        cdef IntPoint cid = IntPoint_from_cIntPoint( _cid )
        cdef IntPoint cellid = IntPoint(0, 0, 0)

        cdef double xij2

        cdef double hi2, hj2

        hi2 = radius_scale * d_h.data[d_idx]
        hi2 *= hi2

        cdef int ierr

        # reset the nbr array length. This should avoid a realloc
        if prealloc:
            nbrs.length = 0
        else:
            nbrs.reset()
        count = 0

        cdef int ix, iy, iz
        for ix in [cid.data.x -1, cid.data.x, cid.data.x + 1]:
            for iy in [cid.data.y - 1, cid.data.y, cid.data.y + 1]:
                for iz in [cid.data.z - 1, cid.data.z, cid.data.z + 1]:
                    cellid.data.x = ix; cellid.data.y = iy; cellid.data.z = iz

                    ierr = PyDict_Contains( cells, cellid )
                    if ierr == 1:

                        cell = <Cell>PyDict_GetItem( cells, cellid )
                        lindices = <UIntArray>PyList_GetItem( cell.lindices, src_index )

                        for indexj in range( lindices.length ):
                            j = lindices.data[indexj]

                            xij2 = norm2( s_x.data[j]-xi.x,
                                          s_y.data[j]-xi.y,
                                          s_z.data[j]-xi.z )

                            hj2 = radius_scale * s_h.data[j]
                            hj2 *= hj2

                            # select neighbor
                            if ( (xij2 < hi2) or (xij2 < hj2) ):
                                if prealloc:
                                    nbrs.data[count] = j
                                    count += 1
                                else:
                                    nbrs.append( j )
        if prealloc:
            nbrs.length = count

    cpdef count_n_part_per_cell(self):
        """Count the number of particles in each cell"""
        cdef int n_arrays = self.narrays
        cdef int n_cells = self.n_cells
        cdef list cell_keys = self._cell_keys
        cdef dict cells = self.cells
        cdef list n_part_per_cell = self.n_part_per_cell

        # Locals
        cdef Cell cell
        cdef int i, j
        cdef IntArray n_part_per_cell_indices
        cdef UIntArray cell_indices
        cdef IntPoint cell_key

        # iterate over all cells and count the number of particles
        for i in range( n_arrays ):
            n_part_per_cell_indices = <IntArray>PyList_GetItem(n_part_per_cell, i)
            n_part_per_cell_indices.resize(n_cells)

            for j in range( n_cells ):
                cell_key = <IntPoint>PyList_GetItem(cell_keys, j)
                cell = <Cell>PyDict_GetItem(cells, cell_key)
                cell_indices = <UIntArray>PyList_GetItem(cell.lindices, i)

                # store the number of particles in this cell
                n_part_per_cell_indices.data[j] = cell_indices.length


cdef class LinkedListNNPS(NNPS):
    """Nearest neighbor query class using the linked list method.
    """
    def __init__(self, int dim, list particles, double radius_scale=2.0,
                 int ghost_layers=1, domain=None,
                 bint fixed_h=False, bint warn=True, bint cache=False):
        """Constructor for NNPS

        Parameters:
        -----------

        dim : int
            Number of dimension.

        particles : list
            The list of particle arrays we are working on

        radius_scale : double, default (2)
            Optional kernel radius scale. Defaults to 2

        ghost_layers : int
            Optional number of layers to share in parallel

        domain : DomainManager, default (None)
            Optional limits for the domain

        fixed_h : bint
            Optional flag to use constant cell sizes throughout.

        warn : bint
            Flag to warn when extending particle lists

        cache : bint
            Flag to set if we want to cache neighbor calls. This costs
            storage but speeds up neighbor calculations.
        """
        # initialize the base class
        NNPS.__init__(
            self, dim, particles, radius_scale, ghost_layers, domain, warn,
            cache
        )

        # initialize the head and next for each particle array
        self.heads = [UIntArray() for i in range(self.narrays)]
        self.nexts = [UIntArray() for i in range(self.narrays)]

        # flag for constant smoothing lengths
        self.fixed_h = fixed_h

        # defaults
        self.ncells_per_dim = IntArray(3)
        self.n_cells = 0

        # compute the intial box sort for all local particles. The
        # DomainManager.setup_domain method is called to compute the
        # cell size.
        self.domain.update()
        self.update()

    cdef _bin(self, int pa_index, UIntArray indices):
        """Bin a given particle array with indices.

        Parameters:
        -----------

        pa_index : int
            Index of the particle array corresponding to the particles list

        indices : UIntArray
            Subset of particles to bin

        """
        cdef NNPSParticleArrayWrapper pa_wrapper = self.pa_wrappers[ pa_index ]
        cdef DoubleArray x = pa_wrapper.x
        cdef DoubleArray y = pa_wrapper.y
        cdef DoubleArray z = pa_wrapper.z

        cdef DoubleArray xmin = self.xmin
        cdef DoubleArray xmax = self.xmax
        cdef IntArray ncells_per_dim = self.ncells_per_dim
        cdef int dim = self.dim

        # the head and next arrays for this particle array
        cdef UIntArray head = self.heads[ pa_index ]
        cdef UIntArray next = self.nexts[ pa_index ]
        cdef double cell_size = self.cell_size

        cdef UIntArray lindices, gindices
        cdef size_t num_particles, indexi, i

        # point and flattened index
        cdef cPoint pnt = cPoint_new(0, 0, 0)
        cdef int _cid

        # flattened cell index
        cdef int cell_index

        # now bin the particles
        num_particles = indices.length
        for indexi in range(num_particles):
            i = indices.data[indexi]

            # the flattened index is considered relative to the
            # minimum along each co-ordinate direction
            pnt.x = x.data[i] - xmin.data[0]
            pnt.y = y.data[i] - xmin.data[1]
            pnt.z = z.data[i] - xmin.data[2]

            # flattened cell index
            _cid = flatten( find_cell_id( pnt, cell_size ), ncells_per_dim, dim )

            # insert this particle
            next.data[ i ] = head.data[ _cid ]
            head.data[_cid] = i

    cpdef _refresh(self):
        """Refresh the head and next arrays locally"""
        cdef DomainManager domain = self.domain
        cdef int narrays = self.narrays
        cdef DoubleArray xmin = self.xmin
        cdef DoubleArray xmax = self.xmax
        cdef double cell_size = self.cell_size
        cdef double cell_size1 = 1./cell_size
        cdef list pa_wrappers = self.pa_wrappers
        cdef NNPSParticleArrayWrapper pa_wrapper
        cdef list heads = self.heads
        cdef list nexts = self.nexts
        cdef int dim = self.dim

        # locals
        cdef int i, j, np
        cdef int ncx, ncy, ncz, _ncells
        cdef UIntArray head, next

        # calculate the number of cells.
        ncx = <int>ceil( cell_size1*(xmax.data[0] - xmin.data[0]) )
        ncy = <int>ceil( cell_size1*(xmax.data[1] - xmin.data[1]) )
        ncz = <int>ceil( cell_size1*(xmax.data[2] - xmin.data[2]) )

        if ncx < 0 or ncy < 0 or ncz < 0:
            msg = 'LinkedListNNPS: Number of cells is negative '\
                   '(%s, %s, %s).'%(ncx, ncy, ncz)
            raise RuntimeError(msg)

        # number of cells along each coordinate direction
        self.ncells_per_dim.data[0] = ncx
        self.ncells_per_dim.data[1] = ncy
        self.ncells_per_dim.data[2] = ncz

        # total number of cells
        _ncells = ncx
        if dim == 2: _ncells = ncx * ncy
        if dim == 3: _ncells = ncx * ncy * ncz
        self.n_cells = <int>_ncells

        if _ncells < 0 or _ncells > 2**28:
            # unsigned ints are 4 bytes, which means 2**28 cells requires 1GB.
            msg = "ERROR: LinkedListNNPS requires too many cells (%s)."%_ncells
            raise RuntimeError(msg)

        # initialize the head and next arrays
        for i in range(narrays):
            pa_wrapper = pa_wrappers[i]
            np = pa_wrapper.get_number_of_particles()

            # re-size the head and next arrays
            head = <UIntArray>PyList_GetItem(heads, i)
            next = <UIntArray>PyList_GetItem(nexts, i )

            head.resize( _ncells )
            next.resize( np )

            # UINT_MAX is used to indicate an invalid index
            for j in range(_ncells):
                head.data[j] = UINT_MAX

            for j in range(np):
                next.data[j] = UINT_MAX

    cpdef get_particles_in_cell(
        self, int cell_index, int pa_index, UIntArray indices):
        """Return the indices for the particles within this cell"""
        cdef UIntArray head = self.heads[ pa_index ]
        cdef UIntArray next = self.nexts[ pa_index ]

        cdef ZOLTAN_ID_TYPE _next

        # reset the indices length
        indices.reset()

        # get the first particle in this cell
        _next = head.data[ cell_index ]
        while( _next != UINT_MAX ):
            # add to the list of particle indices and find next
            indices.append( _next )
            _next = next.data[ _next ]

    cpdef count_n_part_per_cell(self):
        "Count the number of particles in the cell"
        cdef int narrays = self.narrays
        cdef int n_cells = self.n_cells
        cdef list heads = self.heads
        cdef list nexts = self.nexts
        cdef list n_part_per_cell = self.n_part_per_cell

        # Locals
        cdef unsigned int _next
        cdef IntArray n_part_per_cell_indices
        cdef UIntArray head, next
        cdef int i, j, count

        for i in range(narrays):
            head = <UIntArray>PyList_GetItem( heads, i )
            next = <UIntArray>PyList_GetItem( nexts, i )

            n_part_per_cell_indices = <IntArray>PyList_GetItem(
                n_part_per_cell, i)
            n_part_per_cell_indices.resize(n_cells)

            for j in range(n_cells):
                count = 0
                _next = head.data[ j ]
                while (_next != UINT_MAX):
                    count = count + 1
                    _next = next.data[_next]

                n_part_per_cell_indices.data[j] = count

    ######################################################################
    # Neighbor location routines
    ######################################################################
    cpdef get_nearest_particles_no_cache(self, int src_index, int dst_index,
                            size_t d_idx, UIntArray nbrs, bint prealloc):
        """Utility function to get near-neighbors for a particle.

        Parameters:
        -----------

        src_index : int
            Index of the source particle array in the particles list

        dst_index : int
            Index of the destination particle array in the particles list

        d_idx : int (input)
            Destination particle for which neighbors are sought.

        nbrs : UIntArray (output)
            Neighbors for the requested particle are stored here.

        prealloc : bool
            Specifies if the neighbor array already has pre-allocated space
            for the neighbor list.  In this case the neighbors are directly set
            in the given array without resetting or appending to the array.
            This improves performance when the neighbors are cached.
        """
        # src and dst particle arrays
        cdef NNPSParticleArrayWrapper src = self.pa_wrappers[ src_index ]
        cdef NNPSParticleArrayWrapper dst = self.pa_wrappers[ dst_index ]

        # src and dst linked lists
        cdef UIntArray next = self.nexts[ src_index ]
        cdef UIntArray head = self.heads[ src_index ]

        # Number of cells
        cdef IntArray ncells_per_dim = self.ncells_per_dim
        cdef int n_cells = self.n_cells
        cdef int dim = self.dim

        # cell shifts
        cdef IntArray shifts = self.cell_shifts

        # Source data arrays
        cdef DoubleArray s_x = src.x
        cdef DoubleArray s_y = src.y
        cdef DoubleArray s_z = src.z
        cdef DoubleArray s_h = src.h
        cdef UIntArray s_gid = src.gid

        # Destination particle arrays
        cdef DoubleArray d_x = dst.x
        cdef DoubleArray d_y = dst.y
        cdef DoubleArray d_z = dst.z
        cdef DoubleArray d_h = dst.h
        cdef UIntArray d_gid = dst.gid

        # minimum values for the particle distribution
        cdef DoubleArray xmin = self.xmin

        # cell size and radius
        cdef double radius_scale = self.radius_scale
        cdef double cell_size = self.cell_size

        # locals
        cdef UIntArray lindices
        cdef size_t indexj
        cdef double xij2
        cdef double hi2, hj2
        cdef int ierr, nnbrs
        cdef unsigned int _next, count
        cdef int ix, iy, iz

        # get the un-flattened index for the destination particle with
        # respect to the minimum
        cdef cPoint xi = cPoint_new(d_x.data[d_idx] - xmin.data[0],
                                    d_y.data[d_idx] - xmin.data[1],
                                    d_z.data[d_idx] - xmin.data[2])

        cdef cIntPoint _cid = find_cell_id( xi, cell_size )
        cdef cIntPoint cid = cIntPoint_new(0, 0, 0)
        cdef int cell_index

        # this is the physica position of the particle that will be
        # used in pairwise searching
        xi.x = xi.x + xmin.data[0]
        xi.y = xi.y + xmin.data[1]
        xi.z = xi.z + xmin.data[2]

        # gather search radius
        hi2 = radius_scale * d_h.data[d_idx]
        hi2 *= hi2

        # reset the length of the nbr array
        if prealloc:
            nbrs.length = 0
        else:
            nbrs.reset()

        count = 0
        # Begin search through neighboring cells
        for ix in range(3):
            for iy in range(3):
                for iz in range(3):
                    cid.x = _cid.x + shifts.data[ix]
                    cid.y = _cid.y + shifts.data[iy]
                    cid.z = _cid.z + shifts.data[iz]

                    # Only consider valid cell indices
                    cell_index = get_valid_cell_index(cid, ncells_per_dim, dim, n_cells)
                    if cell_index > -1:

                        # get the first particle and begin iteration
                        _next = head.data[ cell_index ]
                        while( _next != UINT_MAX ):
                            hj2 = radius_scale * s_h.data[_next]
                            hj2 *= hj2

                            xij2 = norm2( s_x.data[_next]-xi.x,
                                          s_y.data[_next]-xi.y,
                                          s_z.data[_next]-xi.z )

                            # select neighbor
                            if ( (xij2 < hi2) or (xij2 < hj2) ):
                                if prealloc:
                                    nbrs.data[count] = _next
                                    count += 1
                                else:
                                    nbrs.append( _next )

                            # get the 'next' particle in this cell
                            _next = next.data[_next]
        if prealloc:
            nbrs.length = count

    cpdef get_particles_in_neighboring_cells(
        self, int cell_index, int pa_index, UIntArray nbrs):
        """Return indices for particles in neighboring cells.

        Parameters:
        -----------

        cell_index : int
            Cell index in the range [0,ncells_tot]

        pa_index : int
            Index of the particle array in the particles list

        nbrs : UIntArray (output)
            Neighbors for the requested particle are stored here.

        """
        # src linked lists
        cdef UIntArray next = self.nexts[ pa_index ]
        cdef UIntArray head = self.heads[ pa_index ]

        # Number of cells in each dimension and the total ncells
        cdef IntArray ncells_per_dim = self.ncells_per_dim
        cdef int n_cells = self.n_cells
        cdef int dim = self.dim

        # cell shifts
        cdef IntArray shifts = self.cell_shifts

        # locals
        cdef size_t indexj
        cdef unsigned int _next
        cdef int nnbrs, ix, iy, iz

        # get the un-flattened index for this cell
        cdef cIntPoint _cid = unflatten( cell_index, ncells_per_dim, dim )
        cdef cIntPoint cid
        cdef int _cell_index

        # rest the nbr array length
        nbrs.reset()

        # Begin search through neighboring cells
        for ix in range(3):
            for iy in range(3):
                for iz in range(3):
                    cid.x = _cid.x + shifts.data[ix]
                    cid.y = _cid.y + shifts.data[iy]
                    cid.z = _cid.z + shifts.data[iz]

                    # only use a valid cell index
                    if ( (cid.x > -1) and (cid.y > -1) and (cid.z > -1) ):
                        _cell_index = flatten( cid, ncells_per_dim, dim )
                        if -1 < _cell_index < n_cells:

                            # get the first particle and begin iteration
                            _next = head.data[ _cell_index ]
                            while( _next != UINT_MAX ):
                                nbrs.append( _next )

                                # get the 'next' particle in this cell
                                _next = next.data[_next]
