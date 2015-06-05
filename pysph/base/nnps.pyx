#cython: embedsignature=True
# Library imports.
import numpy as np
cimport numpy as np

# Cython imports
from cython.operator cimport dereference as deref, preincrement as inc
from cython.parallel import parallel, prange, threadid

# malloc and friends
from libcpp.map cimport map
from libcpp.pair cimport pair
from libcpp.vector cimport vector

cdef extern from "<algorithm>" namespace "std" nogil:
    void sort[Iter, Compare](Iter first, Iter last, Compare comp)
    void sort[Iter](Iter first, Iter last)

# cpython
from cpython.dict cimport PyDict_Clear, PyDict_Contains, PyDict_GetItem
from cpython.list cimport PyList_GetItem

# Cython for compiler directives
cimport cython

cdef extern from 'math.h':
    int abs(int) nogil
    double ceil(double) nogil
    double floor(double) nogil
    double fabs(double) nogil
    double fmax(double, double) nogil
    double fmin(double, double) nogil

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


cdef extern from 'limits.h':
    cdef unsigned int UINT_MAX
    cdef int INT_MAX

# ZOLTAN ID TYPE AND PTR
ctypedef unsigned int ZOLTAN_ID_TYPE
ctypedef unsigned int* ZOLTAN_ID_PTR

# Particle Tag information
from pyzoltan.core.carray cimport BaseArray, aligned_malloc, aligned_free
from utils import ParticleTAGS

cdef int Local = ParticleTAGS.Local
cdef int Remote = ParticleTAGS.Remote
cdef int Ghost = ParticleTAGS.Ghost

ctypedef pair[unsigned int, unsigned int] id_gid_pair_t


cdef inline bint _compare_gids(id_gid_pair_t x, id_gid_pair_t y) nogil:
    return y.second > x.second

cdef inline double norm2(double x, double y, double z) nogil:
    return x*x + y*y + z*z


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline long flatten_raw(int x, int y, int z, int* ncells_per_dim,
        int dim) nogil:
    """Return a flattened index for a cell

    The flattening is determined using the row-order indexing commonly
    employed in SPH. This would need to be changed for hash functions
    based on alternate orderings.

    """
    cdef long ncx = ncells_per_dim[0]
    cdef long ncy = ncells_per_dim[1]

    return <long>( x + ncx * y + ncx*ncy * z )


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline long flatten(cIntPoint cid, IntArray ncells_per_dim, int dim) nogil:
    """Return a flattened index for a cell

    The flattening is determined using the row-order indexing commonly
    employed in SPH. This would need to be changed for hash functions
    based on alternate orderings.

    """
    return flatten_raw(cid.x, cid.y, cid.z, ncells_per_dim.data, dim)

def py_flatten(IntPoint cid, IntArray ncells_per_dim, int dim):
    """Python wrapper"""
    cdef cIntPoint _cid = cid.data
    cdef int flattened_index = flatten( _cid, ncells_per_dim, dim )
    return flattened_index


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline long get_valid_cell_index(int cid_x, int cid_y, int cid_z,
        int* ncells_per_dim, int dim, int n_cells) nogil:
    """Return the flattened index for a valid cell"""
    cdef long ncy = ncells_per_dim[1]
    cdef long ncz = ncells_per_dim[2]

    cdef long cell_index = -1

    # basic test for valid indices. Since we bin the particles with
    # respect to the origin, negative indices can never occur.
    cdef bint is_valid = (cid_x > -1) and (cid_y > -1) and (cid_z > -1)

    # additional check for 1D. This is because we search in all 26
    # neighboring cells for neighbors. In 1D this can be problematic
    # since (ncy = ncz = 0) which means (ncy=1 or ncz=1) will also
    # result in a valid cell with a flattened index < ncells
    if dim == 1:
        if ( (cid_y > ncy) or (cid_z > ncz) ):
            is_valid = False

    # Given the validity of the cells, return the flattened cell index
    if is_valid:
        cell_index = flatten_raw(cid_x, cid_y, cid_z, ncells_per_dim, dim)

        if not (-1 < cell_index < n_cells):
            cell_index = -1

    return cell_index

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

@cython.cdivision(True)
cdef inline int real_to_int(double real_val, double step) nogil:
    """ Return the bin index to which the given position belongs.

    Parameters
    ----------
    val -- The coordinate location to bin
    step -- the bin size

    Examples
    --------
    >>> real_to_int(1.5, 1.0)
    1
    >>> real_to_int(-0.5, 1.0)
    -1
    """
    cdef int ret_val = <int>floor( real_val/step )

    return ret_val


cdef void find_cell_id_raw(double x, double y, double z, double
                           cell_size, int *ix, int *iy, int *iz) nogil:
    """ Find the cell index for the corresponding point

    Parameters
    ----------
    x, y, z: double
        the point for which the index is sought
    cell_size : double
        the cell size to use
    ix, iy, iz : int*
        output parameter holding the cell index

    Notes
    ------
    Performs a box sort based on the point and cell size

    Uses the function  `real_to_int`

    """
    ix[0] = real_to_int(x, cell_size)
    iy[0] = real_to_int(y, cell_size)
    iz[0] = real_to_int(z, cell_size)


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
                and self.dst_index != dst_index:
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
        """Setup the context before asing for neighbors.  The `dst_index`
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


##############################################################################
cdef class DictBoxSortNNPS(NNPS):
    """Nearest neighbor query class using the box-sort algorithm using a
    dictionary.

    NNPS bins all local particles using the box sort algorithm in
    Cells. The cells are stored in a dictionary 'cells' which is keyed
    on the spatial index (IntPoint) of the cell.

    """
    def __init__(self, int dim, list particles, double radius_scale=2.0,
                 int ghost_layers=1, domain=None, cache=False,
                 sort_gids=False):
        """Constructor for NNPS

        Parameters
        ----------

        dim : int
            Number of dimensions.

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
        # initialize the base class
        NNPS.__init__(
            self, dim, particles, radius_scale, ghost_layers, domain,
            cache, sort_gids
        )

        # initialize the cells dict
        self.cells = {}

        # compute the intial box sort. First, the Domain Manager's
        # update method is called to comptue the maximum smoothing
        # length for particle binning.
        self.domain.update()
        self.update()

        msg = """WARNING: The cache will currently work only if
        find_nearest_neighbors works and can be used without requiring
        the GIL.  The DictBoxSort does not work with OpenMP since it uses
        Python objects and cannot release the GIL.  Until this is fixed,
        we cannot use caching, or parallel neighbor finding using this
        DictBoxSortNNPS, use the more efficient LinkedListNNPS instead.
        Disabling caching for now.
        """
        if cache:
            print(msg)
        self.use_cache = False


    #### Public protocol ################################################

    cpdef get_nearest_particles_no_cache(self, int src_index, int dst_index,
                            size_t d_idx, UIntArray nbrs, bint prealloc):
        """Utility function to get near-neighbors for a particle.

        Parameters
        ----------

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

        if self.sort_gids:
            self._sort_neighbors(nbrs.data, count, s_gid.data)

    #### Private protocol ################################################

    cpdef _refresh(self):
        """Clear the cells dict"""
        cdef dict cells = self.cells
        PyDict_Clear( cells )

    cpdef _bin(self, int pa_index, UIntArray indices):
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
        self._cell_keys = list(cells.keys())


#############################################################################
cdef class LinkedListNNPS(NNPS):
    """Nearest neighbor query class using the linked list method.
    """
    def __init__(self, int dim, list particles, double radius_scale=2.0,
                 int ghost_layers=1, domain=None,
                 bint fixed_h=False, bint cache=False, bint sort_gids=False):
        """Constructor for NNPS

        Parameters
        ----------

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

        cache : bint
            Flag to set if we want to cache neighbor calls. This costs
            storage but speeds up neighbor calculations.

        sort_gids : bint, default (False)
            Flag to sort neighbors using gids (if they are available). 
            This is useful when comparing parallel results with those
            from a serial run.
        """
        # initialize the base class
        NNPS.__init__(
            self, dim, particles, radius_scale, ghost_layers, domain,
            cache, sort_gids
        )

        # initialize the head and next for each particle array
        self.heads = [UIntArray() for i in range(self.narrays)]
        self.nexts = [UIntArray() for i in range(self.narrays)]

        # flag for constant smoothing lengths
        self.fixed_h = fixed_h

        # defaults
        self.ncells_per_dim = IntArray(3)
        self.n_cells = 0
        self.sort_gids = sort_gids

        # compute the intial box sort for all local particles. The
        # DomainManager.setup_domain method is called to compute the
        # cell size.
        self.domain.update()
        self.update()

    #### Public protocol ################################################

    cdef void find_nearest_neighbors(self, size_t d_idx, UIntArray nbrs) nogil:
        """Low level, high-performance non-gil method to find neighbors.
        This requires that `set_context()` be called beforehand.  This method
        does not reset the neighbors array before it appends the
        neighbors to it.

        """
        # Number of cells
        cdef int n_cells = self.n_cells
        cdef int dim = self.dim

        # cell shifts
        cdef int* shifts = self.cell_shifts.data

        # Source data arrays
        cdef double* s_x = self.src.x.data
        cdef double* s_y = self.src.y.data
        cdef double* s_z = self.src.z.data
        cdef double* s_h = self.src.h.data
        cdef unsigned int* s_gid = self.src.gid.data

        # Destination particle arrays
        cdef double* d_x = self.dst.x.data
        cdef double* d_y = self.dst.y.data
        cdef double* d_z = self.dst.z.data
        cdef double* d_h = self.dst.h.data
        cdef unsigned int* d_gid = self.dst.gid.data

        cdef unsigned int* head = self.head.data
        cdef unsigned int* next = self.next.data

        # minimum values for the particle distribution
        cdef double* xmin = self.xmin.data

        # cell size and radius
        cdef double radius_scale = self.radius_scale
        cdef double cell_size = self.cell_size

        # locals
        cdef size_t indexj
        cdef double xij2
        cdef double hi2, hj2
        cdef int ierr, nnbrs
        cdef unsigned int _next
        cdef int ix, iy, iz

        # this is the physica position of the particle that will be
        # used in pairwise searching
        cdef double x = d_x[d_idx]
        cdef double y = d_y[d_idx]
        cdef double z = d_z[d_idx]

        # get the un-flattened index for the destination particle with
        # respect to the minimum
        cdef int _cid_x, _cid_y, _cid_z
        find_cell_id_raw(
            x - xmin[0], y - xmin[1], z - xmin[2],
            cell_size, &_cid_x, &_cid_y, &_cid_z
        )

        cdef int cid_x, cid_y, cid_z
        cdef long cell_index, orig_length
        cid_x = cid_y = cid_z = 0

        # gather search radius
        hi2 = radius_scale * d_h[d_idx]
        hi2 *= hi2

        orig_length = nbrs.length

        # Begin search through neighboring cells
        for ix in range(3):
            for iy in range(3):
                for iz in range(3):
                    cid_x = _cid_x + shifts[ix]
                    cid_y = _cid_y + shifts[iy]
                    cid_z = _cid_z + shifts[iz]

                    # Only consider valid cell indices
                    cell_index = self._get_valid_cell_index(
                        cid_x, cid_y, cid_z,
                        self.ncells_per_dim.data, dim, n_cells
                    )
                    if cell_index > -1:

                        # get the first particle and begin iteration
                        _next = head[ cell_index ]
                        while( _next != UINT_MAX ):
                            hj2 = radius_scale * s_h[_next]
                            hj2 *= hj2

                            xij2 = norm2( s_x[_next]-x,
                                          s_y[_next]-y,
                                          s_z[_next]-z )

                            # select neighbor
                            if ( (xij2 < hi2) or (xij2 < hj2) ):
                                nbrs.c_append(_next)

                            # get the 'next' particle in this cell
                            _next = next[_next]
        if self.sort_gids:
            self._sort_neighbors(
                &nbrs.data[orig_length], nbrs.length - orig_length, s_gid
            )

    cpdef get_nearest_particles_no_cache(self, int src_index, int dst_index,
                            size_t d_idx, UIntArray nbrs, bint prealloc):
        """Utility function to get near-neighbors for a particle.

        Parameters
        ----------

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
        self.set_context(src_index, dst_index)

        # reset the length of the nbr array
        if prealloc:
            nbrs.length = 0
        else:
            nbrs.reset()

        self.find_nearest_neighbors(d_idx, nbrs)

    cpdef get_spatially_ordered_indices(self, int pa_index, LongArray indices):
        cdef UIntArray head = self.heads[pa_index]
        cdef UIntArray next = self.nexts[pa_index]
        cdef ZOLTAN_ID_TYPE _next
        indices.reset()
        cdef long i

        for i in range(self.n_cells):
            _next = head.data[i]
            while (_next != UINT_MAX):
                indices.append(<long>_next)
                _next = next.data[_next]

    cpdef set_context(self, int src_index, int dst_index):
        """Setup the context before asking for neighbors.  The `dst_index`
        represents the particles for whom the neighbors are to be determined
        from the particle array with index `src_index`.

        Parameters
        ----------

         src_index: int: the source index of the particle array.
         dst_index: int: the destination index of the particle array.
        """
        NNPS.set_context(self, src_index, dst_index)

        # Set the current context.
        self.src = self.pa_wrappers[ src_index ]
        self.dst = self.pa_wrappers[ dst_index ]

        # next and head linked lists
        self.next = self.nexts[ src_index ]
        self.head = self.heads[ src_index ]


    #### Private protocol ################################################

    cpdef _bin(self, int pa_index, UIntArray indices):
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
            _cid = self._get_flattened_cell_index(pnt, cell_size)

            # insert this particle
            next.data[ i ] = head.data[ _cid ]
            head.data[_cid] = i

    cdef long _get_flattened_cell_index(self, cPoint pnt, double cell_size):
        return flatten(
            find_cell_id(pnt, cell_size), self.ncells_per_dim, self.dim
        )

    cpdef long _get_number_of_cells(self) except -1:
        cdef double cell_size = self.cell_size
        cdef double cell_size1 = 1./cell_size
        cdef DoubleArray xmin = self.xmin
        cdef DoubleArray xmax = self.xmax
        cdef int ncx, ncy, ncz
        cdef long _ncells
        cdef int dim = self.dim

        # calculate the number of cells.
        ncx = <int>ceil( cell_size1*(xmax.data[0] - xmin.data[0]) )
        ncy = <int>ceil( cell_size1*(xmax.data[1] - xmin.data[1]) )
        ncz = <int>ceil( cell_size1*(xmax.data[2] - xmin.data[2]) )

        if ncx < 0 or ncy < 0 or ncz < 0:
            msg = 'LinkedListNNPS: Number of cells is negative '\
                   '(%s, %s, %s).'%(ncx, ncy, ncz)
            raise RuntimeError(msg)
            return -1

        # number of cells along each coordinate direction
        self.ncells_per_dim.data[0] = ncx
        self.ncells_per_dim.data[1] = ncy
        self.ncells_per_dim.data[2] = ncz

        # total number of cells
        _ncells = ncx
        if dim == 2: _ncells = ncx * ncy
        if dim == 3: _ncells = ncx * ncy * ncz
        return _ncells

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef long _get_valid_cell_index(self, int cid_x, int cid_y, int cid_z,
            int* ncells_per_dim, int dim, int n_cells) nogil:
        return get_valid_cell_index(
            cid_x, cid_y, cid_z, ncells_per_dim, dim, n_cells
        )

    cpdef long _count_occupied_cells(self, long n_cells) except -1:
        if n_cells < 0 or n_cells > 2**28:
            # unsigned ints are 4 bytes, which means 2**28 cells requires 1GB.
            msg = "ERROR: LinkedListNNPS requires too many cells (%s)."%n_cells
            raise RuntimeError(msg)
            return -1

        return n_cells

    cpdef _refresh(self):
        """Refresh the head and next arrays locally"""
        cdef DomainManager domain = self.domain
        cdef int narrays = self.narrays
        cdef list pa_wrappers = self.pa_wrappers
        cdef NNPSParticleArrayWrapper pa_wrapper
        cdef list heads = self.heads
        cdef list nexts = self.nexts

        # locals
        cdef int i, j, np
        cdef long _ncells
        cdef UIntArray head, next

        _ncells = self._get_number_of_cells()

        _ncells = self._count_occupied_cells(_ncells)

        self.n_cells = <int>_ncells

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


#############################################################################
cdef class BoxSortNNPS(LinkedListNNPS):

    """Nearest neighbor query class using the box sort method but which
    uses the LinkedList algorithm.  This makes this very fast but
    perhaps not as safe as the DictBoxSortNNPS.  All this class does 
    is to use a std::map to obtain a linear cell index from the actual 
    flattened cell index.
    """

    #### Private protocol ################################################

    cdef long _get_flattened_cell_index(self, cPoint pnt, double cell_size):
        cdef long cell_id = flatten(
            find_cell_id(pnt, cell_size), self.ncells_per_dim, self.dim
        )
        return self.cell_to_index[cell_id]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef inline long _get_valid_cell_index(self, int cid_x, int cid_y, int cid_z,
            int* ncells_per_dim, int dim, int n_cells) nogil:
        """Return the flattened index for a valid cell"""
        cdef long ncy = ncells_per_dim[1]
        cdef long ncz = ncells_per_dim[2]

        cdef long cell_id
        cdef long cell_index = -1

        # basic test for valid indices. Since we bin the particles with
        # respect to the origin, negative indices can never occur.
        cdef bint is_valid = (cid_x > -1) and (cid_y > -1) and (cid_z > -1)

        # additional check for 1D. This is because we search in all 26
        # neighboring cells for neighbors. In 1D this can be problematic
        # since (ncy = ncz = 0) which means (ncy=1 or ncz=1) will also
        # result in a valid cell with a flattened index < ncells
        if dim == 1:
            if ( (cid_y > ncy) or (cid_z > ncz) ):
                is_valid = False

        # Given the validity of the cells, return the flattened cell index
        cdef map[long, int].iterator it
        if is_valid:
            cell_id = flatten_raw(cid_x, cid_y, cid_z, ncells_per_dim, dim)
            if cell_id > -1:
                it = self.cell_to_index.find(cell_id)
                if it != self.cell_to_index.end():
                    cell_index = deref(it).second

        return cell_index

    cpdef long _count_occupied_cells(self, long n_cells) except -1:
        cdef list pa_wrappers = self.pa_wrappers
        cdef NNPSParticleArrayWrapper pa_wrapper
        cdef int np
        cdef DoubleArray x, y, z
        cdef DoubleArray xmin = self.xmin
        cdef DoubleArray xmax = self.xmax
        cdef IntArray ncells_per_dim = self.ncells_per_dim
        cdef int narrays = self.narrays
        cdef int dim = self.dim
        cdef double cell_size = self.cell_size

        cdef cPoint pnt = cPoint_new(0, 0, 0)
        cdef long _cid
        cdef map[long, int] _cid_to_index
        cdef pair[long, int] _entry

        # flattened cell index
        cdef int i, j
        cdef long cell_index

        for j in range(narrays):
            pa_wrapper = pa_wrappers[j]
            np = pa_wrapper.get_number_of_particles()
            x = pa_wrapper.x
            y = pa_wrapper.y
            z = pa_wrapper.z

            for i in range(np):
                # the flattened index is considered relative to the
                # minimum along each co-ordinate direction
                pnt.x = x.data[i] - xmin.data[0]
                pnt.y = y.data[i] - xmin.data[1]
                pnt.z = z.data[i] - xmin.data[2]

                # flattened cell index
                _cid = flatten( find_cell_id( pnt, cell_size ), ncells_per_dim, dim )
                _entry.first = _cid
                _entry.second = 1
                _cid_to_index.insert(_entry)

        cdef map[long, int].iterator it = _cid_to_index.begin()
        cdef int count = 0
        while it != _cid_to_index.end():
            _entry = deref(it)
            _cid_to_index[_entry.first] = count
            count += 1
            inc(it)

        self.cell_to_index = _cid_to_index
        return count

