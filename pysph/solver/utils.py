"""
Module contains some common functions.
"""

# standard imports
import pickle
import numpy
import sys
import os
import platform
import commands
import tempfile
import zipfile
from numpy.lib import format

HAS_PBAR = True
try:
    import progressbar
except ImportError:
    HAS_PBAR = False

import pysph

def check_array(x, y):
    """Check if two arrays are equal with an absolute tolerance of
    1e-16."""
    return numpy.allclose(x, y, atol=1e-16, rtol=0)


def zipfile_factory(*args, **kwargs):
    if sys.version_info >= (2, 5):
        kwargs['allowZip64'] = True
    return zipfile.ZipFile(*args, **kwargs)

def savez(file, *args, **kwds):
    """
    Save several arrays into a single file in uncompressed ``.npz`` format.

    If arguments are passed in with no keywords, the corresponding variable
    names, in the .npz file, are 'arr_0', 'arr_1', etc. If keyword arguments
    are given, the corresponding variable names, in the ``.npz`` file will
    match the keyword names.

    Parameters
    ----------
    file : str or file
        Either the file name (string) or an open file (file-like object)
        where the data will be saved. If file is a string, the ``.npz``
        extension will be appended to the file name if it is not already there.
    *args : Arguments, optional
        Arrays to save to the file. Since it is not possible for Python to
        know the names of the arrays outside `savez`, the arrays will be saved
        with names "arr_0", "arr_1", and so on. These arguments can be any
        expression.
    **kwds : Keyword arguments, optional
        Arrays to save to the file. Arrays will be saved in the file with the
        keyword names.

    Returns
    -------
    None

    See Also
    --------
    save : Save a single array to a binary file in NumPy format.
    savetxt : Save an array to a file as plain text.

    Notes
    -----
    The ``.npz`` file format is a zipped archive of files named after the
    variables they contain.  The archive is not compressed and each file
    in the archive contains one variable in ``.npy`` format. For a
    description of the ``.npy`` format, see `format`.

    When opening the saved ``.npz`` file with `load` a `NpzFile` object is
    returned. This is a dictionary-like object which can be queried for
    its list of arrays (with the ``.files`` attribute), and for the arrays
    themselves.

    Examples
    --------
    >>> from tempfile import TemporaryFile
    >>> outfile = TemporaryFile()
    >>> x = np.arange(10)
    >>> y = np.sin(x)

    Using `savez` with *args, the arrays are saved with default names.

    >>> np.savez(outfile, x, y)
    >>> outfile.seek(0) # Only needed here to simulate closing & reopening file
    >>> npzfile = np.load(outfile)
    >>> npzfile.files
    ['arr_1', 'arr_0']
    >>> npzfile['arr_0']
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    Using `savez` with **kwds, the arrays are saved with the keyword names.

    >>> outfile = TemporaryFile()
    >>> np.savez(outfile, x=x, y=y)
    >>> outfile.seek(0)
    >>> npzfile = np.load(outfile)
    >>> npzfile.files
    ['y', 'x']
    >>> npzfile['x']
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    See Also
    --------
    numpy.savez_compressed : Save several arrays into a compressed .npz file
    format

    """
    _savez(file, args, kwds, False)

def savez_compressed(file, *args, **kwds):
    """
    Save several arrays into a single file in compressed ``.npz`` format.

    If keyword arguments are given, then filenames are taken from the keywords.
    If arguments are passed in with no keywords, then stored file names are
    arr_0, arr_1, etc.

    Parameters
    ----------
    file : string
        File name of .npz file.
    args : Arguments
        Function arguments.
    kwds : Keyword arguments
        Keywords.

    See Also
    --------
    numpy.savez : Save several arrays into an uncompressed .npz file format

    """
    _savez(file, args, kwds, True)

def _savez(file, args, kwds, compress):
    if isinstance(file, basestring):
        if not file.endswith('.npz'):
            file = file + '.npz'

    namedict = kwds
    for i, val in enumerate(args):
        key = 'arr_%d' % i
        if key in namedict.keys():
            msg = "Cannot use un-named variables and keyword %s" % key
            raise ValueError, msg
        namedict[key] = val

    if compress:
        compression = zipfile.ZIP_DEFLATED
    else:
        compression = zipfile.ZIP_STORED

    zip = zipfile_factory(file, mode="w", compression=compression)

    # Stage arrays in a temporary file on disk, before writing to zip.
    fd, tmpfile = tempfile.mkstemp(suffix='-numpy.npy')
    os.close(fd)
    try:
        for key, val in namedict.iteritems():
            fname = key + '.npy'
            fid = open(tmpfile, 'wb')
            try:
                format.write_array(fid, numpy.asanyarray(val))
                fid.close()
                fid = None
                zip.write(tmpfile, arcname=fname)
            finally:
                if fid:
                    fid.close()
    finally:
        os.remove(tmpfile)

    zip.close()

#############################################################################


def get_distributed_particles(pa, comm, cell_size):
    # FIXME: this can be removed once the examples all use Application. 
    from pysph.parallel.load_balancer import LoadBalancer
    rank = comm.Get_rank()
    num_procs = comm.Get_size()

    if rank == 0:
        lb = LoadBalancer.distribute_particles(pa, num_procs=num_procs, 
                                               block_size=cell_size)
    else:
        lb = None

    particles = comm.scatter(lb, root=0)

    return particles


################################################################################
# `PBar` class.
############################################################################### 
class PBar(object):
    """A simple wrapper around the progressbar so it works if a user has
    it installed or not.
    """
    def __init__(self, maxval, show=True):
        bar = None
        self.count = 0
        self.maxval = maxval
        self.show = show 
        if HAS_PBAR and show:
            widgets = [progressbar.Percentage(), ' ', progressbar.Bar(),
                       progressbar.ETA()]
            bar = progressbar.ProgressBar(widgets=widgets,
                                          maxval=maxval).start()
        self.bar = bar

    def update(self):
        self.count += 1
        if self.bar is not None:
            self.bar.update(self.count)
        elif self.show:
            sys.stderr.write('\r%d%%'%int(self.count*100/self.maxval))
            sys.stderr.flush()

    def finish(self):
        if self.bar is not None:
            self.bar.finish()
        elif self.show:
            sys.stderr.write('\r100%\n')

            sys.stderr.flush()

##############################################################################
# friendly mkdir  from http://code.activestate.com/recipes/82465/.
############################################################################## 
def mkdir(newdir):
    """works the way a good mkdir should :)
        - already exists, silently complete
        - regular file in the way, raise an exception
        - parent directory(ies) does not exist, make them as well
    """
    if os.path.isdir(newdir):
        pass

    elif os.path.isfile(newdir):
        raise OSError("a file with the same name as the desired " \
                      "dir, '%s', already exists." % newdir)

    else:
        head, tail = os.path.split(newdir)

        if head and not os.path.isdir(head):
            mkdir(head)

        if tail:
            try:
                os.mkdir(newdir)
            # To prevent race in mpi runs
            except OSError as e:
                import errno
                if e.errno == errno.EEXIST and os.path.isdir(newdir):
                    pass
                else:
                    raise
                


##############################################################################
# read pickled data from a file
############################################################################## 
def get_pickled_data(fname):
    
    f = open(fname, 'r')
    data = pickle.load(f)
    f.close()

    return data


def get_pysph_root():
    return os.path.split(pysph.__file__)[0]    
    

##############################################################################
# Load an output file
############################################################################## 
def load(fname):
    """ Load and return data from an  output (.npz) file dumped by PySPH.

    For output file version 1, the function returns a dictionary with
    the keys:

    solver_data : Solver constants at the time of output like time,
    time step and iteration count.

    arrays : ParticleArrays keyed on names with the ParticleArray
    object as value.

    """
    from pysph.base.utils import get_particle_array_wcsph
    data = numpy.load(fname)

    ret = {"arrays":{}}

    if not 'version' in data.files:
        msg = "Wrong file type! No version nnumber recorded."
        raise RuntimeError(msg)
    
    version = data['version']

    if version == 1:

        arrays = data["arrays"]
        arrays.shape = (1,)
        arrays = arrays[0]

        solver_data = data["solver_data"]
        solver_data.shape = (1,)
        solver_data = solver_data[0]

        for array_name in arrays:
            array = get_particle_array_wcsph(name=array_name,
                                             cl_precision="single",
                                             **arrays[array_name])
            
            ret["arrays"][array_name] = array
            
        ret["solver_data"] = solver_data

    else:
        raise RuntimeError("Version not understood!")

    return ret

def load_and_concatenate(prefix,nprocs=1,directory=".",count=None):
    """Load the results from multiple files.

    Given a filename prefix and the number of processors, return a
    concatenated version of the dictionary returned via load.

    Parameters:
    -----------

    prefix : str
        A filename prefix for the output file.

    nprocs : int
        The number of processors (files) to read

    directory : str
        The directory for the files

    count : int
        The file iteration count to read. If None, the last available
        one is read

    """

    if count is None:
        counts = [i.rsplit('_',1)[1][:-4] for i in os.listdir(directory) if i.startswith(prefix) and i.endswith('.npz')]
        counts = sorted( [int(i) for i in counts] )
        count = counts[-1]

    arrays_by_rank = {}
    
    for rank in range(nprocs):
        fname = os.path.join(directory, prefix+'_'+str(rank)+'_'+str(count)+'.npz')

        data = load(fname)
        arrays_by_rank[rank] = data["arrays"]

    arrays = _concatenate_arrays(arrays_by_rank, nprocs)

    data["arrays"] = arrays

    return data

def _concatenate_arrays(arrays_by_rank, nprocs):
    """Concatenate arrays into one single particle array. """

    if nprocs <= 0:
        return 0

    array_names = arrays_by_rank[0].keys()
    first_processors_arrays = arrays_by_rank[0]
    
    if nprocs > 1:
        ret = {}
        for array_name in array_names:
            first_array = first_processors_arrays[array_name]
            for rank in range(1,nprocs):
                other_processors_arrays = arrays_by_rank[rank]
                other_array = other_processors_arrays[array_name]

                # append the other array to the first array
                first_array.append_parray(other_array)

                # remove the non local particles
                first_array.remove_tagged_particles(1)
                
            ret[array_name] = first_array

    else:
        ret = arrays_by_rank[0]

    return ret

# SPH interpolation of data
from pyzoltan.core.carray import UIntArray
from pysph.parallel._kernels import Gaussian
from pysph.base.nnps import NNPS
from pysph.base.point import Point
class SPHInterpolate(object):
    """Class to perform SPH interpolation

    Given solution data on possibly a scattered set, SPHInterpolate
    can be used to interpolate solution data on a regular grid.

    """
    def __init__(self, dim, dst, src, kernel=None):
        self.dst = dst; self.src = src
        if kernel is None:
            self.kernel = Gaussian(dim)
        
        # create the neighbor locator object
        self.nnps = nnps = NNPS(dim=dim, particles=[dst, src], radius_scale=self.kernel.radius)
        nnps.update()
        
    def interpolate(self, arr):
        """Interpolate data given in arr onto coordinate positions"""
        # the result array
        np = self.dst.get_number_of_particles()
        result = numpy.zeros(np)

        nbrs = UIntArray()

        # source arrays
        src = self.src
        sx, sy, sz, sh = src.x, src.y, src.z, src.h

        # dest arrays
        dst = self.dst
        dx, dy, dz, dh = dst.x, dst.y, dst.z, dst.h

        # kernel
        kernel = self.kernel

        for i in range(np):
            xi = Point( dx[i], dy[i], dz[i] )

            self.nnps.get_nearest_particles(src_index=1, dst_index=0, d_idx=i, nbrs=nbrs)
            nnbrs = nbrs._length

            _wij = 0.0; _sum = 0.0
            for indexj in range(nnbrs):
                j  = nbrs[indexj]
                xj = Point( sx[j], sy[j], sz[j] )

                hij = 0.5 * (sh[j] + dh[i])
                
                wij = kernel.py_function(xi, xj, hij)
                _wij += wij
                _sum += arr[j] * wij

            # save the result
            result[i] = _sum/_wij
            
        return result
