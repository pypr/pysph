"""
Module contains some common functions.
"""

# standard imports
import pickle
import numpy
import sys
import os

from pysph.base.particle_array import ParticleArray
from pysph.base.utils import get_particle_array, get_particles_info
from pysph.solver.output import load, dump, output_formats

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

def get_array_by_name(arrays, name):
    """Given a list of arrays and the name of the desired array, return the
    desired array.
    """
    for array in arrays:
        if array.name == name:
            return array

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

    def update(self, delta=1):
        self.count += delta
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


class FloatPBar(object):
    def __init__(self, t_initial, t_final, show=True):
        self.ticks = 1000
        self.bar = PBar(self.ticks, show)
        self.t_initial = t_initial
        self.t_final = t_final
        self.t_diff = t_final - t_initial

    def update(self, time):
        expected_count = int((time - self.t_initial)/self.t_diff*self.ticks)
        expected_count = min(expected_count, self.ticks)
        diff = max(expected_count - self.bar.count, 0)
        if diff > 0:
            self.bar.update(diff)

    def finish(self):
        self.bar.finish()

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


def get_pysph_root():
    return os.path.split(pysph.__file__)[0]

def dump_v1(filename, particles, solver_data, detailed_output=False,
         only_real=True, mpi_comm=None):
    """Dump the given particles and solver data to the given filename using
    version 1.  This is mainly used only for testing that we can continue
    to load older versions of the data files.
    """

    all_array_data = {}
    output_data = {"arrays":all_array_data, "solver_data":solver_data}

    for array in particles:
        all_array_data[array.name] = array.get_property_arrays(
            all=detailed_output, only_real=only_real)

    # Gather particle data on root
    if mpi_comm is not None:
        all_array_data = _gather_array_data(all_array_data, mpi_comm)

    output_data['arrays'] = all_array_data

    if mpi_comm is None or mpi_comm.Get_rank() == 0:
        numpy.savez(filename, version=1, **output_data)


def load_and_concatenate(prefix,nprocs=1,directory=".",count=None):
    """Load the results from multiple files.

    Given a filename prefix and the number of processors, return a
    concatenated version of the dictionary returned via load.

    Parameters
    ----------

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
from pysph.base.kernels import Gaussian, get_compiled_kernel
from pysph.base.nnps import LinkedListNNPS as NNPS
class SPHInterpolate(object):
    """Class to perform SPH interpolation

    Given solution data on possibly a scattered set, SPHInterpolate
    can be used to interpolate solution data on a regular grid.

    """
    def __init__(self, dim, dst, src, kernel=None):
        self.dst = dst; self.src = src
        if kernel is None:
            self.kernel = get_compiled_kernel(Gaussian(dim))

        # create the neighbor locator object
        self.nnps = nnps = NNPS(dim=dim, particles=[dst, src],
                                radius_scale=self.kernel.radius_scale)
        nnps.update()

    def interpolate(self, arr):
        """Interpolate data given in arr onto coordinate positions"""
        # the result array
        np = self.dst.get_number_of_particles()
        result = numpy.zeros(np)

        nbrs = UIntArray()

        # source arrays
        src = self.src
        sx, sy, sz, sh = src.get('x', 'y', 'z', 'h', only_real_particles=False)

        # dest arrays
        dst = self.dst
        dx, dy, dz, dh = dst.x, dst.y, dst.z, dst.h

        # kernel
        kernel = self.kernel

        for i in range(np):
            self.nnps.get_nearest_particles(src_index=1, dst_index=0, d_idx=i, nbrs=nbrs)
            nnbrs = nbrs.length

            _wij = 0.0; _sum = 0.0
            for indexj in range(nnbrs):
                j  = nbrs[indexj]

                hij = 0.5 * (sh[j] + dh[i])

                wij = kernel.kernel(dx[i], dy[i], dz[i], sx[j], sy[j], sz[j], hij)
                _wij += wij
                _sum += arr[j] * wij

            # save the result
            result[i] = _sum/_wij

        return result

    def gradient(self, arr):
        """Compute the gradient on the interpolated data"""
        # the result array
        np = self.dst.get_number_of_particles()

        # result arrays
        resultx = numpy.zeros(np)
        resulty = numpy.zeros(np)
        resultz = numpy.zeros(np)

        nbrs = UIntArray()

        # data arrays
        # source arrays
        src = self.src
        sx, sy, sz, sh, srho, sm = src.get('x', 'y', 'z', 'h', 'rho', 'm', only_real_particles=False)

        # dest arrays
        dst = self.dst
        dx, dy, dz, dh = dst.x, dst.y, dst.z, dst.h

        # kernel
        kernel = self.kernel

        for i in range(np):
            self.nnps.get_nearest_particles(src_index=1, dst_index=0, d_idx=i, nbrs=nbrs)
            nnbrs = nbrs.length

            _wij = 0.0; _sumx = 0.0; _sumy = 0.0; _sumz = 0.0
            for indexj in range(nnbrs):
                j  = nbrs[indexj]

                hij = 0.5 * (sh[j] + dh[i])

                wij = kernel.kernel(dx[i], dy[i], dz[i], sx[j], sy[j], sz[j], hij)

                rhoj = srho[j]; mj = sm[j]
                fji = arr[j] - arr[i]

                # kernel gradient
                dwij = kernel.gradient(dx[i], dy[i], dz[i], sx[j], sy[j], sz[j], hij)

                _wij += wij
                tmp = 1./rhoj * fji * mj

                _sumx += tmp * dwij.x
                _sumy += tmp * dwij.y
                _sumz += tmp * dwij.z

            # save the result
            resultx[i] = _sumx
            resulty[i] = _sumy
            resultz[i] = _sumz

        return resultx, resulty, resultz

def get_files(dirname=None, fname=None, endswith=output_formats):
    """Get all solution files in a given directory, `dirname`.

    Parameters
    ----------

    dirname: str
        Name of directory.
    fname: str
        An initial part of the filename, if not specified use the first
        part of the dirname.
    endswith: str
        The extension of the file to load.
    """

    if dirname is None:
        return []

    if fname is None:
        fname = dirname.split("_output")[0]

    path = os.path.abspath( dirname )
    files = os.listdir( path )

    # get all the output files in the directory
    files = [f for f in files if f.startswith(fname) and f.endswith(endswith)]
    files = [os.path.join(path, f) for f in files]

    # sort the files
    def _key_func(arg):
        a = os.path.splitext(arg)[0]
        return int(a[a.rfind('_')+1:])

    files.sort(key=_key_func)

    return files


def iter_output(files, *arrays):
    """Given an iterable of the solution files, this loads the files, and
    yields the solver data and the requested arrays.

    If arrays is not supplied, it returns a dictionary of the arrays.

    Parameters
    ----------

    files : iterable
        Iterates over the list of desired files

    *arrays : strings
        Optional series of array names of arrays to return.

    Examples
    --------

    >>> files = get_files('elliptical_drop_output')
    >>> for solver_data, arrays in iter_output(files):
    ...     print(solver_data['t'], arrays.keys())

    >>> files = get_files('elliptical_drop_output')
    >>> for solver_data, fluid in iter_output(files, 'fluid'):
    ...     print(solver_data['t'], fluid.name)

    """
    for file in files:
        data = load(file)
        solver_data = data['solver_data']
        if len(arrays) == 0:
            yield solver_data, data['arrays']
        else:
            _arrays = [data['arrays'][x] for x in arrays]
            yield [solver_data] + _arrays

def _sort_key(arg):
    a = os.path.splitext(arg)[0]
    return int(a[a.rfind('_')+1:])

def remove_irrelevant_files(files):
    """Remove any npz files that are not output files.

    That is, the file should not end with a '_number.npz'.  This allows users
    to dump other .npz of .hdf5 files in the output while post-processing without
    breaking.
    """
    result = []
    for f in files:
        try:
            _sort_key(f)
        except ValueError:
            pass
        else:
            result.append(f)
    return result

