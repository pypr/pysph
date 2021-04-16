"""
Module contains some common functions.
"""

# standard imports
import errno
from glob import glob
import os
import socket
import sys
import time

import numpy

import pysph
from pysph.solver.output import load, dump, output_formats  # noqa: 401
from pysph.solver.output import gather_array_data as _gather_array_data

ASCII_FMT = " 123456789#"
try:
    uni_chr = unichr
except NameError:
    uni_chr = chr
UTF_FMT = u" " + u''.join(map(uni_chr, range(0x258F, 0x2587, -1)))


def _supports_unicode(fp):
    # Taken somewhat from the tqdm package.
    if not hasattr(fp, 'encoding'):
        return False
    else:
        encoding = fp.encoding
        try:
            u'\u2588\u2589'.encode(encoding)
        except UnicodeEncodeError:
            return False
        except Exception:
            try:
                return (encoding.lower().startswith('utf-')
                        or ('U8' == encoding))
            except:
                return False
        else:
            return True


def get_free_port(start, skip=None):
    """Return an integer that is an available port for a service. Start at the
    given `start` value and `skip` any specified values.
    """
    skip = () if skip is None else skip
    x = start
    while x < 65536:
        if x in skip:
            x += 1
        else:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(('', x))
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    return x
                except socket.error as e:
                    if e.errno == errno.EADDRINUSE:
                        x += 1
                    else:
                        raise


def is_using_ipython():
    """Return True if the code is being run from an IPython session or
    notebook.
    """
    try:
        # If this is being run inside an IPython console or notebook
        # then this is defined.
        __IPYTHON__
    except NameError:
        return False
    else:
        return True


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


def fmt_time(time):
    mm, ss = divmod(time, 60)
    hh, mm = divmod(mm, 60)
    if hh > 0:
        s = "%d:%02d:%02d" % (hh, mm, ss)
    else:
        s = "%02d:%02.1f" % (mm, ss)
    return s


class ProgressBar(object):
    def __init__(self, ti, tf, show=True, file=None, ascii=False):
        if file is None:
            self.file = sys.stdout
        self.ti = ti
        self.tf = tf
        self.t = 0.0
        self.dt = 1.0
        self.start = time.time()
        self.count = 0
        self.iter_inc = 1
        self.show = show
        self.ascii = ascii
        if not ascii and not _supports_unicode(self.file):
            self.ascii = True
        if not self.file.isatty() and not is_using_ipython():
            self.show = False
        self.display()

    def _fmt_bar(self, percent, width):
        chars = ASCII_FMT if self.ascii else UTF_FMT
        nsyms = len(chars) - 1
        tens, ones = divmod(int(percent/100 * width * nsyms), nsyms)
        end = chars[ones] if ones > 0 else ''
        return (chars[-1]*tens + end).ljust(width)

    def _fmt_iters(self, iters):
        if iters < 1e3:
            s = '%d' % iters
        elif iters < 1e6:
            s = '%.1fk' % (iters/1e3)
        elif iters < 1e9:
            s = '%.1fM' % (iters/1e6)
        return s

    def display(self):
        if self.show:
            elapsed = time.time() - self.start
            if self.t > 0:
                eta = (self.tf - self.t)/self.t * elapsed
            else:
                eta = 0.0
            percent = int(round(self.t/self.tf*100))
            bar = self._fmt_bar(percent, 20)
            secsperit = elapsed/self.count if self.count > 0 else 0
            out = ('{percent:3}%|{bar}|'
                   ' {iters}it | {time:.1e}s [{elapsed}<{eta} | '
                   '{secsperit:.3f}s/it]').format(
                bar=bar, percent=percent, iters=self._fmt_iters(self.count),
                time=self.t, elapsed=fmt_time(elapsed), eta=fmt_time(eta),
                secsperit=secsperit
            )
            self.file.write('\r%s' % out.ljust(70))
            self.file.flush()

    def update(self, t, iter_inc=1):
        '''Set the current time and update the number of iterations.
        '''
        self.dt = t - self.t
        self.iter_inc = iter_inc
        self.count += iter_inc
        self.t = t
        self.display()

    def finish(self):
        self.display()
        if self.show:
            self.file.write('\n')


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
        raise OSError("a file with the same name as the desired "
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
    output_data = {"arrays": all_array_data, "solver_data": solver_data}

    for array in particles:
        all_array_data[array.name] = array.get_property_arrays(
            all=detailed_output, only_real=only_real
        )

    # Gather particle data on root
    if mpi_comm is not None:
        all_array_data = _gather_array_data(all_array_data, mpi_comm)

    output_data['arrays'] = all_array_data

    if mpi_comm is None or mpi_comm.Get_rank() == 0:
        numpy.savez(filename, version=1, **output_data)


def load_and_concatenate(prefix, nprocs=1, directory=".", count=None):
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
        counts = [i.rsplit('_', 1)[1][:-4] for i in os.listdir(directory)
                  if i.startswith(prefix) and i.endswith('.npz')]
        counts = sorted([int(i) for i in counts])
        count = counts[-1]

    arrays_by_rank = {}

    for rank in range(nprocs):
        fname = os.path.join(
            directory, prefix + '_' + str(rank) + '_' + str(count) + '.npz'
        )

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
            for rank in range(1, nprocs):
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

    path = os.path.abspath(dirname)

    if fname is None:
        infos = glob(os.path.join(path, "*.info"))
        if infos:
            fname = os.path.splitext(os.path.basename(infos[0]))[0]
        else:
            fname = os.path.basename(path).split('_output')[0]

    files = glob(os.path.join(path, "%s*.*" % fname))
    files = [f for f in files if f.endswith(endswith)]

    # sort the files
    files.sort(key=_sort_key)

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
    return int(a[a.rfind('_') + 1:])


def remove_irrelevant_files(files):
    """Remove any npz files that are not output files.

    That is, the file should not end with a '_number.npz'. This allows users to
    dump other .npz of .hdf5 files in the output while post-processing without
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
