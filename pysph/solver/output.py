"""
An interface to output the data in various format
"""

import numpy
import os
import sys

from pysph.base.particle_array import ParticleArray
from pysph.base.utils import get_particles_info, get_particle_array
from pysph import has_h5py

output_formats = ('hdf5', 'npz')
COMPRESSION_LEVEL = 6


def _to_str(s):
    if isinstance(s, bytes) and sys.version_info[0] > 2:
        return s.decode('utf-8')
    else:
        return str(s)


def gather_array_data(all_array_data, comm):
    """Given array_data from the current processor and an MPI
    communicator,return a joined array_data from all processors
    on rank 0 and the same array_data on the other machines.
    """

    array_names = all_array_data.keys()

    # gather the data from all processors
    collected_data = comm.gather(all_array_data, root=0)

    if comm.Get_rank() == 0:
        all_array_data = {}
        size = comm.Get_size()

        # concatenate the arrays
        for array_name in array_names:
            array_data = {}
            all_array_data[array_name] = array_data

            _props = collected_data[0][array_name].keys()
            for prop in _props:
                data = [collected_data[pid][array_name][prop]
                        for pid in range(size)]
                prop_arr = numpy.concatenate(data)
                array_data[prop] = prop_arr
    return all_array_data


class Output(object):
    """ Class that handles output for simulation """
    def __init__(self, detailed_output=False, only_real=True, mpi_comm=None,
                 compress=False):
        self.compress = compress
        self.detailed_output = detailed_output
        self.only_real = only_real
        self.mpi_comm = mpi_comm

    def dump(self, fname, particles, solver_data):
        self.particle_data = dict(get_particles_info(particles))
        self.all_array_data = {}
        for array in particles:
            self.all_array_data[array.name] = array.get_property_arrays(
                all=self.detailed_output,
                only_real=self.only_real
                )
        mpi_comm = self.mpi_comm
        if mpi_comm is not None:
            self.all_array_data = gather_array_data(
                self.all_array_data, mpi_comm
            )
        self.solver_data = solver_data
        if mpi_comm is None or mpi_comm.Get_rank() == 0:
            self._dump(fname)

    def load(self, fname):
        return self._load(fname)

    def _dump(self, fname):
        """ Implement the method for writing the output to a file here """
        raise NotImplementedError()

    def _load(self, fname):
        """ Implement the method for loading from file here """
        raise NotImplementedError()


def _dict_bytes_to_str(d):
    # This craziness is needed as if the npz file is saved in Python2
    # then all the strings are bytes and if this is loaded in Python 3,
    # the keys will be bytes and not strings leading to strange errors.
    res = {}
    for key, value in d.items():
        if isinstance(value, dict):
            value = _dict_bytes_to_str(value)
        if isinstance(value, bytes):
            value = _to_str(value)
        if isinstance(value, list):
            if value and isinstance(value[0], bytes):
                value = [_to_str(x) for x in value]
        res[_to_str(key)] = value
    return res


def _get_dict_from_arrays(arrays):
    arrays.shape = (1,)
    res = arrays[0]
    if res and isinstance(list(res.keys())[0], bytes):
        return _dict_bytes_to_str(res)
    else:
        return res


class NumpyOutput(Output):

    def _dump(self, filename):
        save_method = numpy.savez_compressed if self.compress else numpy.savez
        output_data = {"particles": self.particle_data,
                       "solver_data": self.solver_data}
        for name, arrays in self.all_array_data.items():
            self.particle_data[name]["arrays"] = arrays
        save_method(filename, version=2, **output_data)

    def _load(self, fname):
        data = numpy.load(fname, encoding='bytes', allow_pickle=True)

        if 'version' not in data.files:
            msg = "Wrong file type! No version number recorded."
            raise RuntimeError(msg)
        ret = {}
        ret["arrays"] = {}
        version = data['version']
        solver_data = _get_dict_from_arrays(data["solver_data"])
        ret["solver_data"] = solver_data

        if version == 1:
            arrays = _get_dict_from_arrays(data["arrays"])
            for array_name in arrays:
                array = get_particle_array(name=array_name,
                                           **arrays[array_name])
                ret["arrays"][array_name] = array

        elif version == 2:
            particles = _get_dict_from_arrays(data["particles"])

            for array_name, array_info in particles.items():
                for prop, data in array_info['arrays'].items():
                    array_info['properties'][prop]['data'] = data
                array = ParticleArray(name=array_name,
                                      constants=array_info["constants"],
                                      **array_info["properties"])
                array.set_output_arrays(
                    array_info.get('output_property_arrays', [])
                )
                ret["arrays"][array_name] = array

        else:
            raise RuntimeError("Version not understood!")
        return ret


class HDFOutput(Output):

    def _dump(self, filename):
        import h5py
        with h5py.File(filename, 'w') as f:
            solver_grp = f.create_group('solver_data')
            particles_grp = f.create_group('particles')
            for ptype, pdata in self.particle_data.items():
                ptype_grp = particles_grp.create_group(ptype)
                arrays_grp = ptype_grp.create_group('arrays')
                data = self.all_array_data[ptype]
                self._set_constants(pdata, ptype_grp)
                self._set_properties(pdata, arrays_grp, data)
            self._set_solver_data(solver_grp)

    def _load(self, fname):
        if has_h5py():
            import h5py
        else:
            msg = "Install python-h5py to load this file"
            raise ImportError(msg)

        ret = {}
        with h5py.File(fname, 'r') as f:
            solver_grp = f['solver_data']
            particles_grp = f['particles']
            ret["solver_data"] = self._get_solver_data(solver_grp)
            ret["arrays"] = self._get_particles(particles_grp)
        return ret

    def _get_particles(self, grp):

        particles = {}
        for name, prop_array in grp.items():
            output_array = []
            const_grp = prop_array['constants']
            arrays_grp = prop_array['arrays']
            constants = self._get_constants(const_grp)
            array = ParticleArray(_to_str(name), constants=constants)

            for pname, h5obj in arrays_grp.items():
                prop_name = _to_str(h5obj.attrs['name'])
                type_ = _to_str(h5obj.attrs['type'])
                default = h5obj.attrs['default']
                stride = h5obj.attrs.get('stride', 1)
                if h5obj.attrs['stored']:
                    output_array.append(_to_str(pname))
                    array.add_property(
                        prop_name, type=type_, default=default,
                        data=numpy.array(h5obj),
                        stride=stride
                    )
                else:
                    array.add_property(prop_name, type=type_, stride=stride)
            array.set_output_arrays(output_array)
            particles[str(name)] = array
        return particles

    def _get_solver_data(self, grp):
        solver_data = {}
        for name, value in grp.attrs.items():
            solver_data[_to_str(name)] = value
        return solver_data

    def _get_constants(self, grp):
        constants = {}
        for const_name, const_data in grp.items():
            constants[_to_str(const_name)] = numpy.array(const_data)
        return constants

    def _get_compress_options(self):
        if self.compress:
            return dict(compression="gzip",
                        compression_opts=COMPRESSION_LEVEL)
        else:
            return {}

    def _set_constants(self, pdata, ptype_grp):
        pconstants = pdata['constants']
        constGroup = ptype_grp.create_group('constants')
        c_kw = self._get_compress_options()
        for constName, constArray in pconstants.items():
            constGroup.create_dataset(constName, data=constArray, **c_kw)

    def _set_properties(self, pdata, ptype_grp, data):
        c_kw = self._get_compress_options()
        for propname, attributes in pdata['properties'].items():
            if propname in data:
                array = data[propname]
                prop = ptype_grp.create_dataset(propname, data=array, **c_kw)
                prop.attrs['stored'] = True
            else:
                prop = ptype_grp.create_dataset(propname, (0,))
                prop.attrs['stored'] = False

            for attname, value in attributes.items():
                if value is None:
                    value = 'None'
                prop.attrs[attname] = value

    def _set_solver_data(self, grp):
        for name, data in self.solver_data.items():
            grp.attrs[name] = data


def load(fname):
    """
    Load the output data

    Parameters
    ----------
    fname: str
        Name of the file or full path


    Examples
    --------
    >>> data = load('elliptical_drop_100.npz')
    >>> data.keys()
    ['arrays', 'solver_data']
    >>> arrays = data['arrays']
    >>> arrays.keys()
    ['fluid']
    >>> fluid = arrays['fluid']
    >>> type(fluid)
    pysph.base.particle_array.ParticleArray
    >>> data['solver_data']
    {'count': 100, 'dt': 4.6416394784204199e-05, 't': 0.0039955855395528766}
    """

    if fname.endswith('npz'):
        output = NumpyOutput()
    elif fname.endswith('hdf5'):
        output = HDFOutput()
    if os.path.isfile(fname):
        return output.load(fname)
    else:
        msg = "File not present"
        raise RuntimeError(msg)


def dump(filename, particles, solver_data, detailed_output=False,
         only_real=True, mpi_comm=None, compress=False):

    """
    Dump the given particles and solver data to the given filename.

    Parameters
    ----------

    filename: str
        Filename to dump to.

    particles: sequence(ParticleArray)
        Sequence of particle arrays to dump.

    solver_data: dict
        Additional information to dump about solver state.

    detailed_output: bool
        Specifies if all arrays should be dumped.

    only_real: bool
        Only dump the real particles.

    mpi_comm: mpi4pi.MPI.Intracomm
        An MPI communicator to use for parallel commmunications.

    compress: bool
        Specify if the  file is to be compressed or not.

    If `mpi_comm` is not passed or is set to None the local particles alone
    are dumped, otherwise only rank 0 dumps the output.

    """
    if filename.endswith(output_formats):
        fname = os.path.splitext(filename)[0]
    else:
        fname = filename
        filename = fname + '.hdf5'
    if filename.endswith('hdf5') and has_h5py():
        file_format = 'hdf5'
        output = HDFOutput(detailed_output, only_real, mpi_comm, compress)
    else:
        output = NumpyOutput(detailed_output, only_real, mpi_comm, compress)
        file_format = 'npz'
    filename = fname + '.' + file_format
    output.dump(filename, particles, solver_data)
