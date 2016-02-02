"""
An interface to output the data in various format
"""

import numpy
import os

from pysph.base.particle_array import ParticleArray
from pysph.base.utils import get_particles_info, get_particle_array
from pysph import has_h5py, has_pyvisfile

output_formats = ('hdf5', 'npz')


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
            self.all_array_data = self._gather_array_data(
                    self.all_array_data, mpi_comm
                    )
        self.solver_data = solver_data
        self._dump(fname)

    def load(self, fname):
        return self._load(fname);

    def _gather_array_data(self, all_array_data, comm):
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


class NumpyOutput(Output):

    def _dump(self, filename):
        save_method = numpy.savez_compressed if self.compress else numpy.savez
        output_data = {"particles": self.particle_data,
                       "solver_data": self.solver_data}
        for name, arrays in self.all_array_data.items():
            self.particle_data[name]["arrays"] = arrays
        save_method(filename, version=2, **output_data)

    def _load(self, fname):
        def _get_dict_from_arrays(arrays):
            arrays.shape = (1,)
            return arrays[0]
        data = numpy.load(fname)

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
                array = ParticleArray(name=array_name,
                                      constants=array_info["constants"],
                                      **array_info["arrays"])
                array.set_output_arrays(
                    array_info.get('output_property_arrays', [])
                )
                for prop, prop_info in array_info["properties"].items():
                    if prop not in array_info["arrays"]:
                        array.add_property(**prop_info)
                ret["arrays"][array_name] = array

        else:
            raise RuntimeError("Version not understood!")
        return ret


class HDFOutput(Output):

    def _dump(self, filename):
        import h5py
        f = h5py.File(filename, 'w')
        for ptype, pdata in self.particle_data.items():
            ptype_grp = f.create_group(ptype)
            data = self.all_array_data[ptype]
            self._set_constants(pdata, ptype_grp)
            self._set_properties(pdata, ptype_grp, data)
        self._set_attributes(f)
        f.close()

    def _load(self, fname):
        if has_h5py():
            import h5py
        else:
            msg = "Install python-h5py to load this file"
            raise(msg)
        f = h5py.File(fname, 'r')
        solver_data = {}
        particles = {}

        for name, value in f.attrs.items():
            solver_data[str(name)] = value

        for name, prop_array in f.items():
            output_array = []
            constants = {}
            const_grp = prop_array['constants']
            for const_name, const_data in const_grp.items():
                constants[str(const_name)] = numpy.array(const_data)
            array = ParticleArray(str(name), constants=constants)
            for pname, h5obj in prop_array.items():
                if type(h5obj) is h5py.Group:
                    continue
                prop_name = str(h5obj.attrs['name'])
                type_ = h5obj.attrs['type']
                default = h5obj.attrs['default']
                if h5obj.attrs['datapresent']:
                    output_array.append(str(pname))
                    array.add_property(
                            prop_name, type_, default, numpy.array(h5obj))
                else:
                    array.add_property(prop_name, type_)
            array.set_output_arrays(output_array)
            particles[str(name)] = array
        f.close()
        ret = {}
        ret["arrays"] = particles
        ret["solver_data"] = solver_data
        return ret


    def _set_constants(self, pdata, ptype_grp):
        pconstants = pdata['constants']
        constGroup = ptype_grp.create_group('constants')
        for constName, constArray in pconstants.items():
            constGroup.create_dataset(constName, data=constArray)

    def _set_properties(self, pdata, ptype_grp, data):
        for propname, attributes in pdata['properties'].items():
            if propname in data:
                array = data[propname]
                prop = ptype_grp.create_dataset(
                        propname, data=array)
                prop.attrs['datapresent'] = True
            else:
                prop = ptype_grp.create_dataset(propname, (0,))
                prop.attrs['datapresent'] = False

            for attname, value in attributes.items():
                if value is None:
                    value = 'None'
                prop.attrs[attname] = value

    def _set_attributes(self, f):
        for name, data in self.solver_data.items():
            f.attrs[name] = data


def load(fname, file_format = '', directory = '.'):
    """
    Load the output data. It can handle full file path, single file name
    with directory_path. If only one format is saved it can determine the file 
    format also.

    
    Parameters
    ----------
    fname: str
        Name of the file or full path

    file_format: str
        File format to load. Provide no argument for auto searching .If 
        mutiple format are present if throws an error.

    directory: str
        Directory to search for fname

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

    if fname.endswith('npz') or file_format is 'npz':
        output = NumpyOutput();
        if not fname.endswith('npz'):
            fname = fname + '.' + 'npz'
    if fname.endswith('hdf5') or file_format is 'hdf5':
        output = HDFOutput();
        if not fname.endswith('hdf5'):
            fname = fname + '.' + 'hdf5'
    if os.path.isfile(fname):
        return output.load(fname)
  
    filename = os.path.join(directory, fname)
    if os.path.isfile(filename):
        return output.load(filename)
    flist = [fname + '.' + i for i in output_formats]
    flist = [i for i in flist if os.path.isfile(i)]
    flist2 = [filename + '.' + i for i in  output_formats]
    flist2 = [i for i in flist2 if os.path.isfile(i)]
    flist =list(set(flist+flist2))
    if len(flist) is 1:
        fname = flist[0]
        return load(fname)
    else:
        msg = "Too many files or no file present (Try giving the file format)"
        raise RuntimeError(msg)
    

def dump(filename, particles, solver_data, detailed_output=False,
        only_real=True, mpi_comm=None, compress=False, file_format='hdf5'):

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
        Specify if the npz file is to be compressed or not.

    file_format: str ('npz', 'hdf5']
        The format of file to be saved.
    If `mpi_comm` is not passed or is set to None the local particles alone
    are dumped, otherwise only rank 0 dumps the output.

    """
 
    if file_format == 'hdf5' and has_h5py():
        output = HDFOutput(detailed_output, only_real, mpi_comm, compress)
    else:
        output = NumpyOutput(detailed_output, only_real, mpi_comm, compress)
        file_format = 'npz'
    if not filename.endswith(file_format):
        filename = filename + '.' +  file_format
    output.dump(filename, particles, solver_data)
   
