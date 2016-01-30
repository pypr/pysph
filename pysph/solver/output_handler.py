"""
An interface to output the data in various format
"""

import numpy
import os

from pysph.base.particle_array import ParticleArray
from pysph.base.utils import get_particles_info, get_particle_array
from pysph import has_h5py, has_pyvisfile, has_tvtk

output_formats = ('hdf5', 'npz', 'tvtk')


class OutputHandler(object):
    """ Class that handles output for simulation """
    def __init__(self, output_directory='', output_format='hdf5',
                 detailed_output=False, only_real=True, compress=False,
                 mpi_comm=None):
        self.compress = compress
        self.detailed_output = detailed_output
        self.only_real = only_real
        self.mpi_comm = mpi_comm
        self.output_directory = output_directory
        self.get_output_format(output_format)

    def set_output_directory(self, path):
        self.output_directory = path

    def get_output_format(self, output_format):
        self._dump_function = None
        self._load_function = None

        if output_format == 'hdf5' and has_h5py():
            global h5py
            import h5py
            self._dump_function = self._dump_hdf5
            self._load_function = self._load_hdf5
            self.file_format = output_format

        if output_format == 'tvtk' and has_tvtk():
            global tvtk
            import tvtk
            self._dump_function = self._dump_tvtk
            self._load_function = self._load_tvtk
            self.file_format = output_format

        if self._dump_function is None:
            self._dump_function = self._dump_numpy
            self._load_function = self._load_numpy
            self.save_func = numpy.savez
            self.file_format = 'npz'
            if self.compress:
                self.save_func = numpy.savez_compressed

    def dump(self, fname, particles, solver_data):
        filename = os.path.join(self.output_directory, fname)
        filename = filename + '.' + self.file_format
        self.particle_data = dict(get_particles_info(particles))
        self.all_array_data = {}
        for array in particles:
            self.all_array_data[array.name] = array.get_property_arrays(
                all=self.detailed_output,
                only_real=self.only_real)
        mpi_comm = self.mpi_comm
        if mpi_comm is not None:
            self.all_array_data = self._gather_array_data(
                    self.all_array_data, mpi_comm)
        self.solver_data = solver_data
        self._dump_function(filename)

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

    def load(self, fname):
        if fname.endswith('npz'):
            return self._load_numpy(fname)
        if fname.endswith('hdf5'):
            return self._load_hdf5(fname)
        fname = self.get_file(fname)
        return self._load_function(fname)

    def get_file(self, fname):
        available_files = [i for i in os.listdir(self.output_directory)
                           if i.startswith(fname) and
                           i.endswith(output_formats)]
        if len(available_files) > 1:
            available_files = [i for i in available_files
                               if i.endswith(file_format)]

        if len(available_files) == 0:
            msg = "Default format file not present"
            raise RuntimeError(msg)

        elif len(available_files) > 1:
            msg = "Too many files for same format present"
            raise RuntimeError(msg)

        else:
            filename = available_files[0]
        filename = os.path.join(self.output_directory, filename)
        return filename

    def _dump_numpy(self, filename):
        output_data = {"particles": self.particle_data,
                       "solver_data": self.solver_data}
        for name, arrays in self.all_array_data.items():
            self.particle_data[name]["arrays"] = arrays
        self.save_func(filename, version=2, **output_data)

    def _dump_hdf5(self, filename):
        f = h5py.File(filename, 'w')
        for ptype, pdata in self.particle_data.items():
            ptype_grp = f.create_group(ptype)
            data = self.all_array_data[ptype]
            self.set_constants(pdata, ptype_grp)
            self.set_properties(pdata, ptype_grp, data)
        self.set_attributes(f)
        f.close()

    def set_constants(self, pdata, ptype_grp):
        pconstants = pdata['constants']
        constGroup = ptype_grp.create_group('constants')
        for constName, constArray in pconstants.items():
            constGroup.create_dataset(constName, data=constArray)

    def set_properties(self, pdata, ptype_grp, data):
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

    def set_attributes(self, f):
        for name, data in self.solver_data.items():
            f.attrs[name] = data

    def _load_numpy(self, fname):
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

    def _load_hdf5(self, fname):
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
                output_array.reverse()
            array.set_output_arrays(output_array)
            particles[str(name)] = array
        f.close()
        ret = {}
        ret["arrays"] = particles
        ret["solver_data"] = solver_data
        return ret

    def _load_tvtk(self, filename):
        pass

    def _dump_tvtk(self, filename):
        pass


def load(fname):
    output_handler = OutputHandler()
    return output_handler.load(fname)
