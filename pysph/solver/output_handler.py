"""
An interface to output the data in various format
cylo pyvisfile

"""

import numpy
import os

def has_h5py():
    try:
        import h5py
        HAS_H5PY = True
    except ImportError:
        HAS_H5PY = False
    return HAS_H5PY

from pysph.base.particle_array import ParticleArray
from pysph.base.utils import get_particles_info


class OutputHandler(object):
    """ Class that handles output for simulation """
    def __init__(self, output_directory, prefered_format='npz',
                 detailed_output=False, only_real=True, compress=False,
                 mpi_comm=None):
        self.compress = compress
        self.detailed_output = detailed_output
        self.only_real = only_real
        self.mpi_comm = mpi_comm
        self.output_directory = output_directory
        self.get_prefered_format(prefered_format)
        self.filename = ''

    def set_output_directory(self, path):
        self.output_directory = path

    def get_prefered_format(self, prefered_format):
        self._dump_function = None
        self._load_function = None

        if prefered_format == 'hdf5' and has_h5py():
            self._dump_function = self._dump_hdf5
            self._load_function = self._load_hdf5
            self.file_format = prefered_format

        if self._dump_function is None:
            self._dump_function = self._dump_numpy
            self._load_function = self._load_numpy
            self.save_func = numpy.savez
            self.file_format = 'npz'
            if self.compress:
                self.save_func = numpy.savez_compressed

    def dump(self, fname, particles, solver_data):
        self.filename = os.path.join(self.output_directory, fname)
        self.filename = self.filename + '.'  + self.file_format
        self._dump_function(particles, solver_data)

    def load(self,fname):
        if fname.endswith('npz'):
            return self._load_numpy(fname)
        if fname.endswith('hdf5'):
            return self._load_hdf5(fname)
        return self._load_function(fname+'.'+self.file_format)

    def _dump_numpy(self, particles, solver_data):
        particle_data = dict(get_particles_info(particles))
        output_data = {"particles": particle_data,
                       "solver_data": solver_data}
        all_array_data = {}
        for array in particles:
            all_array_data[array.name] = array.get_property_arrays(
                all=self.detailed_output,
                only_real=self.only_real)
        mpi_comm = self.mpi_comm
        if mpi_comm is not None:
            all_array_data = _gather_array_data(all_array_data, mpi_comm)
        for name, arrays in all_array_data.items():
            particle_data[name]["arrays"] = arrays
        self.save_func(self.filename, version=2, **output_data)

    def _dump_hdf5(self, particles, solver_data):
        all_array_data = {}
        particle_data = dict(get_particles_info(particles))
        for array in particles:
            all_array_data[array.name] = array.get_property_arrays(
                all=self.detailed_output,
                only_real=self.only_real)      
        mpi_comm = self.mpi_comm
        if mpi_comm is not None:
            all_array_data = _gather_array_data(all_array_data, mpi_comm)

        with h5py.File(self.filename, 'w') as f:
            for ptype_, pdata in particle_data.items():
                ptype = f.create_group(ptype_)
                data = all_array_data[ptype_]
                pconstants = pdata['constants']
                constGroup = ptype.create_group('constants')
                for constName, constArray in pconstants.items():
                    constGroup.create_dataset(constName, data=constArray)

                for propname, attributes  in pdata['properties'].items():
                    if propname in data:
                        array = data[propname]
                        prop = ptype.create_dataset(propname, data=array)
                        prop.attrs['datapresent'] = True
                    else:
                        prop = ptype.create_dataset(propname,(0,))
                        prop.attrs['datapresent'] = False
                        

                    for attname, value in attributes.items():
                        if value is None:
                            value = 'None'
                        prop.attrs[attname] = value
            for name, data in solver_data.items():
                f.attrs[name] = data

    def _load_numpy(self,fname):
        def _get_dict_from_arrays(arrays):
            arrays.shape = (1,)
            return arrays[0]
        data = numpy.load(fname)
        
        if not 'version' in data.files:
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

        return  ret["arrays"],ret["solver_data"]

    def _load_hdf5(self,fname):
        f = h5py.File(fname,'r')
        solver_data = {}
        particles = {}
        for name,value in f.attrs.items():
            solver_data[name] = value
        for name,prop_array in f.items():
            output_array = []
            constants = {}
            const_grp = prop_array['constants']
            for const_name,const_data in const_grp.items():
                constants[const_name] = numpy.array(const_data)
            arrays =  ParticleArray(name, constants = constants)
            for pname,h5obj in prop_array.items():
                if type(h5obj) is h5py.Group:
                    continue
                prop_name = h5obj.attrs['name']
                type_ = h5obj.attrs['type']
                default = h5obj.attrs['default']
                if h5obj.attrs['datapresent'] == True:
                    output_array.append(pname)
                    arrays.add_property(
                            prop_name,type_,default,numpy.array(h5obj))
                else:
                    arrays.add_property(prop_name,type_)
            arrays.set_output_arrays(output_array)
            particles[name] = arrays
        return particles,solver_data

