#cython: embedsignature=True
"""
A `ParticleArray` represents a collection of particles.

"""

# logging imports
import logging
logger = logging.getLogger()

# numpy imports
cimport numpy
import numpy

# PyZoltan imports
from cyarray.carray cimport *

# Local imports
from cpython cimport PyObject
from cpython cimport *
from cython cimport *

from compyle.config import get_config
from compyle.array import Array, get_backend, to_device
from pysph.base.device_helper import DeviceHelper

# Maximum value of an unsigned int
cdef extern from "limits.h":
    cdef unsigned int UINT_MAX

_UINT_MAX = UINT_MAX

# Declares various tags for particles, and functions to check them.

# Note that these tags are the ones set in the 'tag' property of the
# particles, in a particle array. To define additional discrete
# properties, one can add another integer property to the particles in
# the particle array while creating them.

# These tags could be considered as 'system tags' used internally to
# distinguish among different kinds of particles. If more tags are
# needed for a particular application, add them as mentioned above.

# The is_* functions defined below are to be used in Python for tests
# etc. Cython modules can directly use the enum name.

cpdef bint is_local(int tag):
    return tag == Local

cpdef bint is_remote(int tag):
    return tag == Remote

cpdef bint is_ghost(int tag):
    return tag == Ghost

cpdef int get_local_tag():
    return Local

cpdef int get_remote_tag():
    return Remote

cpdef int get_ghost_tag():
    return Ghost

cdef class ParticleArray:
    """
    Class to represent a collection of particles.

    Attributes
    ----------
    name : str
        name of this particle array.
    properties : dict
        dictionary of {prop_name:carray}.
    constants : dict
        dictionary of {const_name: carray}

    Examples
    --------

    There are many ways to create a ParticleArray::

        >>> p = ParticleArray(name='fluid', x=[1.,2., 3., 4.])
        >>> p.name
        'fluid'
        >>> p.x, p.tag, p.pid, p.gid

    For a full specification of properties with their type etc.::

        >>> p = ParticleArray(name='fluid',
        ...                   x=dict(data=[1,2], type='int', default=1))
        >>> p.get_carray('x').get_c_type()
        'int'

    The default value is what is used to set the default value when a new
    particle is added and the arrays auto-resized.

    To create constants that are not resized with added/removed particles::

        >>> p = ParticleArray(name='f', x=[1,2], constants={'c':[0.,0.,0.]})

    """
    ######################################################################
    # `object` interface
    ######################################################################
    def __init__(self, str name='', default_particle_tag=Local,
                 constants=None, backend=None, **props):
        """Constructor

        Parameters
        ----------

        name : str
            name of this particle array.

        default_particle_tag : int
            one of `Local`, `Remote` or `Ghost`

        constants : dict
            dictionary of constant arrays for the entire particle array.  These
            must be arrays and are not resized when particles are added or
            removed.  These are stored as CArrays internally.

        props :
            any additional keyword arguments are taken to be properties, one
            for each property.

        """
        self.backend = get_backend(backend)
        self.time = 0.0
        self.name = name

        self.properties = {}
        self.default_values = {'tag':default_particle_tag}
        self.stride = {}

        self._initialize(**props)

        self.constants = {}
        if constants is not None:
            for const, data in constants.items():
                self.add_constant(name=const, data=data)

        # default lb_props are all the arrays
        self.lb_props = None

        # list of output property arrays
        self.output_property_arrays = []

        if self.backend is not 'cython':
            h = DeviceHelper(self, backend=self.backend)
            self.set_device_helper(h)
        else:
            self.gpu = None

    def __getattr__(self, name):
        """Convenience, to access particle property arrays as an attribute

        A numpy array is returned. Look at the get() functions documentation for
        care on using this numpy array.

        """
        if name in self.properties:
            return self._get_real_particle_prop(name)
        elif name in self.constants:
            return self.constants[name].get_npy_array()
        else:
            msg = "ParticleArray %s has no property/constant %s."\
                    %(self.name, name)
            raise AttributeError(msg)

    def __setattr__(self, name, value):
        """Convenience, to set particle property arrays as an attribute """
        self.set(**{name:value})

    def __reduce__(self):
        """Implemented to facilitate pickling of extension types """
        d = {}
        d['name'] = self.name
        props = {}
        default_values = {}

        for prop, arr in self.properties.items():
            pinfo = {}
            pinfo['name'] = prop
            pinfo['type'] = arr.get_c_type()
            pinfo['data'] = arr.get_npy_array()
            pinfo['default'] = self.default_values[prop]
            pinfo['stride'] = self.stride.get(prop, 1)
            props[prop] = pinfo

        d['properties'] = props

        props = {}
        for prop, arr in self.constants.items():
            pinfo = dict(name=prop, data=arr)
            props[prop] = pinfo

        d['constants'] = props

        return (ParticleArray, (), d)

    def __setstate__(self, d):
        """ Load the particle array object from the saved dictionary """
        self.properties = {}
        self.constants = {}
        self.property_arrays = []
        self.default_values = {}
        self.num_real_particles = 0

        self.name = d['name']
        props = d['properties']
        for prop in props:
            self.add_property(**props[prop])

        consts = d['constants']
        for prop in consts:
            self.add_constant(**consts[prop])

        self.num_real_particles = numpy.sum(props['tag']['data']==Local)

    def _initialize(self, **props):
        """Initialize the particle array with the given properties.

        Parameters
        ----------

        props : dict
            dictionary containing various property arrays. All these
            arrays are expected to be numpy arrays or objects that can
            be converted to numpy arrays.

        Notes
        -----

         - This will clear any existing data.
         - As a rule internal arrays will always be either long or double
           arrays.

        """
        cdef int nprop, nparticles
        cdef bint tag_present = False
        cdef numpy.ndarray a, arr, npyarr
        cdef IntArray tagarr
        cdef str name

        self.clear()

        nprop = len(props)

        if nprop == 0:
            return

        # Iterate over all props to find the maximum number of values passed.
        nv = 0
        _props = {}
        for name, prop in props.items():
            if isinstance(prop, dict):
                if 'data' in prop:
                    d = prop['data']
                    stride = prop.get('stride', 1)
                    if d is not None:
                        d = numpy.ravel(prop['data'])
                        prop['data'] = d
                        nv = max(nv, len(d)//stride)
                _props[name] = prop
            elif prop is not None:
                d = numpy.ravel(prop)
                _props[name] = d
                nv = max(nv, len(d))
            else:
                _props[name] = prop
        props.update(_props)

        # add the properties
        for name, prop in props.items():
            if isinstance(prop, dict):
                prop_info = prop
                prop_info['name'] = name
                data = prop.get('data', None)
                if data is not None:
                    if nv > 1 and len(prop['data']) == 1:
                        prop_info['data'] = numpy.ones(nv)*data
            else:
                if nv > 1 and len(prop) == 1:
                    prop = numpy.ones(nv)*prop
                prop_info = dict(name=name, data=prop)
            self.add_property(**prop_info)

        self.align_particles()

    ######################################################################
    # `Public` interface
    ######################################################################
    def update_backend(self, backend=None):
        self.backend = get_backend(backend)

    def set_output_arrays(self, list props):
        """Set the list of output arrays for this ParticleArray

        Parameters
        ----------

        props : list
            The list of property arrays

        Notes
        -----

        In PySPH, the solver obtains the list of property arrays to
        output by calling the `ParticleArray.get_property_arrays`
        method. If detailed output is not requested, the
        `output_property_arrays` attribute is used to determine the
        arrays that will be written to file

        """
        # first check if the arrays are valid and raise a warning
        for prop in props:
            self._check_property(prop)

        self.output_property_arrays = props

    def add_output_arrays(self, list props):
        """Append props to the existing list of output arrays

        Parameters
        ----------

        props : list
            The additional list of property arrays to save

        """
        # first check if the arrays are valid and raise a warning
        for prop in props:
            self._check_property(prop)

        # add to the existing list
        self.output_property_arrays.extend(props)
        self.output_property_arrays = list( set(self.output_property_arrays) )

    def get_property_arrays(self, all=True, only_real=True):
        """Return a dictionary of arrays held by the `ParticleArray` container.

        This does not include the constants.

        Parameters
        ----------

        all : bint
            Flag to select all arrays

        only_real : bint
            Flag to select Local/Remote particles

        Notes
        -----

        The dictionary returned is keyed on the property name and the
        value is the NumPy array representing the data. If `all` is
        set to False, the list of arrays is determined by the
        `output_property_arrays` data attribute.

        """
        # the arrays dictionary to be returned
        arrays = {}

        # the list of properties
        props = self.output_property_arrays
        if all or len(props) == 0:
            props = list(self.properties.keys())

        # number of particles
        num_particles = self.get_number_of_particles(only_real)
        if self.gpu is not None and self.backend is not 'cython':
            self.gpu.pull(*props)

        # add the property arrays
        for prop in props:
            stride = self.stride.get(prop, 1)
            prop_array = self.properties[prop].get_npy_array()
            arrays[prop] = prop_array[:num_particles*stride]

        return arrays

    cpdef set_num_real_particles(self, long value):
        self.num_real_particles = value

    cpdef has_array(self, str arr_name):
        """Returns true if the array arr_name is present """
        return self.properties.has_key(arr_name)

    def clear(self):
        """Clear all data held by this array """
        self.properties = {'tag':IntArray(0), 'pid':IntArray(0), 'gid':UIntArray(0)}
        tag_def_values = self.default_values['tag']
        self.default_values.clear()
        self.default_values = {'tag':tag_def_values, 'pid':0, 'gid':_UINT_MAX}

    cpdef set_time(self, double time):
        self.time = time

    cpdef double get_time(self):
        return self.time

    cpdef set_name(self, str name):
        self.name = name

    def set_lb_props(self, list lb_props):
        self.lb_props = lb_props

    cpdef get_lb_props(self):
        """Return the properties that are to be load balanced.  If none are
        explicitly set by the user, return all of the properties.
        """
        if self.lb_props is None:
            return list(self.properties.keys())
        else:
            return self.lb_props

    cpdef int get_number_of_particles(self, bint real=False):
        """ Return the number of particles """
        if self.gpu is not None and self.backend is not 'cython':
            return self.gpu.get_number_of_particles()
        if real:
            return self.num_real_particles
        else:
            if 'tag' in self.properties:
                return self.properties['tag'].length
            elif len(self.properties) > 0:
                prop = list(self.properties.keys())[0]
                stride = self.stride.get(prop, 1)
                return self.properties[prop].length//stride
            else:
                return 0

    cpdef remove_particles(self, indices, align=True):
        """ Remove particles whose indices are given in index_list.

        We repeatedly interchange the values of the last element and values from
        the index_list and reduce the size of the array by one. This is done for
        every property that is being maintained.

        Parameters
        ----------

        indices : array
            an array of indices, this array can be a list, numpy array
            or a LongArray.

        Notes
        -----

        Pseudo-code for the implementation::

            if index_list.length > number of particles
                raise ValueError

            sorted_indices <- index_list sorted in ascending order.

            for every every array in property_array
                array.remove(sorted_indices)

        """
        if self.gpu is not None and self.backend is not 'cython':
            if type(indices) != Array:
                if isinstance(indices, BaseArray):
                    indices = indices.get_npy_array()
                else:
                    indices = numpy.asarray(indices)
                indices = to_device(indices.astype(numpy.uint32),
                                    backend=self.backend)
            return self.gpu.remove_particles(indices, align=align)

        cdef BaseArray index_list
        if isinstance(indices, BaseArray):
            index_list = indices
        else:
            indices = numpy.asarray(indices)
            index_list = LongArray(indices.size)
            index_list.set_data(indices)

        cdef str msg
        cdef numpy.ndarray sorted_indices
        cdef BaseArray prop_array
        cdef int num_arrays, i
        cdef list property_arrays

        if index_list.length > self.get_number_of_particles():
            msg = 'Number of particles to be removed is greater than'
            msg += 'number of particles in array'
            raise ValueError(msg)

        sorted_indices = numpy.sort(index_list.get_npy_array())
        num_arrays = len(self.properties)

        for name, prop_array in self.properties.items():
            stride = self.stride.get(name, 1)
            prop_array.remove(sorted_indices, 1, stride)

        if index_list.length > 0 and align:
            self.align_particles()

    cpdef remove_tagged_particles(self, int tag, bint align=True):
        """ Remove particles that have the given tag.

        Parameters
        ----------

        tag : int
            the type of particles that need to be removed.

        """
        if self.gpu is not None and self.backend is not 'cython':
            return self.gpu.remove_tagged_particles(tag, align=align)
        cdef LongArray indices = LongArray()
        cdef IntArray tag_array = self.properties['tag']
        cdef int *tagarrptr = tag_array.get_data_ptr()
        cdef int i

        # find the indices of the particles to be removed.
        for i in range(tag_array.length):
            if tagarrptr[i] == tag:
                indices.append(i)

        # remove the particles.
        self.remove_particles(indices, align=align)

    def add_particles(self, align=True, **particle_props):
        """
        Add particles in particle_array to self.

        Parameters
        ----------

        particle_props : dict
            a dictionary containing numpy arrays for various particle
            properties.

        Notes
        -----

         - all properties should have same length arrays.
         - all properties should already be present in this particles array.
           if new properties are seen, an exception will be raised.

        """
        cdef str prop
        cdef BaseArray arr

        if len(particle_props) == 0:
            return 0

        if self.gpu is not None and self.backend is not 'cython':
            gpu_particle_props = {}
            for prop, ary in particle_props.items():
                if prop in self.gpu.properties:
                    dtype = self.gpu.get_device_array(prop).dtype
                else:
                    dtype = self.default_values[prop]
                gpu_particle_props[prop] = to_device(
                        numpy.array(ary, dtype=dtype),
                        backend=self.backend)
            return self.gpu.add_particles(align=align, **gpu_particle_props)

        cdef int num_extra_particles, old_num_particles, new_num_particles
        cdef numpy.ndarray s_arr, nparr

        # check if the input properties are valid.
        prop = ''
        for prop in particle_props:
            self._check_property(prop)

        if len(prop) == 0:
            num_extra_particles = 0
        else:
            stride = self.stride.get(prop, 1)
            num_extra_particles = len(particle_props[prop])//stride
        old_num_particles = self.get_number_of_particles()
        new_num_particles = num_extra_particles + old_num_particles

        for prop in self.properties:
            arr = <BaseArray>PyDict_GetItem(self.properties, prop)
            stride = self.stride.get(prop, 1)

            if PyDict_Contains(particle_props, prop)== 1:
                d_type = arr.get_npy_array().dtype
                s_arr = numpy.asarray(particle_props[prop], dtype=d_type)
                arr.extend(s_arr)
            else:
                arr.resize(new_num_particles*stride)
                # set the properties of the new particles to the default ones.
                nparr = arr.get_npy_array()
                nparr[old_num_particles*stride:] = self.default_values[prop]

        if num_extra_particles > 0 and align:
            # make sure particles are aligned properly.
            self.align_particles()

        return 0

    cpdef int append_parray(self, ParticleArray parray, bint align=True,
            bint update_constants=False) except -1:
        """ Add particles from a particle array

        properties that are not there in self will be added
        """
        if parray.get_number_of_particles() == 0:
            return 0

        if self.gpu is not None and self.backend is not 'cython':
            self.gpu.append_parray(parray, align=align,
                                   update_constants=update_constants)
            return 0

        cdef int num_extra_particles = parray.get_number_of_particles()
        cdef int old_num_particles = self.get_number_of_particles()
        cdef int new_num_particles = num_extra_particles + old_num_particles
        cdef str prop_name
        cdef BaseArray arr, source, dest
        cdef numpy.ndarray nparr_dest, nparr_source

        # extend current arrays by the required number of particles
        self.extend(num_extra_particles)

        for prop_name in parray.properties:
            if PyDict_Contains(self.properties, prop_name):
                arr = <BaseArray>PyDict_GetItem(self.properties, prop_name)
            else:
                arr = None

            stride = self.stride.get(prop_name, 1)
            if arr is not None:
                source = <BaseArray>PyDict_GetItem(parray.properties, prop_name)
                nparr_source = source.get_npy_array()
                nparr_dest = arr.get_npy_array()
                nparr_dest[old_num_particles*stride:] = nparr_source
            else:
                # meaning this property is not there in self.
                stride = parray.stride.get(prop_name, 1)
                self.add_property(name=prop_name,
                                  type=parray.properties[prop_name].get_c_type(),
                                  default=parray.default_values[prop_name],
                                  stride=stride
                                  )
                # now add the values to the end of the created array
                dest = <BaseArray>PyDict_GetItem(self.properties, prop_name)
                nparr_dest = dest.get_npy_array()
                source = <BaseArray>PyDict_GetItem(parray.properties, prop_name)
                nparr_source = source.get_npy_array()
                nparr_dest[old_num_particles*stride:] = nparr_source

        if update_constants:
            for const in parray.constants:
                self.constants.setdefault(const, parray.constants[const])

        if num_extra_particles > 0 and align:
            self.align_particles()

        return 0

    cpdef extend(self, int num_particles):
        """ Increase the total number of particles by the requested amount

        New particles are added at the end of the list, you will have to manually
        call align_particles later in order to update the number of particles.
        """
        if num_particles <= 0:
            return

        if self.gpu is not None and self.backend is not 'cython':
            self.gpu.extend(num_particles)
            return 0

        cdef int old_size = self.get_number_of_particles()
        cdef int new_size = old_size + num_particles
        cdef BaseArray arr
        cdef numpy.ndarray nparr
        cdef int stride

        for key in self.properties:
            stride = self.stride.get(key, 1)
            arr = self.properties[key]
            arr.resize(new_size*stride)
            nparr = arr.get_npy_array()
            nparr[old_size*stride:] = self.default_values[key]

    cdef numpy.ndarray _get_real_particle_prop(self, str prop_name):
        """ get the npy array of property corresponding to only real particles

        No checks are performed. Only call this after making sure that the
        property required already exists.
        """
        cdef BaseArray prop_array
        prop_array = self.properties.get(prop_name)
        stride = self.stride.get(prop_name, 1)
        if prop_array is not None:
            return prop_array.get_npy_array()[:self.num_real_particles*stride]
        else:
            return None

    def get(self, *args, only_real_particles=True):
        """ Return the numpy array/constant for the  property names in
        the arguments.

        Parameters
        ----------

        only_real_particles : bool
            indicates if properties of only real particles need to be
            returned or all particles to be returned. By default only
            real particles will be returned.
         args : additional args
            a list of property names.

        Notes
        -----

        The returned numpy array does **NOT** own its data. Other
        operations may be performed.

        Returns
        -------

        Numpy array.

        """
        cdef int nargs = len(args)
        cdef list result = []
        cdef str arg
        cdef int i, stride
        cdef BaseArray arg_array

        if nargs == 0:
            return

        if only_real_particles == True:
            for i in range(nargs):
                arg = args[i]
                self._check_property(arg)
                stride = self.stride.get(arg, 1)
                if arg in self.properties:
                    arg_array = self.properties[arg]
                    result.append(
                        arg_array.get_npy_array()[:self.num_real_particles*stride])

                else:
                    result.append(self.constants[arg].get_npy_array())
        else:
            for i in range(nargs):
                arg = args[i]
                self._check_property(arg)

                if arg in self.properties:
                    arg_array = self.properties[arg]
                    result.append(arg_array.get_npy_array())
                else:
                    result.append(self.constants[arg].get_npy_array())

        if nargs == 1:
            return result[0]
        else:
            return tuple(result)

    def set_device_helper(self, gpu):
        """Set the device helper to push/pull from a hardware accelerator.
        """
        self.gpu = gpu

    def set(self, **props):
        """ Set properties from numpy arrays like objects

        Parameters
        ----------

        props : dict
            a dictionary of properties containing the arrays to be set.

        Notes
        -----

         - the properties being set must already be present in the properties
           dict.
         - the size of the data should match the array already present.

        """
        cdef str prop
        cdef BaseArray prop_array
        cdef int nprops = len(props)
        cdef list prop_names = list(props.keys())
        cdef int i

        for i in range(nprops):
            prop = prop_names[i]
            self._check_property(prop)

        for prop in props:
            proparr = numpy.asarray(props[prop])
            if self.properties.has_key(prop):
                prop_array = self.properties[prop]
                prop_array.set_data(proparr)
            elif self.constants.has_key(prop):
                prop_array = self.constants[prop]
                prop_array.set_data(proparr)

            # if the tag property is being set, the alignment will have to be
            # changed.

    cpdef BaseArray get_carray(self, str prop):
        """Return the c-array for the property or constant.
        """
        if PyDict_Contains(self.properties, prop) == 1:
            return <BaseArray>PyDict_GetItem(self.properties, prop)
        elif PyDict_Contains(self.constants, prop) == 1:
            return <BaseArray>PyDict_GetItem(self.constants, prop)
        else:
            raise KeyError(
                'Property/constant "%s" not present in particle array.' % prop
            )

    cpdef add_constant(self, str name, data):
        """Add a constant property to the particle array.

        A constant property is an array but has a fixed size in that it is
        never resized as particles are added or removed.  These properties
        are always stored internally as CArrays.

        An example of where this is useful is if you need to calculate the
        center of mass of a solid body or the net force on the body.

        Parameters
        ----------

        name : str
            name of the constant
        data : array-like
            the value for the data.
        """
        if name in self.constants:
            raise RuntimeError('Constant called "%s" already exists.'%name)
        if name in self.properties:
            raise RuntimeError('Property called "%s" already exists.'%name)

        array_data = numpy.ravel(data)
        self.constants[name] = self._create_c_array_from_npy_array(array_data)
        if self.gpu is not None:
            self.gpu.add_const(name, self.constants[name])

    cpdef add_property(self, str name, str type='double', default=None, data=None,
                       stride=1):
        """Add a new property to the particle array.

        If a `default` is not supplied 0 is assumed.  The stride is useful when
        many elements are needed per particle.  For example if stride is 3 then
        3 elements are allocated per particle.


        Parameters
        ----------

        name : str
            compulsory name of property.
        type : str
            specifying the data type of this property ('double', 'int'
            etc.)
        default : value
            specifying the default value of this property.
        data : ndarray
            specifying the data associated with each particle.
        stride : int
            the number of elements per particle.

        Notes
        -----

        If there are no particles currently in the particle array, and a
        new property with some particles is added, all the remaining
        properties will be resized to the size of the newly added array.

        If there are some particles in the particle array, and a new
        property is added without any particles, then this new property will
        be resized according to the current size.

        If there are some particles in the particle array and a new property
        is added with a different number of particles, then an error will be
        raised.

        Warning
        -------

            - it is best not to add properties with data when you already have
              particles in the particle array. Reason is that, the particles in
              the particle array will be stored so that the 'Real' particles are
              in the top of the list and later the dummy ones. The data in your
              property array should be matched to the particles appropriately.
              This may not always be possible when there are particles of
              different type in the particle array.
            - Properties without any values can be added anytime.
            - While initializing particle arrays only using the add_property
              function, you will have to call align_particles manually to make
              sure particles are aligned properly.

        """
        cdef str prop_name=name, data_type=type
        cdef bint array_size_proper = False

        if data is not None:
            try:
                len(data)
            except TypeError:
                data = numpy.ones(self.get_number_of_particles()*stride)*data
            else:
                data = numpy.ravel(data)

        # make sure the size of the supplied array is consistent.
        if (data is None or self.get_number_of_particles() == 0 or
            len(data) == 0):
            array_size_proper = True
        else:
            if (self.get_number_of_particles() == len(data)//stride) and \
               (len(data) % stride == 0):
                array_size_proper = True

        if array_size_proper == False:
            msg = 'Array sizes incompatible for property: %s' % name
            logger.error(msg)
            raise ValueError(msg)

        # setup the default values
        if default is None:
            if prop_name not in self.properties:
                default = 0
            else:
                default = self.default_values[prop_name]

        self.default_values[prop_name] = default
        if stride != 1:
            self.stride[name] = stride

        # array sizes are compatible, now resize the required arrays
        # appropriately and add.
        if self.get_number_of_particles() == 0:
            if data is None or len(data) == 0:
                # if a property with that name already exists, do not do
                # anything.
                if prop_name not in self.properties:
                    # just add the property with a zero array.
                    self.properties[prop_name] = self._create_carray(
                        data_type, 0, default)
            else:
                # new property has been added with some particles, while no
                # particles are currently present. First resize the current
                # properties to this new length, and then add this new
                # property.
                n_elem = len(data)//stride
                for prop in self.properties:
                    arr = self.properties[prop]
                    prop_stride = self.stride.get(prop, 1)
                    arr.resize(n_elem*prop_stride)
                    arr.get_npy_array()[:] = self.default_values[prop]
                if prop_name == 'tag':
                    arr = numpy.asarray(data)
                    self.num_real_particles = numpy.sum(arr==Local)
                else:
                    self.num_real_particles = n_elem

                if self.properties.has_key(prop_name):
                    # just add the particles to the already existing array.
                    d_type = self.properties[prop_name].get_npy_array().dtype
                    arr = numpy.asarray(data, dtype=d_type)
                    self.properties[prop_name].set_data(arr)
                else:
                    # now add the new property array
                    # if a type was specifed create that type of array
                    if data_type is None:
                        # get an array for this data
                        self.properties[prop_name] = (
                            self._create_c_array_from_npy_array(data))
                    else:
                        arr = self._create_carray(data_type, len(data), default)
                        np_arr = arr.get_npy_array()
                        np_arr[:] = data
                        self.properties[prop_name] = arr
        else:
            if data is None or len(data) == 0:
                # new property is added without any initial data, resize it to
                # current particle count.
                if not self.properties.has_key(prop_name):
                    arr = self._create_carray(
                        data_type, self.get_number_of_particles()*stride,
                        default
                    )
                    self.properties[prop_name] = arr
            else:
                if self.properties.has_key(prop_name):
                    d_type = self.properties[prop_name].get_npy_array().dtype
                    arr = numpy.asarray(data, dtype=d_type)
                    self.properties[prop_name].set_data(arr)
                else:
                    if data_type is None:
                        # just add the property array
                        self.properties[prop_name] = (
                            self._create_c_array_from_npy_array(data))
                    else:
                        arr = self._create_carray(data_type, len(data), default)
                        np_arr = arr.get_npy_array()
                        arr.get_npy_array()[:] = data
                        self.properties[prop_name] = arr

        if self.gpu is not None:
            self.gpu.add_prop(prop_name, self.properties[prop_name])
            if self.gpu.get_number_of_particles() == 0:
                self.gpu.push()

    ######################################################################
    # Non-public interface
    ######################################################################
    def _create_carray(self, str data_type, int size, default=0):
        """Create a carray of the requested type, and of requested size

        Parameters
        ----------

        data_type : str
            string representing the 'c' data type - eg. 'int' for
            integers.
        size : int
            the size of the requested array
        default : value
            the default value to initialize the array with.

        """
        cdef BaseArray arr
        if data_type == None:
            arr = DoubleArray(size)
        elif data_type == 'double':
            arr = DoubleArray(size)
        elif data_type == 'long':
            arr = LongArray(size)
        elif data_type == 'float':
            arr = FloatArray(size)
        elif data_type == 'int':
            arr = IntArray(size)
        elif data_type == 'unsigned int':
            arr = UIntArray(size)
        else:
            logger.error('Trying to create carray of unknown '
                   'datatype: %s' %data_type)

        if size > 0:
            arr.get_npy_array()[:] = default

        return arr

    cdef _check_property(self, str prop):
        """Check if a property is present or not """
        if (PyDict_Contains(self.properties, prop) or
            PyDict_Contains(self.constants, prop)):
            return
        else:
            raise AttributeError, 'property %s not present'%(prop)

    cdef object _create_c_array_from_npy_array(self, numpy.ndarray np_array):
        """Create and return  a carray array from the given numpy array.

        Notes
        -----

         - this function is used only when a C array needs to be
           created (in the initialize function).

        """
        cdef int np = np_array.size
        cdef object a
        if np_array.dtype == numpy.int32 or np_array.dtype == numpy.int64:
            a = LongArray(np)
            a.set_data(np_array)
        elif np_array.dtype == numpy.float32:
            a = FloatArray(np)
            a.set_data(np_array)
        elif np_array.dtype == numpy.double:
            a = DoubleArray(np)
            a.set_data(np_array)
        else:
            msg = 'unknown numpy data type passed %s'%(np_array.dtype)
            raise TypeError, msg

        return a

    cpdef int align_particles(self) except -1:
        """Moves all 'Local' particles to the beginning of the array

        This makes retrieving numpy slices of properties of 'Local'
        particles possible. This facility will be required frequently.

        Notes
        -----

        Pseudo-code::

            index_arr = LongArray(n)

            next_insert = 0
            for i from 0 to n
                p <- ith particle
                if p is Local
                    if i != next_insert
                        tmp = index_arr[next_insert]
                        index_arr[next_insert] = i
                        index_arr[i] = tmp
                        next_insert += 1
                    else
                        index_arr[i] = i
                        next_insert += 1
                else
                    index_arr[i] = i

             # we now have the new index assignment.
             # swap the required values as needed.
             for every property array:
                 for i from 0 to n:
                     if index_arr[i] != i:
                         tmp = prop[i]
                         prop[i] = prop[index_arr[i]]
                         prop[index_arr[i]] = tmp
        """
        if self.gpu is not None and self.backend is not 'cython':
            self.gpu.align_particles()
            return 0

        cdef size_t i, num_particles
        cdef size_t next_insert
        cdef int tmp
        cdef IntArray tag_arr
        cdef LongArray index_array
        cdef BaseArray arr
        cdef long num_real_particles = 0
        cdef long num_moves = 0

        next_insert = 0
        num_particles = self.get_number_of_particles()

        tag_arr = self.get_carray('tag')

        # malloc the new index array
        index_array = LongArray(num_particles)

        for i in range(num_particles):
            if tag_arr.data[i] == Local:
                num_real_particles += 1
                if i != next_insert:
                    tmp = index_array.data[next_insert]
                    index_array.data[next_insert] = i
                    index_array.data[i] = tmp
                    next_insert += 1
                    num_moves += 1
                else:
                    index_array.data[i] = i
                    next_insert += 1
            else:
                index_array.data[i] = i

        self.num_real_particles = num_real_particles
        # we now have the aligned indices. Rearrange the particles
        # accordingly.

        if num_moves > 0:
            for name, arr in self.properties.items():
                stride = self.stride.get(name, 1)
                arr.c_align_array(index_array, stride)

    cpdef ParticleArray empty_clone(self, props=None):
        """Creates an empty clone of the particle array
        """
        if self.gpu is not None and self.backend is not 'cython':
            return self.gpu.empty_clone(props=props)

        cdef ParticleArray result_array = ParticleArray()
        cdef list output_arrays
        cdef BaseArray src_arr
        cdef int stride
        cdef str prop_type, prop_name

        if props is None:
            prop_names = self.properties
        else:
            prop_names = props

        for const in self.constants:
            result_array.add_constant(const, data=self.constants[const])

        for prop_name in prop_names:
            prop_type = self.properties[prop_name].get_c_type()
            prop_default = self.default_values[prop_name]
            stride = self.stride.get(prop_name, 1)
            result_array.add_property(name=prop_name,
                                      type=prop_type,
                                      default=prop_default,
                                      stride=stride)

        result_array.set_name(self.name)

        if props is None:
            output_arrays = list(self.output_property_arrays)
        else:
            output_arrays = list(
                set(props).intersection(
                    self.output_property_arrays
                )
            )

        result_array.set_output_arrays(output_arrays)
        return result_array

    cpdef ensure_properties(self, ParticleArray src, list props=None):
        """Ensure that the particle array has the same properties as the
        one given.

        Note that this does not check for any constants but only properties.

        If the optional props argument is passed it only checks for these.
        """
        prop_names = props if props else src.properties.keys()

        for prop_name in prop_names:
            if prop_name not in self.properties:
                prop_type = src.properties[prop_name].get_c_type()
                prop_default = src.default_values[prop_name]
                stride = src.stride.get(prop_name, 1)
                self.add_property(
                    name=prop_name, type=prop_type,
                    default=prop_default, stride=stride
                )

    cpdef ParticleArray extract_particles(self, indices,
            ParticleArray dest_array=None, bint align=True, list props=None):
        """Create new particle array for particles with indices in index_array

        Parameters
        ----------

        indices : list/array/LongArray
            indices of particles to be extracted (can be a LongArray or
            list/numpy array).

        dest_array: ParticleArray
            optional Particle array to populate. Note that this array should
            have the necessary properties. If none is passed a new particle
            array is created and returned.

        align: bool
            Specify if the destination particle array is to be aligned after
            particle extraction.

        props : list
            the list of properties to extract, if None all properties
            are extracted.

        Notes
        -----

        The algorithm is as follows:

             - create a new particle array with the required properties.
             - resize the new array to the desired length (index_array.length)
             - copy the properties from the existing array to the new array.

        """
        if self.gpu is not None and self.backend is not 'cython':
            if type(indices) != Array:
                indices = to_device(
                        numpy.array(indices, dtype=numpy.uint32),
                        backend=self.backend)
            return self.gpu.extract_particles(indices, dest_array=dest_array,
                                              align=align, props=props)

        if not dest_array:
            dest_array = self.empty_clone(props=props)

        cdef BaseArray index_array
        if isinstance(indices, BaseArray):
            index_array = indices
        else:
            indices = numpy.asarray(indices)
            index_array = LongArray(indices.size)
            index_array.set_data(indices)

        cdef list prop_names, output_arrays
        cdef BaseArray dst_prop_array, src_prop_array
        cdef str prop_type, prop
        cdef int stride, start_idx

        if props is None:
            prop_names = list(self.properties.keys())
        else:
            prop_names = props

        # now we have the result array setup.
        # resize it
        if index_array.length == 0:
            return dest_array

        start_idx = dest_array.get_number_of_particles()

        dest_array.extend(index_array.length)

        # copy the required indices for each property.
        for prop in prop_names:
            src_prop_array = self.get_carray(prop)
            dst_prop_array = dest_array.get_carray(prop)
            stride = self.stride.get(prop, 1)
            src_prop_array.copy_values(index_array, dst_prop_array,
                                       stride, stride*start_idx)

        if align:
            dest_array.align_particles()

        return dest_array

    cpdef set_tag(self, long tag_value, LongArray indices):
        """Set value of tag to tag_value for the particles in indices """
        cdef LongArray tag_array = self.get_carray('tag')
        cdef int i

        for i in range(indices.length):
            tag_array.data[indices.data[i]] = tag_value

    cpdef copy_properties(self, ParticleArray source, long start_index=-1,
                          long end_index=-1):
        """ Copy properties from source to self

        Parameters
        ----------

        source : ParticleArray
            the particle array from where to copy.
        start_index : long
            the first particle in self which maps to the 0th particle in
            source
        end_index : long
            the index of first particle from start_index that
            is not copied

        """
        cdef BaseArray src_array, dst_array
        for prop_name in source.properties:
            if prop_name in self.properties:
                src_array = source.get_carray(prop_name)
                dst_array = self.get_carray(prop_name)
                stride = self.stride.get(prop_name, 1)
                dst_array.copy_subset(src_array, start_index, end_index, stride)

    cpdef copy_over_properties(self, dict props):
        """ Copy the properties from one set to another.

        Parameters
        ----------

        props : dict
            A mapping between the properties to be copied.

        Examples
        --------

        To save the properties 'x' and 'y' to say 'x0' and 'y0'::

            >>> pa.copy_over_properties(props = {'x':'x0', 'y':'y0'}

        """
        cdef DoubleArray dst, src
        cdef str prop
        cdef int stride

        cdef long np = self.get_number_of_particles()
        cdef long i

        for prop in props:

            src = self.get_carray(prop)
            dst = self.get_carray(props[prop])
            stride = self.stride.get(prop, 1)

            for i in range(np*stride):
                dst.data[i] = src.data[i]

    cpdef set_to_zero(self, list props):

        cdef long np = self.get_number_of_particles()
        cdef long i
        cdef int stride

        cdef DoubleArray prop_arr
        cdef str prop

        for prop in props:
            prop_arr = self.get_carray(prop)
            stride = self.stride.get(prop, 1)

            for i in range(np*stride):
                prop_arr.data[i] = 0.0

    cpdef set_pid(self, int pid):
        """Set the processor id for all particles """
        cdef IntArray pid_arr = self.properties['pid']
        cdef long a

        for a in range(pid_arr.length):
            pid_arr.data[a] = pid

    cpdef remove_property(self, str prop_name):
        """Removes property prop_name from the particle array """

        if self.properties.has_key(prop_name):
            self.properties.pop(prop_name)
            self.default_values.pop(prop_name)
        if prop_name in self.output_property_arrays:
            self.output_property_arrays.remove(prop_name)
        if self.gpu is not None:
            return self.gpu.remove_prop(prop_name)

    def update_min_max(self, props=None):
        """Update the min,max values of all properties """
        if self.gpu is not None and self.backend is not 'cython':
            backend = self.gpu
        else:
            backend = self

        if props:
            for prop in props:
                array = backend.properties[prop]
                array.update_min_max()
        else:
            for array in backend.properties.values():
                array.update_min_max()

    cpdef resize(self, long size):
        """Resize all arrays to the new size.  Note that this does not
        update the number of particles, as this just resizes the internal arrays.
        To do that, you need to call `align_particles`.

        """
        if self.gpu is not None and self.backend is not 'cython':
            return self.gpu.resize(size)

        for prop, array in self.properties.items():
            stride = self.stride.get(prop, 1)
            array.resize(size*stride)


# End of ParticleArray class
##############################################################################
