cimport numpy as np

from pyzoltan.core.carray cimport BaseArray, UIntArray, IntArray, LongArray

# ParticleTag
# Declares various tags for particles, and functions to check them.

# Note that these tags are the ones set in the 'tag' property of the
# particles, in a particle array. To define additional discrete properties,
# one can add another integer property to the particles in the particle array
# while creating them.

# These tags could be considered as 'system tags' used internally to
# distinguish among different kinds of particles. If more tags are needed for
# a particular application, add them as mentioned above.

# The is_* functions defined below are to be used in Python for tests
# etc. Cython modules can directly use the enum name.

cdef enum ParticleTag:
    Local = 0
    Remote
    Ghost

cpdef bint is_local(int tag)
cpdef bint is_remote(int tag)
cpdef bint is_ghost(int tag)

cpdef int get_local_tag()
cpdef int get_remote_tag()
cpdef int get_ghost_tag()

cdef class ParticleArray:
    """
    Maintains various properties for particles.
    """
    # dictionary to hold the properties held per particle
    cdef public dict properties
    cdef public list property_arrays

    # list of output property arrays
    cdef public list output_property_arrays

    # dictionary to hold the constants for all the particles
    cdef public dict constants

    # default value associated with each property
    cdef public dict default_values

    # name associated with this particle array
    cdef public str name

    # indicates if coordinates of particles has changed.
    cdef public bint is_dirty

    # indicate if the particle configuration has changed.
    cdef public bint indices_invalid

    # the number of real particles.
    cdef public long num_real_particles

    # a list of props to be used for load balancing
    cdef list lb_props

    ########################################
    # OpenCL related attributes.



    # dictionary to hold the OpenCL properties for a particle
    cdef public dict cl_properties

    # bool indicating CL is setup
    cdef public bint cl_setup_done

    # time for the particle array
    cdef public double time

    cdef object _create_c_array_from_npy_array(self, np.ndarray arr)
    cdef _check_property(self, str)

    cdef np.ndarray _get_real_particle_prop(self, str prop)

    # set/get the time
    cpdef set_time(self, double time)
    cpdef double get_time(self)

    cpdef set_name(self, str name)

    cpdef get_lb_props(self)

    cpdef set_dirty(self, bint val)
    cpdef set_indices_invalid(self, bint val)

    cpdef BaseArray get_carray(self, str prop)

    cpdef int get_number_of_particles(self, bint real=*)
    cpdef remove_particles(self, indices)
    cpdef remove_tagged_particles(self, int tag)

    # function to add any property
    cpdef add_constant(self, str name, data)
    cpdef add_property(self, str name, str type=*, default=*, data=*)
    cpdef remove_property(self, str prop_name)

    # increase the number of particles by num_particles
    cpdef extend(self, int num_particles)

    cpdef has_array(self, str arr_name)

    # aligns all the real particles in contiguous positions starting from 0
    cpdef int align_particles(self) except -1

    # add particles from the parray to self.
    cpdef int append_parray(self, ParticleArray parray) except -1

    # create a new particle array with the given particles indices and the
    # properties.
    cpdef ParticleArray extract_particles(self, indices, list props=*)

    # set the tag value for the particles
    cpdef set_tag(self, long tag_value, LongArray indices)

    cpdef copy_properties(self, ParticleArray source, long start_index=*, long
                          end_index=*)

    # copy properties from one set of variables to another
    cpdef copy_over_properties(self, dict props)

    # set the pid for all local particles
    cpdef set_pid(self, int pid)

    # set the specified properties to zero
    cpdef set_to_zero(self, list props)

    # perform an update on a particle
    cpdef update_property(self, BaseArray a, BaseArray a0,
                          BaseArray acc, double dt)

    # resize all arrays to a new size
    cpdef resize(self, long size)
