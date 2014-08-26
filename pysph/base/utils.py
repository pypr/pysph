try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

import numpy
from particle_array import ParticleArray, \
    get_local_tag, get_remote_tag, get_ghost_tag

from pyzoltan.core.carray import LongArray

UINT_MAX = (1<<32) - 1

# Internal tags used in PySPH (defined in particle_array.pxd)
class ParticleTAGS:
    Local = get_local_tag()
    Remote = get_remote_tag()
    Ghost = get_ghost_tag()

def arange_long(start, stop=-1):
    """ Creates a LongArray working same as builtin range with upto 2 arguments
    both expected to be positive
    """

    if stop == -1:
        arange = LongArray(start)
        for i in range(start):
            arange.data[i] = i
        return arange
    else:
        size = stop-start
        arange = LongArray(size)
        for i in range(size):
            arange.data[i] = start + i
        return arange


def get_particle_array(
    additional_props=None, **props):
    """ Create and return a particle array with default properties

    Parameters
    ----------

    props : dict
        A dictionary of properties requested.

    Example Usage:
    --------------
    In [1]: import particle

    In [2]: x = linspace(0,1,10)

    In [3]: pa = particle.get_particle_array(x=x)

    In [4]: pa
    Out[4]: <pysph.base.particle_array.ParticleArray object at 0x9ec302c>

    """

    # handle the name separately
    if props.has_key('name'):
        name = props['name']
        props.pop('name')
    else:
        name = "array"

    np = 0
    nprops = len(props)

    prop_dict = {}
    for prop in props.keys():
        data = numpy.asarray(props[prop])
        np = data.size

        if prop in ['tag']:
            prop_dict[prop] = {'data':data,
                               'type':'int',
                               'name':prop}

        if prop in ['pid']:
            prop_dict[prop] = {'data':data,
                               'type':'int',
                               'name':prop}

        if prop in ['gid']:
            prop_dict[prop] = {'data':data.astype(numpy.uint32),
                               'type':'unsigned int',
                               'name':prop}
        else:
            prop_dict[prop] = {'data':data,
                               'type':'double',
                               'name':prop}

    # default properties for an SPH particle
    default_props = ['x', 'y', 'z', 'u', 'v', 'w', 'm', 'h', 'rho', 'p',
                     'au', 'av', 'aw', 'gid', 'pid', 'tag']

    # add any additional props
    if additional_props:
        default_props.extend( additional_props )
        default_props = list( set(default_props) )

    # Add the default props
    for prop in default_props:
        if not prop in prop_dict:
            if prop in ["pid"]:
                prop_dict[prop] = {'name':prop, 'type':'int',
                                   'default':0}

            elif prop in ['gid']:
                data = numpy.ones(shape=np, dtype=numpy.uint32)
                data[:] = UINT_MAX

                prop_dict[prop] = {'name':prop, 'type':'unsigned int',
                                   'data':data, 'default':UINT_MAX}

            elif prop in ['tag']:
                prop_dict[prop] = {'name':prop, 'type':'int'}

            else:
                prop_dict[prop] = {'name':prop, 'type':'double',
                                   'default':0}

    # create the particle array
    pa = ParticleArray(name=name, **prop_dict)

    # default property arrays to save out. Any reasonable SPH particle
    # should define these
    pa.set_output_arrays( ['x', 'y', 'z', 'u', 'v', 'w', 'rho', 'm', 'h',
                           'pid', 'gid', 'tag'] )

    return pa

def get_particle_array_wcsph(**props):
    """Return a particle array for the WCSPH formulation"""

    # handle the name separately
    if props.has_key('name'):
        name = props['name']
        props.pop('name')
    else:
        name = ""

    nprops = len(props)
    np = 0

    prop_dict = {}
    for prop in props.keys():
        data = numpy.asarray(props[prop])
        np = data.size

        if prop in ['tag']:
            prop_dict[prop] = {'data':data,
                               'type':'int',
                               'name':prop}

        if prop in ['pid']:
            prop_dict[prop] = {'data':data,
                               'type':'int',
                               'name':prop}

        if prop in ['gid']:
            prop_dict[prop] = {'data':data.astype(numpy.uint32),
                               'type': 'unsigned int',
                               'name':prop}
        else:
            prop_dict[prop] = {'data':data,
                               'type':'double',
                               'name':prop}

    default_props = ['x', 'y', 'z', 'u', 'v', 'w', 'h', 'rho', 'm',
                     'p', 'cs', 'ax', 'ay', 'az', 'au', 'av', 'aw',
                     'x0','y0', 'z0','u0', 'v0','w0',
                     'arho', 'rho0', 'div', 'gid','pid', 'tag']

    for prop in default_props:
        if not prop in prop_dict:
            if prop in ["pid"]:
                prop_dict[prop] = {'name':prop, 'type':'int',
                                   'default':0}

            elif prop in ['gid']:
                data = numpy.ones(shape=np, dtype=numpy.uint32)
                data[:] = UINT_MAX

                prop_dict[prop] = {'name':prop, 'type':'unsigned int',
                                   'data':data}

            elif prop in ['tag']:
                prop_dict[prop] = {'name':prop, 'type':'int',}

            else:
                prop_dict[prop] = {'name':prop, 'type':'double',
                                   'default':0}

    # create the particle array
    pa = ParticleArray(name=name, **prop_dict)

    # default property arrays to save out. 
    pa.set_output_arrays( ['x', 'y', 'z', 'u', 'v', 'w', 'rho', 'm', 'h',
                           'pid', 'gid', 'tag', 'p'] )

    return pa

def get_particle_array_tvf_fluid(**props):
    "Get the fluid array for the transport velocity formulation"
    tv_props = ['uf', 'vf', 'wf','uhat', 'vhat', 'what',
                'auhat', 'avhat', 'awhat', 'vmag2', 'V']

    pa = get_particle_array(additional_props=tv_props, **props)
    pa.set_output_arrays( ['x', 'y', 'z', 'u', 'v', 'w', 'rho', 'p', 'h',
                           'm', 'au', 'av', 'aw', 'V', 'vmag2'] )

    return pa

def get_particle_array_tvf_solid(**props):
    "Get the solid array for the transport velocity formulation"
    tv_props = ['u0', 'v0', 'w0', 'V', 'wij', 'ax', 'ay', 'az']

    return get_particle_array(additional_props=tv_props, **props)

def get_particle_array_gasd(**props):
    "Get the particle array with requisite properties for gas-dynamics"
    required_props = [
        'x', 'y', 'z', 'u', 'v', 'w', 'rho', 'h', 'm', 'cs', 'p', 'e',
        'au', 'av', 'aw', 'arho', 'ae', 'am', 'ah', 'x0', 'y0', 'z0', 'u0', 'v0', 'w0',
        'rho0', 'e0', 'h0', 'div', 'grhox', 'grhoy', 'grhoz', 'dwdh', 'omega',
        'converged']

    pa = get_particle_array( additional_props=required_props, **props )

    pa.set_output_arrays( ['x', 'y', 'u', 'v', 'rho', 'm', 'h', 'cs', 'p', 'e',
                           'au', 'av', 'ae', 'pid', 'gid', 'tag', 'dwdh'] )

    return pa

def get_particles_info(particles):
    """Return the array information for a list of particles.

    This function returns a dictionary containing the property
    information for a list of particles. This dict can be used for
    example to set-up dummy/empty particle arrays.

    """
    info = OrderedDict()
    for parray in particles:
        info[ parray.name ] = {}
        for prop_name, prop in parray.properties.iteritems():
            
            info[ parray.name ][prop_name] = {
                'name':prop_name, 'type':prop.get_c_type(),
                'default':parray.default_values[prop_name],
                'data':None}

    return info

def create_dummy_particles(info):
    """Returns a replica (empty) of a list of particles"""
    particles = []
    for name, prop_dict in info.iteritems():
        particles.append( ParticleArray(name=name, **prop_dict) )

    return particles

