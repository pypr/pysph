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


def get_particle_array(additional_props=None, constants=None, **props):
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
    pa = ParticleArray(name=name, constants=constants, **prop_dict)

    # default property arrays to save out. Any reasonable SPH particle
    # should define these
    pa.set_output_arrays( ['x', 'y', 'z', 'u', 'v', 'w', 'rho', 'm', 'h',
                           'pid', 'gid', 'tag'] )

    return pa

def get_particle_array_wcsph(constants=None, **props):
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
    pa = ParticleArray(name=name, constants=constants, **prop_dict)

    # default property arrays to save out.
    pa.set_output_arrays( ['x', 'y', 'z', 'u', 'v', 'w', 'rho', 'm', 'h',
                           'pid', 'gid', 'tag', 'p'] )

    return pa

def get_particle_array_iisph(constants=None, **props):
    """Get a particle array for the IISPH formulation."""
    iisph_props = ['uadv', 'vadv', 'wadv', 'rho_adv',
                 'au', 'av', 'aw','ax', 'ay', 'az',
                 'dii0', 'dii1', 'dii2', 'V',
                 'aii', 'dijpj0', 'dijpj1', 'dijpj2', 'p', 'p0', 'piter',
                 'compression'
                 ]
    # Used to calculate the total compression first index is count and second
    # the compression.
    consts = {'tmp_comp': [0.0, 0.0]}
    if constants:
        consts.update(constants)

    pa = get_particle_array(
        constants=consts, additional_props=iisph_props, **props
    )
    pa.set_output_arrays( ['x', 'y', 'z', 'u', 'v', 'w', 'rho', 'h', 'm',
                           'p', 'pid', 'au', 'av', 'aw', 'tag', 'gid', 'V'] )
    return pa

def get_particle_array_rigid_body(constants=None, **props):
    extra_props = ['au', 'av', 'aw', 'V', 'fx', 'fy', 'fz', 'x0', 'y0', 'z0']
    consts = {'total_mass':0.0,
              'cm': [0.0, 0.0, 0.0],

              # The mi are also used to temporarily reduce mass (1), center of
              # mass (3) and the interia components (6), total force (3), total
              # torque (3).
              'mi': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              'force': [0.0, 0.0, 0.0],
              'torque': [0.0, 0.0, 0.0],
              # velocity, acceleration of CM.
              'vc': [0.0, 0.0, 0.0],
              'ac': [0.0, 0.0, 0.0],
              'vc0': [0.0, 0.0, 0.0],
              # angular velocity, acceleration of body.
              'omega': [0.0, 0.0, 0.0],
              'omega0': [0.0, 0.0, 0.0],
              'omega_dot': [0.0, 0.0, 0.0]
              }
    if constants:
        consts.update(constants)
    pa = get_particle_array(constants=consts, additional_props=extra_props,
                            **props)
    pa.set_output_arrays( ['x', 'y', 'z', 'u', 'v', 'w', 'rho', 'h', 'm',
                           'p', 'pid', 'au', 'av', 'aw', 'tag', 'gid', 'V',
                           'fx', 'fy', 'fz'] )
    return pa

def get_particle_array_tvf_fluid(constants=None, **props):
    "Get the fluid array for the transport velocity formulation"
    tv_props = ['uhat', 'vhat', 'what',
                'auhat', 'avhat', 'awhat', 'vmag2', 'V']

    pa = get_particle_array(
        constants=constants, additional_props=tv_props, **props
    )
    pa.set_output_arrays( ['x', 'y', 'z', 'u', 'v', 'w', 'rho', 'p', 'h',
                           'm', 'au', 'av', 'aw', 'V', 'vmag2'] )

    return pa

def get_particle_array_tvf_solid(constants=None, **props):
    "Get the solid array for the transport velocity formulation"
    tv_props = ['u0', 'v0', 'w0', 'V', 'wij', 'ax', 'ay', 'az',
                'uf', 'vf', 'wf', 'ug', 'vg', 'wg']

    return get_particle_array(
        constants=constants, additional_props=tv_props, **props
    )

def get_particle_array_gasd(constants=None, **props):
    "Get the particle array with requisite properties for gas-dynamics"
    required_props = [
        'x', 'y', 'z', 'u', 'v', 'w', 'rho', 'h', 'm', 'cs', 'p', 'e',
        'au', 'av', 'aw', 'arho', 'ae', 'am', 'ah', 'x0', 'y0', 'z0', 'u0', 'v0', 'w0',
        'rho0', 'e0', 'h0', 'div', 'grhox', 'grhoy', 'grhoz', 'dwdh', 'omega',
        'converged', 'alpha1', 'alpha10', 'aalpha1', 'alpha2', 'alpha20', 'aalpha2',
        'del2e']

    pa = get_particle_array(
        constants=constants, additional_props=required_props, **props
    )

    # set the intial smoothing length h0 to the particle smoothing
    # length. This can result in an annoying error in the density
    # iterations which require the h0 array
    pa.h0[:] = pa.h[:]

    pa.set_output_arrays(['x', 'y', 'u', 'v', 'rho', 'm', 'h', 'cs', 'p', 'e',
                          'au', 'av', 'ae', 'pid', 'gid', 'tag', 'dwdh',
                          'alpha1', 'alpha2'] )

    return pa

def get_particles_info(particles):
    """Return the array information for a list of particles.

    This function returns a dictionary containing the property
    information for a list of particles. This dict can be used for
    example to set-up dummy/empty particle arrays.

    """
    info = OrderedDict()
    for parray in particles:
        prop_info = {}
        for prop_name, prop in parray.properties.iteritems():
            prop_info[prop_name] = {
                'name':prop_name, 'type':prop.get_c_type(),
                'default':parray.default_values[prop_name],
                'data':None}
        const_info = {}
        for c_name, value in parray.constants.iteritems():
            const_info[c_name] = value.get_npy_array()
        info[ parray.name ] = dict(
            properties=prop_info, constants=const_info,
            output_property_arrays=parray.output_property_arrays
        )

    return info

def create_dummy_particles(info):
    """Returns a replica (empty) of a list of particles"""
    particles = []
    for name, pa_data in info.iteritems():
        prop_dict = pa_data['properties']
        constants = pa_data['constants']
        pa = ParticleArray(name=name, constants=constants, **prop_dict)
        pa.set_output_arrays(pa_data['output_property_arrays'])
        particles.append(pa)

    return particles
