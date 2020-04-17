try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

import numpy
from .particle_array import ParticleArray, \
    get_local_tag, get_remote_tag, get_ghost_tag

from cyarray.api import LongArray

UINT_MAX = (1 << 32) - 1


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


# A collection of default properties for all SPH arrays.
DEFAULT_PROPS = set(
    ('x', 'y', 'z', 'u', 'v', 'w', 'm', 'h', 'rho', 'p',
     'au', 'av', 'aw', 'gid', 'pid', 'tag')
)


def get_particle_array(additional_props=None, constants=None, backend=None,
                       **props):
    """Create and return a particle array with default properties.

    The default properties are ['x', 'y', 'z', 'u', 'v', 'w', 'm', 'h', 'rho',
    'p', 'au', 'av', 'aw', 'gid', 'pid', 'tag'], this set is available in
    `DEFAULT_PROPS`.


    Parameters
    ----------

    additional_props : list
        If specified, add these properties.

    constants : dict
        Any constants to be added to the particle array.

    Other Parameters
    ----------------
    props : dict
        Additional keywords passed are set as the property arrays.

    Examples
    --------

    >>> x = linspace(0,1,10)
    >>> pa = get_particle_array(name='fluid', x=x)
    >>> pa.properties.keys()
    ['x', 'z', 'rho', 'pid', 'v', 'tag', 'm', 'p', 'gid', 'au',
     'aw', 'av', 'y', 'u', 'w', 'h']
    >>> pa1 = get_particle_array(name='fluid', additional_props=['xx', 'yy'])

    >>> pa = get_particle_array(name='fluid', x=x, constants={'alpha': 1.0})
    >>> pa.constants.keys()
    ['alpha']

    """

    # handle the name separately
    if 'name' in props:
        name = props['name']
        props.pop('name')
    else:
        name = "array"

    # default properties for an SPH particle
    default_props = set(DEFAULT_PROPS)

    # add any additional props to the default_props
    if additional_props:
        default_props = default_props.union(additional_props)

    np = 0

    prop_dict = {}
    for prop in props.keys():
        data = numpy.asarray(props[prop])
        np = data.size

        if prop in ['tag', 'pid']:
            prop_dict[prop] = {'data': data,
                               'type': 'int',
                               'name': prop}
        elif prop in ['gid']:
            prop_dict[prop] = {'data': data.astype(numpy.uint32),
                               'type': 'unsigned int',
                               'name': prop}
        else:
            prop_dict[prop] = {'data': data,
                               'type': 'double',
                               'name': prop}

    # Add the default props
    for prop in default_props:
        if prop not in prop_dict:
            if prop in ["pid"]:
                prop_dict[prop] = {'name': prop, 'type': 'int',
                                   'default': 0}
            elif prop in ['tag']:
                prop_dict[prop] = {'name': prop, 'type': 'int',
                                   'default': ParticleTAGS.Local}
            elif prop in ['gid']:
                data = numpy.ones(shape=np, dtype=numpy.uint32)
                data[:] = UINT_MAX

                prop_dict[prop] = {'name': prop, 'type': 'unsigned int',
                                   'data': data, 'default': UINT_MAX}

            else:
                prop_dict[prop] = {'name': prop, 'type': 'double',
                                   'default': 0}

    # create the particle array
    pa = ParticleArray(name=name, constants=constants, backend=backend,
                       **prop_dict)

    # default property arrays to save out. Any reasonable SPH particle
    # should define these
    pa.set_output_arrays(['x', 'y', 'z', 'u', 'v', 'w', 'rho', 'm', 'h',
                          'pid', 'gid', 'tag'])

    return pa


def get_particle_array_wcsph(constants=None, **props):
    """Return a particle array for the WCSPH formulation.

    This sets the default properties to be::

        ['x', 'y', 'z', 'u', 'v', 'w', 'h', 'rho', 'm', 'p', 'cs', 'ax', 'ay',
        'az', 'au', 'av', 'aw', 'x0','y0', 'z0','u0', 'v0','w0', 'arho',
        'rho0', 'div', 'gid','pid', 'tag']

    Parameters
    ----------
    constants : dict
        Dictionary of constants

    Other Parameters
    ----------------
    props : dict
        Additional keywords passed are set as the property arrays.

    See Also
    --------
    get_particle_array

    """

    wcsph_props = ['cs', 'ax', 'ay', 'az', 'arho', 'x0', 'y0', 'z0',
                   'u0', 'v0', 'w0', 'rho0', 'div', 'dt_cfl', 'dt_force']

    pa = get_particle_array(
        constants=constants, additional_props=wcsph_props, **props
    )

    # default property arrays to save out.
    pa.set_output_arrays([
        'x', 'y', 'z', 'u', 'v', 'w', 'rho', 'm', 'h',
        'pid', 'gid', 'tag', 'p'
    ])

    return pa


def get_particle_array_iisph(constants=None, **props):
    """Get a particle array for the IISPH formulation.

    The default properties are::

        ['x', 'y', 'z', 'u', 'v', 'w', 'm', 'h', 'rho', 'p', 'au', 'av', 'aw',
        'gid', 'pid', 'tag' 'uadv', 'vadv', 'wadv', 'rho_adv', 'au', 'av',
        'aw','ax', 'ay', 'az', 'dii0', 'dii1', 'dii2', 'V', 'aii', 'dijpj0',
        'dijpj1', 'dijpj2', 'p', 'p0', 'piter', 'compression'
         ]

    Parameters
    ----------
    constants : dict
        Dictionary of constants

    Other Parameters
    ----------------
    props : dict
        Additional keywords passed are set as the property arrays.

    See Also
    --------
    get_particle_array

    """
    iisph_props = ['uadv', 'vadv', 'wadv', 'rho_adv',
                   'au', 'av', 'aw', 'ax', 'ay', 'az',
                   'dii0', 'dii1', 'dii2', 'V', 'dt_cfl', 'dt_force',
                   'aii', 'dijpj0', 'dijpj1', 'dijpj2', 'p', 'p0', 'piter',
                   'compression']
    # Used to calculate the total compression first index is count and second
    # the compression.
    consts = {'tmp_comp': [0.0, 0.0]}
    if constants:
        consts.update(constants)

    pa = get_particle_array(
        constants=consts, additional_props=iisph_props, **props
    )
    pa.set_output_arrays(['x', 'y', 'z', 'u', 'v', 'w', 'rho', 'h', 'm',
                          'p', 'pid', 'au', 'av', 'aw', 'tag', 'gid', 'V'])
    return pa


def get_particle_array_rigid_body(constants=None, **props):
    """Return a particle array for a rigid body motion.

    For multiple bodies, add a body_id property starting at index 0 with each
    index denoting the body to which the particle corresponds to.

    Parameters
    ----------
    constants : dict
        Dictionary of constants

    Other Parameters
    ----------------
    props : dict
        Additional keywords passed are set as the property arrays.

    See Also
    --------
    get_particle_array

    """
    extra_props = ['au', 'av', 'aw', 'V', 'fx', 'fy', 'fz', 'x0', 'y0', 'z0',
                   'tang_disp_x', 'tang_disp_y', 'tang_disp_z', 'tang_disp_x0',
                   'tang_disp_y0', 'tang_disp_z0', 'tang_velocity_x',
                   'tang_velocity_y', 'rad_s',
                   'tang_velocity_z',  'nx', 'ny', 'nz']

    body_id = props.pop('body_id', None)
    nb = 1 if body_id is None else numpy.max(body_id) + 1

    consts = {'total_mass': numpy.zeros(nb, dtype=float),
              'num_body': numpy.asarray(nb, dtype=int),
              'cm': numpy.zeros(3*nb, dtype=float),

              # The mi are also used to temporarily reduce mass (1), center of
              # mass (3) and the interia components (6), total force (3), total
              # torque (3).
              'mi': numpy.zeros(16*nb, dtype=float),
              'force': numpy.zeros(3*nb, dtype=float),
              'torque': numpy.zeros(3*nb, dtype=float),
              # velocity, acceleration of CM.
              'vc': numpy.zeros(3*nb, dtype=float),
              'ac': numpy.zeros(3*nb, dtype=float),
              'vc0': numpy.zeros(3*nb, dtype=float),
              # angular velocity, acceleration of body.
              'omega': numpy.zeros(3*nb, dtype=float),
              'omega0': numpy.zeros(3*nb, dtype=float),
              'omega_dot': numpy.zeros(3*nb, dtype=float)
              }
    if constants:
        consts.update(constants)
    pa = get_particle_array(constants=consts, additional_props=extra_props,
                            **props)
    pa.add_property('body_id', type='int', data=body_id)
    pa.set_output_arrays(['x', 'y', 'z', 'u', 'v', 'w', 'rho', 'h', 'm',
                          'p', 'pid', 'au', 'av', 'aw', 'tag', 'gid', 'V',
                          'fx', 'fy', 'fz', 'body_id'])
    return pa


def get_particle_array_tvf_fluid(constants=None, **props):
    """Return a particle array for the TVF formulation for a fluid.

    Parameters
    ----------
    constants : dict
        Dictionary of constants

    Other Parameters
    ----------------
    props : dict
        Additional keywords passed are set as the property arrays.

    See Also
    --------
    get_particle_array

    """
    tv_props = ['uhat', 'vhat', 'what',
                'auhat', 'avhat', 'awhat', 'vmag2', 'V']

    pa = get_particle_array(
        constants=constants, additional_props=tv_props, **props
    )
    pa.set_output_arrays(['x', 'y', 'z', 'u', 'v', 'w', 'rho', 'p', 'h',
                          'm', 'au', 'av', 'aw', 'V', 'vmag2', 'pid', 'gid',
                          'tag'])

    return pa


def get_particle_array_tvf_solid(constants=None, **props):
    """Return a particle array for the TVF formulation for a solid.

    Parameters
    ----------
    constants : dict
        Dictionary of constants

    Other Parameters
    ----------------
    props : dict
        Additional keywords passed are set as the property arrays.

    See Also
    --------
    get_particle_array

    """
    tv_props = ['u0', 'v0', 'w0', 'V', 'wij', 'ax', 'ay', 'az',
                'uf', 'vf', 'wf', 'ug', 'vg', 'wg']

    pa = get_particle_array(
        constants=constants, additional_props=tv_props, **props
    )
    pa.set_output_arrays(
        ['x', 'y', 'z', 'u', 'v', 'w', 'rho', 'p', 'h', 'm', 'V',
         'pid', 'gid', 'tag']
    )
    return pa


def get_particle_array_gasd(constants=None, **props):
    """Return a particle array for a Gas Dynamics problem.

    Parameters
    ----------
    constants : dict
        Dictionary of constants

    Other Parameters
    ----------------
    props : dict
        Additional keywords passed are set as the property arrays.

    See Also
    --------
    get_particle_array

    """
    required_props = [
        'x', 'y', 'z', 'u', 'v', 'w', 'rho', 'h', 'm', 'cs', 'p', 'e',
        'au', 'av', 'aw', 'arho', 'ae', 'am', 'ah', 'x0', 'y0', 'z0',
        'u0', 'v0', 'w0', 'rho0', 'e0', 'h0', 'div', 'dt_cfl',
        'grhox', 'grhoy', 'grhoz', 'dwdh', 'omega', 'converged',
        'alpha1', 'alpha10', 'aalpha1', 'alpha2', 'alpha20', 'aalpha2',
        'del2e'
    ]

    pa = get_particle_array(
        constants=constants, additional_props=required_props, **props
    )

    # set the intial smoothing length h0 to the particle smoothing
    # length. This can result in an annoying error in the density
    # iterations which require the h0 array
    pa.h0[:] = pa.h[:]

    pa.set_output_arrays(['x', 'y', 'u', 'v', 'rho', 'm', 'h', 'cs', 'p', 'e',
                          'au', 'av', 'ae', 'pid', 'gid', 'tag', 'dwdh',
                          'alpha1', 'alpha2'])

    return pa


def get_particles_info(particles):
    """Return the array information for a list of particles.

    Returns
    -------

    An OrderedDict containing the property information for a list of
    particles. This dict can be used for example to set-up dummy/empty
    particle arrays.

    """
    info = OrderedDict()
    for parray in particles:
        prop_info = {}
        for prop_name, prop in parray.properties.items():
            prop_info[prop_name] = {
                'name': prop_name, 'type': prop.get_c_type(),
                'default': parray.default_values[prop_name],
                'stride': parray.stride.get(prop_name, 1),
                'data': None}
        const_info = {}
        if parray.gpu is not None:
            parray.gpu.pull(*list(parray.constants.keys()))
        for c_name, value in parray.constants.items():
            const_info[c_name] = value.get_npy_array()
        info[parray.name] = dict(
            properties=prop_info, constants=const_info,
            output_property_arrays=parray.output_property_arrays
        )

    return info


def create_dummy_particles(info):
    """Returns a replica (empty) of a list of particles"""
    particles = []
    for name, pa_data in info.items():
        prop_dict = pa_data['properties']
        constants = pa_data['constants']
        pa = ParticleArray(name=name, constants=constants, **prop_dict)
        pa.set_output_arrays(pa_data['output_property_arrays'])
        particles.append(pa)

    return particles


def is_overloaded_method(method):
    """Returns True if the given method is overloaded from any of its bases.
    """
    method_name = method.__name__
    klass = method.__self__.__class__
    count = 0
    prev = None
    for base in klass.mro():
        if hasattr(base, method_name):
            method = getattr(base, method_name)
            if method != prev:
                prev = method
                count += 1
        if count > 1:
            break

    return count > 1
