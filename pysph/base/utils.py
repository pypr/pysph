import numpy
from particle_array import ParticleArray

from pyzoltan.core.carray import LongArray

UINT_MAX = (1<<32) - 1

class ParticleTAGS:
    Local = 0
    Remote = 1
    Ghost = 2

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
        

def get_particle_array(cl_precision="double", **props):
    """ Create and return a particle array with default properties 
    
    Parameters
    ----------

    cl_precision : {'single', 'double'}
        Precision to use in OpenCL (default: 'double').

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
        
    nprops = len(props)

    prop_dict = {}

    # handle the name separately
    if props.has_key('name'):
        name = props['name']
        props.pop('name')
    else:
        name = "array"

    default_props = {'x':0.0, 'y':0.0, 'z':0.0, 'u':0.0, 'v':0.0 ,
                     'w':0.0, 'm':1.0, 'h':1.0, 'p':0.0,
                     'rho':1.0, 'au':0.0, 'av':0.0, 'aw':0.0,
                     'gid':UINT_MAX, 'pid':0, 'tag':0}
    
    #Add the properties requested
    np = 0

    constants = {}

    for prop in props.keys():
        if prop == 'name':
            pass
        else:
            if not isinstance(props[prop], numpy.ndarray):
                constants[prop] = props[prop]
                continue

            data = numpy.asarray(props[prop])
            if prop in ['gid']:
                prop_dict[prop] = {'data':data,
                                   'type': 'unsigned int',
                                   'name':prop}

            if prop in ['tag']:
                prop_dict[prop] = {'data':data,
                                   'type': 'long',
                                   'name':prop}

            if prop in ['pid']:
                prop_dict[prop] = {'data':data,
                                   'type':'int',
                                   'name':prop}
            
            else:
                prop_dict[prop] = {'data':data, 'type':'double'}
            
    # Add the default props
    for prop in default_props:
        if prop not in props.keys():

            if prop in ["pid"]:
                prop_dict[prop] = {'name':prop, 'type':'int',
                                   'default':default_props[prop]}

            elif prop in ['gid']:
                data = numpy.ones(shape=np, dtype=numpy.uint32)
                data[:] = UINT_MAX

                prop_dict[prop] = {'name':prop, 'type':'unsigned int',
                                   'data':data}

            elif prop in ['tag']:
                prop_dict[prop] = {'name':prop, 'type':'long',
                                   'default':default_props[prop]}

            else:
                prop_dict[prop] = {'name':prop, 'type':'double',
                                   'default':default_props[prop]}

    pa = ParticleArray(name=name, 
                       cl_precision=cl_precision, **prop_dict)

    # add the constants
    for prop in constants:
        pa.constants[prop] = constants[prop]

    return pa

def get_particle_array_wcsph(cl_precision="single", **props):
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
                               'type': 'long',
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
                prop_dict[prop] = {'name':prop, 'type':'long',}

            else:
                prop_dict[prop] = {'name':prop, 'type':'double',
                                   'default':0}

    # create the particle array
    pa = ParticleArray(name=name, **prop_dict)

    return pa
