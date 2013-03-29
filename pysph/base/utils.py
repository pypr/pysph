import numpy
from particle_array import ParticleArray
from carray import LongArray

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
    name = ""

    default_props = {'x':0.0, 'y':0.0, 'z':0.0, 'u':0.0, 'v':0.0 ,
                     'w':0.0, 'm':1.0, 'h':1.0, 'p':0.0,'e':0.0,
                     'rho':1.0, 'cs':0.0}
    
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
            
            else:
                data = numpy.asarray(props[prop])
                prop_dict[prop] = {'data':data, 'type':'double'}
            
    # Add the default props
    for prop in default_props:
        if prop not in props.keys():
            prop_dict[prop] = {'name':prop, 'type':'double',
                               'default':default_props[prop]}

    # handle the name separately
    if props.has_key('name'):
        name = props['name']

    pa = ParticleArray(name=name, 
                       cl_precision=cl_precision, **prop_dict)

    # add the constants
    for prop in constants:
        pa.constants[prop] = constants[prop]

    return pa
