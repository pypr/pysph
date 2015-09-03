"""Utility functions to interact with SPHysics particle data"""

from os.path import basename
import numpy
from pysph.base.utils import get_particle_array_wcsph as gpa

# Post-process module for VTK output

def sphysics2pysph(partfile, indat='INDAT', dim=3, vtk=True):
    """Load an SPHysics part file and input data

    Parameters:

    partfile : str
        SPHysics part file (eq IPART, PART_00032, etc)

    indat : str
        SPHysics input data file

    dim : int
        Dimension for SPHysics files

    vtk : bint
        Flag to dump VTK output

    Notes:

    The dimension is very important as the SPHysics particle data is
    different in the 2D and 3D cases.

    """
    data = numpy.loadtxt(partfile)

    # sanity check on the input file and problem dimension
    ncols = data.shape[-1]
    if ( (ncols == 9) and (dim == 2) ):
        raise RuntimeError('Possiblly inconsistent dim and SPHysics part file')

    input_data = numpy.loadtxt(indat)

    partbase = basename(partfile)

    if partbase.startswith('IPART'):
        fileno = 0
    else:
        fileno = int( partbase.split('_')[-1] )

    # number of fluid and total number of particles. This is very
    # dangerous and relies on the SPHysics manual (pg. 38)
    dx = float( input_data[21] )
    dy = float( input_data[22] )
    dz = float( input_data[23] )
    h = float( input_data[24] )

    np = int(input_data[25])
    nb = int(input_data[26])
    nbf = int(input_data[27])

    # now load the individual arrays
    if dim == 3:
        x = data[:, 0]; y = data[:, 1]; z = data[:, 2]
        u = data[:, 3]; v = data[:, 4]; w = data[:, 5]

        rho = data[:, 6]; p = data[:, 7]; m = data[:, 8]

    else:
        x = data[:, 0]; z = data[:, 1]
        u = data[:, 2]; w = data[:, 3]

        rho = data[:, 4]; p = data[:, 5]; m = data[:, 6]

    # smoothing lengths
    h = numpy.ones_like(x) * h

    # now create the PySPH arrays
    fluid = gpa(
        name='fluid', x=x[nb:], y=y[nb:], z=z[nb:], u=u[nb:],
        v=v[nb:], w=w[nb:], rho=rho[nb:], p=p[nb:], m=m[nb:],
        h=h[nb:])

    solid = gpa(
        name='boundary', x=x[:nb], y=y[:nb], z=z[:nb], u=u[:nb],
        v=v[:nb], w=w[:nb], rho=rho[:nb], p=p[:nb], m=m[:nb],
        h=h[:nb])

    # PySPH arrays
    arrays = [fluid, solid]

    # Dump out vtk files for Paraview viewing
    if vtk:
        from .pprocess import PySPH2VTK
        props = ['u', 'v', 'w', 'rho', 'p', 'vmag', 'tag']
        pysph2vtk = PySPH2VTK(arrays, fileno=fileno)

        pysph2vtk.write_vtk('fluid', props)
        pysph2vtk.write_vtk('boundary', props)

    # return the list of arrays
    return arrays
