"""Utility functions to read Daniel Price's NDSPMHD solution files"""

import struct
from pysph.base.utils import get_particle_array_gasd as gpa

from .fortranfile import FortranFile

def ndspmhd2pysph(fname, dim=2, read_type=False):
    """Read output data file from NDSPMHD

    Parameters:
    
    fname : str
        NDSPMHD data filename

    dim : int
        Problem dimension

    read_type : bint
        Flag to read the `type` property for particles

    Returns the ParticleArray representation of the data that can be
    used in PySPH.

    """
    f = FortranFile(fname)

    # get the header length
    header_length = f._header_length
    endian = f.ENDIAN
    
    # get the length of the record to be read
    length = f._read_check()

    # now read the individual entries:

    # current time : double
    t = f._read_exactly(8)
    t = struct.unpack(endian+"1d", t)[0]

    # number of particles and number printed : int
    npart = f._read_exactly(4)
    nprint = f._read_exactly(4)

    npart = struct.unpack(endian+"1i", npart)[0]
    nprint = struct.unpack(endian+"1i", nprint)[0]

    # gamma and hfact : double
    gamma = f._read_exactly(8)
    hfact = f._read_exactly(8)

    gamma = struct.unpack(endian+"1d", gamma)[0]
    hfact = struct.unpack(endian+"1d", hfact)[0]

    # ndim, ndimV : int
    ndim = f._read_exactly(4)
    ndimV = f._read_exactly(4)

    # ncollumns, iformat, ibound : int
    nc = f._read_exactly(4)
    ifmt = f._read_exactly(4)
    ib1 = f._read_exactly(4)
    ib2 = f._read_exactly(4)

    nc = struct.unpack(endian+"1i", nc)[0]

    # xmin, xmax : double
    xmin1 = f._read_exactly(8)
    xmin2 = f._read_exactly(8)
    xmax1 = f._read_exactly(8)
    xmax2 = f._read_exactly(8)

    # n : int
    n = f._read_exactly(4)
    n = struct.unpack(endian+"1i", n)[0]

    # geometry type
    geom = f._read_exactly(n)

    # end reading this header
    f._read_check()

    # Now go on to the arrays. Remember, there are 16 entries
    # correcponding to the columns

    x = f.readReals(prec="d")
    y = f.readReals(prec="d")
    u = f.readReals(prec="d")
    v = f.readReals(prec="d")
    w = f.readReals(prec="d")

    h = f.readReals(prec="d")
    rho = f.readReals(prec="d")
    e = f.readReals(prec="d")
    m = f.readReals(prec="d")

    alpha1 = f.readReals(prec="d")
    alpha2 = f.readReals(prec="d")

    p = f.readReals(prec="d")
    drhobdtbrho = f.readReals("d")
    gradh = f.readReals("d")

    au = f.readReals("d")
    av = f.readReals("d")
    aw = f.readReals("d")

    # By default, NDSPMHD does not output the type array. You need to
    # add this to the output routine if you want it.
    if read_type:
        type = f.readInts(prec="i")

    # now create the particle array
    pa = gpa(name='fluid', x=x, y=y, m=m, h=h, rho=rho, e=e, p=p, 
             u=u, v=v, w=w, au=au, av=av, aw=aw, div=drhobdtbrho)

    return pa
