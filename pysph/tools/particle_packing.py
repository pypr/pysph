'''API for motion of points on the boundary of
the geometry'''

from pysph.sph.equation import Equation, Group
from pysph.sph.integrator_step import IntegratorStep
from pysph.sph.scheme import Scheme
from pysph.base.utils import get_particle_array
from pysph.tools.geometry import remove_overlap_particles
from itertools import combinations
from math import sqrt, sin
from compyle.api import declare
import numpy


def get_bounding_box(dx, x, y, z=[0], L=0.0, B=0.0, H=0.0):
    """Returns the bounding box required by the packing method
    """
    xmax, xmin, ymax, ymin, zmax, zmin = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    if x is not None:
        xmax = max(x)
        xmin = min(x)
        ymax = max(y)
        ymin = min(y)
        zmax = max(z)
        zmin = min(z)

    if L < 1e-14:
        lenx = dx * int((xmax - xmin)/dx)
        leny = dx * int((ymax - ymin)/dx)
        lenz = dx * int((zmax - zmin)/dx)
    else:
        lenx = L
        leny = B
        lenz = H

    b0x = xmin - 0.5 * lenx
    b1x = xmax + 0.5 * lenx
    b0y = ymin - 0.5 * leny
    b1y = ymax + 0.5 * leny
    b0z = zmin - 0.5 * lenz
    b1z = zmax + 0.5 * lenz

    return b0x, b1x, b0y, b1y, b0z, b1z


def calculate_normal_2d_surface(boundary, shift, isclosed=True):
    """Calculate normals from the set of points arranged in a sequence

    Parameters
    ----------

    boundary: 2 dimensional numpy array of x and y coordinates
    shift: the signed shift value
    isclosed: True if the boundary is closed curve

    Returns
    -------

    xn: x coordinate of normal
    yn: y coordinate of normal
    x: shifted x coordinate of curve
    y: shifted y coordinate of curve
    area: length of the curve
    """
    import numpy as np
    xb = boundary[0]
    yb = boundary[1]

    xn = np.zeros(len(xb))
    yn = np.zeros(len(xb))
    xn0 = np.zeros(len(xb))
    yn0 = np.zeros(len(xb))
    xn1 = np.zeros(len(xb))
    yn1 = np.zeros(len(xb))
    x = np.zeros(len(xb))
    y = np.zeros(len(xb))
    area = np.zeros(len(xb))

    for i in range(len(xb)):
        next = i + 1
        prev = i - 1
        if isclosed:
            if i + 1 == len(xb):
                next = 0
            if i - 1 == -1:
                prev = len(xb) - 1
        else:
            if i + 1 == len(xb):
                next = i
            if i - 1 == -1:
                prev = 0
        dx0 = xb[next] - xb[i]
        dy0 = yb[next] - yb[i]
        d0 = sqrt(dx0**2 + dy0**2)
        xn0[i] = dy0 / (d0 + 1e-6)
        yn0[i] = -dx0 / (d0 + 1e-6)
        dx1 = xb[i] - xb[prev]
        dy1 = yb[i] - yb[prev]
        d1 = sqrt(dx1**2 + dy1**2)
        xn1[i] = dy1 / (d1 + 1e-6)
        yn1[i] = -dx1 / (d1 + 1e-6)
        area[i] = 0.5 * (d0 + d1)
        if not isclosed:
            if i == 0:
                xn1[i] = xn0[i]
                yn1[i] = yn0[i]
            elif i == len(xb) - 1:
                xn0[i] = xn1[i]
                yn0[i] = yn1[i]
        xn[i] = 0.5 * (xn0[i] + xn1[i])
        yn[i] = 0.5 * (yn0[i] + yn1[i])

    d = np.sqrt(xn**2 + yn**2)
    x = xb - shift * 0.5 * (xn0 + xn1) / d**2
    y = yb - shift * 0.5 * (yn0 + yn1) / d**2
    xn = xn / d
    yn = yn / d
    return xn, yn, x, y, area


def shift_surface_inside(x, y, shift, xn, yn,
                         z=numpy.array([0]),
                         zn=numpy.array([0])):
    """Shifted coodinated along xn, yn and zn

    Parameters
    ----------

    x: x coordinate of the boundary
    y: y coordinate of the boundary
    shift: the signed shift value
    xn: x coordinate of normal
    yn: y coordinate of normal
    z: z coordinated of the boundary
    zn: z coordinate of normal

    Returns
    -------

    x0: shifted x coordinate of boundary
    y0: shifted y coordinate of boundary
    z0: shifted z coordinate of boundary
    """
    x0 = x - shift * xn
    y0 = y - shift * yn
    z0 = z - shift * zn
    if len(z) == 1:
        return x0, y0
    else:
        return x0, y0, z0


def repair_boundary(x, y, hard):
    """Repair boundary near hard point

    Parameters
    ----------

    x: x coordinate of the boundary
    y: y coordinate of the boundary
    hard: list of hard points
    """
    N = len(x)
    for id in hard:
        id1 = id - 1
        id2 = id - 2
        id3 = id - 3
        id4 = id - 4
        if id1 < 0:
            id1 = N + id1
        if id2 < 0:
            id2 = N + id2
        if id3 < 0:
            id3 = N + id3
        if id4 < 0:
            id4 = N + id4
        dx = x[id] - x[id4]
        x[id3] = x[id] - 0.8 * dx
        x[id2] = x[id] - 0.5 * dx
        x[id1] = x[id] - 0.2 * dx
        dy = y[id] - y[id4]
        y[id3] = y[id] - 0.8 * dy
        y[id2] = y[id] - 0.5 * dy
        y[id1] = y[id] - 0.2 * dy

        d = (x[id] - x[id1])**2 + (y[id] - y[id1])**2
        id1 = id + 1
        id2 = id + 2
        id3 = id + 3
        id4 = id + 4
        if id1 >= N:
            id1 = id1 - N
        if id2 >= N:
            id2 = id2 - N
        if id3 >= N:
            id3 = id3 - N
        if id4 >= N:
            id4 = id4 - N
        dx = x[id4] - x[id]
        dy = y[id4] - y[id]
        d0 = dx**2 + dy**2
        fac = sqrt(d / d0)
        x[id3] = x[id] + 0.8 * dx
        x[id2] = x[id] + 0.5 * dx
        x[id1] = x[id] + fac * dx
        y[id3] = y[id] + 0.8 * dy
        y[id2] = y[id] + 0.5 * dy
        y[id1] = y[id] + fac * dy


def create_frozen_container_outer(
        dx, hdx, rho, bound, layers=8, dim=2, name='frozen'):
    """function to create frozen particle lattice of given
    size and spacing

    Parameters
    ----------

    dx: required resolution
    hdx: h/dx ratio
    rho: density of the particles
    L: length of the domain
    B: Breadth of the domain
    l: number of layers
    H: height of the domain
    dim: dimension of the problem
    name: name of the particle array

    Returns
    -------

    frozen: pysph.base.utils.ParticleArray class instance
    """
    import numpy as np
    eps = dx/10
    h = hdx * dx
    m = rho * dx**dim
    nl = layers * dx
    b = bound

    if dim == 2:
        x0, y0 = np.mgrid[b[0] + dx:b[1] - eps:2 * dx,
                          b[2]:b[3] + dx / 2 - eps:dx]
        x1, y1 = np.mgrid[b[0]:b[1] + dx - eps:2 * dx,
                          b[2] + dx / 2:b[3] - eps:dx]
        x0, y0 = [t.ravel() for t in (x0, y0)]
        x1, y1 = [t.ravel() for t in (x1, y1)]
        x = np.concatenate((x0, x1))
        y = np.concatenate((y0, y1))
        z = np.zeros_like(x)
    elif dim == 3:
        x0, y0, z0 = np.mgrid[b[0] + dx:b[1] - eps:2 * dx,
                              b[2]:b[3] - eps + dx / 2:dx,
                              b[4]:b[5] - eps + dx / 2:dx]
        x1, y1, z1 = np.mgrid[b[0]:b[1] - eps + dx:2 * dx,
                              b[2] + dx / 2:b[3] - eps:dx,
                              b[4] + dx / 2:b[5] - eps:dx]
        x0, y0, z0 = [t.ravel() for t in (x0, y0, z0)]
        x1, y1, z1 = [t.ravel() for t in (x1, y1, z1)]
        x = np.concatenate((x0, x1))
        y = np.concatenate((y0, y1))
        z = np.concatenate((z0, z1))

    inner = get_particle_array(x=x, y=y, z=z, m=m, rho=rho, h=h, name=name)

    if dim == 2:
        x0, y0 = np.mgrid[b[0] - 2 * nl:b[1] - eps + 2 * nl:dx,
                          b[2] - 2 * nl + dx / 2:b[3] - eps + 2 * nl:dx]
        print(x[0])
        x, y = [t.ravel() for t in (x0, y0)]
        z = np.zeros_like(x)
        cond = ~((x - (b[0] - eps - nl) > 1e-14) &
                 (x - (b[1] - eps + nl) < 1e-14) &
                 (y - (b[2] - eps - nl) > 1e-14) &
                 (y - (b[3] - eps + nl) < 1e-14))

    elif dim == 3:
        x0, y0, z0 = np.mgrid[b[0] - 2 * nl:b[1] - eps + 2 * nl:dx,
                              b[2] - 2 * nl + dx / 2:b[3] - eps + 2 * nl:dx,
                              b[4] - 2 * nl + dx / 2:b[5] - eps + 2 * nl:dx]
        x, y, z = [t.ravel() for t in (x0, y0, z0)]
        cond = ~((x - (b[0] - eps - nl) > 1e-14) &
                 (x - (b[1] + eps + nl) < 1e-14) &
                 (y - (b[2] - eps - nl) > 1e-14) &
                 (y - (b[3] + eps + nl) < 1e-14) &
                 (z - (b[4] - eps - nl) > 1e-14) &
                 (z - (b[5] + eps + nl) < 1e-14))

    frozen = get_particle_array(
        x=x[cond], y=y[cond], z=z[cond], m=m, rho=rho, h=h, name=name)
    ids = np.where(inner.h > -1)[0]
    print(ids)
    inner.extract_particles(ids, frozen)
    return frozen


def create_free_particles_outer(dx, hdx, rho, bound, layers=8, dim=2,
                                name='free'):
    """function to create free particle lattice of given size and spacing

    Parameters
    ----------

    dx: required resolution
    hdx: h/dx ratio
    rho: density of the particles
    L: length of the domain
    B: Breadth of the domain
    H: height of the domain
    dim: dimension of the problem
    name: name of the particle array

    Returns
    -------

    free: pysph.base.utils.ParticleArray class instance
    """
    import numpy as np
    eps = dx / 10
    h = hdx * dx
    m = rho * dx**dim
    nl = layers * dx
    b = bound
    if dim == 2:
        x0, y0 = np.mgrid[b[0] - nl + dx:b[1] - eps + nl:2 * dx,
                          b[2] - nl + dx / 2:b[3] - eps + nl:dx]
        x1, y1 = np.mgrid[b[0] - nl:b[1] - eps + nl:2 * dx,
                          b[2] - nl:b[3] - eps + nl:dx]
        x0, y0 = [t.ravel() for t in (x0, y0)]
        x1, y1 = [t.ravel() for t in (x1, y1)]
        x = np.concatenate((x0, x1))
        y = np.concatenate((y0, y1))
        z = np.zeros_like(x)
        print(x[0])
        cond = ~((x - b[0] + eps > 1e-14) & (x - b[1] - eps < 1e-14) &
                 (y - b[2] + eps > 1e-14) & (y - b[3] - eps < 1e-14))

    elif dim == 3:
        x0, y0, z0 = np.mgrid[b[0] - nl + dx:b[1] - eps + nl:2 * dx,
                              b[2] - nl + dx / 2:b[3] - eps + nl:dx,
                              b[4] - nl + dx / 2:b[5] - eps + nl:dx]
        x1, y1, z1 = np.mgrid[b[0] - nl:b[1] - eps + nl:2 * dx,
                              b[2] - nl:b[3] - eps + nl:dx,
                              b[4] - nl:b[5] - eps + nl:dx]
        x0, y0, z0 = [t.ravel() for t in (x0, y0, z0)]
        x1, y1, z1 = [t.ravel() for t in (x1, y1, z1)]
        x = np.concatenate((x0, x1))
        y = np.concatenate((y0, y1))
        z = np.concatenate((z0, z1))
        cond = ~((x - b[0] + eps > 1e-14) & (x - b[1] - eps < 1e-14) &
                 (y - b[2] + eps > 1e-14) & (y - b[3] - eps < 1e-14) &
                 (z - b[4] + eps > 1e-14) & (z - b[5] - eps < 1e-14))

    free = get_particle_array(
        x=x[cond], y=y[cond], z=z[cond], m=m,
        rho=rho, h=h, name=name)
    return free


def create_frozen_container_rect(
        dx, hdx, rho, bound, layers=3, dim=2, name='frozen'):
    """function to create frozen particle lattice of given
    size and spacing

    Parameters
    ----------

    dx: required resolution
    hdx: h/dx ratio
    rho: density of the particles
    L: length of the domain
    B: Breadth of the domain
    l: number of layers
    H: height of the domain
    dim: dimension of the problem
    name: name of the particle array

    Returns
    -------

    frozen: pysph.base.utils.ParticleArray class instance
    """
    import numpy as np
    eps = dx/10
    h = hdx * dx
    m = rho * dx**dim
    nl = layers * dx
    b = bound
    if dim == 2:
        x0, y0 = np.mgrid[b[0] - nl + dx / 2:b[1] - eps + nl:dx,
                          b[2] - nl + dx / 2:b[3] - eps + nl:dx]
        x, y = [t.ravel() for t in (x0, y0)]
        z = np.zeros_like(x)
        cond = ~((x - b[0] > 1e-14) & (x - b[1] < 1e-14) &
                 (y - b[2] > 1e-14) & (y - b[3] < 1e-14))

    elif dim == 3:
        x0, y0, z0 = np.mgrid[b[0] - nl + dx / 2:b[1] + nl:dx,
                              b[2] - nl + dx / 2:b[3] + nl:dx,
                              b[4] - nl + dx / 2:b[5] + nl:dx]
        x, y, z = [t.ravel() for t in (x0, y0, z0)]
        cond = ~((x - b[0] > 1e-14) & (x - b[1] < 1e-14) &
                 (y - b[2] > 1e-14) & (y - b[3] < 1e-14) &
                 (z - b[4] > 1e-14) & (z - b[5] < 1e-14))

    frozen = get_particle_array(
        x=x[cond], y=y[cond], z=z[cond], m=m,
        rho=rho, h=h, name=name)
    return frozen


def create_free_particles_rect(dx, hdx, rho, bound, dim=2, name='free'):
    """function to create free particle lattice of given size and spacing

    Parameters
    ----------

    dx: required resolution
    hdx: h/dx ratio
    rho: density of the particles
    L: length of the domain
    B: Breadth of the domain
    H: height of the domain
    dim: dimension of the problem
    name: name of the particle array

    Returns
    -------

    free: pysph.base.utils.ParticleArray class instance
    """
    import numpy as np
    eps = dx/10
    h = hdx * dx
    m = rho * dx**dim
    b = bound
    if dim == 2:
        x0, y0 = np.mgrid[b[0] + dx / 2:b[1]:dx,
                          b[2] + dx / 2:b[3]:dx]
        x, y = [t.ravel() for t in (x0, y0)]
        z = np.zeros_like(x)
    elif dim == 3:
        x0, y0, z0 = np.mgrid[b[0] + dx / 2:b[1]:dx,
                              b[2] + dx / 2:b[3]:dx,
                              b[4] + dx / 2:b[5]:dx]
        x, y, z = [t.ravel() for t in (x0, y0, z0)]
    free = get_particle_array(x=x, y=y, z=z, m=m, rho=rho, h=h, name=name)
    return free


def create_frozen_container(dx, hdx, rho, bound,
                            layers=3, dim=2,
                            name='frozen'):
    """function to create frozen particle lattice of given
    size and spacing

    Parameters
    ----------

    dx: required resolution
    hdx: h/dx ratio
    rho: density of the particles
    L: length of the domain
    B: Breadth of the domain
    layers: number of layers
    H: height of the domain
    dim: dimension of the problem
    name: name of the particle array

    Returns
    -------

    frozen: pysph.base.utils.ParticleArray class instance
    """
    import numpy as np
    eps = dx/10
    h = hdx * dx
    m = rho * dx**dim
    nl = layers * dx
    b = bound
    if dim == 2:
        x0, y0 = np.mgrid[b[0] - nl + dx:b[1] - eps + nl:2 * dx,
                          b[2] - nl + dx / 2:b[3] - eps + nl:dx]
        x1, y1 = np.mgrid[b[0] - nl:b[1] - eps + nl:2 * dx,
                          b[2] - nl:b[3] - eps + nl:dx]
        x0, y0 = [t.ravel() for t in (x0, y0)]
        x1, y1 = [t.ravel() for t in (x1, y1)]
        x = np.concatenate((x0, x1))
        y = np.concatenate((y0, y1))
        z = np.zeros_like(x)
        cond = ~((x - b[0] + eps > 1e-14) & (x - b[1] - eps < 1e-14) &
                 (y - b[2] + eps > 1e-14) & (y - b[3] - eps < 1e-14))

    elif dim == 3:
        x0, y0, z0 = np.mgrid[b[0] - nl + dx:b[1] - eps + nl:2 * dx,
                              b[2] - nl + dx / 2:b[3] - eps + nl:dx,
                              b[4] - nl + dx / 2:b[5] - eps + nl:dx]
        x1, y1, z1 = np.mgrid[b[0] - nl:b[1] - eps + nl:2 * dx,
                              b[2] - nl:b[3] - eps + nl:dx,
                              b[4] - nl:b[5] - eps + nl:dx]
        x0, y0, z0 = [t.ravel() for t in (x0, y0, z0)]
        x1, y1, z1 = [t.ravel() for t in (x1, y1, z1)]
        x = np.concatenate((x0, x1))
        y = np.concatenate((y0, y1))
        z = np.concatenate((z0, z1))
        cond = ~((x - b[0] + eps > 1e-14) & (x - b[1] - eps < 1e-14) &
                 (y - b[2] + eps > 1e-14) & (y - b[3] - eps < 1e-14) &
                 (z - b[4] + eps > 1e-14) & (z - b[5] - eps < 1e-14))

    frozen = get_particle_array(
        x=x[cond], y=y[cond], z=z[cond], m=m,
        rho=rho, h=h, name=name)
    return frozen


def create_free_particles(dx, hdx, rho, bound, dim=2, name='free'):
    """function to create free particle lattice of given size and spacing

    Parameters
    ----------

    dx: required resolution
    hdx: h/dx ratio
    rho: density of the particles
    L: length of the domain
    B: Breadth of the domain
    H: height of the domain
    dim: dimension of the problem
    name: name of the particle array

    Returns
    -------

    free: pysph.base.utils.ParticleArray class instance
    """
    import numpy as np
    eps = dx/10
    h = hdx * dx
    m = rho * dx**dim
    b = bound
    if dim == 2:
        x0, y0 = np.mgrid[b[0] + dx:b[1] - eps:2 * dx,
                          b[2]:b[3] - eps + dx / 2:dx]
        x1, y1 = np.mgrid[b[0]:b[1] - eps + dx:2 * dx,
                          b[2] + dx / 2:b[3] - eps:dx]
        x0, y0 = [t.ravel() for t in (x0, y0)]
        x1, y1 = [t.ravel() for t in (x1, y1)]
        x = np.concatenate((x0, x1))
        y = np.concatenate((y0, y1))
        z = np.zeros_like(x)
    elif dim == 3:
        x0, y0, z0 = np.mgrid[b[0] + dx:b[1] - eps:2 * dx,
                              b[2]:b[3] - eps + dx / 2:dx,
                              b[4]:b[5] - eps + dx / 2:dx]
        x1, y1, z1 = np.mgrid[b[0]:b[1] - eps + dx:2 * dx,
                              b[2] + dx / 2:b[3] - eps:dx,
                              b[4] + dx / 2:b[5] - eps:dx]
        x0, y0, z0 = [t.ravel() for t in (x0, y0, z0)]
        x1, y1, z1 = [t.ravel() for t in (x1, y1, z1)]
        x = np.concatenate((x0, x1))
        y = np.concatenate((y0, y1))
        z = np.concatenate((z0, z1))
    free = get_particle_array(x=x, y=y, z=z, m=m, rho=rho, h=h, name=name)
    return free


def create_surface_from_stl(
        filename, dx, hdx, rho, scale=1.0, shift=True, name='solid_nodes',
        hard={}, invert=False):
    """function to create solid nodes particle from a 3D Stl

    Parameters
    ----------

    filename: stl file name
    dx: required resolution
    hdx: h/dx ratio
    rho: density of the particles
    scale: scale factor
    shift: True if the geometry is to be shifted
    name: name of the particle array
    hard: dict of hard pint index with coordinates
    invert: True if normals to be inverted

    Returns
    -------

    solid_nodes: pysph.base.utils.ParticleArray class instance
    """
    import numpy as np
    from stl import stl
    from pysph.tools.geometry import evaluate_area_of_triangle
    import meshio
    data = meshio.read(filename, file_format="stl")
    triangles = data.cells_dict['triangle']
    normals = data.cell_data['facet_normals'][0]
    xc, yc, zc = [], [], []
    xn, yn, zn = normals[:, 0], normals[:, 1], normals[:, 2]
    area = []
    for tri in triangles:
        centroid = np.average(data.points[tri], axis=0)
        xc.append(centroid[0])
        yc.append(centroid[1])
        zc.append(centroid[2])
        _area = evaluate_area_of_triangle(data.points[tri])
        area.append(_area)

    xc, yc, zc, area = [np.array(t) for t in (xc, yc, zc, area)]

    h = hdx * dx
    m = dx * dx * dx * rho

    xx = scale * xc
    yx = scale * yc
    zx = scale * zc

    if invert:
        xn = -xn
        yn = -xn
        zn = -xn
    area = area * scale**2
    n_pnts = sum(area) / dx**2
    d = np.sqrt(xn**2 + yn**2 + zn**2)
    if shift:
        x, y, z = shift_surface_inside(
            x=xc, y=yc, z=zc, shift=dx / 2, xn=xn / d,
            yn=yn / d, zn=zn / d)
    else:
        x, y, z = xc.copy(), yc.copy(), zc.copy()
    solid_nodes = get_particle_array(
        x=x, y=y, z=z, m=m, rho=rho, h=h, name=name, xn=xn / d,
        yn=yn / d, zn=zn / d, area=area, hard=0.0)
    solid_nodes.add_property('xc')
    solid_nodes.add_property('yc')
    solid_nodes.add_property('zc')
    solid_nodes.add_constant('n_pnts', n_pnts)
    solid_nodes.xc[:] = xc
    solid_nodes.yc[:] = yc
    solid_nodes.zc[:] = zc
    for id in hard:
        solid_nodes.hard[id] = 1.0
    return solid_nodes


def create_surface_from_file(
        filename, points, dx, hdx, rho, isclosed, shift=True, invert=False,
        name='solid_nodes', hard={}):
    """function to create solid nodes particle from a x,y data file

    Parameters
    ----------

    filename: stl file name
    dx: required resolution
    hdx: h/dx ratio
    rho: density of the particles
    isclosed: True if the curve is closed
    shift: True if the geometry is to be shifted
    invert: True if normals to be inverted
    name: name of the particle array
    hard: dict of hard pint index with coordinates

    Returns
    -------

    solid_nodes: pysph.base.utils.ParticleArray class instance
    """
    import numpy as np
    boundary = None
    if filename is not None:
        xa, ya = np.loadtxt(filename, unpack=True)
        boundary = [xa, ya]
    else:
        boundary = points
        xa, ya = boundary[0], boundary[1]
    h = hdx * dx
    m = dx * dx * rho
    fact = 1.0
    _shift = 0.0
    if invert:
        fact = -1.0 * fact
    if shift:
        _shift = fact * dx / 2
    xn, yn, x, y, area = calculate_normal_2d_surface(
        boundary, _shift, isclosed=isclosed)
    repair_boundary(x, y, hard)
    n_pnts = sum(area) / dx
    solid_nodes = get_particle_array(
        x=x, y=y, z=0, m=m, rho=rho, h=h, xn=fact * xn,
        yn=fact * yn, zn=0, name=name, area=area, hard=0)
    solid_nodes.add_constant('n_pnts', n_pnts)
    for id in hard:
        solid_nodes.hard[id] = 1.0
    return solid_nodes


# intergrator
class InteriorStep(IntegratorStep):
    """Euler integrator for free particles
    """
    def stage1(self, d_idx, d_x, d_y, d_z,
               d_u, d_v, d_w, d_au, d_av, d_aw,
               dt=0.0):
        d_x[d_idx] = d_x[d_idx] + dt * d_u[d_idx]
        d_y[d_idx] = d_y[d_idx] + dt * d_v[d_idx]
        d_z[d_idx] = d_z[d_idx] + dt * d_w[d_idx]

        d_u[d_idx] = d_u[d_idx] + dt * d_au[d_idx]
        d_v[d_idx] = d_v[d_idx] + dt * d_av[d_idx]
        d_w[d_idx] = d_w[d_idx] + dt * d_aw[d_idx]


class SolidStep(IntegratorStep):
    """Euler integrator for boundary particles
    """
    def stage1(self, d_idx, d_x, d_y, d_z, d_u, d_v,
               d_w, d_au, d_av, d_aw, d_xr, d_yr,
               d_zr, d_hard, dt=0.0):
        if d_hard[d_idx] < 0.5:
            V = d_u[d_idx] * d_xr[d_idx] + d_v[d_idx] * d_yr[d_idx] + d_w[
                d_idx] * d_zr[d_idx]
            d_x[d_idx] = d_x[d_idx] + dt * d_xr[d_idx] * V
            d_y[d_idx] = d_y[d_idx] + dt * d_yr[d_idx] * V
            d_z[d_idx] = d_z[d_idx] + dt * d_zr[d_idx] * V

            d_u[d_idx] = d_u[d_idx] + dt * d_au[d_idx]
            d_v[d_idx] = d_v[d_idx] + dt * d_av[d_idx]
            d_w[d_idx] = d_w[d_idx] + dt * d_aw[d_idx]


# Equations
# used in post process
class FindExternalParticles(Equation):
    """Tag the neighboring particles inside and
     outside based on projection

    Parameters:

    dest: pysph.base.utils.ParticleArray
        Free particles array

    sources: list of pysph.base.utils.ParticleArray
        Boundary nodes array
    """
    def initialize(self, d_idx, d_interior):
        d_interior[d_idx] = 0

    def loop(self, d_idx, s_idx, s_xn, s_yn, s_zn, XIJ, d_neartag, d_interior,
             RIJ, d_h, s_hard):
        if d_neartag[d_idx] == s_idx:
            proj = XIJ[0] * s_xn[s_idx] + XIJ[1] * s_yn[s_idx] + XIJ[2] * s_zn[
                s_idx]
            if proj > 1e-14:
                d_interior[d_idx] = 1
            else:
                d_interior[d_idx] = -1


class FindNearNodes(Equation):
    """Tag the neighboring particles inside if the dest particle is interior

    Parameters:

    dest: pysph.base.utils.ParticleArray
        Free particles array

    sources: list of pysph.base.utils.ParticleArray
        Free particles array
    """
    def loop(self, d_idx, s_idx, d_interior, s_interior, d_neartag):
        if d_interior[d_idx] == 0:
            if s_interior[s_idx] == 1:
                d_interior[d_idx] = 1
            elif s_interior[s_idx] == -1:
                d_interior[d_idx] = -1


###########################################################
class SPHApprox(Equation):
    """Standard SPH approximation

    Parameters:

    dest: pysph.base.utils.ParticleArray
        any particles array

    sources: list of pysph.base.utils.ParticleArray
        any particles array other than dest
    """
    def loop(self, d_idx, d_f, s_m, s_rho, s_f, s_idx, WIJ):
        d_f[d_idx] += s_f[s_idx] * s_m[s_idx] * WIJ / s_rho[s_idx]


class SPHDerivativeApprox(Equation):
    """Standard SPH derivative approximation

    Parameters:

    dest: pysph.base.utils.ParticleArray
        any particles array

    sources: list of pysph.base.utils.ParticleArray
        any particles array other than dest
    """
    def loop(self, d_idx, d_df, s_m, s_rho, s_f, s_idx, DWIJ):
        d_df[d_idx] += s_f[s_idx] * s_m[s_idx] * DWIJ[0] / s_rho[s_idx]


class FindNearestNode(Equation):
    """Find nearest boundary node

    Parameters:

    dest: pysph.base.utils.ParticleArray
        Free or boundary particle

    sources: list of pysph.base.utils.ParticleArray
        Boundary nodes array
    """
    def __init__(self, dest, sources, fac=1.0):
        self.fac = fac
        super(FindNearestNode, self).__init__(dest, sources)

    def initialize(self, d_idx, d_nearest, d_neartag, d_xn, d_yn, d_zn,
                   d_hard):
        d_nearest[d_idx] = 10000.0
        d_neartag[d_idx] = -1
        if d_hard[d_idx] < 0.5:
            d_xn[d_idx] = 0.0
            d_yn[d_idx] = 0.0
            d_zn[d_idx] = 0.0

    def loop(self, d_idx, s_idx, RIJ, d_nearest, d_neartag, t, d_hard, d_h,
             d_xn, d_yn, d_zn, s_xn, s_yn, s_zn, s_hard):
        if (RIJ < d_nearest[d_idx]) and (d_hard[d_idx] < 0.5):
            if (s_hard[s_idx] < 0.5) and (RIJ - self.fac * d_h[d_idx] < 1e-14):
                d_nearest[d_idx] = RIJ
                d_neartag[d_idx] = s_idx
                d_xn[d_idx] = s_xn[s_idx]
                d_yn[d_idx] = s_yn[s_idx]
                d_zn[d_idx] = s_zn[s_idx]


class EvaluateAdaptiveTime(Equation):
    """Find new timestep

    Parameters:

    dest: pysph.base.utils.ParticleArray
        Free or boundary particle

    sources: None
    """
    def initialize(self, d_idx, d_dt_adapt, d_u, d_v, d_w, d_au, d_av, d_aw,
                   dt, t, d_h, d_pb, d_nu):
        if t < 1e-14:
            d_dt_adapt[d_idx] = 1e-7
        else:
            dt_vel = 10000
            dt_visc = 10000
            Vx = d_u[d_idx] + d_au[d_idx] * dt
            Vy = d_v[d_idx] + d_av[d_idx] * dt
            Vz = d_w[d_idx] + d_aw[d_idx] * dt
            V = sqrt(Vx**2 + Vy**2 + Vz**2)
            if V > 1e-14:
                dt_visc = sqrt(0.1 * d_h[d_idx] / d_nu[0] / V)
            dt_pb = 0.1 * d_h[d_idx] / sqrt(d_pb[0])
            dt_ = min(dt_vel, dt_visc, dt_pb)

            d_dt_adapt[d_idx] = dt_


class SummationDensity(Equation):
    """Standard summation density

    Parameters:

    dest: pysph.base.utils.ParticleArray
        Free or boundary particle

    sources: list of pysph.base.utils.ParticleArray
        All particle arrays (not nodes)
    """
    def initialize(self, d_idx, d_V, d_rho):
        d_V[d_idx] = 0.0
        d_rho[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_V, d_rho, d_m, s_m, WIJ):
        d_V[d_idx] += WIJ
        d_rho[d_idx] += s_m[s_idx] * WIJ


class NumberDensityGradient(Equation):
    """Evaluate the number density gradient

    Parameters:

    dest: pysph.base.utils.ParticleArray
        Free or boundary particle

    sources: list of pysph.base.utils.ParticleArray
        All particle arrays (not nodes)
    """
    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_m, d_rho, s_rho, d_pb, d_au, d_av, d_aw,
             d_V, s_V, DWIJ, XIJ, s_m):

        Vi = d_m[d_idx] / d_rho[d_idx]
        Vj = s_m[s_idx] / s_rho[s_idx]

        tmp = -d_pb[0] * Vi * Vj / d_m[d_idx]

        d_au[d_idx] += tmp * DWIJ[0]
        d_av[d_idx] += tmp * DWIJ[1]
        d_aw[d_idx] += tmp * DWIJ[2]


class ViscousDamping(Equation):
    """Evaluate acceleration due to damping

    Parameters:

    dest: pysph.base.utils.ParticleArray
        Free or boundary particle

    sources: list of pysph.base.utils.ParticleArray
        All particle arrays (not nodes)
    """
    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def post_loop(self, d_idx, d_rho, d_m, d_V, d_au, d_av, d_aw, d_nu, d_u,
                  d_v, d_w, s_m, t):

        etai = d_nu[0]
        d_au[d_idx] += -etai * d_u[d_idx]
        d_av[d_idx] += -etai * d_v[d_idx]
        d_aw[d_idx] += -etai * d_w[d_idx]


class RepulsionForce(Equation):
    """accelearion due to repulsion force

    Parameters:

    dest: pysph.base.utils.ParticleArray
        free or boundary particles

    sources: pysph.base.utils.ParticleArray
        all particles (not nodes)

    hdx: double
        h/dx ratio
    """
    def __init__(self, dest, sources, hdx):
        self.hdx = hdx
        super(RepulsionForce, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, d_au, d_av, d_aw, RIJ, XIJ, d_h, d_k):
        if RIJ > 1e-14:
            rij = RIJ
            xij = XIJ[0]
            yij = XIJ[1]
            zij = XIJ[2]
            if RIJ - d_h[d_idx] * 0.5 < 1e-14:
                rij = d_h[d_idx] * 0.5
                xij = xij / RIJ * rij
                yij = yij / RIJ * rij
                zij = zij / RIJ * rij
            c = self.hdx * d_h[d_idx] * 2. / 3.
            tmp = 12. * d_k[0] * (3 * c**2 / rij**5 - 2 * c / rij**4)
            if tmp > 1e-14:
                d_au[d_idx] += tmp * xij
                d_av[d_idx] += tmp * yij
                d_aw[d_idx] += tmp * zij


class FindBoundaryNodeDirection(Equation):
    """direction of motion of boundary particle

    Parameters:

    dest: pysph.base.utils.ParticleArray
        boundary particles

    sources: pysph.base.utils.ParticleArray
        corresponding boundary node array
    """
    def initialize(self, d_idx, d_dmin, d_xr, d_yr, d_zr):
        d_dmin[d_idx] = 10000
        d_xr[d_idx] = 0.0
        d_yr[d_idx] = 0.0
        d_zr[d_idx] = 0.0

    def loop(self, d_idx, d_dmin, XIJ, RIJ, VIJ, d_xr, d_yr, d_zr):
        dist = RIJ
        dir = VIJ[0] * XIJ[0] + VIJ[1] * XIJ[1] + VIJ[2] * XIJ[2]
        if dir < -1e-14:
            if d_dmin[d_idx] - dist > 1e-14:
                d_dmin[d_idx] = dist
                d_xr[d_idx] = -XIJ[0] / RIJ
                d_yr[d_idx] = -XIJ[1] / RIJ
                d_zr[d_idx] = -XIJ[2] / RIJ


class FilterLayers(Equation):
    """tag free particles near solid nodes

    Parameters:

    dest: pysph.base.utils.ParticleArray
        Boundary nodes

    sources: list of pysph.base.utils.ParticleArray
        all free particles
    """
    def loop(self, s_filter, s_idx, t):
        if t < 1e-14:
            s_filter[s_idx] = 1


class FindNearestNodeToHardPoint(Equation):
    """Find free particles to a geometry node

    Parameters:

    dest: pysph.base.utils.ParticleArray
        Boundary nodes

    sources: list of pysph.base.utils.ParticleArray
        all free particles
    """
    def initialize(self, d_idx, d_nearest, d_neartag, d_xn, d_yn, d_zn):
        d_nearest[d_idx] = 10000.0
        d_neartag[d_idx] = -1

    def loop(self, d_idx, s_idx, RIJ, d_nearest, d_neartag, t, d_hard, d_xn,
             d_yn, d_zn, s_xn, s_yn, s_zn, d_h):
        if (RIJ < d_nearest[d_idx]):
            d_nearest[d_idx] = RIJ
            d_neartag[d_idx] = s_idx


class ProjectionToSurfaceBoundary(Equation):
    """find the distance perpendicular to the surface for
    each free particle

    Parameters:

    dest: pysph.base.utils.ParticleArray
        free particles

    sources: list of pysph.base.utils.ParticleArray
        geometry nodes
    """
    def loop(self, d_idx, s_idx, XIJ, s_xn, s_yn, s_zn,
             d_neartag, d_u, d_v, d_w, d_h, d_nearest):
        projection = XIJ[0] * s_xn[s_idx] + XIJ[1] * s_yn[s_idx] + XIJ[
            2] * s_zn[s_idx]
        if (d_neartag[d_idx] == s_idx):
            d_nearest[d_idx] = projection


class ParticlePacking(Scheme):
    """An API for a hybrid particle packing scheme.

    Methods
    -------
    create_frozen_container(L, B, H=0, l=5, name='frozen')
        Creates the outer container of ghost particles

    create_free_particles(L, B, H=0, name='free')
        Creates array of free particles

    create_agitators(free, solid_nodes, L, B, H=0, name='agitator')
        Creates agitator particles [Not used]

    create_boundary_node(filename, scale=1.0, shift=True,
                         invert=False, name='solid_nodes', isclosed=True)
        Creates boundary node arrays

    _is_volume_converged(pa)
        Checks for convergence

    post_process(free, solid, solid_nodes, frozen, dx, filename)
        Method to split free particles into solid and fluid

    setup_hardpoints(pa_solid_nodes, pa_fluid, pa_solid)
        Sets a free particle near to a hard point on the boundary node

    _project_particles_to_boundary(pa_fluid, pa_solid)
        Method to project free particles to surface

    remove_duplicates(neartag, ids)
        Remove duplicates from the near tagged free particles list

    freeze_particles(pa_fluid, pa_frozen)
        freezes particles outside the filtered layers

    _check(particles, pa_fluid)
        Checks for convergence and decreases projection frequency

    post_step(particles, solver)
        methods runs post every iteration
    """
    def __init__(self, fluids, solids, frozen, dim, hdx=1.2,
                 dx=0.1, nu=None, pb=None, k=None, dfreq=-1,
                 hardpoints=None, use_prediction=None,
                 filter_layers=None, reduce_dfreq=None, tol=None):
        """Parameters
        ----------

        fluids: list
            list of free particles

        solids: dict
            dict with boundary and boundary nodes pair

        frozen: list
            list of frozen particles arrays

        dim: int
            dimension of the problem

        hdx: double
            h/dx ratio

        dx: double
            expected resolution

        nu: double
            damping constant

        pb: double
            background pressure

        k: double
            repulsion force constant

        dfreq: int
            projection frequency

        rho0: double
            expected density

        hardpoints: dict
            dictionary of hard point index and correspinding normal

        use_prediction: bool
            if true use prediction to projection first set of particles

        filter_layers: bool
            if True filter layers near to the surface to be packed

        reduce_dfreq: bool
            if True the projection frequency is reduced when projection stops

        tol : float
            tolerance for the convergence criteria
        """
        import numpy as np
        self.fluids = fluids
        self.solids = solids
        self.frozen = frozen
        self.solver = None
        self.dx = dx
        self.hdx = hdx
        self.dim = dim
        self.nu = nu
        self.pb = pb
        self.k = k
        self.hardpoints = {} if hardpoints is None else hardpoints
        self.rho0 = 1.0
        self.cutoff = 0.95
        self.nu_max = None
        self.dfreq = dfreq
        self.surface_points = 0
        self.do_check = False
        self.use_prediction = True if use_prediction is None\
            else use_prediction
        self.filter_layers = True if filter_layers is None\
            else filter_layers
        self.reduce_dfreq = True if reduce_dfreq is None\
            else reduce_dfreq
        self.converge = []
        self.blank = []
        self.divs = 10
        self.tol = tol

    def add_user_options(self, group):
        from pysph.sph.scheme import add_bool_argument

        group.add_argument("--dfreq", action="store", type=int, dest="dfreq",
                           default=None, help="particle deletion frequency.")

        group.add_argument("--pb", action="store", type=float,
                           dest="pb", default=None, help="Background pressure")

        group.add_argument("--nu", action="store", type=float,
                           dest="nu", default=None,
                           help="Dynamic viscosity")

        group.add_argument("--k", action="store", type=float,
                           dest="k", default=None,
                           help="Spring Constant")

        group.add_argument("--dx", action="store", type=float,
                           dest="dx", default=None,
                           help="Set particle spacing value")

        group.add_argument("--tol", action="store", type=float,
                           dest="tol", default=None,
                           help="tolerance for convergence")

        add_bool_argument(group, 'use-prediction',
                          dest='use_prediction',
                          help='use predicted number of points',
                          default=None)

        add_bool_argument(group, 'filter-layers',
                          dest='filter_layers',
                          help='use layered arrangement for packing',
                          default=None)

        add_bool_argument(group, 'reduce-dfreq',
                          dest='reduce_dfreq',
                          help='reduce update frequency for faster conv',
                          default=None)

    def consume_user_options(self, options):
        vars = [
            'dfreq', 'pb', 'nu', 'k', 'dx', 'use_prediction',
            'filter_layers', 'reduce_dfreq', 'tol'
        ]

        data = dict((var, self._smart_getattr(options, var)) for var in vars)
        self.configure(**data)

        dx = self.dx
        if self.pb is None:
            self.pb = 1.0

        if self.nu is None:
            if self.dim == 2:
                self.nu = .2/dx
            elif self.dim == 3:
                self.nu = 0.5/dx

        if self.k is None:
            if self.dim == 2:
                self.k = 0.004 * dx
            elif self.dim == 3:
                self.k = 0.006 * dx

        print(self.pb, self.k, self.nu)

        if self.dfreq < 0:
            self.dfreq = 50

        if self.tol is None:
            self.tol = 1e-2

    def create_frozen_container(self,
                                bound,
                                layers=5,
                                name='frozen',
                                outer=False,
                                rect=False):
        if outer:
            return create_frozen_container_outer(self.dx,
                                                 self.hdx,
                                                 1.0,
                                                 bound,
                                                 layers=layers,
                                                 dim=self.dim,
                                                 name=name)
        elif rect:
            return create_frozen_container_rect(self.dx,
                                                self.hdx,
                                                1.0,
                                                bound,
                                                layers=layers,
                                                dim=self.dim,
                                                name=name)
        else:
            return create_frozen_container(self.dx,
                                           self.hdx,
                                           1.0,
                                           bound,
                                           layers=layers,
                                           dim=self.dim,
                                           name=name)

    def create_free_particles(self,
                              bound,
                              layers=5,
                              name='free',
                              outer=False,
                              rect=False):
        if outer:
            return create_free_particles_outer(self.dx,
                                               self.hdx,
                                               1.0,
                                               bound,
                                               layers=layers,
                                               dim=self.dim,
                                               name=name)
        elif rect:
            return create_free_particles_rect(self.dx,
                                              self.hdx,
                                              1.0,
                                              bound,
                                              dim=self.dim,
                                              name=name)
        else:
            return create_free_particles(self.dx,
                                         self.hdx,
                                         1.0,
                                         bound,
                                         dim=self.dim,
                                         name=name)

    def create_boundary_node(self,
                             filename,
                             points=None,
                             scale=1.0,
                             shift=True,
                             invert=False,
                             name='solid_nodes',
                             isclosed=True):
        if self.dim == 2:
            return create_surface_from_file(filename,
                                            points,
                                            self.dx,
                                            self.hdx,
                                            1.0,
                                            isclosed,
                                            invert=invert,
                                            shift=shift,
                                            name=name,
                                            hard=self.hardpoints)
        elif self.dim == 3:
            return create_surface_from_stl(filename,
                                           self.dx,
                                           self.hdx,
                                           1.0,
                                           scale=scale,
                                           shift=shift,
                                           name=name,
                                           hard=self.hardpoints,
                                           invert=invert)

    def configure_solver(self, kernel=None, integrator_cls=None,
                         extra_steppers=None, **kw):
        from pysph.base.kernels import QuinticSpline
        if kernel is None:
            kernel = QuinticSpline(dim=self.dim)

        steppers = {}
        if extra_steppers is not None:
            steppers.update(extra_steppers)

        from pysph.sph.integrator import EulerIntegrator

        cls = EulerIntegrator
        for name in self.fluids:
            if name not in steppers:
                steppers[name] = InteriorStep()

        for name in self.solids:
            if name not in steppers:
                steppers[name] = SolidStep()

        integrator = cls(**steppers)

        from pysph.solver.solver import Solver
        self.solver = Solver(
            dim=self.dim, integrator=integrator, kernel=kernel, n_damp=10,
            adaptive_timestep=True, pfreq=3000, tf=200, max_steps=40000, **kw)

    def get_equations(self):

        all = self.fluids + list(self.solids.keys()) + self.frozen
        equations = []

        g1 = []
        for name in self.fluids:
            g1.append(
                FindNearestNode(dest=name, sources=list(self.solids.values())))
        for name in self.solids:
            g1.append(FindNearestNode(dest=name, sources=[self.solids[name]]))
            if self.filter_layers:
                g1.append(
                    FilterLayers(dest=self.solids[name], sources=self.fluids))
            if self.hardpoints:
                g1.append(
                    FindNearestNodeToHardPoint(dest=self.solids[name],
                                               sources=self.fluids))
        if len(self.solids.keys()) > 0:
            equations.append(Group(equations=g1, real=False))

        g2 = []
        for name in self.solids:
            g2.append(
                ProjectionToSurfaceBoundary(dest=name,
                                            sources=[self.solids[name]]))
        for name in self.fluids:
            if len(self.solids.keys()) > 0:
                g2.append(
                    ProjectionToSurfaceBoundary(dest=name,
                                                sources=list(
                                                    self.solids.values())))
        for name in self.solids:
            g2.append(
                FindBoundaryNodeDirection(dest=name,
                                          sources=[self.solids[name]]))
        for name in all:
            g2.append(SummationDensity(dest=name, sources=all))
        equations.append(Group(equations=g2, real=False))

        g3 = []
        dest = self.fluids + list(self.solids.keys())
        for name in dest:
            g3.append(NumberDensityGradient(dest=name, sources=all))
            if self.nu > 1e-14:
                g3.append(ViscousDamping(dest=name, sources=all))
        for name in self.fluids:
            g3.append(
                RepulsionForce(dest=name,
                               sources=all,
                               hdx=self.cutoff / self.hdx))
        source = self.fluids + self.frozen
        for name in self.solids:
            g3.append(
                RepulsionForce(dest=name,
                               sources=source,
                               hdx=self.cutoff / self.hdx))
        for name in self.solids:
            g3.append(
                RepulsionForce(dest=name,
                               sources=[name],
                               hdx=self.cutoff / self.hdx))
        equations.append(Group(equations=g3, real=False))

        g4 = []
        dest = self.fluids + list(self.solids.keys())
        for name in dest:
            g4.append(EvaluateAdaptiveTime(dest=name, sources=[]))
        equations.append(Group(equations=g4, real=False))

        print(equations)
        return equations

    def setup_properties(self, particles, clean=True):
        """Add following properties to the particle array

        x, y, z: coordinate of the position
        u, v, w: components of velocity
        p: pressure
        V: Volume
        h: support radius
        m: mass
        rho: density
        au, av, aw: components of acceleration
        dt_adapt: adaptive time step
        xr, yr, zr: unit direction of boundary particle motion
        x0, y0, z0: position at previous time step
        area: area of the boundary nodes
        hard: 1 if the particle is hard pint
        xc, yc, zc: centroid of boundary nodes
        xn, yn, zn: normal of boundary nodes
        dmin: minimum distance value
        nearest: distance of nearest particle
        neartag: id of nearest particle
        filter: 1 if the particle is near boundary node
        nu: damping constant
        pb: background pressure
        k: repulsion force constant
        """
        props = [
            'x', 'y', 'z', 'u', 'v', 'w', 'p', 'V', 'h', 'm', 'rho', 'au',
            'av', 'aw', 'dt_adapt', 'xr', 'yr', 'zr', 'x0', 'y0', 'z0',
            'area', 'hard', 'xc', 'yc', 'zc',
            'xn', 'yn', 'zn', 'dmin', 'nearest'
        ]
        output_props = [
            'x', 'y', 'z', 'u', 'v', 'w', 'V', 'rho', 'xn', 'yn', 'zn', 'm',
            'au', 'av', 'aw', 'h', 'xc', 'yc', 'zc', 'neartag', 'nearest',
            'filter', 'hard'
        ]

        newarr = []
        for pa in particles:
            prop_to_ensure = props.copy()
            self._ensure_properties(pa, prop_to_ensure, clean=False)
            pa.add_property('neartag', type='int')
            pa.add_property('filter', type='int')
            pa.add_constant('nu', self.nu)
            pa.add_constant('pb', self.pb)
            pa.add_constant('k', self.k)
            pa.set_output_arrays(output_props)
        particles.extend(newarr)

    def _is_volume_converged(self, pa):
        import numpy as np
        u = pa.u
        v = pa.v
        w = pa.w
        h = pa.h[0]
        vel = np.sqrt(u**2 + v**2 + w**2)
        maxvel = max(vel)
        rel_dist = maxvel * self.solver.dt / h * 100
        self.converge.append([rel_dist, self.solver.t])
        print('\nConvergence = ', rel_dist)
        if (rel_dist - self.tol < 1e-14) and (len(self.converge) > 10):
            self.solver.tf = self.solver.t

    def post_process(self, free, solid, solid_nodes, frozen, dx, filename):
        import numpy as np
        free_n = free.name

        if solid is None:
            import os
            xs, ys, zs, xf, yf, zf = None, None, None, None, None, None
            if os.path.exists(filename):
                data = np.load(filename)
                xs = data['xs']
                ys = data['ys']
                zs = data['zs']
                xf = data['xf']
                yf = data['yf']
                zf = data['zf']
                xf = np.concatenate((xf, free.x))
                yf = np.concatenate((yf, free.y))
                zf = np.concatenate((zf, free.z))
            else:
                xf = free.x
                yf = free.y
                zf = free.z
                xs = []
                ys = []
                zs = []
            return np.savez(filename, xs=xs, ys=ys, zs=zs, xf=xf, yf=yf, zf=zf)

        solid_nodes_n = solid_nodes.name
        frozen_n = frozen.name

        xb, yb, zb = None, None, None
        xi, yi, zi = None, None, None
        xf, yf, zf = None, None, None

        from pysph.tools.sph_evaluator import SPHEvaluator
        from pysph.sph.equation import Group

        arrays = []
        for pa in [free, solid_nodes, frozen]:
            arrays.append(pa)
            pa.add_property('interior', type='int')

        eqns = [
            Group(equations=[
                FindNearestNode(dest=free_n, sources=[solid_nodes_n], fac=4.0)
            ]),
            Group(equations=[
                FindExternalParticles(dest=free_n, sources=[solid_nodes_n])
            ]),
            Group(equations=[
                FindNearNodes(dest=free_n, sources=[free_n, frozen_n]),
                FindNearNodes(dest=frozen_n, sources=[free_n, frozen_n])
            ],
                  min_iterations=4, max_iterations=5, iterate=True)
        ]
        print(eqns)

        spheval = SPHEvaluator(
            arrays, eqns, kernel=self.solver.kernel,
            dim=self.dim, backend='cython')
        spheval.evaluate()

        isinterior = free.interior
        external = isinterior == 1
        internal = isinterior == -1
        xi = free.x[external]
        yi = free.y[external]
        zi = free.z[external]

        xf = free.x[internal]
        yf = free.y[internal]
        zf = free.z[internal]

        isinterior = frozen.interior
        filter = frozen.filter
        external = (isinterior == 1) & (filter < 0.5)
        internal = (isinterior == -1) & (filter < 0.5)
        f_xi = frozen.x[external]
        f_yi = frozen.y[external]
        f_zi = frozen.z[external]

        f_xf = frozen.x[internal]
        f_yf = frozen.y[internal]
        f_zf = frozen.z[internal]

        xb = solid.x
        yb = solid.y
        zb = solid.z

        xfluid = np.concatenate((xi, f_xi))
        yfluid = np.concatenate((yi, f_yi))
        zfluid = np.concatenate((zi, f_zi))

        xsolid = np.concatenate((xf, xb, f_xf))
        ysolid = np.concatenate((yf, yb, f_yf))
        zsolid = np.concatenate((zf, zb, f_zf))

        np.savez(filename, xs=xsolid, ys=ysolid, zs=zsolid,
                 xf=xfluid, yf=yfluid, zf=zfluid)

    def setup_hardpoints(self, pa_solid_nodes, pa_fluid, pa_solid):
        if self.solver.t < 1e-14:
            # replace nearest point from hard pint
            import numpy as np
            xh = pa_solid_nodes.x
            yh = pa_solid_nodes.y
            zh = pa_solid_nodes.z
            neartag = pa_solid_nodes.neartag
            h = pa_solid_nodes.h[0]
            m = pa_solid_nodes.m[0]
            rho = pa_solid_nodes.rho[0]

            x = []
            y = []
            z = []
            xn = []
            yn = []
            zn = []
            ids = []

            for id in list(self.hardpoints.keys()):
                x.append(xh[id])
                y.append(yh[id])
                z.append(zh[id])
                normal = self.hardpoints[id]
                xn.append(normal[0])
                yn.append(normal[1])
                zn.append(normal[2])
                ids.append(neartag[id])

            pa_fluid.remove_particles(ids)

            pa_add = get_particle_array(
                x=x, y=y, z=z, xn=xn, yn=yn, zn=zn, m=m, h=h, rho=rho,
                hard=1.0)
            pa_solid.add_particles(**pa_add.get_property_arrays())

    def _project_particles_to_boundary(self, pa_fluid, pa_solid):

        import numpy as np
        N = len(pa_solid.x)
        proj = pa_fluid.nearest
        neartag = pa_fluid.neartag
        h = pa_fluid.h[0]
        ids = None
        if ((self.solver.count == 0 and self.use_prediction)
                or (self.solver.count == self.dfreq and self.hardpoints
                    and self.use_prediction)):
            ids = np.argsort(abs(proj))
            n_pnts = int(0.90 * self.surface_points)
            ids = ids[0:n_pnts]
        else:
            num = self.divs
            if self.do_check:
                if len(self.blank) > 10:
                    if len(self.blank) == 11:
                        self.dfreq = 30
                    self.divs = 13
            for i in range(num):
                dist = 0.05 * float(i + 1) * h / self.hdx
                ids = np.where(abs(proj) - dist < 1e-14)[0]
                if len(ids) > 0.1 * self.surface_points:
                    ids = np.argsort(abs(proj))
                    num = int(0.1 * self.surface_points + 0.5)
                    ids = ids[0:num]
                if len(ids) > 0 and self.do_check and i > 9:
                    num = int(0.1 * (self.surface_points - N))
                    if num < 2:
                        num = 1
                    filter = 2
                    if num < len(ids):
                        filter = int(len(ids) / num)
                    ids = ids[0::filter]
                if len(ids) > 0:
                    break

        if len(ids) > 0:
            self.blank = []
        else:
            self.blank.append(0)
        uids = self.remove_duplicates(neartag[ids], ids)
        pa_fluid.extract_particles(uids, pa_solid)
        pa_fluid.remove_particles(uids)

        hard = pa_solid.hard
        cond = hard < 0.5
        proj = pa_solid.nearest
        xn = pa_solid.xn
        yn = pa_solid.yn
        zn = pa_solid.zn
        pa_solid.x[cond] -= proj[cond] * xn[cond]
        pa_solid.y[cond] -= proj[cond] * yn[cond]
        pa_solid.z[cond] -= proj[cond] * zn[cond]

    def remove_duplicates(self, neartag, ids):
        if len(ids) > 0:
            unique_ids = []
            new_tag = []
            for i in range(len(ids)):
                id = ids[i]
                tag = neartag[i]
                if tag not in new_tag:
                    new_tag.append(tag)
                    unique_ids.append(id)

            return unique_ids
        else:
            return ids

    def freeze_particles(self, pa_fluid, pa_frozen):
        import numpy as np

        # remove fluids
        filter = pa_fluid.filter
        ids = np.where(filter < 0.5)[0]
        pa_fluid.extract_particles(ids, pa_frozen)
        pa_fluid.remove_particles(ids)

        # remove it from adaptive time calc
        pa_frozen.dt_adapt[:] = np.inf

    def _check(self, particles, pa_fluid):
        import numpy as np
        if len(self.blank) > 11:
            self._is_volume_converged(pa_fluid)
            if self.dfreq > 2 and self.reduce_dfreq:
                self.dfreq = self.dfreq - 1

    def post_step(self, particles, solver):
        import numpy as np
        lookup = list(self.solids.values())
        pa_fluid = None
        pa_solid = None
        pa_frozen = None
        pa_solid_nodes = None
        dfreq = self.dfreq
        t = solver.t
        if solver.count % (dfreq) == 0:
            for name in self.solids:
                for pa in particles:
                    if name == pa.name:
                        pa_solid = pa
                    elif self.solids[name] == pa.name:
                        pa_solid_nodes = pa
                        self.surface_points = int(pa.n_pnts[0])
                        print('\npredicted surface pnts', self.surface_points)
                    elif self.fluids[0] == pa.name:
                        pa_fluid = pa
                    elif self.frozen[0] == pa.name:
                        pa_frozen = pa

                if t < 1e-14:
                    pa_frozen.filter[:] = 1  # done to remerge the fluid back
                if self.hardpoints:
                    self.setup_hardpoints(pa_solid_nodes, pa_fluid, pa_solid)
                if self.filter_layers and t < 1e-14:
                    self.freeze_particles(pa_fluid, pa_frozen)

                N = len(pa_solid.x)
                print('no of solids particles', N)
                if N > 1.5 * self.surface_points:
                    print(
                        'boundary attracting too many particles - increase'
                        'viscosity'
                    )
                    import sys
                    sys.exit()

                if self.hardpoints and solver.count == 0:
                    return
                self._project_particles_to_boundary(pa_fluid, pa_solid)

                if self.filter_layers:
                    self.do_check = True

                if len(self.blank) == 5:
                    self.dfreq = 5
                    self.do_check = True

                if (self.do_check):
                    self._check(particles, pa_fluid)

            if len(self.solids.keys()) == 0:
                for pa in particles:
                    if self.fluids[0] == pa.name:
                        pa_fluid = pa
                self._is_volume_converged(pa_fluid)
