from pysph.base.particle_array import ParticleArray
from pysph.base import nnps
import numpy as np
from stl import mesh
from numpy.linalg import norm
from pyzoltan.core.carray import UIntArray


class ZeroAreaTriangleException(Exception):
    pass


class PolygonMeshError(ValueError):
    pass


def _point_sign(x1, y1, x2, y2, x3, y3):
    return (x1 - x3) * (y2 - y3) - (x2 - x3) * (y1 - y3)


def _in_triangle(px, py, vx1, vy1, vx2, vy2, vx3, vy3):
    b1 = _point_sign(px, py, vx1, vy1, vx2, vy2) < 0
    b2 = _point_sign(px, py, vx2, vy2, vx3, vy3) < 0
    b3 = _point_sign(px, py, vx3, vy3, vx1, vy1) < 0

    return ((b1 == b2) and (b2 == b3))


def _interp_2d(v0, v1, dx):
    l = norm(v0 - v1)
    p = np.linspace(0., 1., int(l / dx + 2)).reshape(-1, 1)
    return v0.reshape(1, -1) * p + v1.reshape(1, -1) * (1. - p)


def _get_triangle_sides(triangle, dx):
    """Interpolate points on the sides of the triangle"""
    sides = np.vstack([_interp_2d(triangle[0], triangle[1], dx),
                       _interp_2d(triangle[1], triangle[2], dx),
                       _interp_2d(triangle[2], triangle[0], dx)])
    return sides[:, 0], sides[:, 1], sides[:, 2]


def _projection_parameters(x, a, b, p):
    """Parameters of projection of x on surface (sa + tb + p)
    """
    return np.dot(x - p, a), np.dot(x - p, b)


def _fill_triangle(triangle, h=0.1):
    EPS = np.finfo(float).eps
    if triangle.shape[0] != 3:
        raise PolygonMeshError("non-triangular meshes are not supported: "
                               "dim({}) ! = 3.".format(triangle))

    if norm(triangle[0] - triangle[1]) < EPS or \
       norm(triangle[1] - triangle[2]) < EPS or \
       norm(triangle[2] - triangle[0]) < EPS:
        raise ZeroAreaTriangleException(
            "unable to interpolate in zero area triangle: {}".format(triangle))

    # Surface will be of the form v = sa + tb + p
    # p is (arbitrarily) taken as the centroid of the triangle

    p = (triangle[0] + triangle[1] + triangle[2]) / 3
    a = (triangle[1] - triangle[0]) / norm(triangle[1] - triangle[0])
    b = np.cross(np.cross(a, triangle[2] - triangle[1]), a)
    b /= norm(b)
    st = np.array([_projection_parameters(triangle[0], a, b, p),
                   _projection_parameters(triangle[1], a, b, p),
                   _projection_parameters(triangle[2], a, b, p)])

    st_min, st_max = np.min(st, axis=0), np.max(st, axis=0)
    s_mesh, t_mesh = np.meshgrid(np.arange(st_min[0], st_max[0], h),
                                 np.arange(st_min[1], st_max[1], h))

    s_mesh, t_mesh = s_mesh.ravel(), t_mesh.ravel()
    mask = np.empty(len(s_mesh), dtype='bool')

    for i in range(len(s_mesh)):
        mask[i] = _in_triangle(s_mesh[i], t_mesh[i],
                               st[0, 0], st[0, 1],
                               st[1, 0], st[1, 1],
                               st[2, 0], st[2, 1])

    s_mesh, t_mesh = s_mesh[mask], t_mesh[mask]
    st_mesh = np.vstack([s_mesh, t_mesh])

    # Final mesh coordinates generated from parameters s and t

    result = np.dot(s_mesh.reshape(-1, 1), a.reshape(-1, 1).T) + \
        np.dot(t_mesh.reshape(-1, 1), b.reshape(-1, 1).T) + p
    return result[:, 0], result[:, 1], result[:, 2]


def _get_neighbouring_particles(pa_src, pa_dst, radius_scale):
    """
    Parameters
    ----------
    pa_src : Source particle array
    pa_dst : Destination particle array
    """
    pa_list = [pa_dst, pa_src]
    nps = nnps.LinkedListNNPS(dim=3, particles=pa_list,
                              radius_scale=radius_scale)

    nps.set_context(src_index=1, dst_index=0)

    n = len(pa_dst.x)
    nbrs = UIntArray()
    grid_set = set()
    for i in range(n):
        nps.get_nearest_particles(1, 0, i, nbrs)
        neighbours = list(nbrs.get_npy_array())
        for neighbour in neighbours:
            grid_set.add(neighbour)

    idx = list(grid_set)

    return pa_src.x[idx], pa_src.y[idx], pa_src.z[idx]


def _get_stl_mesh(stl_fname, dx_triangle):
    """Interpolates points within triangles in the stl file"""
    m = mesh.Mesh.from_file(stl_fname)
    x_list, y_list, z_list = [], [], []
    for i in range(len(m.vectors)):
        try:
            x1, y1, z1 = _fill_triangle(m.vectors[i], dx_triangle)
            x2, y2, z2 = _get_triangle_sides(m.vectors[i], dx_triangle)
            x_list.append(np.r_[x1, x2])
            y_list.append(np.r_[y1, y2])
            z_list.append(np.r_[z1, z2])
        except ZeroAreaTriangleException as e:
            print(e)
            print("Skipping triangle {}".format(i))
            continue
    x = np.concatenate(x_list)
    y = np.concatenate(y_list)
    z = np.concatenate(z_list)

    return x, y, z


def get_stl_surface(stl_fname, dx_sph, h_sph, radius_scale=1.0,
                    dx_triangle=None):
    """Generate points to cover surface described by stl file

    The function generates a grid with a spacing of dx_sph and keeps points at a
    distance ``radius_scale * h_sph`` around the surface described by the STL
    file.

    The algorithm for this is straightforward and consists of the following
    steps:

    1. Interpolate a set of points over the STL triangles
    2. Create a grid that covers the entire STL object
    3. Remove points more than ``h_sph * radius_scale`` distance from a surface

    Parameters
    ----------

    stl_fname : str
        File name of STL file
    dx_sph : float
        Spacing in generated grid points
    h_sph : float
        Smoothing length
    radius_scale : float
        Kernel radius scale
    dx_triangle : float, optional
        By default, dx_triangle = 0.5 * dx_sph

    Returns
    -------

    x : ndarray
        1d numpy array with x coordinates of surface grid
    y : ndarray
        1d numpy array with y coordinates of surface grid
    z : ndarray
        1d numpy array with z coordinates of surface grid

    Raises
    ------

    PolygonMeshError
        If polygons in STL file are not all triangles
    """
    if dx_triangle is None:
        dx_triangle = 0.5 * dx_sph

    x, y, z = _get_stl_mesh(stl_fname, dx_triangle)
    pa_mesh = ParticleArray(name='mesh', x=x, y=y, z=z, h=h_sph)

    offset = radius_scale * h_sph
    x_grid, y_grid, z_grid = np.meshgrid(
        np.arange(x.min() - offset, x.max() + offset, dx_sph),
        np.arange(y.min() - offset, y.max() + offset, dx_sph),
        np.arange(z.min() - offset, z.max() + offset, dx_sph)
    )

    pa_grid = ParticleArray(name='grid', x=x_grid, y=y_grid, z=z_grid, h=h_sph)

    x_grid, y_grid, z_grid = _get_neighbouring_particles(pa_grid, pa_mesh,
                                                         radius_scale)
    return x_grid, y_grid, z_grid
