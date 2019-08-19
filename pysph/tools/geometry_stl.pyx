from pysph.base.particle_array import ParticleArray
from pysph.base import nnps
import numpy as np
from stl import mesh
from numpy.linalg import norm
from cyarray.api import UIntArray
cimport cython
cimport numpy as np

DTYPE = np.float
ctypedef np.float_t DTYPE_t


class ZeroAreaTriangleException(Exception):
    pass


class PolygonMeshError(ValueError):
    pass


cpdef _point_sign(double x1, double y1, double x2,
                double y2, double x3, double y3):
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

    # Final mesh coordinates generated from parameters s and t

    result = np.dot(s_mesh.reshape(-1, 1), a.reshape(-1, 1).T) + \
        np.dot(t_mesh.reshape(-1, 1), b.reshape(-1, 1).T) + p
    return result[:, 0], result[:, 1], result[:, 2]


def _get_stl_mesh(stl_fname, dx_triangle, uniform = False):
    """Interpolates points within triangles in the stl file"""
    m = mesh.Mesh.from_file(stl_fname)
    x_list, y_list, z_list = [], [], []
    for i in range(len(m.vectors)):
        x1, y1, z1 = _fill_triangle(m.vectors[i], dx_triangle)
        x2, y2, z2 = _get_triangle_sides(m.vectors[i], dx_triangle)
        x_list.append(np.r_[x1, x2])
        y_list.append(np.r_[y1, y2])
        z_list.append(np.r_[z1, z2])

    x = np.concatenate(x_list)
    y = np.concatenate(y_list)
    z = np.concatenate(z_list)
    if uniform:
        return x, y, z, x_list, y_list, z_list, m
    else:
        return x, y, z


def remove_repeated_points(x, y, z, dx_triangle):
    EPS = np.finfo(float).eps
    pa_mesh = ParticleArray(name="mesh", x=x, y=y, z=z, h=EPS)
    pa_grid = ParticleArray(name="grid", x=x, y=y, z=z, h=EPS)
    pa_list = [pa_mesh, pa_grid]
    nps = nnps.LinkedListNNPS(dim=3, particles=pa_list, radius_scale=1)
    cdef int src_index = 1, dst_index = 0
    nps.set_context(src_index=1, dst_index=0)
    nbrs = UIntArray()
    cdef list idx = []
    cdef int i = 0
    for i in range(len(x)):
        nps.get_nearest_particles(src_index, dst_index, i, nbrs)
        neighbours = nbrs.get_npy_array()
        idx.append(neighbours.min())
    idx_set = set(idx)
    idx = list(idx_set)
    return pa_mesh.x[idx], pa_mesh.y[idx], pa_mesh.z[idx]


def prism(tri_normal, tri_points, dx_sph):
    """
    Parameters
    ----------
    tri_normal : outward normal of triangle
    tri_points : points forming the triangle
    dx_sph :  grid spacing
    Returns
    -------
    prism_normals : 5X3 array containing the 5 outward normals of the prism
    prism _points : 6X3 array containing the 6 points used to define the prism
    prism_face_centres : 5X3 array containing the centres of the 5 faces
                         of the prism
    """
    cdef np.ndarray prism_normals = np.zeros((5, 3), dtype=DTYPE)
    cdef np.ndarray prism_points = np.zeros((6, 3), dtype=DTYPE)
    cdef np.ndarray prism_face_centres = np.zeros((5, 3), dtype=DTYPE)
    cdef int sign = 1
    cdef int m, n

    # unit normals of the triangular faces of the prism.
    prism_normals[0] = tri_normal / norm(tri_normal)
    prism_normals[4] = -1 * prism_normals[0]

    # distance between triangular faces of prism
    cdef float h = 1.5 * dx_sph

    for m in range(3):
        prism_points[m] = tri_points[m]

    # second triangular face at a distance h from STL triangle.
    for m in range(3):
        for n in range(3):
            prism_points[m+3][n] = tri_points[m][n] - tri_normal[n]*h

    #need to determine if point orientation is clockwise or anticlockwise
    #to determine normals direction.
    normal_tri_cross = np.cross(prism_points[2]-prism_points[0],
                                prism_points[1]-prism_points[0])
    if np.dot(tri_normal, normal_tri_cross) < 0:
        sign = 1  # clockwise
    else:
        sign = -1  # anti-clockwise

    # Normals of the reactangular faces of the prism.
    prism_normals[1] = sign * np.cross(prism_points[1]-prism_points[0],
                                       prism_points[1]-prism_points[4])

    prism_normals[2] = sign * np.cross(prism_points[2]-prism_points[0],
                                       prism_points[5]-prism_points[2])

    prism_normals[3] = sign * np.cross(prism_points[1]-prism_points[2],
                                       prism_points[5]-prism_points[2])

    # centroids of the triangles
    prism_face_centres[0] = (prism_points[0] +
                             prism_points[1] +
                             prism_points[2])/3

    prism_face_centres[4] = (prism_points[3] +
                             prism_points[4] +
                             prism_points[5])/3

    # centres of the rectangular faces
    prism_face_centres[1] = (prism_points[0] + prism_points[3] +
                             prism_points[1] + prism_points[4])/4

    prism_face_centres[2] = (prism_points[0] + prism_points[3] +
                             prism_points[2] + prism_points[5])/4

    prism_face_centres[3] = (prism_points[1] + prism_points[2] +
                             prism_points[4] + prism_points[5])/4

    return prism_normals, prism_points, prism_face_centres


def get_points_from_mgrid(pa_grid, pa_mesh, x_list, y_list, z_list,
                          float radius_scale, float dx_sph, my_mesh):
    """
    The function finds nearest neighbours for a given mesh on a given grid
    and only returns those points which lie within the volume of the stl
    object
    Parameters
    ----------
    pa_grid : Source particle array
    pa_mesh : Destination particle array
    x_list, y_list, z_list : Coordinates of surface points for each triangle
    """
    pa_list = [pa_mesh, pa_grid]
    nps = nnps.LinkedListNNPS(dim=3, particles=pa_list,
                              radius_scale=radius_scale)
    cdef int src_index = 1, dst_index = 0
    nps.set_context(src_index=1, dst_index=0)
    nbrs = UIntArray()
    cdef np.ndarray prism_normals = np.zeros((5, 3), dtype=DTYPE)
    cdef np.ndarray prism_face_centres = np.zeros((5, 3), dtype=DTYPE)
    cdef np.ndarray prism_points = np.zeros((6, 3), dtype=DTYPE)
    cdef list selected_points = []
    cdef int counter = 0, l = 0
    # Iterating over each triangle
    for i in range(np.shape(x_list)[0]):
        prism_normals, prism_points, prism_face_centres = prism(
            my_mesh.normals[i], my_mesh.vectors[i], dx_sph
        )
        # Iterating over surface points in triangle to find nearest
        # neighbour on grid.
        for j in range(len(x_list[i])):
            nps.get_nearest_particles(src_index, dst_index, counter, nbrs)
            neighbours = nbrs.get_npy_array()
            l = len(neighbours)
            for t in range(l):
                point = np.array([pa_grid.x[neighbours[t]],
                                  pa_grid.y[neighbours[t]],
                                  pa_grid.z[neighbours[t]]])
                # determining whether point is within prism.
                if inside_prism(point, prism_normals,
                                prism_points, prism_face_centres):
                    selected_points.append(neighbours[t])
            counter = counter + 1
    idx_set = set(selected_points)
    idx = list(idx_set)
    return pa_grid.x[idx], pa_grid.y[idx], pa_grid.z[idx]


cdef bint inside_prism(double[:] point, double[:,:] prism_normals,
                        double[:,:] prism_points, double[:,:] prism_face_centres):
    """ Identifies whether a point is within the corresponding prism by checking
        if all dot products of the normals of the prism with the vector joining
        the point and a point on the corresponding side is negative
    """
    if dot(prism_normals[0], point, prism_face_centres[0]) > 0:
        return False
    if dot(prism_normals[4], point, prism_face_centres[4]) > 0:
        return False
    if dot(prism_normals[1], point, prism_face_centres[1]) > 0:
        return False
    if dot(prism_normals[2], point, prism_face_centres[2]) > 0:
        return False
    if dot(prism_normals[3], point, prism_face_centres[3]) > 0:
        return False
    return True


cdef double dot(double[:] normal, double[:] point, double[:] face_centre):
    return normal[0]*(point[0]-face_centre[0]) + \
           normal[1]*(point[1]-face_centre[1]) + \
           normal[2]*(point[2]-face_centre[2])

def get_stl_surface_uniform(stl_fname, dx_sph=1, h_sph=1,
                            radius_scale=1.0, dx_triangle=None):
    """Generate points to cover surface described by stl file
    The function generates a grid with a spacing of dx_sph and keeps points
    on the grid which lie within the STL object.

    The algorithm for this is straightforward and consists of the following
    steps:
    1. Interpolate a set of points over the STL triangles
    2. Create a grid that covers the entire STL object
    3. Remove grid points generated outside the given STL object by checking
       if the points lie within prisms formed by the triangles.
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

    x, y, z, x_list, y_list, z_list, my_mesh = \
        _get_stl_mesh(stl_fname, dx_triangle, unifrom = True)
    pa_mesh = ParticleArray(name='mesh', x=x, y=y, z=z, h=h_sph)
    offset = radius_scale * h_sph
    x_grid, y_grid, z_grid = np.meshgrid(
        np.arange(x.min() - offset, x.max() + offset, dx_sph),
        np.arange(y.min() - offset, y.max() + offset, dx_sph),
        np.arange(z.min() - offset, z.max() + offset, dx_sph)
    )

    pa_grid = ParticleArray(name='grid', x=x_grid, y=y_grid, z=z_grid, h=h_sph)
    xf, yf, zf = get_points_from_mgrid(pa_grid, pa_mesh, x_list,
                                       y_list, z_list, radius_scale,
                                       dx_sph, my_mesh)
    return xf, yf, zf


def get_stl_surface(stl_fname, dx_triangle, radius_scale=1.0):
    """ Generate points to cover surface described by stl file
    Returns
    -------
    x : ndarray
        1d numpy array with x coordinates of surface grid
    y : ndarray
        1d numpy array with y coordinates of surface grid
    z : ndarray
        1d numpy array with z coordinates of surface grid
    """

    x, y, z = _get_stl_mesh(stl_fname, dx_triangle, uniform = False)
    xf, yf, zf = remove_repeated_points(x, y, z, dx_triangle)
    return xf, yf, zf
