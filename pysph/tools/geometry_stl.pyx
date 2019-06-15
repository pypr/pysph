from pysph.base.particle_array import ParticleArray
from pysph.base import nnps
import numpy as np
from stl import mesh
from numpy.linalg import norm
from cyarray.api import UIntArray
from mayavi import mlab
cimport cython
cimport numpy as np

DTYPE = np.float
DTYPE1 = np.int
ctypedef np.float_t DTYPE_t

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
    # st_mesh = np.vstack([s_mesh, t_mesh])

    # Final mesh coordinates generated from parameters s and t

    result = np.dot(s_mesh.reshape(-1, 1), a.reshape(-1, 1).T) + \
        np.dot(t_mesh.reshape(-1, 1), b.reshape(-1, 1).T) + p
    return result[:, 0], result[:, 1], result[:, 2]


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
    points = []
    for i in range(len(x)):
        point = [x[i], y[i], z[i]]
        if point in points:
            pass
        else:
            points.append([x[i], y[i], z[i]])
    x, y, z = [], [], []
    for i in range(np.shape(points)[0]):
        x.append(points[i][0])
        y.append(points[i][1])
        z.append(points[i][2])
    points = []
    for i in range(len(x)):
        point = [x[i], y[i], z[i]]
        if point in points:
            pass
        else:
            points.append([x[i], y[i], z[i]])
    x, y, z = [], [], []
    for i in range(np.shape(points)[0]):
        x.append(points[i][0])
        y.append(points[i][1])
        z.append(points[i][2])
    return np.array(x), np.array(y), np.array(z)


def _get_stl_mesh_uniform(stl_fname, dx_triangle):
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
    sizes = []  # contains number of points corresponding to each triangle
    for i in range(np.shape(x_list)[0]):
        sizes.append(np.shape(x_list[i])[0])
    sizes = np.array(sizes)
    sizes[0] = sizes[0]-1
    for k in range(len(sizes)-1):
        sizes[k+1] = sizes[k]+sizes[k+1]
    x = np.concatenate(x_list)
    y = np.concatenate(y_list)
    z = np.concatenate(z_list)
    return x, y, z, sizes


def updater(my_mesh, normal_5, points, radius_scale, dx_sph, tracker):
    cdef int m,n
    cdef float mag = norm(my_mesh.normals[tracker])
    normal_5[0] = my_mesh.normals[tracker] / norm(my_mesh.normals[tracker])
    normal_5[4] = -1*normal_5[0]
    cdef float h = 1.5 * radius_scale * dx_sph
    for m in range(3):
        for n in range(3):
            points[n][m] = my_mesh.vectors[tracker][n][m]
    for m in range(3):
        for n in range(3):
            points[m+3][n] = my_mesh.vectors[tracker][m][n] - normal_5[0][n]*h
    normal_5[1] = np.cross(points[1]-points[0], points[1]-points[4])
    normal_5[2] = np.cross(points[2]-points[0], points[5]-points[2] )
    normal_5[3] = np.cross(points[1]-points[2], points[5]-points[2])
    for i in range(1,4):
        normal_5[i] = normal_5[i] / norm(normal_5[i])
    return normal_5, points

def _get_neighbouring_particles(pa_src, pa_dst, float radius_scale,
                                np.ndarray sizes, float dx_sph, stl_fname):
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
    cdef int n = len(pa_dst.x)
    nbrs = UIntArray()
    cdef list grid_set = []
    cdef list remover = []
    cdef int tracker = 0
    my_mesh = mesh.Mesh.from_file(stl_fname)
    cdef int n_trig = np.shape(my_mesh.vectors)[0]
    cdef np.ndarray centroids = np.zeros((n_trig, 3), dtype=DTYPE)
    cdef int i
    cdef np.ndarray normal_5 = np.zeros((5,3), dtype=DTYPE)
    cdef np.ndarray point_2 = np.zeros((6,3), dtype=DTYPE)
    normal_5, point_2 = updater(my_mesh, normal_5, point_2, radius_scale, dx_sph, tracker)
    for i in range(n):
        nps.get_nearest_particles(1, 0, i, nbrs)
        neighbours = nbrs.get_npy_array()
        for neighbour in neighbours:
            point = np.array([pa_src.x[neighbour],
                pa_src.y[neighbour],
                pa_src.z[neighbour]])
            grid_set.append(neighbour)
            if elemental_volume(point, normal_5, point_2,my_mesh, dx_sph, radius_scale,i):
                remover.append(0)
            else:
                remover.append(1)
        if i == sizes[tracker]:
            if not(sizes[tracker] == sizes[-1]):
                tracker = tracker+1
                normal_5, point_2 = updater(my_mesh, normal_5, point_2, radius_scale, dx_sph, tracker)
    x, y, z = [], [], []
    for i in grid_set:
        x.append(pa_src.x[i])
        y.append(pa_src.y[i])
        z.append(pa_src.z[i])
    return x, y, z, remover

def elemental_volume(np.ndarray point, np.ndarray normal_5,np.ndarray points,
                     my_mesh, float dx_sph, float radius_scale, i):
    """ Identifies whether a point is within h thickness of it's
    corresponding side"""
    if np.dot(normal_5[0], point - (points[0]+points[1]+points[2])/3) > 0:
        return False
    elif np.dot(normal_5[4], point - (points[5]+points[4]+points[3])/3) > 0:
        return False
    elif np.dot(normal_5[1], point - (points[0]+points[3]+points[1]+points[4])/4) > 0:
        return False
    elif np.dot(normal_5[2], point - (points[0]+points[3]+points[2]+points[5])/4) > 0:
        return False
    elif np.dot(normal_5[3], point - (points[1]+points[2]+points[4]+points[5])/4) > 0:
        return False

    return True

def remove_exterior(x, y, z, remover):
    """ removes points which lie outside the given volume"""
    remove_point = []
    for i in range(len(x)):
        if remover[i] == 1:
            remove_point.append(i)

    x = np.delete(x, remove_point)
    y = np.delete(y, remove_point)
    z = np.delete(z, remove_point)
    return x, y, z

def remove_repeated(x, y, z):
    cdef int n = len(x)
    cdef list points = []
    for i in range(n):
        point = [x[i], y[i], z[i]]
        points.append(point)
    points_uni = set(tuple(i) for i in points)
    points = list(points_uni)
    cdef list xf =[], yf =[], zf = []
    for i in range(len(points)):
        xf.append(points[i][0])
        yf.append(points[i][1])
        zf.append(points[i][2])
    return xf, yf, zf


def get_stl_surface_uniform(stl_fname="stl_files/bit_yes_up.stl", dx_sph=1, h_sph=1,
                            radius_scale=1.0, dx_triangle=None):
    """ Returns coordinates of uniform distribution of
     particles for given stl"""
    if dx_triangle is None:
        dx_triangle = 0.5 * dx_sph

    x, y, z ,sizes = _get_stl_mesh_uniform(stl_fname, dx_triangle)
    pa_mesh = ParticleArray(name='mesh', x=x, y=y, z=z, h=h_sph)
    offset = radius_scale * h_sph
    x_grid, y_grid, z_grid = np.meshgrid(
        np.arange(x.min() - offset, x.max() + offset, dx_sph),
        np.arange(y.min() - offset, y.max() + offset, dx_sph),
        np.arange(z.min() - offset, z.max() + offset, dx_sph)
    )

    pa_grid = ParticleArray(name='grid', x=x_grid, y=y_grid, z=z_grid, h=h_sph)
    x_grid, y_grid, z_grid, remover = \
        _get_neighbouring_particles(pa_grid, pa_mesh, radius_scale,
                                    sizes, dx_sph, stl_fname)
    xf, yf, zf = remove_exterior(x_grid, y_grid, z_grid, remover)
    xf, yf, zf = remove_repeated(xf, yf, zf)
    return xf, yf, zf


def get_stl_surface(stl_fname, dx_triangle, radius_scale=1.0):
    """ Returns coordinates of particle distribution along surface
    of stl file"""

    x, y, z = _get_stl_mesh(stl_fname, dx_triangle)
    return x, y, z
