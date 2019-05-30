from pysph.base.particle_array import ParticleArray
from pysph.base import nnps
import numpy as np
from stl import mesh
from numpy.linalg import norm
from cyarray.api import UIntArray
import time
from mayavi import mlab


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
    return x, y, z


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
    x = np.concatenate(x_list)
    y = np.concatenate(y_list)
    z = np.concatenate(z_list)
    centroids_x = []
    centroids_y = []
    centroids_z = []
    for i in range(np.shape(m.vectors)[0]):
        centroid = ((m.vectors[i][0]+m.vectors[i][1]+m.vectors[i][2]))/3
        centroids_x.append(centroid[0])
        centroids_y.append(centroid[1])
        centroids_z.append(centroid[2])
    return x, y, z, centroids_x, centroids_y, centroids_z, sizes


def _get_neighbouring_particles(pa_src, pa_dst, radius_scale, sizes):
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
    sizes = np.array(sizes)
    sizes[0] = sizes[0]-1
    for k in range(len(sizes)-1):
        sizes[k+1] = sizes[k]+sizes[k+1]
    n = len(pa_dst.x)
    nbrs = UIntArray()
    grid_set = []
    size_count = 0
    tracker = 0
    new_sizes = []  # contains new no. of points corresponding to each triangle
    mulitple_occur = []  # contains the triangles corresponding to every point.
    # When finding neighbouring particles it is possible that a
    # neighbouring point obatined might be the nearest neighbour
    # of mulitple points corresponding to different triangles.
    for i in range(n):
        nps.get_nearest_particles(1, 0, i, nbrs)
        neighbours = list(nbrs.get_npy_array())
        for neighbour in neighbours:
            if neighbour in grid_set:
                if tracker in mulitple_occur[grid_set.index(neighbour)]:
                    pass
                else:
                    mulitple_occur[grid_set.index(neighbour)].append(tracker)
            else:
                grid_set.append(neighbour)
                mulitple_occur.append(list([tracker]))
                size_count = size_count+1
        if(i == sizes[tracker]):
            tracker = tracker+1
            new_sizes.append(size_count)
    idx = list(grid_set)
    return pa_src.x[idx], pa_src.y[idx], pa_src.z[idx], \
        new_sizes, mulitple_occur


def elemental_volume(point, normal, face, radius_scale, dx_sph, centroid,
                     multiple_occur, centroids_x, centroids_y, centroids_z, u,
                     v, w):
    """ Identifies whether a point is within h thickness of it's
    corresponding side"""
    mag = norm(normal)
    normal = normal/mag
    h = 1.5*radius_scale*dx_sph
    face_1 = np.zeros_like(face)
    for k in range(np.shape(face)[1]):
        for i in range(np.shape(face)[0]):
            face_1[i][k] = face[i][k]-normal[k]*h
    centroid_2 = (face_1[0]+face_1[1]+face_1[2])/3
    for i in multiple_occur:
        centroid = np.array([centroids_x[i], centroids_y[i], centroids_z[i]])
        normal = np.array([u[i], v[i], w[i]])
        if(np.dot(point - centroid, normal) > 0):
            return True

    if(np.dot(point - centroid, normal) <= 0 and np.dot(point - centroid_2,
       normal) > 0):
        return False
    else:
        return True


def remove_exterior(x, y, z, centroids_x, centroids_y, centroids_z, stl_fname,
                    sizes, u, v, w, radius_scale, dx_sph, multiple_occur):
    """ removes points which lie outside the given volume"""
    remove_point = []
    my_mesh = mesh.Mesh.from_file(stl_fname)
    i = 0
    centroid = np.array([centroids_x[i], centroids_y[i], centroids_z[i]])
    normal = np.array([u[i], v[i], w[i]])
    for j in range(len(x)):
        point = np.array([x[j], y[j], z[j]])
        if(elemental_volume(point, normal, my_mesh.vectors[i], radius_scale,
                            dx_sph, centroid, multiple_occur[j], centroids_x,
                            centroids_y, centroids_z, u, v, w)):
                remove_point.append(j)
        if(j == sizes[i]):
            i = i+1
            centroid = np.array([centroids_x[i],
                                centroids_y[i],
                                centroids_z[i]])
            normal = np.array([u[i], v[i], w[i]])
    x = np.delete(x, remove_point)
    y = np.delete(y, remove_point)
    z = np.delete(z, remove_point)
    return x, y, z


def fix_normals(x, y, z, centroids_x, centroids_y, centroids_z, stl_fname):
    """ used to ensure that all normal's are outward normals"""
    my_mesh = mesh.Mesh.from_file(stl_fname)
    u, v, w = [], [], []
    for i in range(np.shape(my_mesh.vectors)[0]):
        u.append(my_mesh.normals[i][0])
        v.append(my_mesh.normals[i][1])
        w.append(my_mesh.normals[i][2])
    fig1 = mlab.figure(bgcolor=(0, 0, 0))
    fig1.scene.disable_render = True
    mlab.points3d(x, y, z, color=(0.1, 0.6, 0.3), opacity=0.3)
    outline = mlab.outline(line_width=3)
    outline.outline_mode = "cornered"
    outline_size = 0.007
    outline.bounds = (centroids_x[0]-outline_size, centroids_x[0]+outline_size,
                      centroids_y[0]-outline_size, centroids_y[0]+outline_size,
                      centroids_z[0]-outline_size, centroids_z[0]+outline_size)
    normals = mlab.quiver3d(centroids_x, centroids_y, centroids_z,
                            u, v, w, figure=fig1)
    centroids = mlab.points3d(centroids_x, centroids_y, centroids_z,
                              color=(0.5, 0.5, 0.5))
    fig1.scene.disable_render = False
    glyph_normals = centroids.glyph.glyph_source.glyph_source.output.points. \
        to_array()

    def picker_callback(picker):
        if picker.actor in centroids.actor.actors:
            centroid_id = picker.point_id//glyph_normals.shape[0]
            if centroid_id != -1:
                x, y, z = centroids_x[centroid_id], centroids_y[centroid_id], \
                    centroids_z[centroid_id]
                u[centroid_id], v[centroid_id], w[centroid_id] = \
                    -1*u[centroid_id], -1*v[centroid_id], -1*w[centroid_id]
                normals.mlab_source.reset(u=u, v=v, w=w)
                outline.bounds = (x-outline_size, x+outline_size,
                                  y-outline_size, y+outline_size,
                                  z-outline_size, z+outline_size)
    picker = fig1.on_mouse_pick(picker_callback)
    picker.tolerance = 0.01
    mlab.title("Click on centroid to invert normal")
    mlab.show()
    return u, v, w


def get_stl_surface_uniform(stl_fname, dx_sph, h_sph, radius_scale=1.0,
                            dx_triangle=None):
    """ Returns coordinates of uniform distribution of
     particles for given stl"""
    if dx_triangle is None:
        dx_triangle = 0.5 * dx_sph

    x, y, z, centroids_x, centroids_y, centroids_z, sizes = \
        _get_stl_mesh_uniform(stl_fname, dx_triangle)
    u, v, w = fix_normals(x, y, z, centroids_x, centroids_y, centroids_z,
                          stl_fname)
    pa_mesh = ParticleArray(name='mesh', x=x, y=y, z=z, h=h_sph)
    offset = radius_scale * h_sph
    x_grid, y_grid, z_grid = np.meshgrid(
        np.arange(x.min() - offset, x.max() + offset, dx_sph),
        np.arange(y.min() - offset, y.max() + offset, dx_sph),
        np.arange(z.min() - offset, z.max() + offset, dx_sph)
    )

    pa_grid = ParticleArray(name='grid', x=x_grid, y=y_grid, z=z_grid, h=h_sph)
    x_grid, y_grid, z_grid, new_sizes, mulitple_occur = \
        _get_neighbouring_particles(pa_grid, pa_mesh, radius_scale, sizes)
    xf, yf, zf = remove_exterior(x_grid, y_grid, z_grid,
                                 centroids_x, centroids_y, centroids_z,
                                 stl_fname, new_sizes,
                                 u, v, w,
                                 radius_scale, dx_sph, mulitple_occur)
    return xf, yf, zf


def get_stl_surface(stl_fname, dx_sph, h_sph, radius_scale=1.0,
                    dx_triangle=None):
    """ Returns coordinates of particle distribution along surface
    of stl file"""
    if dx_triangle is None:
        dx_triangle = 0.5 * dx_sph

    x, y, z = _get_stl_mesh(stl_fname, dx_triangle)
    return x, y, z
