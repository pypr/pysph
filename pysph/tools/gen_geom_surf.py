'''
Functions can be used to generate points to describe a given input
mesh.

Supported mesh formats: All file formats supported by meshio.
(https://github.com/nschloe/meshio)
'''

import numpy as np
import meshio
from pysph.tools.geom_surf_points import get_surface_points,\
                             get_surface_points_uniform


class Mesh:
    def __init__(self, file_name, file_type=None):
        if file_type is None:
            self.mesh = meshio.read(file_name)
        else:
            self.mesh = meshio.read(file_name, file_type)

        self.cells = np.array([], dtype=int).reshape(0, 3)

    def extract_connectivity_info(self):
        cell_blocks = self.mesh.cells
        for block in cell_blocks:
            self.cells = np.concatenate((self.cells, block.data))

        return self.cells

    def extract_coordinates(self):
        x, y, z = self.mesh.points.T
        self.x, self.y, self.z = x, y, z

        return x, y, z

    def compute_normals(self):
        n = self.cells.shape[0]
        self.normals = np.zeros((n, 3))
        points = self.mesh.points

        for i in range(n):
            idx = self.cells[i]
            pts = np.array([points[idx[0]],
                            points[idx[1]],
                            points[idx[2]]])

            normals = np.cross(pts[1] - pts[0], pts[2] - pts[0])
            nrm = np.linalg.norm(normals)
            self.normals[i] = normals/nrm

        return self.normals


def gen_surf_points(file_name, dx):
    '''
    Generates points with a spacing dx to describe the surface of the
    input mesh file.

    Supported file formats: Refer to https://github.com/nschloe/meshio

    Only works with triangle meshes.
    Parameters
    ----------
    file_name : string
        Mesh file name
    dx : float
        Required spacing between generated particles
    Returns
    -------
    xf, yf, zf : ndarray
        1d numpy arrays with x, y, z coordinates of covered surface
    '''
    mesh = Mesh(file_name)
    cells = mesh.extract_connectivity_info()
    x, y, z = mesh.extract_coordinates()

    xf, yf, zf = get_surface_points(x, y, z, cells, dx)
    return xf, yf, zf


def gen_surf_points_uniform(file_name, dx_sph, h_sph,
                            radius_scale=1.0, dx_triangle=None,
                            file_format=None):
    '''
    Generates points on a grid of spacing dx to descibe the input mesh file
    Supported file formats: Refer to https://github.com/nschloe/meshio

    Only works with triangle meshes.
    Parameters
    ----------
    file_name : string
        Mesh file name
    dx_sph : float
        Grid spacing
    h_sph : float
        Smoothing length
    radius_scale : float, optional
        Kernel radius scale
    dx_triangle : float, optional
        By default, dx_triangle = 0.5 * dx_sph
    file_format : str
        Mesh file format
    Returns
    -------
    xf, yf, zf : ndarray
        1d numpy arrays with x, y, z coordinates of covered surface grid
    '''
    mesh = Mesh(file_name)
    cells = mesh.extract_connectivity_info()
    x, y, z = mesh.extract_coordinates()

    if file_format is 'stl':
        normals = mesh.mesh.cell_data['facet_normals'][0]
    else:
        normals = mesh.compute_normals()

    xf, yf, zf = get_surface_points_uniform(x, y, z, cells, normals,
                                            dx_sph, h_sph, radius_scale,
                                            dx_triangle)
    return xf, yf, zf
