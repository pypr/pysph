'''
Functions can be used to generate points to describe a given input
mesh.

Supported mesh formats: All file formats supported by meshio.
(https://github.com/nschloe/meshio)
'''

import numpy as np
import meshio
from pysph.tools.mesh_tools import surface_points, surf_points_uniform


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

        idx = self.cells
        pts = points[idx]
        a = pts[:, 1] - pts[:, 0]
        b = pts[:, 2] - pts[:, 0]

        normals = np.cross(a, b)
        mag = np.linalg.norm(normals, axis=1)
        mag.shape = (n, 1)
        self.normals = normals/mag

        return self.normals


def mesh2points(file_name, dx, file_format=None, uniform=False):
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
    file_format : str
        Mesh file format
    uniform : bool
        If True generates points on a grid of spacing dx

    Returns
    -------
    xf, yf, zf : ndarray
        1d numpy arrays with x, y, z coordinates of covered surface
    '''
    mesh = Mesh(file_name)
    cells = mesh.extract_connectivity_info()
    x, y, z = mesh.extract_coordinates()

    if uniform is False:
        xf, yf, zf = surface_points(x, y, z, cells, dx)

    else:
        if file_format is 'stl':
            normals = mesh.mesh.cell_data['facet_normals'][0]
        else:
            normals = mesh.compute_normals()

        xf, yf, zf = surf_points_uniform(x, y, z, cells, normals, dx, dx)

    return xf, yf, zf
