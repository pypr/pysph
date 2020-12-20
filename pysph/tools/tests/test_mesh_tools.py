import numpy as np
import unittest
import pytest
from pysph.base.particle_array import ParticleArray
import pysph.tools.mesh_tools as G
from pysph.base.utils import get_particle_array


# Data of a unit length cube
def cube_data():
    points = np.array([[0., 0., 0.],
                       [0., 1., 0.],
                       [1., 1., 0.],
                       [1., 0., 0.],
                       [0., 0., 1.],
                       [0., 1., 1.],
                       [1., 0., 1.],
                       [1., 1., 1.]])

    x_cube, y_cube, z_cube = points.T

    cells = np.array([[0, 1, 2],
                      [0, 2, 3],
                      [0, 4, 5],
                      [0, 5, 1],
                      [0, 3, 6],
                      [0, 6, 4],
                      [4, 6, 7],
                      [4, 7, 5],
                      [3, 2, 7],
                      [3, 7, 6],
                      [1, 5, 7],
                      [1, 7, 2]])

    normals = np.array([[0.,  0., -1.],
                        [0.,  0., -1.],
                        [-1.,  0.,  0.],
                        [-1.,  0.,  0.],
                        [0., -1.,  0.],
                        [0., -1.,  0.],
                        [0.,  0.,  1.],
                        [0.,  0.,  1.],
                        [1.,  0.,  0.],
                        [1.,  0.,  0.],
                        [0.,  1.,  0.],
                        [0.,  1.,  0.]])

    vectors = np.zeros((len(cells), 3, 3))
    for i, cell in enumerate(cells):
        idx1, idx2, idx3 = cell

        vector = np.array([[x_cube[idx1], y_cube[idx1], z_cube[idx1]],
                           [x_cube[idx2], y_cube[idx2], z_cube[idx2]],
                           [x_cube[idx3], y_cube[idx3], z_cube[idx3]]])
        vectors[i] = vector

    return x_cube, y_cube, z_cube, cells, normals, vectors


class TestGeometry(unittest.TestCase):
    def test_in_triangle(self):
        assert(G._in_triangle(0.5, 0.5, 0.0, 0.0, 1.5, 0.0, 0.0, 1.5) is True)
        assert(G._in_triangle(1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0) is False)

    def test_interp_2d(self):
        # Check interpolation between two points on line y=x
        dx = 0.1
        r = G._interp_2d(np.array([0., 0.]), np.array([1., 1.]), dx)
        # Check if all points satisfy y=x
        np.testing.assert_array_almost_equal(
            r[:, 0] - r[:, 1], np.zeros(r.shape[0]))
        # Check if distance between consecutive points is lesser than dx
        np.testing.assert_array_less(np.linalg.norm(r[1:] - r[0:-1], axis=1),
                                     np.ones(r.shape[0] - 1) * dx)

    def test_fill_triangle(self):
        triangle = np.array([[0., 0., 0.],
                             [1., 0., 0.],
                             [0., 1., 0.]])
        dx_triangle = 0.1
        x, y, z = G._fill_triangle(triangle, dx_triangle)
        EPS = np.finfo(float).eps
        np.testing.assert_array_less(-x, np.zeros(x.shape[0]) + EPS)
        np.testing.assert_array_less(-y, np.zeros(x.shape[0]) + EPS)
        np.testing.assert_array_less(-(x + y), np.ones(x.shape[0]) + EPS)
        np.testing.assert_almost_equal(z, np.zeros(x.shape[0]))

    def test_fill_triangle_throws_zero_area_triangle_exception(self):
        self.assertRaises(G.ZeroAreaTriangleException, G._fill_triangle,
                          np.zeros((3, 3)), 0.5)

    def test_fill_triangle_throws_polygon_mesh_error(self):
        self.assertRaises(G.PolygonMeshError, G._fill_triangle,
                          np.zeros((4, 3)), 0.5)

    def test_get_points_from_mgrid(self):
        """Find neighbouring particles around a unit cube"""
        h = 0.1
        x_cube, y_cube, z_cube, cells, normals, vectors = cube_data()

        x, y, z, x_list, y_list, z_list, vectors = \
        G._get_surface_mesh(x_cube, y_cube, z_cube, cells, h, uniform=True)
        pa_mesh = ParticleArray(name='mesh', x=x, y=y, z=z, h=h)
        offset = h
        x_grid, y_grid, z_grid = np.meshgrid(
            np.arange(x.min() - offset, x.max() + offset, h),
            np.arange(y.min() - offset, y.max() + offset, h),
            np.arange(z.min() - offset, z.max() + offset, h))
        pa_grid = ParticleArray(name='grid', x=x_grid, y=y_grid, z=z_grid, h=h)
        x_grid, y_grid, z_grid = G.get_points_from_mgrid(
            pa_grid, pa_mesh, x_list, y_list, z_list, 1, h, vectors, normals
        )

        for i in range(x.shape[0]):
            assert((x[i] ** 2 + y[i] ** 2 + z[i] ** 2) <= 4)

    def _cube_assert(self, x, y, z, h):
        """Check if x,y,z lie within surface of thickness `h` of a unit cube"""
        def surface1(x, y, z): return min(abs(x), abs(1 - x)) < h and \
            y > -h and y < 1 + h and z > -h and z < 1 + h

        def on_surface(x, y, z): return surface1(x, y, z) or \
            surface1(y, x, z) or surface1(z, x, y)

        for i in range(x.shape[0]):
            assert on_surface(x[i], y[i], z[i])

    def test_get_surface_mesh(self):
        """Check if mesh is generated correctly for unit cube"""

        x_cube, y_cube, z_cube, cells, normals, vectors = cube_data()
        x, y, z = G._get_surface_mesh(x_cube, y_cube, z_cube, cells, 0.1)
        h = np.finfo(float).eps
        self._cube_assert(x, y, z, h)

    def test_get_surface_points(self):
        """Check if surface is generated correctly for unit cube"""
        h = 0.1
        x_cube, y_cube, z_cube, cells, normals, vectors = cube_data()
        x, y, z = G.surface_points(x_cube, y_cube, z_cube, cells, h)
        self._cube_assert(x, y, z, h)

    def test_get_surface_points_uniform(self):
        """Check if uniform surface is generated correctly for unit cube"""
        h = 0.1
        x_cube, y_cube, z_cube, cells, normals, vectors = cube_data()
        x, y, z = G.surf_points_uniform(x_cube, y_cube, z_cube,
                                        cells, normals, 1.0, 1.0)
        self._cube_assert(x, y, z, h)

    def test_prism(self):
        tri_normal = np.array([0, -1, 0])
        tri_points = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1]])
        h = 1/1.5
        prism_normals, prism_points, prism_face_centres = \
            G.prism(tri_normal, tri_points, h)
        assert np.array([-1, 0, 0]) in prism_normals
        assert np.array([0, 1, 0]) in prism_points
        assert np.array([0.5, 0.5, 0]) in prism_face_centres


if __name__ == "__main__":
    unittest.main()
