import numpy as np
import unittest
import pytest
import tempfile
from pysph.base.particle_array import ParticleArray
from stl import mesh
import pysph.tools.geometry_stl as G
from pysph.base.utils import get_particle_array

pytest.importorskip("stl")

cube_stl = """solid cube
  facet normal 0 0 -1
    outer loop
      vertex 0 0 0
      vertex 0 1 0
      vertex 1 1 0
    endloop
  endfacet
  facet normal 0 0 -1
    outer loop
      vertex 0 0 0
     vertex 1 1 0
      vertex 1 0 0
    endloop
  endfacet
  facet normal -1 0 0
    outer loop
      vertex 0 0 0
      vertex 0 0 1
      vertex 0 1 1
    endloop
  endfacet
  facet normal -1 0 0
    outer loop
      vertex 0 0 0
      vertex 0 1 1
      vertex 0 1 0
    endloop
  endfacet
  facet normal 0 -1 0
    outer loop
      vertex 0 0 0
      vertex 1 0 0
      vertex 1 0 1
    endloop
  endfacet
  facet normal 0 -1 0
    outer loop
      vertex 0 0 0
      vertex 1 0 1
      vertex 0 0 1
    endloop
  endfacet
  facet normal 0 0 1
    outer loop
      vertex 0 0 1
      vertex 1 0 1
      vertex 1 1 1
    endloop
  endfacet
  facet normal 0 0 1
    outer loop
      vertex 0 0 1
      vertex 1 1 1
      vertex 0 1 1
    endloop
  endfacet
  facet normal 1 0 0
    outer loop
      vertex 1 0 0
      vertex 1 1 0
      vertex 1 1 1
    endloop
  endfacet
  facet normal 1 0 0
    outer loop
      vertex 1 0 0
      vertex 1 1 1
      vertex 1 0 1
    endloop
  endfacet
  facet normal 0 1 0
    outer loop
      vertex 0 1 0
      vertex 0 1 1
      vertex 1 1 1
    endloop
  endfacet
  facet normal 0 1 0
    outer loop
      vertex 0 1 0
      vertex 1 1 1
      vertex 1 1 0
    endloop
  endfacet
endsolid cube"""


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

    def _generate_cube_stl(self):
        f = tempfile.NamedTemporaryFile(mode='w', delete=False)
        f.write(cube_stl)
        f.close()
        return f.name

    def test_get_neighbouring_particles(self):
        """Find neighbouring particles around a unit cube"""
        h = 0.1
        cube_fname = self._generate_cube_stl()
        x, y, z, c_x, c_y, c_z, sizes = G._get_stl_mesh_uniform(cube_fname, h)
        pa_mesh = ParticleArray(name='mesh', x=x, y=y, z=z, h=h)
        offset = h
        x_grid, y_grid, z_grid = np.meshgrid(
         np.arange(x.min() - offset, x.max() + offset, h),
         np.arange(y.min() - offset, y.max() + offset, h),
         np.arange(z.min() - offset, z.max() + offset, h))
        pa_grid = ParticleArray(name='grid', x=x_grid, y=y_grid, z=z_grid, h=h)
        x_grid, y_grid, z_grid, new_sizes, mulitple_occur = \
        G._get_neighbouring_particles(pa_grid, pa_mesh, 1, sizes)

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

    def test_get_stl_mesh(self):
        """Check if mesh is generated correctly for unit cube"""
        cube_fname = self._generate_cube_stl()
        x, y, z = G._get_stl_mesh(cube_fname, 0.1)
        h = np.finfo(float).eps
        self._cube_assert(x, y, z, h)

    def test_get_stl_surface(self):
        """Check if stl surface is generated correctly for unit cube"""
        cube_fname = self._generate_cube_stl()
        h = 0.1
        x, y, z = G.get_stl_surface(cube_fname, h, h, 1)
        self._cube_assert(x, y, z, h)

    def test_get_stl_surface_uniform(self):
        """Check if stl surface is generated correctly for unit cube"""
        cube_fname = self._generate_cube_stl()
        h = 0.1
        x, y, z = G.get_stl_surface(cube_fname, h, h, 1)
        self._cube_assert(x, y, z, h)


    def test_remove_exterior(self):
        h = 0.1
        cube_fname = self._generate_cube_stl()
        x, y, z, c_x, c_y, c_z, sizes = G._get_stl_mesh_uniform(cube_fname, h)
        pa_mesh = ParticleArray(name='mesh', x=x, y=y, z=z, h=h)
        offset = h
        x_grid, y_grid, z_grid = np.meshgrid(
                np.arange(x.min() - offset, x.max() + offset, h),
                np.arange(y.min() - offset, y.max() + offset, h),
                np.arange(z.min() - offset, z.max() + offset, h)
        )
        pa_grid = ParticleArray(name='grid', x=x_grid, y=y_grid, z=z_grid, h=h)
        x_grid, y_grid, z_grid, new_sizes, mulitple_occur = \
         G._get_neighbouring_particles(pa_grid, pa_mesh, 1, sizes)
        my_mesh = mesh.Mesh.from_file(cube_fname)
        u, v, w = [], [], []
        for i in range(np.shape(my_mesh.vectors)[0]):
            u.append(my_mesh.normals[i][0])
            v.append(my_mesh.normals[i][1])
            w.append(my_mesh.normals[i][2])
        xf, yf, zf = G.remove_exterior(x_grid, y_grid, z_grid,
                                       c_x, c_y, c_z,
                                       cube_fname, new_sizes,
                                       u, v, w,
                                       1, h, mulitple_occur)

        for i in range(len(xf)):
            assert 0 <= xf[i] <= 1
            assert 0 <= yf[i] <= 1
            assert 0 <= zf[i] <= 1


if __name__ == "__main__":
    unittest.main()
