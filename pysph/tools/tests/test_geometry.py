import numpy as np
import unittest

from pysph.tools.geometry import distance, distance_2d
from pysph.tools.geometry import matrix_exp, extrude, translate, rotate
from pysph.tools.geometry import get_2d_wall, get_2d_tank, get_2d_block


class TestGeometry(unittest.TestCase):

    def test_distance(self):
        point_1 = np.random.random_sample(3)
        point_2 = np.random.random_sample(3)
        self.assertAlmostEqual(distance(point_1, point_2),
                               np.linalg.norm(point_1 - point_2))

    def test_distance_2d(self):
        point_1 = np.random.random_sample(2)
        point_2 = np.random.random_sample(2)
        self.assertAlmostEqual(distance_2d(point_1, point_2),
                               np.linalg.norm(point_1 - point_2))

    def test_matrix_exp(self):
        try:
            import scipy.linalg.expm as e
            n = np.random.randint(15)
            mat = np.random.rand(n, n)
            assert np.allclose(e(mat), matrix_exp(mat))
        except ImportError:
            raise unittest.SkipTest()

    def test_extrude(self):
        n = np.random.randint(20)
        x = np.random.random_sample(n)
        y = np.random.random_sample(n)
        dx = np.random.random()
        extrude_dist = np.random.random()
        x_new, y_new, z_new = extrude(x, y, dx, extrude_dist)
        assert np.allclose(x_new[:len(x)], x)
        assert np.allclose(x_new[-len(x):], x)
        assert np.allclose(y_new[:len(y)], y)
        assert np.allclose(y_new[-len(y):], y)

    def test_translate(self):
        n = np.random.randint(100)
        x = np.random.random_sample(n)
        y = np.random.random_sample(n)
        z = np.random.random_sample(n)
        x_shift = np.random.uniform(0.0, n)
        y_shift = np.random.uniform(0.0, n)
        z_shift = np.random.uniform(0.0, n)
        x_new, y_new, z_new = translate(x, y, z, x_shift, y_shift, z_shift)
        assert np.allclose(x_new - x_shift, x)
        assert np.allclose(y_new - y_shift, y)
        assert np.allclose(z_new - z_shift, z)

    def test_rotate(self):
        n = np.random.randint(100)
        x = np.random.random_sample(n)
        y = np.random.random_sample(n)
        z = np.random.random_sample(n)
        angle = np.random.uniform(0.0, 360.0)
        theta = np.pi * angle / 180.0
        axis = np.random.random_sample(3)
        x_new, y_new, z_new = rotate(x, y, z, axis, angle)
        unit_vector = axis / np.linalg.norm(axis)
        mat = np.cross(np.eye(3), unit_vector * theta)
        rotation_matrix = matrix_exp(np.matrix(mat))
        test_points = []
        for xi, yi, zi in zip(x, y, z):
            point = np.array([xi, yi, zi])
            new = np.asarray(np.dot(rotation_matrix, point))
            test_points.append(new[0])
        test_points = np.asarray(test_points)
        x_test = test_points[:, 0]
        y_test = test_points[:, 1]
        z_test = test_points[:, 2]
        assert np.allclose(x_new, x_test)
        assert np.allclose(y_new, y_test)
        assert np.allclose(z_new, z_test)

    def test_get_2d_wall(self):
        dx = np.random.random() / 10.0
        center = np.random.random_sample(2)
        length = np.random.uniform(0.0, 10.0)
        num_layers = np.random.randint(10)
        up = np.random.choice([True, False])
        x, y = get_2d_wall(dx, center, length, num_layers, up)
        x_test = np.arange(-length / 2.0, length / 2.0, dx) + center[0]
        y_test = np.ones_like(x_test) * center[1]
        layer_length = len(x) / num_layers
        value = 1 if up else -1
        assert np.allclose(x_test, x[:len(x_test)])
        assert np.allclose(y_test, y[:len(y_test)])
        assert np.allclose(x[:layer_length], x[-layer_length:])
        assert np.allclose(y[:layer_length],
                           y[-layer_length:] - value * (num_layers - 1) * dx)

    def test_get_2d_tank(self):
        dx = np.random.random() / 10.0
        center = np.random.random_sample(2)
        length = np.random.uniform(0.0, 10.0)
        height = np.random.uniform(0.0, 10.0)
        num_layers = np.random.randint(10)
        outside = np.random.choice([True, False])
        value = 1 if outside else -1
        x, y = get_2d_tank(dx, center, length, height)


if __name__ == "__main__":
    unittest.main()
