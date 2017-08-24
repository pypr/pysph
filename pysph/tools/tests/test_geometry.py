import numpy as np
import unittest

from pysph.tools.geometry import distance, distance_2d
from pysph.tools.geometry import matrix_exp, extrude, translate, rotate
from pysph.tools.geometry import get_2d_wall, get_2d_tank, get_2d_block
from pysph.tools.geometry import get_2d_circle, get_3d_hollow_cylinder
from pysph.tools.geometry import get_3d_sphere, get_3d_block
from pysph.tools.geometry import get_4digit_naca_airfoil, get_naca_wing
from pysph.tools.geometry import get_5digit_naca_airfoil, _get_m_k
from pysph.tools.geometry import remove_overlap_particles
from pysph.base.utils import get_particle_array


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
            n = np.random.randint(1, 10)
            mat = np.random.rand(n, n)
            assert np.allclose(e(mat), matrix_exp(mat))
        except ImportError:
            raise unittest.SkipTest()

    def test_extrude(self):
        n = np.random.randint(1, 20)
        x = np.random.random_sample(n)
        y = np.random.random_sample(n)
        dx = (10.0)**(np.random.uniform(-3, -1))
        extrude_dist = np.random.random()
        x_new, y_new, z_new = extrude(x, y, dx, extrude_dist)
        assert np.allclose(x_new[:len(x)], x)
        assert np.allclose(x_new[-len(x):], x)
        assert np.allclose(y_new[:len(y)], y)
        assert np.allclose(y_new[-len(y):], y)

    def test_translate(self):
        n = np.random.randint(1, 100)
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
        n = np.random.randint(1, 25)
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
        dx = (10.0)**(np.random.uniform(-3, -1))
        center = np.random.random_sample(2)
        length = np.random.randint(50, 200) * dx
        num_layers = np.random.randint(1, 4)
        up = np.random.choice([True, False])
        x, y = get_2d_wall(dx, center, length, num_layers, up)
        x_test = np.arange(-length / 2.0, length / 2.0, dx) + center[0]
        y_test = np.ones_like(x_test) * center[1]
        layer_length = int(len(x) / num_layers)
        value = 1 if up else -1
        assert np.allclose(x_test, x[:len(x_test)])
        assert np.allclose(y_test, y[:len(y_test)])
        assert np.allclose(x[:layer_length], x[-layer_length:])
        assert np.allclose(y[:layer_length],
                           y[-layer_length:] - value * (num_layers - 1) * dx)

    def test_get_2d_tank(self):
        dx = (10.0)**(np.random.uniform(-3, -1))
        center = np.random.random_sample(2)
        length = np.random.randint(50, 200) * dx
        height = np.random.randint(50, 200) * dx
        num_layers = np.random.randint(1, 4)
        x, y = get_2d_tank(dx, center, length, height, num_layers)
        x_len = max(x) - min(x)
        y_len = max(y) - min(y)
        assert abs(x_len - length - (2.0 * num_layers - 1) * dx) <= 1.001 * dx
        assert abs(y_len - height - num_layers * dx) <= 1.001 * dx

    def test_get_2d_circle(self):
        dx = (10.0)**(np.random.uniform(-3, -1))
        center = np.random.random_sample(2)
        radius = np.random.randint(50, 200) * dx
        x, y = get_2d_circle(dx, radius, center)
        count = 0
        for i in range(len(x)):
            point = np.array([x[i], y[i]])
            dist = distance_2d(point, center)
            if dist >= radius * (1.0 + dx * 1.0e-03):
                count += 1
        assert count == 0

    def test_get_2d_block(self):
        dx = (10.0)**(np.random.uniform(-3, -1))
        center = np.random.random_sample(2)
        length = np.random.randint(50, 200) * dx
        height = np.random.randint(50, 200) * dx
        x, y = get_2d_block(dx, length, height, center)
        len_x = max(x) - min(x)
        len_y = max(y) - min(y)
        new_center = np.array(
            [(max(x) + min(x)) / 2.0, (max(y) + min(y)) / 2.0])
        self.assertAlmostEqual(len_x, length)
        self.assertAlmostEqual(len_y, height)
        assert np.allclose(new_center, center)

    def test_get_3d_sphere(self):
        dx = (10.0)**(np.random.uniform(-3, -1))
        center = np.random.random_sample(3)
        radius = np.random.randint(20, 50) * dx
        x, y, z = get_3d_sphere(dx, radius, center)
        count = 0
        for i in range(len(x)):
            point = np.array([x[i], y[i], z[i]])
            dist = distance(point, center)
            if dist >= radius * (1.0 + dx * 1.0e-04):
                count += 1
        assert count == 0

    def test_get_3d_block(self):
        dx = (10.0)**(np.random.uniform(-3, -1))
        center = np.random.random_sample(3)
        length = np.random.randint(50, 200) * dx
        height = np.random.randint(50, 200) * dx
        depth = np.random.randint(50, 200) * dx
        x, y, z = get_3d_block(dx, length, height, depth, center)
        len_x = max(x) - min(x)
        len_y = max(y) - min(y)
        len_z = max(z) - min(z)
        new_center = np.array([(max(x) + min(x)) / 2.0,
                               (max(y) + min(y)) / 2.0,
                               (max(z) + min(z)) / 2.0])
        self.assertAlmostEqual(len_x, length)
        self.assertAlmostEqual(len_y, height)
        self.assertAlmostEqual(len_z, depth)
        assert np.allclose(center, new_center)

    def test_get_3d_hollow_cylinder(self):
        dx = (10)**(np.random.uniform(-3, -1))
        radius = np.random.randint(50, 100) * dx
        length = np.random.randint(50, 100) * dx
        center = np.random.random_sample(3)
        num_layers = np.random.randint(1, 5)
        x, y, z = get_3d_hollow_cylinder(dx, radius, length, center,
                                         num_layers)
        x = x - center[0]
        y = y - center[1]
        z = z - center[2]
        count = 0
        for i in range(len(x)):
            point = np.array([x[i], y[i]])
            dist = distance_2d(point)
            condition_1 = dist >= radius * (1.0 + dx * 1.0e-03)
            condition_2 = abs(z[i]) >= (length / 2.0) * (1.0 + dx * 1.0e-03)
            if condition_1 or condition_2:
                count += 1
        assert count == 0

    def test_get_naca_wing(self):
        dx = (10)**(np.random.uniform(-3, -1))
        c = np.random.uniform(0.5, 2.0)
        span = np.random.uniform(1.0, 4.0)
        series = np.random.choice(['210', '220', '230', '240', '250', '221',
                                   '231', '241', '251'])
        t = np.random.randint(10, 24)
        airfoil_1 = series + str(t)
        camber = str(2 + np.random.randint(5)) + str(2 + np.random.randint(6))
        airfoil_2 = np.random.choice(['00', camber]) + str(t)
        airfoil = np.random.choice([airfoil_1, airfoil_2])
        x, y, z = get_naca_wing(dx, airfoil, span, c)
        count = 0
        for zi in z:
            if abs(zi) > (span + dx) / 2.0:
                count += 1
        assert count == 0

    def test_remove_overlap_particles(self):
        dx_1 = (10)**(np.random.randint(-3, -1))
        dx_2 = (10)**(np.random.randint(-3, -1))
        length = np.random.randint(50, 100) * dx_1
        height = np.random.randint(50, 100) * dx_1
        radius = np.random.randint(50, 100) * dx_2
        x1, y1 = get_2d_block(dx_1, length, height)
        x2, y2 = get_2d_circle(dx_2, radius)
        r1 = np.ones_like(x1) * 100.0
        m1 = r1 * dx_1 * dx_1
        h1 = np.ones_like(x1) * dx_1 * 1.5
        fluid = get_particle_array(name='fluid', x=x1, y=y1, h=h1, rho=r1,
                                   m=m1)
        r2 = np.ones_like(x2) * 100.0
        m2 = r2 * dx_2 * dx_2
        h2 = np.ones_like(x2) * dx_2 * 1.5
        solid = get_particle_array(name='solid', x=x2, y=y2, h=h2, rho=r2,
                                   m=m2)
        remove_overlap_particles(fluid, solid, dx_2, 2)
        x1 = fluid.x
        y1 = fluid.y
        count = 0
        for i in range(len(x1)):
            point = np.array([x1[i], y1[i]])
            dist = distance_2d(point)
            if dist <= radius:
                count += 1
        assert count == 0


if __name__ == "__main__":
    unittest.main()
