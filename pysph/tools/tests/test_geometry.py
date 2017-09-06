import numpy as np
import unittest
import pytest

import pysph.tools.geometry as G
from pysph.base.utils import get_particle_array


class TestGeometry(unittest.TestCase):

    def test_distance(self):
        point_1 = np.random.random_sample(3)
        point_2 = np.random.random_sample(3)
        d1 = G.distance(point_1, point_2)
        d2 = np.linalg.norm(point_1 - point_2)
        assert d1 == pytest.approx(d2)

    def test_distance_2d(self):
        point_1 = np.random.random_sample(2)
        point_2 = np.random.random_sample(2)
        d1 = G.distance_2d(point_1, point_2)
        d2 = np.linalg.norm(point_1 - point_2)
        assert d1 == pytest.approx(d2)

    def test_matrix_exp(self):
        try:
            from scipy import linalg
            e = linalg.expm
            n = np.random.randint(1, 10)
            mat = np.random.rand(n, n)
            assert np.allclose(e(mat), G.matrix_exp(mat))
        except ImportError:
            raise unittest.SkipTest('Scipy is not installed')

    def test_extrude(self):
        n = np.random.randint(1, 20)
        x = np.random.random_sample(n)
        y = np.random.random_sample(n)
        dx = 0.05
        extrude_dist = np.random.random()
        x_new, y_new, z_new = G.extrude(x, y, dx, extrude_dist)
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
        x_new, y_new, z_new = G.translate(x, y, z, x_shift, y_shift, z_shift)
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
        x_new, y_new, z_new = G.rotate(x, y, z, axis, angle)
        unit_vector = axis / np.linalg.norm(axis)
        mat = np.cross(np.eye(3), unit_vector * theta)
        rotation_matrix = G.matrix_exp(np.matrix(mat))
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
        dx = 0.05
        length = 3.0
        center = np.random.random_sample(2)
        num_layers = np.random.randint(1, 4)
        up = np.random.choice([True, False])
        x, y = G.get_2d_wall(dx, center, length, num_layers, up)
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
        dx = 0.05
        length = 5.0
        height = 4.0
        center = np.random.random_sample(2)
        num_layers = np.random.randint(1, 4)
        x, y = G.get_2d_tank(dx, center, length, height, num_layers)
        x_len = max(x) - min(x)
        y_len = max(y) - min(y)
        assert abs(x_len - length - (2.0 * num_layers - 1) * dx) <= 1.001 * dx
        assert abs(y_len - height - num_layers * dx) <= 1.001 * dx

    def test_get_2d_circle(self):
        dx = 0.05
        radius = 3.0
        center = np.random.random_sample(2)
        x, y = G.get_2d_circle(dx, radius, center)
        xc, yc = x - center[0], y - center[1]
        assert not np.any(xc * xc + yc * yc) > radius * radius

    def test_get_2d_block(self):
        dx = 0.05
        length = 5.0
        height = 4.0
        center = np.random.random_sample(2)
        x, y = G.get_2d_block(dx, length, height, center)
        len_x = max(x) - min(x)
        len_y = max(y) - min(y)
        new_center = np.array(
            [(max(x) + min(x)) / 2.0, (max(y) + min(y)) / 2.0])
        assert len_x == pytest.approx(length)
        assert len_y == pytest.approx(height)
        assert np.allclose(new_center, center)

    def test_get_3d_sphere(self):
        dx = 0.23
        radius = 2.0
        center = np.random.random_sample(3)
        x, y, z = G.get_3d_sphere(dx, radius, center)
        xs, ys, zs = x - center[0], y - center[1], z - center[2]
        assert not np.any(xs * xs + ys * ys + zs * zs) > radius * radius

    def test_get_3d_block(self):
        dx = 0.23
        length = 3.0
        height = 2.0
        depth = 1.5
        center = np.random.random_sample(3)
        x, y, z = G.get_3d_block(dx, length, height, depth, center)
        len_x = max(x) - min(x)
        len_y = max(y) - min(y)
        len_z = max(z) - min(z)
        new_center = np.array([(max(x) + min(x)) / 2.0,
                               (max(y) + min(y)) / 2.0,
                               (max(z) + min(z)) / 2.0])
        assert len_x == pytest.approx(length)
        assert len_y == pytest.approx(height)
        assert len_z == pytest.approx(depth)
        assert np.allclose(center, new_center)

    def test_get_3d_hollow_cylinder(self):
        dx = 0.15
        radius = 2.0
        length = 3.0
        center = np.random.random_sample(3)
        num_layers = np.random.randint(1, 5)
        x, y, z = G.get_3d_hollow_cylinder(dx, radius, length, center,
                                           num_layers)
        xc, yc, zc = x - center[0], y - center[1], z - center[2]
        assert not np.any(xc * xc + yc * yc) > radius * radius
        assert not np.any(zc) > (length / 2.0)

    def test_get_4digit_naca_airfoil(self):
        dx = 0.1
        c = 1.5
        camber = '44'
        airfoils = ['0010', '2414', '4420', '3218', '2424']
        for airfoil in airfoils:
            x, y = G.get_4digit_naca_airfoil(dx, airfoil, c)
            count = 0
            t = 0.01 * float(airfoil[2:])
            for xi, yi in zip(x, y):
                yt = (0.2969 * np.sqrt(xi / c) - 0.1260 * (xi / c) -
                      0.3516 * ((xi / c)**2.) + 0.2843 * (xi / c)**3. -
                      0.1015 * ((xi / c)**4.))
                yt = yt * 5.0 * t
                if airfoil[:2] == '00':
                    if abs(yi) > yt + dx:
                        count += 1
                else:
                    m = 0.01 * float(airfoil[0])
                    p = 0.1 * float(airfoil[1])
                    if xi <= p * c:
                        yc = (m / (p**2.0)) * \
                            (2. * p * (xi / c) - (xi / c)**2.)
                        dydx = (2. * m / (p * p)) * (p - xi / c) / c
                    else:
                        yc = (m / ((1. - p) * (1. - p))) * \
                            (1. - 2. * p + 2. * p * (xi / c) - (xi / c)**2.)
                        dydx = (2. * m / ((1. - p) * (1. - p))) * \
                            (p - xi / c) / c
                    theta = np.arctan(dydx)
                    if yi >= 0.0:
                        yu = yc + yt * np.cos(theta)
                        if yi > yu + dx:
                            count += 1
                    else:
                        yl = yc - yt * np.cos(theta)
                        if yi < yl - dx:
                            count += 1
            assert count == 0

    def test_get_5digit_naca_airfoil(self):
        dx = 0.05
        c = 1.5
        series = ['210', '220', '230', '240', '250',
                  '221', '231', '241', '251']
        t = 12
        for s in series:
            airfoil = s + str(t)
            t = 0.01 * t
            x, y = G.get_5digit_naca_airfoil(dx, airfoil, c)
            count = 0
            m, k = G._get_m_k(s)
            for xi, yi in zip(x, y):
                yt = (0.2969 * np.sqrt(xi / c) - 0.1260 * (xi / c) -
                      0.3516 * ((xi / c)**2.) + 0.2843 * (xi / c)**3. -
                      0.1015 * ((xi / c)**4.))
                yt = 5.0 * t * yt
            xn = xi / c
            if xn <= m:
                yc = c * (k / 6.) * (xn**3. - 3. * m *
                                     xn * xn + m * m * (3. - m) * xn)
                dydx = (k / 6.) * (3. * xn * xn -
                                   6. * m * xn + m * m * (3. - m))
            else:
                yc = c * (k * (m**3.) / 6.) * (1. - xn)
                dydx = -(k * (m**3.) / 6.)
            theta = np.arctan(dydx)
            if yi >= 0.0:
                yu = yc + yt * np.cos(theta)
                if yi > yu + dx:
                    count += 1
            else:
                yl = yc - yt * np.cos(theta)
                if yi < yl - dx:
                    count += 1
            assert count == 0

    def test_get_naca_wing(self):
        dx = 0.15
        c = 1.5
        span = 3.0
        airfoils = ['23014', '22112', '0012', '2410']
        for airfoil in airfoils:
            x, y, z = G.get_naca_wing(dx, airfoil, span, c)
            assert not np.any(z) > (span / dx + dx)

    def test_remove_overlap_particles(self):
        dx_1 = 0.1
        dx_2 = 0.15
        length = 4.5
        height = 3.0
        radius = 2.0
        x1, y1 = G.get_2d_block(dx_1, length, height)
        x2, y2 = G.get_2d_circle(dx_2, radius)
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
        G.remove_overlap_particles(fluid, solid, dx_2, 2)
        x1 = fluid.x
        y1 = fluid.y
        count = 0
        for i in range(len(x1)):
            point = np.array([x1[i], y1[i]])
            dist = G.distance_2d(point)
            if dist <= radius:
                count += 1
        assert count == 0


if __name__ == "__main__":
    np.random.seed(0)
    unittest.main()
