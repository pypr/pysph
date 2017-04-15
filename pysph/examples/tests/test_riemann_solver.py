import unittest
import numpy as np
from math import sqrt
from pysph.examples.gas_dynamics import riemann_solver
import numpy.testing as npt


try:
    import scipy
except ImportError:
    scipy = None


class RiemannSolverTestCase(unittest.TestCase):

    def setUp(self):
        riemann_solver.set_gamma(1.4)

    def assert_error(self, given, expected, precision):
        return npt.assert_almost_equal(np.ravel(given), expected, precision)

    @unittest.skipIf(scipy is None, 'No scipy module, skipping this test')
    def test_compute_star_fsolve(self):
        # Sod Shock Tube
        x = riemann_solver.star_pu_fsolve(
            rho_l=1.0, u_l=0.0, p_l=1.0, c_l=sqrt(1.4),
            rho_r=0.125, u_r=0.0, p_r=0.1, c_r=sqrt(1.12)
            )
        self.assert_error(x, (0.30313, 0.92745), 5)
        # Sjogreen
        x = riemann_solver.star_pu_fsolve(
            rho_l=1.0, u_l=-2.0, p_l=0.4, c_l=sqrt(0.56),
            rho_r=1.0, u_r=2.0, p_r=0.4, c_r=sqrt(0.56)
            )
        self.assert_error(x, (0.00189, 0.00000), 5)
        # Blastwave
        x = riemann_solver.star_pu_fsolve(
            rho_l=1.0, u_l=0.0, p_l=1000.0, c_l=sqrt(1400),
            rho_r=1.0, u_r=0.0, p_r=0.01, c_r=sqrt(0.014)
            )
        self.assert_error(x, (460.894, 19.5975), 3)
        # Woodward and Collela
        x = riemann_solver.star_pu_fsolve(
            rho_l=1.0, u_l=0.0, p_l=0.01, c_l=sqrt(0.014),
            rho_r=1.0, u_r=0.0, p_r=100.0, c_r=sqrt(140)
            )
        self.assert_error(x, (46.0950, -6.19633), 4)
        # Sod Shock Tube
        riemann_solver.set_gamma(1.2)
        x = riemann_solver.star_pu_fsolve(
            rho_l=1.0, u_l=0.0, p_l=1.0, c_l=sqrt(1.2),
            rho_r=0.125, u_r=0.0, p_r=0.1, c_r=sqrt(0.96)
            )
        self.assert_error(x, (0.31274, 1.01132), 5)

    def test_compute_star_newton_raphson(self):
        # SodShock Tube
        x = riemann_solver.star_pu_newton_raphson(
            rho_l=1.0, u_l=0.0, p_l=1.0, c_l=sqrt(1.4),
            rho_r=0.125, u_r=0.0, p_r=0.1, c_r=sqrt(1.12)
            )
        self.assert_error(x, (0.30313, 0.92745), 5)

        # Sjogreen
        x = riemann_solver.star_pu_newton_raphson(
            rho_l=1.0, u_l=-2.0, p_l=0.4, c_l=sqrt(0.56),
            rho_r=1.0, u_r=2.0, p_r=0.4, c_r=sqrt(0.56)
            )
        self.assert_error(x, (0.00189, 0.00000), 5)

        # Blastwave
        x = riemann_solver.star_pu_newton_raphson(
            rho_l=1.0, u_l=0.0, p_l=1000.0, c_l=sqrt(1400),
            rho_r=1.0, u_r=0.0, p_r=0.01, c_r=sqrt(0.014)
            )
        self.assert_error(x, (460.894, 19.5975), 3)

        # Woodward and Collela
        x = riemann_solver.star_pu_newton_raphson(
            1.0, 0.0, 0.01, sqrt(0.014),
            1.0, 0.0, 100.0, sqrt(140)
            )
        self.assert_error(x, (46.0950, -6.19633), 4)

    def test_left_rarefaction(self):
        # SodShock Tube
        x = riemann_solver.left_rarefaction(
            rho_l=1.0, u_l=0.0, p_l=1.0, c_l=sqrt(1.4),
            p_star=0.30313, u_star=0.92745, s=-2.0
            )
        self.assert_error(x, (1.0, 0.0, 1.0), 10)
        x = riemann_solver.left_rarefaction(
            rho_l=1.0, u_l=0.0, p_l=1.0, c_l=sqrt(1.4),
            p_star=0.30313, u_star=0.92745, s=-1.6
            )
        self.assert_error(x, (1.0, 0.0, 1.0), 10)
        x = riemann_solver.left_rarefaction(
            rho_l=1.0, u_l=0.0, p_l=1.0, c_l=sqrt(1.4),
            p_star=0.30313, u_star=0.92745, s=0.0
            )
        self.assert_error(x, (0.42632, 0.92745, 0.30313), 5)

        # Blastwave
        x = riemann_solver.left_rarefaction(
            rho_l=1.0, u_l=0.0, p_l=1000.0, c_l=sqrt(1400),
            p_star=460.894, u_star=19.5975, s=-0.5 / 0.012
            )
        self.assert_error(x, (1.0, 0.0, 1000.0), 6)
        x = riemann_solver.left_rarefaction(
            rho_l=1.0, u_l=0.0, p_l=1000.0, c_l=sqrt(1400),
            p_star=460.894, u_star=19.5975, s=0.0 / 0.012
            )
        self.assert_error(x, (0.57506, 19.5975, 460.894), 5)

    def test_right_shock(self):
        # SodShock Tube
        x = riemann_solver.right_shock(
            rho_r=0.125, u_r=0.0, p_r=0.1, c_r=sqrt(1.4),
            p_star=0.30313, u_star=0.92745, s=2.0
            )
        self.assert_error(x, (0.125, 0.0, 0.1), 10)
        x = riemann_solver.right_shock(
            rho_r=0.125, u_r=0.0, p_r=0.1, c_r=sqrt(1.4),
            p_star=0.30313, u_star=0.92745, s=1.6
            )
        self.assert_error(x, (0.26557, 0.92745, 0.30313), 5)

        # Blastwave
        x = riemann_solver.right_shock(
            rho_r=1.0, u_r=0.0, p_r=0.01, c_r=sqrt(0.014),
            p_star=460.894, u_star=19.5975, s=0.5 / 0.012
            )
        self.assert_error(x, (1.0, 0.0, 0.01), 10)
        x = riemann_solver.right_shock(
            rho_r=1.0, u_r=0.0, p_r=0.01, c_r=sqrt(0.014),
            p_star=460.894, u_star=19.5975, s=0.2 / 0.012
            )
        self.assert_error(x, (5.99924, 19.5975, 460.894), 5)

    def test_right_rarefaction(self):
        # Sjogreen
        x = riemann_solver.right_rarefaction(
            rho_r=1.0, u_r=2.0, p_r=0.4, c_r=sqrt(0.56),
            p_star=0.00189, u_star=0.000, s=0.5 / 0.15
            )
        self.assert_error(x, (1.0, 2.0, 0.4), 10)
        x = riemann_solver.right_rarefaction(
            rho_r=1.0, u_r=2.0, p_r=0.4, c_r=sqrt(0.56),
            p_star=0.00189, u_star=0.000, s=0.0 / 0.15
            )
        self.assert_error(x, (0.02185, 0.0, 0.00189), 4)

    def test_left_shock(self):
        # Woodward and Collela
        x = riemann_solver.left_shock(
            rho_l=1.0, u_l=0.0, p_l=0.01, c_l=sqrt(0.014),
            p_star=46.095, u_star=-6.1963, s=-0.5 / 0.035
            )
        self.assert_error(x, (1.0, 0.0, 0.01), 10)
        x = riemann_solver.left_shock(
            rho_l=1.0, u_l=0.0, p_l=0.01, c_l=sqrt(0.014),
            p_star=46.095, u_star=-6.1963, s=-0.2 / 0.035
            )
        self.assert_error(x, (5.99242, -6.1963, 46.095), 5)

        # Test_case 5 (Toro)
        x = riemann_solver.left_shock(
            rho_l=5.99924, u_l=19.5975, p_l=460.894, c_l=sqrt(107.555),
            p_star=1691.64, u_star=8.68975, s=-0.5 / 0.035
            )
        self.assert_error(x, (5.99924, 19.5975, 460.894), 5)
        x = riemann_solver.left_shock(
            rho_l=5.99924, u_l=19.5975, p_l=460.894, c_l=sqrt(107.555),
            p_star=1691.64, u_star=8.68975, s=0.1 / 0.035
            )
        self.assert_error(x, (14.2823, 8.68975, 1691.64), 4)

    def test_left_contact(self):
        # Woodward and Collela
        x = riemann_solver.left_contact(
            rho_l=1.0, u_l=0.0, p_l=0.01, c_l=sqrt(0.014),
            p_star=46.095, u_star=-6.1963, s=-0.2 / 0.035
            )
        self.assert_error(x, (5.99242, -6.1963, 46.095), 5)

        # Test_case 5 (Toro)
        x = riemann_solver.left_contact(
            rho_l=5.99924, u_l=19.5975, p_l=460.894, c_l=sqrt(107.555),
            p_star=1691.64, u_star=8.68975, s=0.1 / 0.035
            )
        self.assert_error(x, (14.2823, 8.68975, 1691.64), 4)

        # SodShock Tube
        x = riemann_solver.left_contact(
            rho_l=1.0, u_l=0.0, p_l=1.0, c_l=sqrt(1.4),
            p_star=0.30313, u_star=0.92745, s=0.0
            )
        self.assert_error(x, (0.42632, 0.92745, 0.30313), 5)

        # Blastwave
        x = riemann_solver.left_contact(
            rho_l=1.0, u_l=0.0, p_l=1000.0, c_l=sqrt(1400),
            p_star=460.894, u_star=19.5975, s=0.0 / 0.012
            )
        self.assert_error(x, (0.57506, 19.5975, 460.894), 5)

    def test_right_contact(self):
        # Sjogreen
        x = riemann_solver.right_contact(
            rho_r=1.0, u_r=2.0, p_r=0.4, c_r=sqrt(0.56),
            p_star=0.00189, u_star=0.000, s=0.0 / 0.15
            )
        self.assert_error(x, (0.02185, 0.0, 0.00189), 4)

        # SodShock Tube
        x = riemann_solver.right_contact(
            rho_r=0.125, u_r=0.0, p_r=0.1, c_r=sqrt(1.4),
            p_star=0.30313, u_star=0.92745, s=1.6
            )
        self.assert_error(x, (0.26557, 0.92745, 0.30313), 5)

        # Blastwave
        x = riemann_solver.right_contact(
            rho_r=1.0, u_r=0.0, p_r=0.01, c_r=sqrt(0.014),
            p_star=460.894, u_star=19.5975, s=0.2 / 0.012
            )
        self.assert_error(x, (5.99924, 19.5975, 460.894), 5)

    def test_different_gamma(self):
        # SodShock Tube(gamma = 1.2)
        riemann_solver.set_gamma(1.2)
        x = riemann_solver.star_pu_newton_raphson(
            rho_l=1.0, u_l=0.0, p_l=1.0, c_l=sqrt(1.2),
            rho_r=0.125, u_r=0.0, p_r=0.1, c_r=sqrt(0.96)
            )
        self.assert_error(x, (0.31274, 1.01132), 5)
        x = riemann_solver.right_contact(
            rho_r=0.125, u_r=0.0, p_r=0.1, c_r=sqrt(1.2),
            p_star=0.31274, u_star=1.01132, s=0.15/0.1
            )
        self.assert_error(x, (0.31323, 1.01132, 0.31274), 5)
        x = riemann_solver.left_contact(
            rho_l=1.0, u_l=0.0, p_l=1.0, c_l=sqrt(1.2),
            p_star=0.31274, u_star=1.01132, s=-0.5/0.1
            )
        self.assert_error(x, (1.0, 0.0, 1.0), 10)

if __name__ == '__main__':
    unittest.main()
