import pytest
from pytest import approx

import pysph.sph.gas_dynamics.riemann_solver as R

solvers = [
    R.ducowicz, R.exact, R.hll_ball, R.hllc,
    R.hllc_ball, R.hlle, R.hllsy, R.llxf, R.roe,
    R.van_leer
]


def _check_shock_tube(solver, **approx_kw):
    # Given
    # Shock tube
    gamma = 1.4
    rhol, pl, ul = 1.0, 1.0, 0.0
    rhor, pr, ur = 0.125, 0.1, 0.0

    result = [0.0, 0.0]
    # When
    solver(rhol, rhor, pl, pr, ul, ur, gamma, niter=20,
           tol=1e-6, result=result)

    # Then
    assert result == approx((0.30313, 0.92745), **approx_kw)


def _check_blastwave(solver, **approx_kw):
    # Given
    gamma = 1.4
    result = [0.0, 0.0]
    rhol, pl, ul = 1.0, 1000.0, 0.0
    rhor, pr, ur = 1.0, 0.01, 0.0

    # When
    solver(rhol, rhor, pl, pr, ul, ur, gamma, niter=20,
           tol=1e-6, result=result)

    # Then
    assert result == approx((460.894, 19.5975), **approx_kw)


def _check_sjogreen(solver, **approx_kw):
    # Given
    gamma = 1.4
    result = [0.0, 0.0]
    rhol, pl, ul = 1.0, 0.4, -2.0
    rhor, pr, ur = 1.0, 0.4, 2.0

    # When
    solver(rhol, rhor, pl, pr, ul, ur, gamma, niter=20,
           tol=1e-6, result=result)

    # Then
    assert result == approx((0.0018938, 0.0), **approx_kw)


def _check_woodward_collela(solver, **approx_kw):
    # Given
    gamma = 1.4
    result = [0.0, 0.0]
    rhol, pl, ul = 1.0, 0.01, 0.0
    rhor, pr, ur = 1.0, 100.0, 0.0

    # When
    solver(rhol, rhor, pl, pr, ul, ur, gamma, niter=20,
           tol=1e-6, result=result)

    # Then
    assert result == approx((46.0950, -6.19633), **approx_kw)


def test_exact_riemann():
    solver = R.exact

    _check_shock_tube(solver, rel=1e-4)
    _check_blastwave(solver, rel=1e-3)
    _check_sjogreen(solver, abs=1e-4)
    _check_woodward_collela(solver, rel=1e-4)


def test_van_leer():
    solver = R.van_leer
    _check_shock_tube(solver, rel=1e-3)
    _check_blastwave(solver, rel=1e-2)
    _check_sjogreen(solver, abs=1e-2)
    _check_woodward_collela(solver, rel=1e-2)


def test_ducowicz():
    solver = R.ducowicz
    _check_shock_tube(solver, rel=0.2)
    _check_blastwave(solver, rel=0.4)
    _check_sjogreen(solver, abs=1e-2)
    _check_woodward_collela(solver, rel=0.4)


# Most other solvers seem rather poor in comparison.
@pytest.mark.parametrize("solver", solvers)
def test_all_solver_api(solver):
    if solver.__name__ in ['roe', 'hllc']:
        rel = 2.0
    else:
        rel = 1.0
    _check_shock_tube(solver, rel=rel)
