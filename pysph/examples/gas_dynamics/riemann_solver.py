"""
Exact solution to Riemann problems.
"""

import numpy
from math import sqrt


def set_gamma(g):
    global gamma, gp1_2g, gm1_2g, gm1_gp1, gm1_2, gm1, gp1
    gamma = g
    gm1_2g = (gamma - 1.0) / (2.0 * gamma)
    gp1_2g = (gamma + 1.0) / (2.0 * gamma)
    gm1_gp1 = (gamma - 1.0) / (gamma + 1.0)
    gm1_2 = (gamma - 1.0) / 2.0
    gm1 = gamma - 1.0
    gp1 = gamma + 1.0


def solve(x_min=-0.5, x_max=0.5, x_0=0.0, t=0.1, p_l=1.0, p_r=0.1, rho_l=1.0,
          rho_r=0.125, u_l=0.0, u_r=0.0, N=101):

    r"""
    Parameters
    ----------
    x_min : float
        the leftmost point of domain
    x_max : float
        the rightmost point of domain
    x_0 : float
        the position of the diaphgram
    t : float
        total time of simulation
    p_l, u_l, rho_l : float
        pressure, velocity, density in the left region
    p_r, u_r, rho_r : float
        pressure, velocity, density in the right region
    N : int
        number of points under study

    The default arguments mentioned correspond to the Sod shock tube case.

    Notes
    -----
    The function returns the exact solution in the order of density, velocity,
    pressure, energy and x-coordinates of the points under study.

    References
    ----------
    .. E.F. Toro, Riemann Solvers and Numerical Methods for Fluid Dynamics,
        Springer (2009), Chapter 4, pp. 115-138

    """
    c_l = sqrt(gamma * p_l / rho_l)
    c_r = sqrt(gamma * p_r / rho_r)

    try:
        import scipy
        print("Using fsolve to solve the non-linear equation")
        p_star, u_star = star_pu_fsolve(rho_l, u_l, p_l, c_l,
                                        rho_r, u_r, p_r, c_r)
    except ImportError:
        print("Using Newton-Raphson method to solve the non-linear equation")
        p_star, u_star = star_pu_newton_raphson(rho_l, u_l, p_l, c_l,
                                                rho_r, u_r, p_r, c_r)

    # check if the discontinuity is inside the domain
    msg = "discontinuity not in domain"
    assert x_0 >= x_min and x_0 <= x_max, msg

    # transform domain according to initial discontinuity
    x_min = x_min - x_0
    x_max = x_max - x_0

    print('p_star=' + str(p_star))
    print('u_star=' + str(u_star))
    x = numpy.linspace(x_min, x_max, N)
    density = []
    pressure = []
    velocity = []
    energy = []
    for i in range(0, N):
        s = x[i] / t
        rho, u, p = complete_solution(rho_l, u_l, p_l, c_l, rho_r, u_r,
                                      p_r, c_r, p_star, u_star, s)
        density.append(rho)
        velocity.append(u)
        pressure.append(p)
        energy.append(p / (gm1 * rho))
    # transform the domain back to original coordinates
    x = x + x_0
    return tuple(map(numpy.asarray, [density, velocity, pressure, energy, x]))


def _flux_fsolve(pressure, rho1, c1, p1):
    if pressure <= p1:  # Rarefaction
        return lambda p: (2 / gm1) * c1 * ((p / p1)**gm1_2g - 1.0)
    else:  # Shock
        return lambda p: (
            (p - p1) * sqrt(((2 / gp1) / rho1) / ((gm1_gp1 * p1) + p))
        )


def star_pu_fsolve(rho_l, u_l, p_l, c_l, rho_r, u_r, p_r, c_r):
    p_min = min(p_l, p_r)
    p_max = max(p_l, p_r)
    f_min = _flux_fsolve(p_min, rho_l, c_l, p_l)(p_min) + \
        _flux_fsolve(p_min, rho_r, c_r, p_r)(p_min) + u_r - u_l
    f_max = _flux_fsolve(p_max, rho_l, c_l, p_l)(p_max) + \
        _flux_fsolve(p_max, rho_r, c_r, p_r)(p_max) + u_r - u_l
    if (f_min > 0 and f_max > 0):
        p_guess = 0.5 * (0 + p_min)
        p_star, u_star = _star_pu(rho_l, u_l, p_l, c_l, rho_r,
                                  u_r, p_r, c_r, p_guess)
    elif(f_min <= 0 and f_max >= 0):
        p_guess = (p_l + p_r) * 0.5
        p_star, u_star = _star_pu(rho_l, u_l, p_l, c_l, rho_r,
                                  u_r, p_r, c_r, p_guess)
    else:
        p_guess = 2 * p_max
        p_star, u_star = _star_pu(rho_l, u_l, p_l, c_l, rho_r,
                                  u_r, p_r, c_r, p_guess)
    return p_star, u_star


def _star_pu(rho_l, u_l, p_l, c_l, rho_r, u_r, p_r, c_r, p_guess):
    """Computes the pressure and velocity in the star region using fsolve
       from scipy module"""
    fl = _flux_fsolve(p_guess, rho_l, c_l, p_l)
    fr = _flux_fsolve(p_guess, rho_r, c_r, p_r)
    f = lambda p: fl(p) + fr(p) + u_r - u_l
    from scipy.optimize import fsolve
    p_star = fsolve(f, 0.0)
    u_star = (
        0.5 * (u_l + u_r + _flux_fsolve(p_star, rho_r, c_r, p_r)(p_star) -
               _flux_fsolve(p_star, rho_l, c_l, p_l)(p_star))
    )
    return p_star, u_star


def _flux_newton(pressure, rho1, c1, p1):
    if pressure <= p1:  # Rarefaction
        flux = (2 / gm1) * c1 * ((pressure / p1)**gm1_2g - 1.0)
        flux_derivative = (1.0 / (rho1 * c1)) * \
            (pressure / p1)**(-gp1_2g)
        return flux, flux_derivative
    else:  # Shock
        flux = (
            (pressure - p1) * sqrt(((2 / gp1) / rho1) /
                                   ((gm1_gp1 * p1) + pressure))
        )
        flux_derivative = (
            (1.0 - 0.5 * (pressure - p1) / ((gm1_gp1 * p1) + pressure)) *
            sqrt(((2 / gp1) / rho1) / ((gm1_gp1 * p1) + pressure))
        )
        return flux, flux_derivative


def star_pu_newton_raphson(rho_l, u_l, p_l, c_l, rho_r, u_r, p_r, c_r):
    tol_pre = 1.0e-06
    nr_iter = 20
    p_start = _compute_guess_p(rho_l, u_l, p_l, c_l, rho_r, u_r, p_r, c_r)
    p_old = p_start
    u_diff = u_r - u_l
    for i in range(nr_iter):
        fL, fLd = _flux_newton(p_old, rho_l, c_l, p_l)
        fR, fRd = _flux_newton(p_old, rho_r, c_r, p_r)
        p = p_old - (fL + fR + u_diff) / (fLd + fRd)
        change = 2.0 * abs((p - p_old) / (p + p_old))
        if change <= tol_pre:
            break
        if p < 0.0:
            p = tol_pre
        p_old = p
    u = 0.5 * (u_l + u_r + fR - fL)
    return p, u


def _compute_guess_p(rho_l, u_l, p_l, c_l, rho_r, u_r, p_r, c_r):
    """ Computes the initial guess for pressure.
    References
    ----------
    E.F. Toro, Riemann Solvers and Numerical Methods for Fluid Dynamics
    Springer (2009), Chapter 9, pp. 297-306
    """
    quser = 2.0
    p_linearized = 0.5 * (p_l + p_r) + 0.5 * (u_l - u_r) * \
        0.25 * (rho_l + rho_r) * (c_l + c_r)
    p_linearized = max(0.0, p_linearized)
    p_min = min(p_l, p_r)
    p_max = max(p_l, p_r)
    qmax = p_max / p_min
    if(
        qmax <= quser and (p_min <= p_linearized and
                           p_linearized <= p_max)
    ):
        """A Primitive Variable Riemann Solver (PMRS)"""
        return p_linearized
    else:
        """A Two-Rarefaction Riemann Solver (TRRS)"""
        if p_linearized < p_min:
            p_lr = (p_l / p_r)**gm1_2g
            u_linearized = (p_lr * u_l / c_l + u_r / c_r + (2 / gm1) *
                            (p_lr - 1.0)) / (p_lr / c_l + 1.0 / c_r)
            return (
                0.5 * (p_l * (1.0 + gm1_2 * (u_l - u_linearized) /
                              c_l)**(1.0 / gm1_2g) +
                       p_r * (1.0 + gm1_2 * (u_linearized - u_r) / c_r) **
                       (1.0 / gm1_2g))
            )
        else:
            """A Two-Shock Riemann Solver (TSRS)"""
            gL = sqrt(((2 / gp1) / rho_l) /
                      (gm1_gp1 * p_l + p_linearized))
            gR = sqrt(((2 / gp1) / rho_r) /
                      (gm1_gp1 * p_r + p_linearized))
            return (gL * p_l + gR * p_r - (u_r - u_l)) / (gL + gR)


def complete_solution(rho_l, u_l, p_l, c_l, rho_r, u_r, p_r, c_r, p_star,
                      u_star, s):
    if s <= u_star:
        rho, u, p = left_contact(rho_l, u_l, p_l, c_l, p_star, u_star, s)
    else:
        rho, u, p = right_contact(rho_r, u_r, p_r, c_r, p_star, u_star, s)
    return rho, u, p


def left_contact(rho_l, u_l, p_l, c_l, p_star, u_star, s):
    if p_star <= p_l:
        rho, u, p = left_rarefaction(rho_l, u_l, p_l, c_l, p_star, u_star, s)
    else:
        rho, u, p = left_shock(rho_l, u_l, p_l, c_l, p_star, u_star, s)
    return rho, u, p


def left_rarefaction(rho_l, u_l, p_l, c_l, p_star, u_star, s):
    s_head = u_l - c_l
    s_tail = u_star - c_l * (p_star / p_l)**gm1_2g
    if s <= s_head:
        rho, u, p = rho_l, u_l, p_l
    elif (s > s_head and s <= s_tail):
        u = (2 / gp1) * (c_l + gm1_2 * u_l + s)
        c = (2 / gp1) * (c_l + gm1_2 * (u_l - s))
        rho = rho_l * (c / c_l)**(2 / gm1)
        p = p_l * (c / c_l)**(1.0 / gm1_2g)
    else:
        rho = rho_l * (p_star / p_l)**(1.0 / gamma)
        u = u_star
        p = p_star
    return rho, u, p


def left_shock(rho_l, u_l, p_l, c_l, p_star, u_star, s):
    sL = u_l - c_l * sqrt(gp1_2g * (p_star / p_l) + gm1_2g)
    if s <= sL:
        rho, u, p = rho_l, u_l, p_l
    else:
        rho_1 = rho_l * ((p_star / p_l) + gm1_gp1) / \
            ((p_star / p_l) * gm1_gp1 + 1.0)
        rho, u, p = rho_1, u_star, p_star
    return rho, u, p


def right_contact(rho_r, u_r, p_r, c_r, p_star, u_star, s):
    if p_star > p_r:
        rho, u, p = right_shock(rho_r, u_r, p_r, c_r, p_star, u_star, s)
    else:
        rho, u, p = right_rarefaction(rho_r, u_r, p_r, c_r, p_star, u_star, s)
    return rho, u, p


def right_rarefaction(rho_r, u_r, p_r, c_r, p_star, u_star, s):
    s_head = u_r + c_r
    s_tail = u_star + c_r * (p_star / p_r)**gm1_2g
    if s >= s_head:
        rho, u, p = rho_r, u_r, p_r
    elif (s < s_head and s > s_tail):
        u = (2 / gp1) * (-c_r + gm1_2 * u_r + s)
        c = (2 / gp1) * (c_r - gm1_2 * (u_r - s))
        rho = rho_r * (c / c_r)**(2 / gm1)
        p = p_r * (c / c_r)**(1.0 / gm1_2g)
    else:
        rho = rho_r * (p_star / p_r)**(1.0 / gamma)
        u = u_star
        p = p_star
    return rho, u, p


def right_shock(rho_r, u_r, p_r, c_r, p_star, u_star, s):
    sR = u_r + c_r * sqrt(gp1_2g * (p_star / p_r) + gm1_2g)
    if s >= sR:
        rho, u, p = rho_r, u_r, p_r
    else:
        rho_1 = rho_r * ((p_star / p_r) + gm1_gp1) / \
            ((p_star / p_r) * gm1_gp1 + 1.0)
        rho, u, p = rho_1, u_star, p_star
    return rho, u, p


if __name__ == '__main__':
    set_gamma(1.4)
    solve()
