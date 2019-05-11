"""GSPH functions"""

from math import sqrt

from compyle.api import declare


def printf(s):
    print(s)


def SIGN(x=0.0, y=0.0):
    if y >= 0:
        return abs(x)
    else:
        return -abs(x)


def riemann_solve(method=1, rhol=0.0, rhor=1.0, pl=0.0, pr=1.0, ul=0.0, ur=1.0,
                  gamma=1.4, niter=20, tol=1e-6, result=[0.0, 0.0]):
    if method == 0:
        return non_diffusive(rhol, rhor, pl, pr, ul, ur, gamma, niter, tol,
                             result)
    elif method == 1:
        return van_leer(rhol, rhor, pl, pr, ul, ur, gamma, niter, tol, result)
    elif method == 2:
        return exact(rhol, rhor, pl, pr, ul, ur, gamma, niter, tol, result)
    elif method == 3:
        return hllc(rhol, rhor, pl, pr, ul, ur, gamma, niter, tol, result)
    elif method == 4:
        return ducowicz(rhol, rhor, pl, pr, ul, ur, gamma, niter, tol, result)
    elif method == 5:
        return hlle(rhol, rhor, pl, pr, ul, ur, gamma, niter, tol, result)
    elif method == 6:
        return roe(rhol, rhor, pl, pr, ul, ur, gamma, niter, tol, result)
    elif method == 7:
        return llxf(rhol, rhor, pl, pr, ul, ur, gamma, niter, tol, result)
    elif method == 8:
        return hllc_ball(rhol, rhor, pl, pr, ul, ur, gamma, niter, tol,
                         result)
    elif method == 9:
        return hll_ball(rhol, rhor, pl, pr, ul, ur, gamma, niter, tol, result)
    elif method == 10:
        return hllsy(rhol, rhor, pl, pr, ul, ur, gamma, niter, tol, result)


def non_diffusive(rhol=0.0, rhor=1.0, pl=0.0, pr=1.0, ul=0.0, ur=1.0,
                  gamma=1.4, niter=20, tol=1e-6, result=[0.0, 0.0]):
    result[0] = 0.5*(pl + pr)
    result[1] = 0.5*(ul + ur)
    return 0


def van_leer(rhol=0.0, rhor=1.0, pl=0.0, pr=1.0,
             ul=0.0, ur=1.0, gamma=1.4, niter=20, tol=1e-6, result=[0.0, 0.0]):
    """Van Leer Riemann solver.


    Parameters
    ----------
    rhol, rhor: double: left and right density.
    pl, pr: double: left and right pressure.
    ul, ur: double: left and right speed.
    gamma: double: Ratio of specific heats.
    niter: int: Max number of iterations to try for iterative schemes.
    tol: double: Error tolerance for convergence.

    result: list: List of length 2. Result will contain computed pstar, ustar

    Returns
    -------

    Returns 0 if the value is computed correctly else 1 if the iterations
    either did not converge or if there is an error.

    """
    if ((rhol < 0) or (rhor < 0) or (pl < 0) or (pr < 0)):
        result[0] = 0.0
        result[1] = 0.0
        return 1

    # local variables
    gamma1, gamma2 = declare('double', 2)
    Vl, Vr, cl, cr, zl, zr, wl, wr = declare('double', 8)
    ustar_l, ustar_r, pstar, pstar_old = declare('double', 4)
    converged, iteration = declare('int', 2)

    gamma2 = 1.0 + gamma
    gamma1 = 0.5 * gamma2/gamma
    smallp = 1e-25

    # initialize variables
    wl = 0.0
    wr = 0.0
    zl = 0.0
    zr = 0.0

    # specific volumes
    Vl = 1./rhol
    Vr = 1./rhor

    # Lagrangean sound speeds
    cl = sqrt(gamma * pl * rhol)
    cr = sqrt(gamma * pr * rhor)

    # Initial guess for pstar
    pstar = pr - pl - cr * (ur - ul)
    pstar = pl + pstar * cl/(cl + cr)

    pstar = max(pstar, smallp)

    # Now Iterate using NR to obtain the star values
    iteration = 0
    while iteration < niter:
        pstar_old = pstar

        wl = 1.0 + gamma1 * (pstar - pl)/pl
        wl = cl * sqrt(wl)

        wr = 1.0 + gamma1 * (pstar - pr)/pr
        wr = cr * sqrt(wr)

        zl = 4.0 * Vl * wl * wl
        zl = -zl * wl/(zl - gamma2*(pstar - pl))

        zr = 4.0 * Vr * wr * wr
        zr = zr * wr/(zr - gamma2*(pstar - pr))

        ustar_l = ul - (pstar - pl)/wl
        ustar_r = ur + (pstar - pr)/wr

        pstar = pstar + (ustar_r - ustar_l)*(zl*zr)/(zr - zl)
        pstar = max(smallp, pstar)

        converged = (abs(pstar - pstar_old)/pstar < tol)
        if converged:
            break

        iteration += 1

    # calculate the averaged ustar
    ustar_l = ul - (pstar - pl)/wl
    ustar_r = ur + (pstar - pr)/wr
    ustar = 0.5 * (ustar_l + ustar_r)

    result[0] = pstar
    result[1] = ustar
    if converged:
        return 0
    else:
        return 1


def prefun_exact(p=0.0, dk=0.0, pk=0.0, ck=0.0, g1=0.0, g2=0.0,
                 g4=0.0, g5=0.0, g6=0.0, result=[0.0, 0.0]):
    """The pressure function.  Updates result with f, fd.
    """

    pratio, f, fd, ak, bk, qrt = declare('double', 6)

    if (p <= pk):
        pratio = p/pk
        f = g4*ck*(pratio**g1 - 1.0)
        fd = (1.0/(dk*ck))*pratio**(-g2)
    else:
        ak = g5/dk
        bk = g6*pk
        qrt = sqrt(ak/(bk+p))
        f = (p-pk)*qrt
        fd = (1.0 - 0.5*(p-pk)/(bk + p))*qrt

    result[0] = f
    result[1] = fd


def exact(rhol=0.0, rhor=1.0, pl=0.0, pr=1.0,
          ul=0.0, ur=1.0, gamma=1.4, niter=20, tol=1e-6, result=[0.0, 0.0]):
    """Exact Riemann solver.

    Solve the Riemann problem for the Euler equations to determine the
    intermediate states.

    Parameters
    ----------
    rhol, rhor: double: left and right density.
    pl, pr: double: left and right pressure.
    ul, ur: double: left and right speed.
    gamma: double: Ratio of specific heats.
    niter: int: Max number of iterations to try for iterative schemes.
    tol: double: Error tolerance for convergence.

    result: list: List of length 2. Result will contain computed pstar, ustar

    Returns
    -------

    Returns 0 if the value is computed correctly else 1 if the iterations
    either did not converge or if there is an error.

    """
    # save derived variables
    tmp1 = 1.0/(2*gamma)
    tmp2 = 1.0/(gamma - 1.0)
    tmp3 = 1.0/(gamma + 1.0)

    gamma1 = (gamma - 1.0) * tmp1
    gamma2 = (gamma + 1.0) * tmp1
    gamma3 = 2*gamma * tmp2
    gamma4 = 2 * tmp2
    gamma5 = 2 * tmp3
    gamma6 = tmp3/tmp2
    gamma7 = 0.5 * (gamma - 1.0)

    # sound speed to the left and right based on the Ideal EOS
    cl, cr = declare('double', 2)
    cl = sqrt(gamma*pl/rhol)
    cr = sqrt(gamma*pr/rhor)

    # Iteration constants
    i = declare('int')
    i = 0

    # local variables
    qser, cup, ppv, pmin, pmax, qmax = declare('double', 6)
    pq, um, ptl, ptr, pm, gel, ger = declare('double', 7)
    pstar, pold, udifff = declare('double', 3)
    p, change = declare('double', 2)

    # initialize variables
    fl, fr = declare('matrix(2)', 2)
    p = 0.0

    # check the initial data
    if (gamma4*(cl+cr) <= (ur-ul)):
        return 1

    # compute the guess pressure 'pm'
    qser = 2.0
    cup = 0.25 * (rhol + rhor)*(cl+cr)
    ppv = 0.5 * (pl + pr) + 0.5*(ul - ur)*cup
    pmin = min(pl, pr)
    pmax = max(pl, pr)
    qmax = pmax/pmin

    ppv = max(0.0, ppv)

    if ((qmax <= qser) and (pmin <= ppv) and (ppv <= pmax)):
        pm = ppv
    elif (ppv < pmin):
        pq = (pl/pr)**gamma1
        um = (pq*ul/cl + ur/cr + gamma4*(pq - 1.0))/(pq/cl + 1.0/cr)
        ptl = 1.0 + gamma7 * (ul - um)/cl
        ptr = 1.0 + gamma7 * (um - ur)/cr
        pm = 0.5*(pl*ptl**gamma3 + pr*ptr**gamma3)
    else:
        gel = sqrt((gamma5/rhol)/(gamma6*pl + ppv))
        ger = sqrt((gamma5/rhor)/(gamma6*pr + ppv))
        pm = (gel*pl + ger*pr - (ur-ul))/(gel + ger)

    # the guessed value is pm
    pstart = pm

    pold = pstart
    udifff = ur-ul

    for i in range(niter):
        prefun_exact(pold, rhol, pl, cl, gamma1, gamma2,
                     gamma4, gamma5, gamma6, fl)
        prefun_exact(pold, rhor, pr, cr, gamma1, gamma2,
                     gamma4, gamma5, gamma6, fr)

        p = pold - (fl[0] + fr[0] + udifff)/(fl[1] + fr[1])
        change = 2.0 * abs((p - pold)/(p + pold))
        if change <= tol:
            break
        pold = p

    if i == niter - 1:
        printf("%s", "Divergence in Newton-Raphson Iteration")
        return 1

    # compute the velocity in the star region 'um'
    um = 0.5 * (ul + ur + fr[0] - fl[0])
    result[0] = p
    result[1] = um
    return 0


def sample(pm=0.0, um=0.0, s=0.0, rhol=1.0, rhor=0.0, pl=1.0, pr=0.0,
           ul=1.0, ur=0.0, gamma=1.4, result=[0.0, 0.0, 0.0]):
    """Sample the solution to the Riemann problem.

    Parameters
    ----------

    pm, um : float
        Pressure and velocity in the star region as returned
        by `exact`

    s : float
        Sampling point (x/t)

    rhol, rhor : float
        Densities on either side of the discontinuity

    pl, pr : float
        Pressures on either side of the discontinuity

    ul, ur : float
        Velocities on either side of the discontinuity

    gamma : float
        Ratio of specific heats

    result : list(double)
        (rho, u, p) : Sampled density, velocity and pressure

    """
    tmp1, tmp2, tmp3 = declare('double', 3)
    g1, g2, g3, g4, g5, g6 = declare('double', 6)
    cl, cr = declare('double', 2)

    # save derived variables
    tmp1 = 1.0/(2*gamma)
    tmp2 = 1.0/(gamma - 1.0)
    tmp3 = 1.0/(gamma + 1.0)

    g1 = (gamma - 1.0) * tmp1
    g2 = (gamma + 1.0) * tmp1
    g3 = 2*gamma * tmp2
    g4 = 2 * tmp2
    g5 = 2 * tmp3
    g6 = tmp3/tmp2
    g7 = 0.5 * (gamma - 1.0)

    # sound speeds at the left and right data states
    cl = sqrt(gamma*pl/rhol)
    cr = sqrt(gamma*pr/rhor)

    if s <= um:
        # sampling point lies to the left of the contact discontinuity
        if (pm <= pl):  # left rarefaction
            # speed of the head of the rarefaction
            shl = ul - cl

            if (s <= shl):
                # sampled point is left state
                rho = rhol
                u = ul
                p = pl
            else:
                cml = cl*(pm/pl)**g1    # eqn (4.54)
                stl = um - cml          # eqn (4.55)

                if (s > stl):
                    # sampled point is Star left state. eqn (4.53)
                    rho = rhol*(pm/pl)**(1.0/gamma)
                    u = um
                    p = pm
                else:
                    # sampled point is inside left fan
                    u = g5 * (cl + g7*ul + s)
                    c = g5 * (cl + g7*(ul - s))

                    rho = rhol*(c/cl)**g4
                    p = pl * (c/cl)**g3

        else:  # pm <= pl
            # left shock
            pml = pm/pl
            sl = ul - cl*sqrt(g2*pml + g1)

            if (s <= sl):
                # sampled point is left data state
                rho = rhol
                u = ul
                p = pl
            else:
                # sampled point is Star left state
                rho = rhol*(pml + g6)/(pml*g6 + 1.0)
                u = um
                p = pm

    else:  # s < um
        # sampling point lies to the right of the contact discontinuity
        if (pm > pr):
            # right shock
            pmr = pm/pr
            sr = ur + cr * sqrt(g2*pmr + g1)

            if (s >= sr):
                # sampled point is right data state
                rho = rhor
                u = ur
                p = pr
            else:
                # sampled point is star right state
                rho = rhor*(pmr + g6)/(pmr*g6 + 1.0)
                u = um
                p = pm
        else:
            # right rarefaction
            shr = ur + cr

            if (s >= shr):
                # sampled point is right state
                rho = rhor
                u = ur
                p = pr
            else:
                cmr = cr*(pm/pr)**g1
                STR = um + cmr

                if (s <= STR):
                    # sampled point is star right
                    rho = rhor*(pm/pr)**(1.0/gamma)
                    u = um
                    p = pm
                else:
                    # sampled point is inside left fan
                    u = g5*(-cr + g7*ur + s)
                    c = g5*(cr - g7*(ur - s))
                    rho = rhor * (c/cr)**g4
                    p = pr*(c/cr)**g3

    result[0] = rho
    result[1] = u
    result[2] = p


def ducowicz(rhol=0.0, rhor=1.0, pl=0.0, pr=1.0,
             ul=0.0, ur=1.0, gamma=1.4, niter=20, tol=1e-6, result=[0.0, 0.0]):
    """Ducowicz Approximate Riemann solver for the Euler equations to
    determine the intermediate states.

    Parameters
    ----------
    rhol, rhor: double: left and right density.
    pl, pr: double: left and right pressure.
    ul, ur: double: left and right speed.
    gamma: double: Ratio of specific heats.
    niter: int: Max number of iterations to try for iterative schemes.
    tol: double: Error tolerance for convergence.

    result: list: List of length 2. Result will contain computed pstar, ustar

    Returns
    -------

    Returns 0 if the value is computed correctly else 1 if the iterations
    either did not converge or if there is an error.

    """
    al, ar = declare('double', 2)
    # strong shock parameters
    al = 0.5 * (gamma + 1.0)
    ar = 0.5 * (gamma + 1.0)

    csl, csr, umin, umax, plmin, prmin, bl, br = declare('double', 8)
    a, b, c, d, pstar, ustar, dd = declare('double', 7)
    # Lagrangian sound speeds
    csl = sqrt(gamma * pl * rhol)
    csr = sqrt(gamma * pr * rhor)

    umin = ur - 0.5 * csr/ar
    umax = ul + 0.5 * csl/al

    plmin = pl - 0.25*rhol*csl*csl/al
    prmin = pr - 0.25*rhor*csr*csr/ar

    bl = rhol*al
    br = rhor*ar

    a = (br-bl) * (prmin-plmin)
    b = br*umin*umin - bl*umax*umax
    c = br*umin - bl*umax
    d = br*bl * (umin-umax)*(umin-umax)

    # Case A : ustar - umin > 0
    dd = sqrt(max(0.0, d-a))
    ustar = (b + prmin-plmin)/(c - SIGN(dd, umax-umin))

    if (((ustar - umin) >= 0.0) and ((ustar - umax) <= 0)):
        pstar = 0.5 * (plmin + prmin + br*abs(ustar-umin)*(ustar-umin) -
                       bl*abs(ustar-umax)*(ustar-umax))
        pstar = max(pstar, 0.0)
        result[0] = pstar
        result[1] = ustar
        return 0

    # Case B: ustar - umin < 0, ustar - umax > 0
    dd = sqrt(max(0.0, d+a))
    ustar = (b - prmin + plmin)/(c - SIGN(dd, umax-umin))
    if (((ustar-umin) <= 0.0) and ((ustar-umax) >= 0.0)):
        pstar = 0.5 * (plmin+prmin + br*abs(ustar-umin)*(ustar-umin) -
                       bl*abs(ustar-umax)*(ustar-umax))
        pstar = max(pstar, 0.0)
        result[0] = pstar
        result[1] = ustar
        return 0

    a = (bl+br)*(plmin-prmin)
    b = bl*umax + br*umin
    c = 1./(bl + br)

    # Case C : ustar-umin >0, ustar-umax > 0
    dd = sqrt(max(0.0, a-d))
    ustar = (b+dd)*c
    if (((ustar-umin) >= 0) and ((ustar-umax) >= 0.0)):
        pstar = 0.5 * (plmin+prmin + br*abs(ustar-umin)*(ustar-umin)
                       - bl*abs(ustar-umax)*(ustar-umax))
        pstar = max(pstar, 0.0)
        result[0] = pstar
        result[1] = ustar
        return 0

    # Case D: ustar - umin < 0, ustar - umax < 0
    dd = sqrt(max(0.0, -a - d))
    ustar = (b-dd)*c
    pstar = 0.5 * (plmin+prmin + br*abs(ustar-umin)*(ustar-umin)
                   - bl*abs(ustar-umax)*(ustar-umax))
    pstar = max(pstar, 0.0)
    result[0] = pstar
    result[1] = ustar
    return 0


def roe(rhol=0.0, rhor=1.0, pl=0.0, pr=1.0, ul=0.0, ur=1.0,
        gamma=1.4, niter=20, tol=1e-6, result=[0.0, 0.0]):
    """Roe's approximate Riemann solver for the Euler equations to
    determine the intermediate states.

    Parameters
    ----------
    rhol, rhor: double: left and right density.
    pl, pr: double: left and right pressure.
    ul, ur: double: left and right speed.
    gamma: double: Ratio of specific heats.
    niter: int: Max number of iterations to try for iterative schemes.
    tol: double: Error tolerance for convergence.

    result: list: List of length 2. Result will contain computed pstar, ustar

    Returns
    -------

    Returns 0 if the value is computed correctly else 1 if the iterations
    either did not converge or if there is an error.

    """

    rrhol, rrhor, denominator, plr, vlr, ulr = declare('double', 6)
    cslr, cslr1 = declare('double', 2)
    # Roe-averaged pressure and specific volume
    rrhol = sqrt(rhol)
    rrhor = sqrt(rhor)

    denominator = 1./(rrhor + rrhol)

    plr = (rrhol*pl + rrhor*pr) * denominator
    vlr = (rrhol/rhol + rrhor/rhor) * denominator
    ulr = (rrhol*ul + rrhor*ur) * denominator

    # average sound speed at the interface
    cslr = sqrt(gamma * plr/vlr)
    cslr1 = 1./cslr

    # the intermediate states
    result[0] = plr - 0.5 * (ur - ul) * cslr
    result[1] = ulr - 0.5 * (pr - pl) * cslr1

    return 0


def llxf(rhol=0.0, rhor=1.0, pl=0.0, pr=1.0, ul=0.0, ur=1.0,
         gamma=1.4, niter=20, tol=1e-6, result=[0.0, 0.0]):
    """Lax Friedrichs approximate Riemann solver for the Euler equations to
    determine the intermediate states.

    Parameters
    ----------
    rhol, rhor: double: left and right density.
    pl, pr: double: left and right pressure.
    ul, ur: double: left and right speed.
    gamma: double: Ratio of specific heats.
    niter: int: Max number of iterations to try for iterative schemes.
    tol: double: Error tolerance for convergence.

    result: list: List of length 2. Result will contain computed pstar, ustar

    Returns
    -------

    Returns 0 if the value is computed correctly else 1 if the iterations
    either did not converge or if there is an error.

    """

    gamma1, csl, csr, cslr, el, El, er, Er, pstar = declare('double', 9)
    gamma1 = 1./(gamma - 1.0)

    # Lagrangian sound speeds
    csl = sqrt(gamma * pl * rhol)
    csr = sqrt(gamma * pr * rhor)
    cslr = max(csr, csl)

    # Total energy on either side
    el = pl*gamma1/rhol
    El = el + 0.5 * ul*ul

    er = pr*gamma1/rhor
    Er = er + 0.5 * ur*ur

    # the intermediate states
    # cdef double ustar = 0.5 * ( ul + ur - 1./cslr * (pr - pl) )
    pstar = 0.5 * (pl + pr - cslr * (ur - ul))
    result[0] = pstar
    result[1] = 1./pstar * (0.5 * ((pl*ul + pr*ur) - cslr*(Er - El)))
    return 0


def hllc(rhol=0.0, rhor=1.0, pl=0.0, pr=1.0, ul=0.0, ur=1.0,
         gamma=1.4, niter=20, tol=1e-6, result=[0.0, 0.0]):
    """Harten-Lax-van Leer-Contact approximate Riemann solver for the Euler
    equations to determine the intermediate states.

    Parameters
    ----------
    rhol, rhor: double: left and right density.
    pl, pr: double: left and right pressure.
    ul, ur: double: left and right speed.
    gamma: double: Ratio of specific heats.
    niter: int: Max number of iterations to try for iterative schemes.
    tol: double: Error tolerance for convergence.

    result: list: List of length 2. Result will contain computed pstar, ustar

    Returns
    -------

    Returns 0 if the value is computed correctly else 1 if the iterations
    either did not converge or if there is an error.

    """

    gamma1, pstar, ustar, mstar, estar = declare('double', 5)
    gamma1 = 1./(gamma - 1.0)

    # Roe-averaged interface velocity
    rrhol = sqrt(rhol)
    rrhor = sqrt(rhor)
    ulr = (rrhol*ul + rrhor*ur)/(rrhol + rrhor)

    # relative velocities at the interface
    vl = ul - ulr
    vr = ur - ulr

    # Sound speeds at the interface
    csl = sqrt(gamma * pl/rhol)
    csr = sqrt(gamma * pr/rhor)
    cslr = (rrhol*csl + rrhor*csr)/(rrhol + rrhor)

    # wave speed estimates at the interface
    sl = min(vl - csl, 0 - cslr)
    sr = max(vr + csr, 0 + cslr)

    sm = rhor*vr*(sr-vr) - rhol*vl*(sl-vl) + pl - pr
    sm /= (rhor*(sr-vr) - rhol*(sl-vl))

    # phat
    phat = rhol*(vl - sl)*(vl - sm) + pl

    # Total energy on either side
    el = pl*gamma1/rhol
    El = rhol*(el + 0.5 * ul*ul)

    er = pr*gamma1/rhor
    Er = rhor*(er + 0.5 * ur*ur)

    # Momentum on either side
    Ml = rhol*ul
    Mr = rhor*ur

    # compute the values based on the wave speeds
    if (sl > 0):
        pstar = pl
        ustar = ul

    elif (sl <= 0.0) and (0.0 < sm):
        mstar = 1./(sl - sm) * ((sl - vl) * Ml + (phat - pl))
        estar = 1./(sl - sm) * ((sl - vl) * El - pl*vl + phat*sm)

        pstar = sm*mstar + phat

        ustar = sm*estar + (sm + ulr)*phat
        ustar /= pstar

    elif (sm <= 0) and (0 < sr):
        mstar = 1./(sr - sm) * ((sr - vr) * Mr + (phat - pr))
        estar = 1./(sr - sm) * ((sr - vr) * Er - pr*vr + phat*sm)

        pstar = sm*mstar + phat

        ustar = sm*estar + (sm + ulr)*phat
        ustar /= pstar

    elif (sr < 0):
        pstar = pr
        ustar = ur

    else:
        printf("%s", "Incorrect wave speeds")
        return 1

    result[0] = pstar
    result[1] = ustar
    return 0


def hllc_ball(rhol=0.0, rhor=1.0, pl=0.0, pr=1.0, ul=0.0, ur=1.0,
              gamma=1.4, niter=20, tol=1e-6, result=[0.0, 0.0]):
    """Harten-Lax-van Leer-Contact approximate Riemann solver for the Euler
    equations to determine the intermediate states.

    Parameters
    ----------
    rhol, rhor: double: left and right density.
    pl, pr: double: left and right pressure.
    ul, ur: double: left and right speed.
    gamma: double: Ratio of specific heats.
    niter: int: Max number of iterations to try for iterative schemes.
    tol: double: Error tolerance for convergence.

    result: list: List of length 2. Result will contain computed pstar, ustar

    Returns
    -------

    Returns 0 if the value is computed correctly else 1 if the iterations
    either did not converge or if there is an error.

    """
    gamma1, csl, csr, cslr, rholr, pstar, ustar = declare('double', 7)
    Sl, Sr, ql, qr, pstar_l, pstar_r = declare('double', 6)

    gamma1 = 0.5 * (gamma + 1.0)/gamma

    # sound speeds in the undisturbed fluid
    csl = sqrt(gamma * pl/rhol)
    csr = sqrt(gamma * pr/rhor)
    cslr = 0.5 * (csl + csr)

    # averaged density
    rholr = 0.5 * (rhol + rhor)

    # provisional intermediate pressure
    pstar = 0.5 * (pl + pr - rholr*cslr*(ur - ul))

    # Wave speed estimates (ustar is taken as the intermediate velocity)
    ustar = 0.5 * (ul + ur - 1./(rholr*cslr)*(pr - pl))

    Hl = pstar/pl
    Hr = pstar/pr

    ql = 1.0
    if (Hl > 1):
        ql = sqrt(1 + gamma1*(Hl - 1.0))

    qr = 1.0
    if (Hr > 1):
        qr = sqrt(1 + gamma1*(Hr - 1.0))

    Sl = ul - csl*ql
    Sr = ur + csr*qr

    # compute the intermediate pressure
    pstar_l = pl + rhol*(ul - Sl)*(ul - ustar)
    pstar_r = pr + rhor*(ur - Sr)*(ur - ustar)

    pstar = 0.5 * (pstar_l + pstar_r)

    # return intermediate states
    result[0] = pstar
    result[1] = ustar
    return 0


def hlle(rhol=0.0, rhor=1.0, pl=0.0, pr=1.0, ul=0.0, ur=1.0,
         gamma=1.4, niter=20, tol=1e-6, result=[0.0, 0.0]):
    """Harten-Lax-van Leer-Einfeldt approximate Riemann solver for the Euler
    equations to determine the intermediate states.

    Parameters
    ----------
    rhol, rhor: double: left and right density.
    pl, pr: double: left and right pressure.
    ul, ur: double: left and right speed.
    gamma: double: Ratio of specific heats.
    niter: int: Max number of iterations to try for iterative schemes.
    tol: double: Error tolerance for convergence.

    result: list: List of length 2. Result will contain computed pstar, ustar

    Returns
    -------

    Returns 0 if the value is computed correctly else 1 if the iterations
    either did not converge or if there is an error.

    """
    gamma1 = 1./(gamma - 1.0)

    # Roe-averaged interface velocity
    rrhol = sqrt(rhol)
    rrhor = sqrt(rhor)
    # ulr = (rrhol*ul + rrhor*ur)/(rrhol + rrhor)

    # lagrangian sound speeds
    csl = sqrt(gamma * pl * rhol)
    csr = sqrt(gamma * pr * rhor)
    cslr = (rrhol*csl + rrhor*csr)/(rrhol + rrhor)

    # wave speed estimates
    sl = min(ul - csl, -cslr)
    sr = max(ur + csr, +cslr)

    smax = max(sl, sr)
    smin = min(sl, sr)
    # cdef double smax = max( (ur-ulr) + csr,  cslr )
    # cdef double smin = max( (ul-ulr) - csl, -cslr )

    # Total energy on either side
    el = pl*gamma1/rhol
    El = el + 0.5 * ul*ul

    er = pr*gamma1/rhor
    Er = er + 0.5 * ur*ur

    # Momentum on either side
    # Ml = rhol*ul
    # Mr = rhor*ur

    # the intermediate states
    pstar = ((smax * pl - smin * pr)/(smax - smin) +
             smax*smin/(smax - smin)*(ur - ul))
    ustar = ((smax * pl*ul - smin * pr*ur)/(smax - smin) +
             smax*smin/(smax - smin)*(Er - El))
    result[0] = pstar
    result[1] = ustar/pstar

    return 0


def hll_ball(rhol=0.0, rhor=1.0, pl=0.0, pr=1.0, ul=0.0, ur=1.0,
             gamma=1.4, niter=20, tol=1e-6, result=[0.0, 0.0]):
    """Harten-Lax-van Leer-Ball approximate Riemann solver for the Euler equations
    to determine the intermediate states.

    Parameters
    ----------
    rhol, rhor: double: left and right density.
    pl, pr: double: left and right pressure.
    ul, ur: double: left and right speed.
    gamma: double: Ratio of specific heats.
    niter: int: Max number of iterations to try for iterative schemes.
    tol: double: Error tolerance for convergence.

    result: list: List of length 2. Result will contain computed pstar, ustar

    Returns
    -------

    Returns 0 if the value is computed correctly else 1 if the iterations
    either did not converge or if there is an error.

    """
    # Roe-averaged pressure and specific volume
    rrhol = sqrt(rhol)
    rrhor = sqrt(rhor)

    denominator = 1./(rrhor + rrhol)

    # sound speeds
    csl = sqrt(gamma * pl/rhol)
    csr = sqrt(gamma * pr/rhor)
    # cslr2 = denominator * (rrhol*csl*csl + rrhor*csr*csr )
    eta = 0.5 * (gamma - 1.0) * (rrhor*rrhol) * denominator * denominator
    betal = abs(ul)
    betar = abs(ur)

    # averaged velocity and cs2
    ulr = (rrhol*ul + rrhor*ur)/(rrhol*rrhor)
    cslr2 = (rrhol*csl*csl + rrhor*csr*csr)/(rrhol*rrhor)
    cslr = sqrt(cslr2 + eta * (betar - betal) * (betar - betal))

    # wave speed estimates
    Sl = min(ulr - cslr, ul - csl)
    Sr = max(ulr + cslr, ur + csr)

    # intermediate states
    # rhostar = 1./(Sr - Sl) * (rhor*(Sr-ur) - rhol*(Sl-ul))
    ustar = ((Sr*Sl*(rhor-rhol) + rhol*ul*Sr - rhor*ur*Sl) /
             (rhol*(ul-Sl) + rhor*(Sr-ur)))
    # pstar = 0.5 * (pl + pr - rhostar*ustar*( (Sr-ustar) + (Sl-ustar) ) + \
    #                                rhor*ur*(ur - Sr) + rhol*ul*(ul-Sl) )
    pstar = (pr*(ustar-Sl) - pl*(ustar-Sr) +
             rhor*ur*(ustar-Sl)*(ur-Sr) -
             rhol*ul*(ustar-Sr)*(ul-Sl))
    pstar = pstar/(Sr - Sl)
    result[0] = pstar
    result[1] = ustar

    return 0


def hllsy(rhol=0.0, rhor=1.0, pl=0.0, pr=1.0, ul=0.0, ur=1.0,
          gamma=1.4, niter=20, tol=1e-6, result=[0.0, 0.0]):
    """HLL Riemann solver defined by Sirotkin and Yoh in 'A Smoothed Particle
    Hydrodynamics method with approximate Riemann solvers for simulation of
    strong explosions' (2013), Computers and Fluids.

    Parameters
    ----------
    rhol, rhor: double: left and right density.
    pl, pr: double: left and right pressure.
    ul, ur: double: left and right speed.
    gamma: double: Ratio of specific heats.
    niter: int: Max number of iterations to try for iterative schemes.
    tol: double: Error tolerance for convergence.

    result: list: List of length 2. Result will contain computed pstar, ustar

    Returns
    -------

    Returns 0 if the value is computed correctly else 1 if the iterations
    either did not converge or if there is an error.

    """

    gamma1 = 1./(gamma - 1.0)

    # Roe-averaging factors
    rrhol = sqrt(rhol)
    rrhor = sqrt(rhor)
    denominator = 1./(rrhor + rrhol)

    # Lagrangian sound speed Eq. (35) in SY13
    csl = sqrt(gamma * pl*rhol)
    csr = sqrt(gamma * pr*rhor)
    cslr = denominator * (rrhol*csl + rrhor*csr)

    # weighting factors Eqs. (33 - 35) in SY13
    bl = max(csl, cslr)
    br = max(csr, cslr)
    wl = br/(bl + br)
    wr = bl/(bl + br)
    wlr = bl*br/(bl + br)

    # Total energy on either side
    el = pl*gamma1/rhol
    El = el + 0.5 * ul*ul

    er = pr*gamma1/rhor
    Er = er + 0.5 * ur*ur

    # intermediate states Eq.(32) in SY13
    pstar = wl*pl + wr*pr - wlr*(ur - ul)
    ustar = wl*(pl*ul) + wr*(pr*ur) - wlr*(Er - El)
    result[0] = pstar
    result[1] = ustar/pstar
    return 0


HELPERS = [
    SIGN, riemann_solve, non_diffusive,
    ducowicz, exact, hll_ball, hllc,
    hllc_ball, hlle, hllsy, llxf, roe,
    van_leer, prefun_exact
]
