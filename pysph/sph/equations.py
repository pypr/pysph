from textwrap import dedent

###############################################################################
# `Equation` class.
###############################################################################
class Equation(object):
    def __init__(self, dest, sources):
        self.dest = dest
        self.sources = sources if sources is not None and len(sources) > 0 \
                                                                else None
        # Does the equation require neighbors or not.
        self.no_source = self.sources is None

        # we assume that positions, mass, density and h are always required
        self.arrays = ['d_x', 'd_y', 'd_z', 'd_m', 'd_rho', 'd_h',
                       's_x', 's_y', 's_z', 's_m', 's_rho', 's_h',]
        
###############################################################################
# `Group` class.
###############################################################################
class Group(object):
    def __init__(self, equations):
        self.equations = equations
        
###############################################################################
# `Variable` class.
###############################################################################
class Variable(object):
    """Should be unique to each equation.
    """
    def __init__(self, type, name, default=None):
        self.type = type
        self.name = name
        self.default = default
        if default is not None:
            declare = 'cdef {0} {1} = {2}'.format(self.type, self.name, 
                                                self.default)
            initialize = '{0} = {1}'.format(self.name, self.default)
        else:
            declare = 'cdef {0} {1}'.format(self.type, self.name)
            initialize = ''
        self.declare = declare
        self.initialize = initialize
        
###############################################################################
# `Temporary` class.
###############################################################################
class Temporary(Variable):
    """Use for temporary variables.  Can be common between equations.
    """
    pass
    

###############################################################################
# `SummationDensity` class.
###############################################################################
class SummationDensity(Equation):
    def cython_code(self):
        variables = [Variable(type='double', name='rho_sum', default=0.0)]
        temp = [Temporary(type='double', name='hab', default=0.0)]
        arrays = ['s_h', 's_m', 's_x', 's_y', 's_z',
                  'd_h', 'd_x', 'd_y', 'd_z', 'd_rho']
        
        loop = dedent('''\
        hab = 0.5*(s_h[s_idx] + d_h[d_idx])
        rho_sum += s_m[s_idx]*KERNEL(d_x[d_idx], d_y[d_idx], d_z[d_idx],
                                     s_x[s_idx], s_y[s_idx], s_z[s_idx], hab)
        ''')
        post = dedent('''\
        d_rho[d_idx] = rho_sum
        ''')
        return dict(variables=variables, temporaries=temp, loop=loop, post=post,
                    arrays=arrays)

                 
class TaitEOS(Equation):
    def __init__(self, dest, sources=[], rho0=1000.0, c0=1.0, gamma=7.0):
        Equation.__init__(self, dest, sources)
        self.rho0 = rho0
        self.c0 = c0
        self.gamma = gamma
        self.gamma1 = 0.5*(gamma - 1.0)

        self.B = rho0*c0*c0/gamma
        
    def cython_code(self):
        temp = [ Temporary(type='double', name='ratio', default=0.0),
                 Temporary(type='double', name='tmp', default=0.0),
                 Temporary(type='double', name='hij', default=0.0)]

        arrays = ['d_rho', 'd_p', 'd_cs']

        loop = dedent("""\
        ratio = d_rho[d_idx]/rho0
        tmp = pow(ratio, GAMMA)

        d_p[d_idx] = B * (tmp -1)
        d_cs[d_idx] = c0 * pow( ratio, GAMMA1 )
        """).replace('B', str(self.B)).replace('c0', str(self.c0))\
                      .replace('rho0', str(self.rho0)).replace('GAMMA1', str(self.gamma1))\
                      .replace('GAMMA', str(self.gamma))

        helpers = dedent("""\
        from libc.math cimport pow
        """)

        return dict(temporaries=temp, loop=loop, arrays=arrays,
                    helper=helpers)

class ContinuityEquation(Equation):
    def cython_code(self):
        variables = [Variable(type='double', name='rho_sum', default=0.0)]
        temp = [ Temporary(type='double[3]', name='grad', default=None),
                 Temporary(type='double[3]', name='xij', default=None),
                 Temporary(type='double', name='vijdotdwij', default=0.0)]

        arrays = self.arrays + ['d_u', 'd_v', 'd_w', 's_u', 's_v', 's_w',
                                'd_arho']

        loop = dedent("""\

        hij = 0.5*(s_h[s_idx] + d_h[d_idx])
        GRADIENT( d_x[d_idx], d_y[d_idx], d_z[d_idx], s_x[s_idx], s_y[s_idx], s_z[s_idx], hij, grad)

        # relative position vector
        xij[0] = d_x[d_idx] - s_x[s_idx]
        xij[1] = d_y[d_idx] - s_y[s_idx]
        xij[2] = d_z[d_idx] - s_z[s_idx]

        vijdotdwij = grad[0]*xij[0] + grad[1]*xij[1] + grad[2]*xij[2]

        #print d_idx, vijdotdwij, grad[0], grad[1], grad[2], rho_sum
        rho_sum += s_m[s_idx]*vijdotdwij
        """)

        post = dedent('''\
        # Density rate
        d_arho[d_idx] = rho_sum
        ''')
        
        return dict(temporaries=temp, loop=loop, arrays=arrays,
                    post=post, variables=variables)

class MomentumEquation(Equation):
    def __init__(self, dest, sources, alpha=1.0, beta=1.0, eta=0.1,
                 gx=0.0, gy=0.0, gz=0.0):
        Equation.__init__(self, dest, sources)
        self.alpha = alpha
        self.beta = beta
        self.eta = eta

        self.gx = gx
        self.gy = gy
        self.gz = gz
        
    def cython_code(self):
        variables = [Variable(type='double', name='usum', default=0.0),
                     Variable(type='double', name='vsum', default=0.0),
                     Variable(type='double', name='wsum', default=0.0)]
        
        temp = [ Temporary(type='double', name='muij', default=0.0),
                 Temporary(type='double', name='piij', default=0.0),
                 Temporary(type='double', name='cij', default=0.0),
                 Temporary(type='double', name='xij2', default=0.0),
                 Temporary(type='double', name='rhoij', default=0.0),
                 Temporary(type='double', name='rhoi21', default=0.0),
                 Temporary(type='double', name='tmp', default=0.0),
                 Temporary(type='double[3]', name='xij', default=None),
                 Temporary(type='double[3]', name='vij', default=None),
                 Temporary(type='double', name='vijdotxij', default=0.0),]

        arrays = self.arrays + ['d_u', 'd_v', 'd_w', 's_u', 's_v', 's_w',
                                'd_p', 's_p', 'd_cs', 's_cs',
                                'd_au', 'd_av', 'd_aw']

        loop = dedent("""\

        hij = 0.5*(s_h[s_idx] + d_h[d_idx])

        # relative position vector
        xij[0] = d_x[d_idx] - s_x[s_idx]
        xij[1] = d_y[d_idx] - s_y[s_idx]
        xij[2] = d_z[d_idx] - s_z[s_idx]

        # relative velocity vector
        vij[0] = d_u[d_idx] - s_u[s_idx]
        vij[1] = d_v[d_idx] - s_v[s_idx]
        vij[2] = d_w[d_idx] - s_w[s_idx]

        vijdotxij = vij[0]*xij[0] + vij[1]*xij[1] + vij[2]*xij[2]

        rhoi21 = 1.0/(d_rho[d_idx]*d_rho[d_idx])
        rhoj21 = 1.0/(s_rho[s_idx]*s_rho[s_idx])

        tmp = d_p[d_idx] * rhoi21 + s_p[s_idx] * rhoj21

        piij = 0.0
        if vijdotxij < 0:
            rhoij = 0.5 * (d_rho[d_idx] + s_rho[s_idx])
            cij = 0.5 * (d_cs[d_idx] + s_cs[s_idx])

            xij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2])

            muij = hij * vijdotxij
            muij = muij/(xij2 + ETA*ETA*hij*hij)

            piij = -ALPHA*cij*muij + BETA*muij*muij
            piij = piij/rhoij

        tmp = tmp + piij
        tmp = -s_m[s_idx] * tmp

        GRADIENT( d_x[d_idx], d_y[d_idx], d_z[d_idx], s_x[s_idx], s_y[s_idx], s_z[s_idx], hij, grad)

        usum = usum + tmp*xij[0]
        vsum = vsum + tmp*xij[1]
        wsum = wsum + tmp*xij[2]

        """).replace('ALPHA', str(self.alpha)).replace('BETA', str(self.beta))\
        .replace('ETA', str(self.eta))

        post = dedent('''\
        # Gradient of Pressure and viscosity
        d_au[d_idx] = usum + GX
        d_av[d_idx] = vsum + GY
        d_aw[d_idx] = wsum + GZ
        ''').replace('GX', str(self.gx)).\
        replace('GY', str(self.gy)).\
        replace('GZ', str(self.gz))
        
        return dict(temporaries=temp, loop=loop, arrays=arrays,
                    post=post, variables=variables)

class XSPHCorrection(Equation):
    def __init__(self, dest, sources, eps=0.5):
        Equation.__init__(self, dest, sources)
        self.eps = eps

        self.arrays = self.arrays + ['d_ax', 'd_ay', 'd_az']
        
    def cython_code(self):
        variables = [Variable(type='double', name='xsum', default=0.0),
                     Variable(type='double', name='ysum', default=0.0),
                     Variable(type='double', name='zsum', default=0.0)]
        
        temp = [ Temporary(type='double', name='tmp', default=0.0),
                 Temporary(type='double', name='wij', default=0.0),
                 Temporary(type='double', name='rhoij', default=0.0),
                 Temporary(type='double', name='hij', default=0.0),
                 Temporary(type='double[3]', name='vij', default=None)]

        loop = dedent("""\

        hij = 0.5*(s_h[s_idx] + d_h[d_idx])
        rhoij = 0.5*(s_rho[s_idx] + d_rho[d_idx])

        # relative velocity vector
        vij[0] = d_u[d_idx] - s_u[s_idx]
        vij[1] = d_v[d_idx] - s_v[s_idx]
        vij[2] = d_w[d_idx] - s_w[s_idx]

        rhoij = 0.5 * (d_rho[d_idx] + s_rho[s_idx])

        wij =  KERNEL(d_x[d_idx], d_y[d_idx], d_z[d_idx],
        s_x[s_idx], s_y[s_idx], s_z[s_idx], hij)

        tmp = EPS*s_m[s_idx]*wij/rhoij

        xsum = xsum - tmp*vij[0]
        ysum = ysum - tmp*vij[1]
        zsum = zsum - tmp*vij[2]

        """).replace('EPS', str(self.eps))

        post = dedent('''\
        # XSPH corrections
        d_ax[d_idx] = xsum + d_u[d_idx]
        d_ay[d_idx] = ysum + d_v[d_idx]
        d_az[d_idx] = zsum + d_w[d_idx]
        ''')
        
        return dict(temporaries=temp, loop=loop, arrays=self.arrays,
                    post=post, variables=variables)    
