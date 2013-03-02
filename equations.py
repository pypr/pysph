from textwrap import dedent

###############################################################################
# `Equation` class.
###############################################################################
class Equation(object):
    def __init__(self, dest, sources):
        self.dest = dest
        self.sources = sources
        
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

        self.B = rho0*c0*c0/gamma
        
    def cython_code(self):
        temp = [ Temporary(type='double', name='ratio', default=0.0),
                 Temporary(type='double', name='gamma1', default=0.0),
                 Temporary(type='double', name='gamma_power1', default=0.0) ]

        constants = [ Variable(type='double', name='gamma', default=7.0),
                      Variable(type='double', name='c0', default=1.0),
                      Variable(type='double', name='rho0', default=1.0),
                      Variable(type='double', name='B', default=1.0) ]
        
        
        arrays = ['d_rho', 'd_p', 'd_cs']

        loop = dedent("""\
        ratio = d_rho[d_idx]/rho0
        gamma1 = 0.5 * (GAMMA - 1.0)
        gamma_power1 = pow( ratio, GAMMA )

        d_p[d_idx] = B * (gamma_power1 -1)
        d_cs[d_idx] = c0 * pow( ratio, gamma_power1 )
        """).replace('B', str(self.B)).replace('c0', str(self.c0))\
                      .replace('rho0', str(self.rho0)).replace('GAMMA', str(self.gamma))

        helpers = dedent("""\
        from libc.math cimport pow
        """)

        return dict(temporaries=temp, loop=loop, arrays=arrays, constants=constants,
                    helper=helpers)
