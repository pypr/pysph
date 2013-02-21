from textwrap import dedent

###############################################################################
# `CubicSpline` class.
###############################################################################
class CubicSpline(object):
    def cython_code(self):
        code = dedent('''\
        cdef double CubicSplineKernel(double x, double y, double h):
            return 1.0
        ''')
        return dict(helper=code)


