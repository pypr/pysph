"""Experimental data for the dam break problem.

The data is extracted from:

Martin and Moyce 1952
"An Experimental Study of the Collapse of Liquid Columns on a Rigid Horizontal
Plane", J. C. Martin and W. J. Moyce, Philososphical Transactions of the Royal
Society of London Series A, 244, 312--324 (1952).

and

"Moving-Particle Semi-Implicit Method for Fragmentation of Incompressible
fluid", S. Koshizuka and Y. Oka, Nuclear Science and Engineering, 123,
421--434 (1996).

"""

import numpy as np
from io import StringIO

# This is the data for n^2=2 and a=1.125 from Figure 3.
mm_data_1 = u"""
0.849   1.245
1.212   1.443
1.602   1.884
2.283   2.689
2.950   3.728
3.598   4.528
3.905   4.999
4.592   5.841
4.961   6.271
5.316   6.717
"""

# This is the data for n^2=2 and a=2.25 from Figure 3.
mm_data_2 = u"""
0.832   1.217
1.219   1.474
1.997   2.292
2.547   2.995
3.345   4.134
4.034   4.944
4.418   5.881
5.091   6.980
5.685   7.945
6.306   8.966
6.822   9.986
7.439   10.963
8.031   11.977
8.633   13.005
9.237   13.970
"""

ko_data = u"""
0.0     1.000
0.381   1.111
0.769   1.252
1.153   1.505
1.537   1.892
1.935   2.241
2.323   2.615
2.719   3.003
3.096   3.624
"""

def get_martin_moyce_1():
    """Returns t*sqrt(2*g/a), z/a for the case where a = 1.125 inches
    """
    # z/a vs t*np.sqrt(2*g/a)
    t, z = np.loadtxt(StringIO(mm_data_1), unpack=True)
    return t, z

def get_martin_moyce_2():
    """Returns t*sqrt(2*g/a), z/a for the case where a = 2.25 inches
    """
    # z/a vs t*np.sqrt(2*g/a)
    t, z = np.loadtxt(StringIO(mm_data_2), unpack=True)
    return t, z


def get_koshizuka_oka_data():
    # z/L vs t*np.sqrt(2*g/L)
    t, z = np.loadtxt(StringIO(ko_data), unpack=True)
    return t, z
