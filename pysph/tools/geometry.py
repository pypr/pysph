from __future__ import division
import numpy as np
import copy
from pysph.base.nnps import LinkedListNNPS
from pysph.base.utils import get_particle_array, get_particle_array_wcsph
from cyarray.api import UIntArray
from numpy.linalg import norm


def distance(point1, point2=np.array([0.0, 0.0, 0.0])):
    return np.sqrt(sum((point1 - point2) * (point1 - point2)))


def distance_2d(point1, point2=np.array([0.0, 0.0])):
    return np.sqrt(sum((point1 - point2) * (point1 - point2)))


def matrix_exp(matrix):
    """
    Exponential of a matrix.

    Finds the exponential of a square matrix of any order using the
    formula exp(A) = I + (A/1!) + (A**2/2!) + (A**3/3!) + .........

    Parameters
    ----------
    matrix : numpy matrix of order nxn (square) filled with numbers

    Returns
    -------
    result : numpy matrix of the same order

    Examples
    --------
    >>>A = np.matrix([[1, 2],[2, 3]])
    >>>matrix_exp(A)
    matrix([[19.68002699, 30.56514746],
            [30.56514746, 50.24517445]])
    >>>B = np.matrix([[0, 0],[0, 0]])
    >>>matrix_exp(B)
    matrix([[1., 0.],
            [0., 1.]])
    """

    matrix = np.asmatrix(matrix)
    tol = 1.0e-16
    result = matrix**(0)
    n = 1
    condition = True
    while condition:
        adding = matrix**(n) / (1.0 * np.math.factorial(n))
        result += adding
        residue = np.sqrt(np.sum(np.square(adding)) /
                          np.sum(np.square(result)))
        condition = (residue > tol)
        n += 1
    return result


def extrude(x, y, dx=0.01, extrude_dist=1.0, z_center=0.0):
    """
    Extrudes a 2d geometry.

    Takes a 2d geometry with x, y values and extrudes it in z direction by the
    amount extrude_dist with z_center as center

    Parameters
    ----------
    x : 1d array object with numbers
    y : 1d array object with numbers
    dx : a number
    extrude_dist : a number
    z_center : a number

    x, y should be of the same length and no x, y pair should be the same

    Returns
    -------
    x_new : 1d numpy array object with new x values
    y_new : 1d numpy array object with new y values
    z_new : 1d numpy array object with z values

    x_new, y_new, z_new are of the same length

    Examples
    --------
    >>>x = np.array([0.0])
    >>>y = np.array([0.0])
    >>>extrude(x, y, 0.1, 0.2, 0.0)
    (array([ 0., 0., 0.]),
     array([ 0., 0., 0.]),
     array([-0.1, 0., 0.1]))
    """

    z = np.arange(z_center - extrude_dist / 2.,
                  z_center + (extrude_dist + dx) / 2., dx)
    x_new = np.tile(np.asarray(x), len(z))
    y_new = np.tile(np.asarray(y), len(z))
    z_new = np.repeat(z, len(x))
    return x_new, y_new, z_new


def translate(x, y, z, x_translate=0.0, y_translate=0.0, z_translate=0.0):
    """
    Translates set of points in 3d cartisean space.

    Takes set of points and translates each and every point by some
    mentioned amount in all the 3 directions.

    Parameters
    ----------
    x : 1d array object with numbers
    y : 1d array object with numbers
    z : 1d array object with numbers
    x_translate : a number
    y_translate : a number
    z_translate : a number

    Returns
    -------
    x_new : 1d numpy array object with new x values
    y_new : 1d numpy array object with new y values
    z_new : 1d numpy array object with new z values

    Examples
    --------
    >>>x = np.array([0.0, 1.0, 2.0])
    >>>y = np.array([-1.0, 0.0, 1.5])
    >>>z = np.array([0.5, -1.5, 0.0])
    >>>translate(x, y, z, 1.0, -0.5, 2.0)
    (array([ 1., 2., 3.]), array([-1.5, -0.5, 1.]), array([2.5, 0.5, 2.]))
    """

    x_new = np.asarray(x) + x_translate
    y_new = np.asarray(y) + y_translate
    z_new = np.asarray(z) + z_translate
    return x_new, y_new, z_new


def rotate(x, y, z, axis=np.array([0.0, 0.0, 1.0]), angle=90.0):
    """
    Rotates set of points in 3d cartisean space.

    Takes set of points and rotates each point with some angle w.r.t
    a mentioned axis.

    Parameters
    ----------
    x : 1d array object with numbers
    y : 1d array object with numbers
    z : 1d array object with numbers
    axis : 1d array with 3 numbers
    angle(in degrees) : number

    Returns
    -------
    x_new : 1d numpy array object with new x values
    y_new : 1d numpy array object with new y values
    z_new : 1d numpy array object with new z values

    Examples
    --------
    >>>x = np.array([0.0, 1.0, 2.0])
    >>>y = np.array([-1.0, 0.0, 1.5])
    >>>z = np.array([0.5, -1.5, 0.0])
    >>>axis = np.array([0.0, 0.0, 1.0])
    >>>rotate(x, y, z, axis, 90.0)
    (array([ 0.29212042, -0.5, 2.31181936]),
     array([ -0.5, 3.31047738, 11.12095476]),
     array([-0.5, -0.5, 3.5]))
    """

    theta = angle * np.pi / 180.0
    unit_vector = np.asarray(axis) / norm(np.asarray(axis))
    matrix = np.cross(np.eye(3), unit_vector * theta)
    rotation_matrix = matrix_exp(np.matrix(matrix))
    new_points = []
    for xi, yi, zi in zip(np.asarray(x), np.asarray(y), np.asarray(z)):
        point = np.array([xi, yi, zi])
        new = np.dot(rotation_matrix, point)
        new_points.append(np.asarray(new)[0])
    new_points = np.array(new_points)
    x_new = new_points[:, 0]
    y_new = new_points[:, 1]
    z_new = new_points[:, 2]
    return x_new, y_new, z_new


def get_2d_wall(dx=0.01, center=np.array([0.0, 0.0]), length=1.0,
                num_layers=1, up=True):
    """
    Generates a 2d wall which is parallel to x-axis. The wall can be
    rotated parallel to any axis using the rotate function. 3d wall
    can be also generated using the extrude function after generating
    particles using this function.

        ^
        |
        |
       y|*******************
        |   wall particles
        |
        |____________________>
                 x

    Parameters
    ----------
    dx : a number which is the spacing required
    center : 1d array like object which is the center of wall
    length : a number which is the length of the wall
    num_layers : Number of layers for the wall
    up : True if the layers have to created on top of base wall

    Returns
    -------
    x : 1d numpy array with x coordinates of the wall
    y : 1d numpy array with y coordinates of the wall
    """

    x = np.arange(-length / 2., length / 2. + dx, dx) + center[0]
    y = np.ones_like(x) * center[1]
    value = 1 if up else -1
    for i in range(1, num_layers):
        y1 = np.ones_like(x) * center[1] + value * i * dx
        y = np.concatenate([y, y1])
    return np.tile(x, num_layers), y


def get_2d_tank(dx=0.05, base_center=np.array([0.0, 0.0]), length=1.0,
                height=1.0, num_layers=1, outside=True, staggered=False,
                top=False):
    """
    Generates an open 2d tank with the base parallel to x-axis and the side
    walls parallel to y-axis. The tank can be rotated to any direction using
    rotate function. 3d tank can be generated using extrude function.

        ^
        |*               *
        |*   2d tank     *
       y|*  particles    *
        |*               *
        |* * * * * * * * *
        |      base
        |____________________>
                 x

    Parameters
    ----------
    dx : a number which is the spacing required
    base_center : 1d array like object which is the center of base wall
    length : a number which is the length of the base
    height : a number which is the length of the side wall
    num_layers : Number of layers for the tank
    outside : A boolean value which decides if the layers are inside or outside
    staggered : A boolean value which decides if the layers are staggered or not
    top : A boolean value which decides if the top is present or not

    Returns
    -------
    x : 1d numpy array with x coordinates of the tank
    y : 1d numpy array with y coordinates of the tank
    """

    dy = dx
    fac = 1 if outside else 0
    if staggered:
        dx = dx/2

    start = fac*(1 - num_layers)*dx
    end = fac*num_layers*dx + (1 - fac) * dx
    x, y = np.mgrid[start:length+end:dx, start:height+end:dy]

    topset = 0 if top else 10*height
    if staggered:
        topset += dx
        y[1::2] += dx

    offset = 0 if outside else (num_layers-1)*dx
    cond = ~((x > offset) & (x < length-offset) &
             (y > offset) & (y < height+topset-offset))
    return x[cond] + base_center[0] - length/2, y[cond] + base_center[1]


def get_2d_circle(dx=0.01, r=0.5, center=np.array([0.0, 0.0])):
    """
    Generates a completely filled 2d circular area.

    Parameters
    ----------
    dx : a number which is the spacing required
    r : a number which is the radius of the circle
    center : 1d array like object which is the center of the circle

    Returns
    -------
    x : 1d numpy array with x coordinates of the circle particles
    y : 1d numpy array with y coordinates of the circle particles
    """

    N = int(2.0 * r / dx) + 1
    x, y = np.mgrid[-r:r:N * 1j, -r:r:N * 1j]
    x, y = np.ravel(x), np.ravel(y)
    condition = (x * x + y * y <= r * r)
    x, y = x[condition], y[condition]
    return x + center[0], y + center[1]


def get_2d_hollow_circle(dx=0.01, r=1.0, center=np.array([0.0, 0.0]),
                         num_layers=2, inside=True):
    """
    Generates a hollow 2d circle with some number of layers either on the
    inside or on the outside of the body which is taken as an argument

    Parameters
    ----------
    dx : a number which is the spacing required
    r : a number which is the radius of the circle
    center : 1d array like object which is the center of the circle
    num_layers : a number (int)
    inside : boolean (True or False). If this is True then the layers
             are generated inside the circle

    Returns
    -------
    x : 1d numpy array with x coordinates of the circle particles
    y : 1d numpy array with y coordinates of the circle particles
    """

    r_grid = r + dx * num_layers
    N = int(2.0 * r_grid / dx) + 1
    x, y = np.mgrid[-r_grid:r_grid:N * 1j, -r_grid:r_grid:N * 1j]
    x, y = np.ravel(x), np.ravel(y)
    if inside:
        cond1 = (x * x + y * y <= r * r)
        cond2 = (x * x + y * y >= (r - num_layers * dx)**2)
    else:
        cond1 = (x * x + y * y >= r * r)
        cond2 = (x * x + y * y <= (r + num_layers * dx)**2)
    cond = cond1 & cond2
    x, y = x[cond], y[cond]
    return x + center[0], y + center[0]


def get_3d_hollow_cylinder(dx=0.01, r=0.5, length=1.0,
                           center=np.array([0.0, 0.0, 0.0]),
                           num_layers=2, inside=True):
    """
    Generates a 3d hollow cylinder which is a extruded geometry
    of the hollow circle with a closed base.

    Parameters
    ----------
    dx : a number which is the spacing required
    r : a number which is the radius of the cylinder
    length : a number which is the length of the cylinder
    center : 1d array like object which is the center of the cylinder
    num_layers : a number (int)
    inside : boolean (True or False). If this is True then the layers
             are generated inside the cylinder

    Returns
    -------
    x : 1d numpy array with x coordinates of the cylinder particles
    y : 1d numpy array with y coordinates of the cylinder particles
    z : 1d numpy array with z coordinates of the cylinder particles
    """

    x_2d, y_2d = get_2d_hollow_circle(dx, r, center[:2], num_layers, inside)
    x, y, z = extrude(x_2d, y_2d, dx, length - dx, center[2] + dx / 2.)
    x_circle, y_circle = get_2d_circle(dx, r, center[:2])
    z_circle = np.ones_like(x_circle) * (center[2] - length / 2.)
    x = np.concatenate([x, x_circle])
    y = np.concatenate([y, y_circle])
    z = np.concatenate([z, z_circle])
    return x, y, z


def get_2d_block(dx=0.01, length=1.0, height=1.0, center=np.array([0., 0.])):
    """
    Generates a 2d rectangular block of particles with axes parallel to
    the coordinate axes.

     ^
     |
     |h * * * * * * *
     |e * * * * * * *
    y|i * * * * * * *
     |g * * * * * * *
     |h * * * * * * *
     |t * * * * * * *
     |  * * * * * * *
     |    length
     |________________>
                      x

    Parameters
    ----------
    dx : a number which is the spacing required
    length : a number which is the length of the block
    height : a number which is the height of the block
    center : 1d array like object which is the center of the block

    Returns
    -------
    x : 1d numpy array with x coordinates of the block particles
    y : 1d numpy array with y coordinates of the block particles
    """

    n1 = int(length / dx) + 1
    n2 = int(height / dx) + 1
    x, y = np.mgrid[-length / 2.:length / 2.:n1 *
                    1j, -height / 2.:height / 2.:n2 * 1j]
    x, y = np.ravel(x), np.ravel(y)
    return x + center[0], y + center[1]


def get_3d_sphere(dx=0.01, r=0.5, center=np.array([0.0, 0.0, 0.0])):
    """
    Generates a 3d sphere.

    Parameters
    ----------
    dx : a number which is the spacing required
    r : a number which is the radius of the sphere
    center : 1d array like object which is the center of the sphere

    Returns
    -------
    x : 1d numpy array with x coordinates of the sphere particles
    y : 1d numpy array with y coordinates of the sphere particles
    z : 1d numpy array with z coordinates of the sphere particles
    """

    N = int(2.0 * r / dx) + 1
    x, y, z = np.mgrid[-r:r:N * 1j, -r:r:N * 1j, -r:r:N * 1j]
    x, y, z = np.ravel(x), np.ravel(y), np.ravel(z)
    cond = (x * x + y * y + z * z <= r * r)
    x, y, z = x[cond], y[cond], z[cond]
    return x + center[0], y + center[1], z + center[2]


def get_3d_block(dx=0.01, length=1.0, height=1.0, depth=1.0,
                 center=np.array([0., 0., 0.])):
    """
    Generates a 3d block of particles with the length, height and depth
    parallel to x, y and z axis respectively.

    Paramters
    ---------
    dx : a number which is the spacing required
    length : a number which is the length of the block
    height : a number which is the height of the block
    depth : a number which is the depth of the block
    center : 1d array like object which is the center of the block

    Returns
    -------
    x : 1d numpy array with x coordinates of the block particles
    y : 1d numpy array with y coordinates of the block particles
    z : 1d numpy array with z coordinates of the block particles
    """

    n1 = int(length / dx) + 1
    n2 = int(height / dx) + 1
    n3 = int(depth / dx) + 1
    x, y, z = np.mgrid[-length / 2.:length / 2.:n1 * 1j, -height /
                       2.:height / 2.:n2 * 1j, -depth / 2.:depth / 2.:n3 * 1j]
    x, y, z = np.ravel(x), np.ravel(y), np.ravel(z)
    return x + center[0], y + center[1], z + center[2]


def get_4digit_naca_airfoil(dx=0.01, airfoil='0012', c=1.0):
    """
    Generates a 4 digit series NACA airfoil. For a 4 digit series airfoil,
    the first digit is the (maximum camber / chord) * 100, second digit is
    (location of maximum camber / chord) * 10 and the third and fourth digits
    are the (maximum thickness / chord) * 100. The particles generated
    using this function will form a solid 2d airfoil.

    Parameters
    ----------
    dx : a number which is the spacing required
    airfoil : a string of 4 characters which is the airfoil name
    c : a number which is the chord of the airfoil

    Returns
    -------
    x : 1d numpy array with x coordinates of the airfoil particles
    y : 1d numpy array with y coordinates of the airfoil particles

    References
    ----------
    https://en.wikipedia.org/wiki/NACA_airfoil
    """

    n = int(c / dx) + 1
    x, y = np.mgrid[0:c:n * 1j, -c / 2.:c / 2.:n * 1j]
    x = np.ravel(x)
    y = np.ravel(y)
    x_naca = []
    y_naca = []
    t = float(airfoil[2:]) * 0.01 * c
    if airfoil[:2] == '00':
        for xi, yi in zip(x, y):
            yt = 5.0 * t * (0.2969 * np.sqrt(xi / c) - 0.1260 * (xi / c) -
                            0.3516 * ((xi / c)**2.) + 0.2843 * ((xi / c)**3.)
                            - 0.1015 * ((xi / c)**4.))
            if abs(yi) <= yt:
                x_naca.append(xi)
                y_naca.append(yi)
    else:
        m = 0.01 * float(airfoil[0])
        p = 0.1 * float(airfoil[1])
        for xi, yi in zip(x, y):
            yt = 5.0 * t * (0.2969 * np.sqrt(xi / c) - 0.1260 * (xi / c) -
                            0.3516 * ((xi / c)**2.) + 0.2843 * ((xi / c)**3.)
                            - 0.1015 * ((xi / c)**4.))
            if xi <= p * c:
                yc = (m / (p * p)) * (2. * p * (xi / c) - (xi / c)**2.)
                dydx = (2. * m / (p * p)) * (p - xi / c) / c
            else:
                yc = (m / ((1. - p) * (1. - p))) * \
                    (1. - 2. * p + 2. * p * (xi / c) - (xi / c)**2.)
                dydx = (2. * m / ((1. - p) * (1. - p))) * (p - xi / c) / c
            theta = np.arctan(dydx)
            if yi >= 0.0:
                yu = yc + yt * np.cos(theta)
                if yi <= yu:
                    xu = xi - yt * np.sin(theta)
                    x_naca.append(xu)
                    y_naca.append(yi)
            else:
                yl = yc - yt * np.cos(theta)
                if yi >= yl:
                    xl = xi + yt * np.sin(theta)
                    x_naca.append(xl)
                    y_naca.append(yi)
    x_naca = np.array(x_naca)
    y_naca = np.array(y_naca)
    return x_naca, y_naca


def _get_m_k(series):
    if series == '210':
        return 0.058, 361.4
    elif series == '220':
        return 0.126, 51.64
    elif series == '230':
        return 0.2025, 15.957
    elif series == '240':
        return 0.290, 6.643
    elif series == '250':
        return 0.391, 3.23
    elif series == '221':
        return 0.130, 51.99
    elif series == '231':
        return 0.217, 15.793
    elif series == '241':
        return 0.318, 6.52
    elif series == '251':
        return 0.441, 3.191


def get_5digit_naca_airfoil(dx=0.01, airfoil='23112', c=1.0):
    """
    Generates a 5 digit series NACA airfoil. For a 5 digit series airfoil,
    the first digit is the design lift coefficient * 20 / 3, second digit is
    (location of maximum camber / chord) * 20, third digit indicates the
    reflexitivity of the camber and the fourth and fifth digits are the
    (maximum thickness / chord) * 100. The particles generated using this
    function will form a solid 2d airfoil.

    Parameters
    ----------
    dx : a number which is the spacing required
    airfoil : a string of 5 characters which is the airfoil name
    c : a number which is the chord of the airfoil

    Returns
    -------
    x : 1d numpy array with x coordinates of the airfoil particles
    y : 1d numpy array with y coordinates of the airfoil particles

    References
    ----------
    https://en.wikipedia.org/wiki/NACA_airfoil
    http://www.aerospaceweb.org/question/airfoils/q0041.shtml
    """

    n = int(c / dx) + 1
    x, y = np.mgrid[0:c:n * 1j, -c / 2.:c / 2.:n * 1j]
    x = np.ravel(x)
    y = np.ravel(y)
    x_naca = []
    y_naca = []
    t = 0.01 * float(airfoil[3:])
    series = airfoil[:3]
    m, k = _get_m_k(series)
    for xi, yi in zip(x, y):
        yt = 5.0 * t * (0.2969 * np.sqrt(xi / c) - 0.1260 * (xi / c) -
                        0.3516 * ((xi / c)**2.) + 0.2843 * ((xi / c)**3.)
                        - 0.1015 * ((xi / c)**4.))
        xn = xi / c
        if xn <= m:
            yc = c * (k / 6.) * (xn**3. - 3. * m *
                                 xn * xn + m * m * (3. - m) * xn)
            dydx = (k / 6.) * (3. * xn * xn - 6. * m * xn + m * m * (3. - m))
        else:
            yc = c * (k * (m**3.) / 6.) * (1. - xn)
            dydx = -(k * (m**3.) / 6.)
        theta = np.arctan(dydx)
        if yi >= 0.0:
            yu = yc + yt * np.cos(theta)
            if yi <= yu:
                xu = xi - yt * np.sin(theta)
                x_naca.append(xu)
                y_naca.append(yi)
        else:
            yl = yc - yt * np.cos(theta)
            if yi >= yl:
                xl = xi + yt * np.sin(theta)
                x_naca.append(xl)
                y_naca.append(yi)
    x_naca = np.array(x_naca)
    y_naca = np.array(y_naca)
    return x_naca, y_naca


def get_naca_wing(dx=0.01, airfoil='0012', span=1.0, chord=1.0):
    """
    Generates a wing using a NACA 4 or 5 digit series airfoil. This will
    generate only a rectangular wing.

    Parameters
    ----------
    dx : a number which is the spacing required
    airfoil : a string of 4 or 5 characters which is the airfoil name
    span : a number which is the span of the wing
    c : a number which is the chord of the wing

    Returns
    -------
    x : 1d numpy array with x coordinates of the airfoil particles
    y : 1d numpy array with y coordinates of the airfoil particles
    z : 1d numpy array with z coordinates of the airfoil particles
    """

    if len(airfoil) == 4:
        x, y = get_4digit_naca_airfoil(dx, airfoil, chord)
    elif len(airfoil) == 5:
        x, y = get_5digit_naca_airfoil(dx, airfoil, chord)
    return extrude(x, y, dx, span)


def find_overlap_particles(fluid_parray, solid_parray, dx_solid, dim=3):
    """This function will take 2 particle arrays as input and will find all the
    particles of the first particle array which are in the vicinity of the
    particles from second particle array. The function will find all the
    particles within the dx_solid vicinity so some particles may be identified
    at the outer surface of the particles from the second particle array.

    The particle arrays should atleast contain x, y and h values for a 2d case
    and atleast x, y, z and h values for a 3d case.

    Parameters
    ----------
    fluid_parray : a pysph particle array object
    solid_parray : a pysph particle array object
    dx_solid : a number which is the dx of the second particle array
    dim : dimensionality of the problem

    Returns
    -------
    list of particle indices to remove from the first array.

    """

    x = fluid_parray.x
    x1 = solid_parray.x
    y = fluid_parray.y
    y1 = solid_parray.y
    z = fluid_parray.z
    z1 = solid_parray.z
    if dim == 2:
        z = np.zeros_like(x)
        z1 = np.zeros_like(x1)
    to_remove = []
    ll_nnps = LinkedListNNPS(dim, [fluid_parray, solid_parray])
    for i in range(len(x)):
        nbrs = UIntArray()
        ll_nnps.get_nearest_particles(1, 0, i, nbrs)
        point_i = np.array([x[i], y[i], z[i]])
        near_points = nbrs.get_npy_array()
        distances = []
        for ind in near_points:
            dest = [x1[ind], y1[ind], z1[ind]]
            distances.append(distance(point_i, dest))
        if len(distances) == 0:
            continue
        elif min(distances) < (dx_solid * (1.0 - 1.0e-07)):
            to_remove.append(i)
    return to_remove


def remove_overlap_particles(fluid_parray, solid_parray, dx_solid, dim=3):
    """
    This function will take 2 particle arrays as input and will remove all
    the particles of the first particle array which are in the vicinity of
    the particles from second particle array. The function will remove all
    the particles within the dx_solid vicinity so some particles are removed
    at the outer surface of the particles from the second particle array.

    The particle arrays should atleast contain x, y and h values for a 2d case
    and atleast x, y, z and h values for a 3d case

    Parameters
    ----------
    fluid_parray : a pysph particle array object
    solid_parray : a pysph particle array object
    dx_solid : a number which is the dx of the second particle array
    dim : dimensionality of the problem

    Returns
    -------
    None
    """

    idx = find_overlap_particles(fluid_parray, solid_parray, dx_solid, dim)
    fluid_parray.remove_particles(idx)


def show_2d(points, **kw):
    """Show two-dimensional geometry data.

    The `points` are a tuple of x, y, z values, the extra keyword arguments are
    passed along to the scatter function.

    """
    import matplotlib.pyplot as plt
    plt.scatter(points[0], points[1], **kw)
    plt.xlabel('X')
    plt.ylabel('Y')


def show_3d(points, **kw):
    """Show two-dimensional geometry data.

    The `points` are a tuple of x, y, z values, the extra keyword arguments are
    passed along to the `mlab.points3d` function.

    """
    from mayavi import mlab
    mlab.points3d(points[0], points[1], points[2], **kw)
    mlab.axes(xlabel='X', ylabel='Y', zlabel='Z')
