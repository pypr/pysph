import numpy as np
from pysph.tools.geometry import get_2d_wall


def get_beach_geometry_2d(dx=0.1, l=3.0, h=1.0, flat_l=1.0, angle=45.0,
                          num_layers=3):
    """
    Generates a beach like geometry which is commonly used for simulations
    related to SPHysics.

    Parameters
    ----------
    dx : Spacing between the particles
    l : Total length of the beach
    h : Height of the wall used at the beach position
    flat_l : Length of the flat part
    angle : Angle of the inclined part
    num_layers : number of layers

    Returns
    -------
    x : 1d numpy array with x coordinates of the beach
    y : 1d numpy array with y coordinates of the beach
    x4 : 1d numpy array with x coordinates of the obstacle
    y4 : 1d numpy array with y coordinates of the obstacle
    """

    theta = np.pi * angle / 180.0
    x1, y1 = get_2d_wall(dx, np.array([(flat_l + dx) / 2.0, 0.]), flat_l,
                         num_layers, False)
    x2 = np.arange(flat_l - l, 0.0, dx * np.cos(theta))
    h2 = (l - flat_l) * np.tan(theta)
    y2_layer = x2 * np.tan(-theta)
    x2 = np.tile(x2, num_layers)
    y2 = []
    for i in range(num_layers):
        y2.append(y2_layer - i * dx)
    y2 = np.ravel(np.array(y2))
    y3 = np.arange(h2 + dx, h + h2, dx)
    x3_layer = np.ones_like(y3) * (flat_l - l)
    y3 = np.tile(y3, num_layers)
    x3 = []
    for i in range(num_layers):
        x3.append(x3_layer - i * dx)
    x3 = np.ravel(np.array(x3))
    x = np.concatenate([x1, x2, x3])
    y = np.concatenate([y1, y2, y3])
    y4 = np.arange(dx, 2.0 * h, dx)
    x4_layer = np.ones_like(y4) * flat_l
    y4 = np.tile(y4, num_layers)
    x4 = []
    for i in range(num_layers):
        x4.append(x4_layer + i * dx)
    x4 = np.ravel(np.array(x4))
    return x, y, x4, y4
