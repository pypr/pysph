"""Simple reader for VTK data sets. This can be used to generate particles
from any input VTK dataset.

This is useful because Gmsh can save its output as VTK datasets which can then
be converted into particles.

Copyright (c) 2015 Prabhu Ramachandran
"""

import numpy as np
from tvtk.api import tvtk

def _read_vtk_file(fname):
    """Given a .vtk file, read it and return the output.
    """
    r = tvtk.DataSetReader(file_name=fname)
    r.update()
    return r.output

def _convert_to_points(dataset, vertices=True, cell_centers=True):
    """Given a VTK dataset, convert it to a set of points that can be used
    for simulation with SPH.

    Parameters
    ----------

     - dataset : tvtk.DataSet
     - vertices: bool: if True, it converts the vertices to points.
     - cell_centers: bool: if True, converts the cell centers to points.

    Returns
    -------

    x, y, z of the points.
    """
    pts = np.array([], dtype=float)
    if vertices:
        pts = np.append(pts, dataset.points.to_array())
    if cell_centers:
        cell_centers = tvtk.CellCenters(input=dataset)
        cell_centers.update()
        p = cell_centers.output.points.to_array()
        pts = np.append(pts, p)
    pts.shape = len(pts)/3, 3
    x, y, z = pts.T
    return x, y, z

def vtk_file_to_points(fname, vertices=True, cell_centers=True):
    """Given a file containing a VTK dataset (currently only an old style .vtk
    file), convert it to a set of points that can be used for simulation with
    SPH.

    Parameters
    ----------

     - fname : str: file name,
     - vertices: bool: if True, it converts the vertices to points.
     - cell_centers: bool: if True, converts the cell centers to points.

    Returns
    -------

    x, y, z of the points.
    """
    dataset = _read_vtk_file(fname)
    return _convert_to_points(dataset, vertices, cell_centers)

def transform_points(x, y, z, transform):
    """Given the coordinates, x, y, z and the TVTK transform instance,
    return the transformed coordinates.
    """
    assert isinstance(transform, tvtk.Transform)

    m = transform.matrix.to_array()
    xt, yt, zt, wt = np.dot(m, np.array((x, y, z, np.ones_like(x))))
    return xt, yt, zt
