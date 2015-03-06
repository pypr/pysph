"""Simple reader for VTK data sets. This can be used to generate particles
from any input VTK dataset.

This is useful because Gmsh can save its output as VTK datasets which can then
be converted into particles.

Copyright (c) 2015 Prabhu Ramachandran
"""

import numpy as np
from tvtk.api import tvtk

def read_vtk_file(fname):
    """Given a .vtk file, read it and return the output.
    """
    r = tvtk.DataSetReader(file_name=fname)
    r.update()
    return r.output

def convert_to_points(dataset, vertices=True, cell_centers=True):
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
