"""Utility module to read input mesh files.  This is primarily for meshes
generated using Gmsh.  This module also provides some simple classes
that allow one to create extruded 3D surfaces by generating a gmsh file
in Python.

There is also a function to read VTK dataset and produce points from them.
This is very useful as Gmsh can generate VTK datasets from its meshes and thus
the meshes can be imported as point clouds that may be used in an SPH
simulation.
"""
# Copyright (c) 2015 Prabhu Ramachandran

import gzip
import json

import numpy as np
import subprocess

import tempfile
from tvtk.api import tvtk

import os
from os.path import exists, expanduser, join
import sys

def _read_vtk_file(fname):
    """Given a .vtk file (or .vtk.gz), read it and return the output.
    """
    if fname.endswith('.vtk.gz'):
        tmpfname = tempfile.mktemp(suffix='.vtk')
        with open(tmpfname, 'wb') as tmpf:
            data = gzip.open(fname).read()
            tmpf.write(data)
        r = tvtk.DataSetReader(file_name=tmpfname)
    else:
        tmpfname = None
        r = tvtk.DataSetReader(file_name=fname)
    r.update()
    if tmpfname is not None:
        os.remove(tmpfname)
    return r.output

def _convert_to_points(dataset, vertices=True, cell_centers=True):
    """Given a VTK dataset, convert it to a set of points that can be used
    for simulation with SPH.

    Parameters
    ----------

    dataset : tvtk.DataSet
    vertices: bool
        If True, it converts the vertices to points.
    cell_centers: bool
        If True, converts the cell centers to points.

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

    fname : str
        File name.
    vertices: bool
        If True, it converts the vertices to points.
    cell_centers: bool
        If True, converts the cell centers to points.

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


class Loop(object):
    """Create a Line Loop in Gmsh parlance but using a turtle-graphics like
    approach.

    Use this to create a 2D closed surface.  The surface is always in the x-y
    plane.

    Examples
    --------

    Here is a simple example::

      >>> l1 = Loop((0.0, 0.0), mesh_size=0.1)
      >>> l1.move(1.0).turn(90).move(1.0).turn(90).move(1.0).turn(90).move(1.0)

    This will create a square shape.

    """

    def __init__(self, start, mesh_size=0.1):
        self.mesh_size = mesh_size
        self.points = [start]
        self.elems = []
        self.tolerance = 1e-12
        self._last_angle = 0.0
        self._last_position = start
        self._index = 1

    ### Public Protocol ###################################
    def turn(self, angle):
        """Turn by angle (in degrees).
        """
        self._last_angle += angle
        return self

    def move(self, dist):
        """Move by given distance at the current angle.
        """
        x0, y0 = self._last_position
        angle = self._last_angle
        rad = np.pi*angle/180.
        x = x0 + dist*np.cos(rad)
        y = y0 + dist*np.sin(rad)
        p1 = self._index
        p2 = self._add_point(x, y)
        self._add_elem('line', (p1, p2))
        return self

    def arc(self, radius, angle=180):
        """Create a circular arc given the radius and for an angle as
        specified.
        """
        x0, y0 = self._last_position
        last_angle = self._last_angle
        rad = np.pi*last_angle/180.
        last = complex(x0, y0)
        dz = complex(np.cos(rad), np.sin(rad))
        cen = last + radius*dz
        rad = np.pi*angle/180.
        end = cen + (last - cen)*complex(np.cos(rad), np.sin(rad))
        s_idx = self._index
        c_idx = self._add_point(cen.real, cen.imag)
        e_idx = self._add_point(end.real, end.imag)
        self._add_elem('circle', (s_idx, c_idx, e_idx))
        return self

    def write(self, fp, point_id_base=0, elem_id_base=0):
        """Write data to given file object `fp`.
        """
        points = self.points
        for i in range(len(points)):
            idx = point_id_base + i + 1
            self._write_point(fp, points[i], idx)

        elems = self.elems
        idx = 0
        loop = []
        for i in range(len(elems)):
            elem = elems[i]
            idx = elem_id_base + i + 1
            loop.append(str(idx))
            kind, data = elem[0], elem[1]
            if kind == 'line':
                self._write_line(fp, data, idx, point_id_base)
            elif kind == 'circle':
                self._write_circle(fp, data, idx, point_id_base)
        idx += 1
        ids = ', '.join(loop)
        s = 'Line Loop({idx}) = {{{ids}}};\n'.format(idx=idx, ids=ids)
        fp.write(s)
        return len(points), len(elems) + 1

    ### Private Protocol #################################
    def _add_elem(self, kind, data):
        self.elems.append((kind, data))

    def _check_for_existing_point(self, x, y):
        tol = self.tolerance
        for i, (xi, yi) in enumerate(self.points):
            if abs(xi - x) < tol and abs(yi - y) < tol:
                return i + 1

    def _add_point(self, x, y):
        last_x, last_y = self._last_position
        pnt = (x, y)
        existing = self._check_for_existing_point(x, y)
        if existing is None:
            self.points.append(pnt)
            self._index += 1
            index = self._index
        else:
            index = existing
            pnt = self.points[existing]
        self._last_position = pnt
        return index

    def _write_point(self, fp, pnt, idx):
        fp.write('Point({idx}) = {{{x}, {y}, 0.0, {mesh_size}}};\n'.\
            format(x=pnt[0], y=pnt[1], idx=idx, mesh_size=self.mesh_size))

    def _write_line(self, fp, data, e_idx, p_idx):
        s = 'Line({idx}) = {{{p1}, {p2}}};\n'.format(
            idx=e_idx, p1=data[0]+p_idx, p2=data[1] +p_idx
        )
        fp.write(s)

    def _write_circle(self, fp, data, e_idx, p_idx):
        s = 'Circle({idx}) = {{{start}, {cen}, {end}}};\n'.format(
            idx=e_idx, start=data[0]+p_idx, cen=data[1]+p_idx, end=data[2] +p_idx
        )
        fp.write(s)


class Surface(object):
    def __init__(self, *loops):
        """Constructor.

        Parameters
        ----------

        loops : tuple(Loop)
            Any additional positional arguments are treated as loop objects.
        """
        self.loops = list(loops)
        self.idx = 0

    def write(self, fp, point_id_base=0, elem_id_base=0):
        pid_base = point_id_base
        eid_base = elem_id_base
        loop_ids = []
        for loop in self.loops:
            np, ne = loop.write(fp, pid_base, eid_base)
            pid_base += np
            eid_base += ne
            loop_ids.append(str(eid_base))
        idx = eid_base + 1
        loop_str = ', '.join(loop_ids)
        fp.write('Plane Surface({idx}) = {{{loop_str}}};\n'.format(
                idx=idx, loop_str=loop_str))
        self.idx = idx
        return pid_base, eid_base


class Extrude(object):
    def __init__(self, dx=0.0, dy=0.0, dz=1.0, surfaces=None):
        """Extrude a given set of surfaces by the displacements given
        along each directions.

        Parameters
        ----------

        dx : float
            Extrusion along x.
        dy : float
            Extrusion along y.
        dz : float
            Extrusion along z.
        surfaces: list
            List of surfaces to extrude.

        """
        self.dx, self.dy, self.dz = dx, dy, dz
        if surfaces is None:
            self.surfaces = []
        else:
            self.surfaces = list(surfaces)

    def write(self, fp, point_id_base=0, elem_id_base=0):
        pid_base = point_id_base
        eid_base = elem_id_base
        surf_ids = []
        for surf in self.surfaces:
            np, ne = surf.write(fp, pid_base, eid_base)
            pid_base += np
            eid_base += ne
            surf_ids.append(str(surf.idx))

        s_ids = ', '.join(surf_ids)
        fp.write(
            'Extrude {{{dx}, {dy}, {dz}}} {{\n'
            '    Surface{{{s_ids}}};\n}}\n'.format(
            dx=self.dx, dy=self.dy, dz=self.dz, s_ids=s_ids
            )
        )
        return pid_base, eid_base


class Gmsh(object):
    def __init__(self, gmsh=None):
        """Construct a Gmsh helper object that can be used to mesh objects.

        Parameters
        ----------

        gmsh: str
            Path to gmsh executable.
        """
        self.config = expanduser(join('~', '.pysph', 'gmsh.json'))
        if gmsh is None:
            if exists(self.config):
                self._read_config()
            else:
                gmsh = self._ask_user_for_gmsh()
                self._set_gmsh(gmsh)
        else:
            self._set_gmsh(gmsh)

    #### Public Protocol ##################################
    def write_geo(self, entities, fp):
        """Write a list of given entities to the given file pointer.
        """
        p_count, e_count = 0, 0
        for entity in entities:
            np, ne = entity.write(fp, p_count, e_count)
            p_count += np
            e_count += ne

    def write_vtk_mesh(self, entities, fname):
        """Write a list of given entities to the given file name.
        """
        tmp_geo = tempfile.mktemp(suffix='.geo')
        try:
            self.write_geo(entities, open(tmp_geo, 'w'))
            self._call_gmsh('-3', tmp_geo, '-o', fname)
        finally:
            os.remove(tmp_geo)

    def get_points(self, entities, vertices=True, cell_centers=False):
        """Given a list of entities, return x, y, z arrays for the position.

        Parameters
        ----------

        entities : list
            List of entities.
        vertices: bool
            If True, it converts the vertices to points.
        cell_centers: bool
            If True, converts the cell centers to points.

        """
        tmp_vtk = tempfile.mktemp(suffix='.vtk')
        try:
            self.write_vtk_mesh(entities, tmp_vtk)
            x, y, z = vtk_file_to_points(
                tmp_vtk, vertices=vertices, cell_centers=cell_centers
            )
            return x, y, z
        finally:
            os.remove(tmp_vtk)

    def get_points_from_geo(self, geo_file_name, vertices=True,
                            cell_centers=False):
        """Given a .geo file, generate a mesh and get the points from the mesh.

        Parameters
        ----------

        geo_file_name: str
            Filename of the .geo file.
        vertices: bool
            If True, it converts the vertices to points.
        cell_centers: bool
            If True, converts the cell centers to points.

        """
        tmp_vtk = tempfile.mktemp(suffix='.vtk')
        try:
            self._call_gmsh('-3', geo_file_name, '-o', tmp_vtk)
            return vtk_file_to_points(
                tmp_vtk, vertices=vertices, cell_centers=cell_centers
            )
        finally:
            os.remove(tmp_vtk)

    #### Private Protocol #################################
    def _ask_user_for_gmsh(self):
        gmsh = input('Please provide the path to gmsh executable: ')
        return gmsh

    def _read_config(self):
        if exists(self.config):
            data = json.load(open(self.config))
            self.gmsh = data['path']

    def _set_gmsh(self, gmsh):
        self.gmsh = gmsh
        data = dict(path=gmsh)
        json.dump(data, open(self.config, 'w'))

    def _call_gmsh(self, *args):
        if self.gmsh is None:
            raise RuntimeError('Gmsh is not configured, set the gmsh path.')
        cmd = [self.gmsh] + list(args)
        subprocess.check_call(cmd)


def example_3d_p(fp=sys.stdout):
    """Creates a 3D "P" with a hole inside it.
    """
    # The exterior of the "P"
    l1 = Loop((0.0, 0.0), mesh_size=0.1)
    l1.turn(-90).move(1.0).turn(90).move(0.2).turn(90).move(0.5)\
       .arc(0.25, -180).turn(90).move(0.2)
    # The inner loop for the hole in the middle.
    l2 = Loop((0.1, -0.25))
    l2.arc(0.1, 90).turn(90).arc(0.1, 90).turn(90)\
      .arc(0.1, 90).turn(90).arc(0.1, 90)
    s = Surface(l1, l2)
    ex = Extrude(0.0, 0.0, 1.0, surfaces=[s])
    ex.write(fp)
    return ex

def example_cube(fp=sys.stdout):
    """Simple example of a cube.
    """
    l1 = Loop((0.0, 0.0), mesh_size=0.1)
    l1.move(1.0).turn(90).move(1.0).turn(90).move(1.0).turn(90).move(1.0)
    s = Surface(l1)
    ex = Extrude(0.0, 0.0, 1.0, surfaces=[s])
    ex.write(fp)
    return ex

def example_plot_3d_p(gmsh):
    """Note: this will only work if you have gmsh installed.
    """
    import io
    fp = io.StringIO()
    ex = example_3d_p(fp)
    g = Gmsh(gmsh)
    x, y, z = g.get_points([ex])
    from mayavi import mlab
    mlab.points3d(x, y, z, color=(1, 0, 0))
