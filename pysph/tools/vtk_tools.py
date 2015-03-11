"""Simple reader for VTK data sets. This can be used to generate particles
from any input VTK dataset.

This is useful because Gmsh can save its output as VTK datasets which can then
be converted into particles.

Copyright (c) 2015 Prabhu Ramachandran
"""

import json

import numpy as np
import subprocess

import tempfile
from tvtk.api import tvtk

import os
from os.path import exists, expanduser, join
import sys

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


class Loop(object):
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
        self._last_angle += angle
        return self

    def move(self, dist):
        x0, y0 = self._last_position
        angle = self._last_angle
        rad = np.pi*angle/180.
        x = x0 + dist*np.cos(rad)
        y = y0 + dist*np.sin(rad)
        p1 = self._index
        p2 = self._add_point(x, y)
        self._add_elem('line', (p1, p2))
        return self

    def arc_to(self, disp):
        x0, y0 = self._last_position
        angle = self._last_angle
        rad = np.pi*angle/180.
        dz = complex(np.cos(rad), np.sin(rad))
        end = disp*dz
        cen = end*0.5
        s_idx = self._index
        c_idx = self._add_point(x0 + cen.real, y0 + cen.imag)
        e_idx = self._add_point(x0 + end.real, y0 + end.imag)
        self._add_elem('circle', (s_idx, c_idx, e_idx))
        return self

    def write(self, fp, point_id_base=0, elem_id_base=0):
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
    def __init__(self, dx=0.0, dy=0.0, dz=1.0, *surfaces):
        self.dx, self.dy, self.dz = dx, dy, dz
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
    def __init__(self, gmsh=None, entities=None):
        self.entities = entities
        self.config = expanduser(join('~', '.pysph', 'gmsh.json'))
        if gmsh is None and exists(self.config):
            self._read_config()
        else:
            self._set_gmsh(gmsh)

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

    def set_entities(self, entities):
        self.entities = entities

    def write_geo(self, fp):
        p_count, e_count = 0, 0
        for entity in self.entities:
            np, ne = entity.write(fp, p_count, e_count)
            p_count += np
            e_count += ne

    def write_vtk_mesh(self, fname):
        tmp_geo = tempfile.mktemp(suffix='.geo')
        try:
            self.write_geo(open(tmp_geo, 'w'))
            self._call_gmsh('-3', tmp_geo, '-o', fname)
        finally:
            os.remove(tmp_geo)

    def get_points(self, vertices=True, cell_centers=False):
        tmp_vtk = tempfile.mktemp(suffix='.vtk')
        try:
            self.write_vtk_mesh(tmp_vtk)
            x, y, z = vtk_file_to_points(
                tmp_vtk, vertices=vertices, cell_centers=cell_centers
            )
            return x, y, z
        finally:
            os.remove(tmp_vtk)

def example(fp=sys.stdout):
    """Creates a 3D "P".
    """
    l1 = Loop((0.0, 0.0), mesh_size=0.1)
    l1.turn(-90).move(1.0).turn(90).move(0.2).turn(90).move(0.5)\
       .arc_to(0.5).turn(90).move(0.2)
    #l2 = Loop((0.15, -0.25))
    #l2.arc_to(0.1).turn(180).arc_to(0.1)
    s = Surface(l1)
    ex = Extrude(0.0, 0.0, 1.0, s)
    ex.write(fp)
    return ex
