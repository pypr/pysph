"""General post-processing utility for solution data"""

TVTK = True
try:
    from tvtk.api import tvtk, write_data
except (ImportError, SystemExit):
    TVTK = False

if TVTK:
    from tvtk.array_handler import array2vtk

from os import path
import numpy as np
import pysph.solver.utils as utils


def get_ke_history(files, array_name):
    t, ke = [], []
    for sd, array in utils.iter_output(files, array_name):
        t.append(sd['t'])
        m, u, v, w = array.get('m', 'u', 'v', 'w')
        _ke = 0.5 * np.sum( m * (u**2 + v**2 + w**2) )
        ke.append(_ke)
    return np.asarray(t), np.asarray(ke)


class Results(object):
    def __init__(self, dirname=None, fname=None, endswith=".npz"):
        self.dirname = dirname
        self.fname = fname
        self.endswith = endswith

        # the starting file number
        self.start = 0

        if ( (dirname is not None) and (fname is not None) ):
            self.load()

    def set_dirname(self, dirname):
        self.dirname=dirname

    def set_fname(self, fname):
        self.fname = fname

    def load(self):
        self.files = files = utils.get_files(
            self.dirname, self.fname, self.endswith)

        self.nfiles = len(files)

    def reload(self):
        self.start = self.nfiles
        self.load()

    def get_ke_history(self, array_name):
        self.t, self.ke = get_ke_history(self.files, array_name)

    def _write_vtk_snapshot(self, mesh, directory, _fname):
        fname = path.join(directory, _fname)
        write_data( mesh, fname )

    def write_vtk(self, array_name, props):
        if not TVTK:
            return

        # create a list of props
        if type(props) != list:
            props = [ props ]

        # create an output folder for the vtk files
        dirname = path.join(self.dirname, 'vtk')
        utils.mkdir(dirname)

        nfiles = self.nfiles
        for i in range(self.start, nfiles):
            f = self.files[i]
            data = utils.load(f)

            array = data['arrays'][array_name]
            num_particles = array.num_real_particles

            # save the points
            points = np.zeros( shape=(num_particles,3) )
            points[:, 0] = array.z
            points[:, 1] = array.y
            points[:, 2] = array.x

            mesh = tvtk.PolyData(points=points)

            # add the scalar props
            for prop in props:
                if prop == 'vmag':
                    u, v, w = array.get('u','v','w')
                    numpy_array = np.sqrt(u**2 + v**2 + w**2)
                else:
                    numpy_array = array.get(prop)

                vtkarray = array2vtk(numpy_array)
                vtkarray.SetName(prop)

                # add the array as point data
                mesh.point_data.add_array(vtkarray)

            # set the last prop as the active scalar
            mesh.point_data.set_active_scalars(props[-1])

            # spit it out
            fileno = data['solver_data']['count']
            _fname = self.fname + '_%s_%s'%(array_name, fileno)

            self._write_vtk_snapshot(mesh, dirname, _fname)

class PySPH2VTK(object):
    """Convert PySPH array data to Paraview legible VTK data"""
    def __init__(self, arrays, dirname='.', fileno=None):
        self.arrays = arrays
        self.dirname = dirname
        self.fileno=fileno

        array_dict = {}
        for array in arrays:
            array_dict[ array.name ] = array

        self.array_dict = array_dict

    def _write_vtk_snapshot(self, mesh, directory, _fname):
        fname = path.join(directory, _fname)
        write_data( mesh, fname )

    def write_vtk(self, array_name, props):
        # ceck if it is possible
        if not TVTK:
            raise RuntimeError('Cannot generate VTK output!')

        # check if the array is legal
        if array_name not in list(self.array_dict.keys()):
            raise RuntimeError('Array %s not defined'%array_name)

        # create a list of props
        if type(props) != list:
            props = [ props ]

        # create an output folder for the vtk files
        dirname = path.join(self.dirname, 'vtk')
        utils.mkdir(dirname)

        array = self.array_dict[array_name]
        num_particles = array.num_real_particles

        # save the points
        points = np.zeros( shape=(num_particles,3) )
        points[:, 0] = array.z
        points[:, 1] = array.y
        points[:, 2] = array.x

        mesh = tvtk.PolyData(points=points)

        # add the scalar props
        for prop in props:
            if prop == 'vmag':
                u, v, w = array.get('u','v','w')
                numpy_array = np.sqrt(u**2 + v**2 + w**2)
            else:
                numpy_array = array.get(prop)

            vtkarray = array2vtk(numpy_array)
            vtkarray.SetName(prop)

            # add the array as point data
            mesh.point_data.add_array(vtkarray)

        # set the last prop as the active scalar
        mesh.point_data.set_active_scalars(props[-1])

        # spit it out
        if self.fileno is None:
            _fname = '%s'%(array_name)
        else:
            _fname = '%s_%03d'%(array_name, self.fileno)

        self._write_vtk_snapshot(mesh, dirname, _fname)
