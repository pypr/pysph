"""General post-processing utility for solution data"""

VV=True
try:
    import visvis
except ImportError:
    VV=False

TVTK=True
try:
    from tvtk.api import tvtk, write_data
except ImportError:
    TVTK = False

if TVTK:
    from tvtk.array_handler import array2vtk

from os import path
import numpy as np
import pysph.solver.utils as utils

class Results(object):
    def __init__(self, dirname=None, fname=None, endswith=".npz"):
        self.dirname = dirname
        self.fname = fname
        self.endswith = endswith

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

    def get_ke_history(self, array_name):
        nfiles = self.nfiles

        self.ke = ke = np.zeros(nfiles, dtype=np.float64)
        self.t = t = np.zeros(nfiles, dtype=np.float64)

        for i in range(nfiles):
            data = utils.load(self.files[i])

            # save the time array
            t[i] = data['solver_data']['t']

            array = data['arrays'][array_name]
            
            m, u, v, w = array.get('m', 'u', 'v', 'w')
            ke[i] = 0.5 * np.sum( m * (u**2 + v**2) )

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
        for i in range(nfiles):
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
