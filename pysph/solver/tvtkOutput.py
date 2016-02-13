from pysph import has_tvtk, has_pyvisfile
from pysph.solver.output import Output
import numpy as np


class VtkOutput(Output):

    def __init__(self, only_real=True, mpi_comm=None, scalar_array=[],
                 vector_array=[]):
        self.set_output_scalar(scalar_array)
        self.set_output_vector(vector_array)
        super(VtkOutput, self).__init__(False, only_real, mpi_comm)

    def set_output_vector(self, vector_array={}):
        """
        Set the vector to dump in vtk output

        Parameter
        ----------

        vector_array: dictionary of list
            dictionary of properties   to dump
            The inner list can only have three elements
            example: {'P': ['x', 'y', 'z'], 'V': ['u', 'v', 'w'}]
        """

        for name, vector in vector_array.items():
            assert(len(vector), 3)
        self.vector_array = vector_array

    def set_output_scalar(self, scalar_array=[]):
        """
        Set the scalars to dump in vtk_output

        Parameter
        ---------
        scalar_array: list
            The set of properties to dump
        """

        self.scalar_array = scalar_array

    def _get_scalars(self, arrays):
        scalars = []
        for prop_name in self.scalar_array:
            scalars.append((prop_name, arrays[prop_name]))
        return scalars

    def _get_vectors(self, arrays):
        vectors = []
        for prop_name, prop_list in self.vector_array.items():
            vec = np.array([arrays[prop_list[0]], arrays[prop_list[1]],
                            arrays[prop_list[2]]])
            data = (prop_name, vec)
            vectors.append(data)
        return vectors

    def _dump(self, filename):
        for ptype, pdata in self.all_array_data.items():
            self._setup_data(pdata)
            self._dump_arrays(filename + '_' + ptype)

    def _setup_data(self, arrays):
        self.numPoints = arrays['x'].size
        self.points = np.array([arrays['x'], arrays['y'],
                                arrays['z']])
        self.data = []
        self.data.extend(self._get_scalars(arrays))
        self.data.extend(self._get_vectors(arrays))


class PyVisFileOutput(VtkOutput):

    def _dump_arrays(self, filename):
        from pyvisfile.vtk import (UnstructuredGrid, DataArray,
                                   AppendedDataXMLGenerator, VTK_VERTEX)
        n = self.numPoints
        da = DataArray("points", self.points)
        grid = UnstructuredGrid((n, da), cells=np.arange(n),
                                cell_types=np.asarray([VTK_VERTEX] * n))
        for name, feild in self.data:
            da = DataArray(name, feild)
            grid.add_pointdata(da)
        with open(filename + '.vtu', "w") as f:
            AppendedDataXMLGenerator(None)(grid).write(f)


class TvtkOutput(VtkOutput):
    def _dump_arrays(self, filename):
        from tvtk.api import tvtk
        n = self.numPoints
        cells = np.arange(n)
        cells.shape = (n, 1)
        cell_type = tvtk.Vertex().cell_type
        ug = tvtk.UnstructuredGrid(points=self.points.transpose())
        ug.set_cells(cell_type, cells)
        from mayavi.core.dataset_manager import DatasetManager
        dsm = DatasetManager(dataset=ug)
        for name, feild in self.data:
            dsm.add_array(feild.transpose(), name)
            dsm.activate(name)
        from tvtk.api import write_data
        write_data(ug, filename)


def dump_vtk(filename, particles, only_real=True, mpi_comm=None,
             scalars=[], vectors={}):
    if has_pyvisfile():
        output = PyVisFileOutput(only_real, mpi_comm, scalars, vectors)
    elif has_tvtk():
        output = TvtkOutput(only_real, mpi_comm, scalars, vectors)
    else:
        msg = 'Tvtk and pyvisfile Not present'
        raise ImportError(msg)
    output.dump(filename, particles)
