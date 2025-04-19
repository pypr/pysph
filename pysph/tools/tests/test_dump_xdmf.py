import shutil
import unittest
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
from pytest import importorskip

from pysph.base.utils import get_particle_array
from pysph.solver.output import dump
from pysph.tools.dump_xdmf import main as dump_xdmf


class TestDumpXDMF(unittest.TestCase):
    def test_dump_xdmf(self, npoints=10, random_seed=0):
        vtk = importorskip('vtk')
        np.random.seed(random_seed)
        tmp_dir = mkdtemp()
        hdf5file = str(Path(tmp_dir) / 'test.hdf5')

        # Assign rho as random data, make a particle array and dump it as hdf5.
        self.rho = np.random.rand(npoints)
        pa = get_particle_array(name='fluid',
                                rho=self.rho,
                                x=np.arange(npoints),
                                y=np.zeros(npoints),
                                z=np.zeros(npoints))
        dump(hdf5file, [pa], {})

        try:
            # Generate XDMF for dumped hdf5 file.
            dump_xdmf([hdf5file, '--combine-particle-arrays', '--outdir', tmp_dir])

            # Retrieve data by reading xdmf file
            xdmffile = Path(hdf5file).with_suffix('.xdmf')
            reader = vtk.vtkXdmfReader()
            reader.SetFileName(xdmffile)
            reader.Update()
            block = reader.GetOutput().GetBlock(0)
            point_data = block.GetPointData()
            array_data = {}
            for i in range(point_data.GetNumberOfArrays()):
                vtk_array = point_data.GetArray(i)
                if vtk_array:
                    array_name = vtk_array.GetName()
                    array_data[array_name] = np.array(vtk_array)

            # Check if retrieved data and generated data is same.
            assert np.allclose(self.rho, array_data['rho'], atol=1e-14), \
                "Expected %s,\n got %s" % (self.rho, array_data['rho'])

            del reader
            # Note: Ideally, the reader need not be deleted as reader.Update()
            # itself will open the file, read its contents, and then close the
            # file.
            # Ref. https://discourse.vtk.org/t/does-a-python-reader-need-to-be-closed/7261/3 # noqa
            #
            # But shutil.rmtree(tmp_dir) still gives the following error on
            # windows:
            # """
            # The process cannot access the file because it is being used by
            # another process: 'C:\\Users\\RUNNER~1\\AppData\\Local\\
            # Temp\\tmpqft4428c\\test.xdmf'
            # """
            # Deleting the reader avoids the error.

        finally:
            shutil.rmtree(tmp_dir)


if __name__ == '__main__':
    unittest.main()
