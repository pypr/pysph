import numpy as np
import shutil
import os
from os.path import join
import socket
from tempfile import mkdtemp
from pysph import has_h5py

try:
    # This is for Python-2.6.x
    from unittest2 import TestCase, main, skipUnless
except ImportError:
    from unittest import TestCase, main, skipUnless

from pysph.base.utils import get_particle_array, get_particle_array_wcsph
from pysph.solver.utils import dump, load, dump_v1, get_files, get_free_port


class TestGetFiles(TestCase):
    def setUp(self):
        self.root = mkdtemp()
        self.fname = 'dam_break_2d'
        self.dirname = join(self.root, self.fname + '_output')
        os.mkdir(self.dirname)
        self.files = [
            join(
                self.dirname,
                self.fname+'_'+str(i)+'.npz'
            )
            for i in range(11)
        ]
        for name in self.files:
            with open(name, 'w') as fp:
                fp.write('')

    def test_get_files(self):
        self.assertEqual(get_files(self.dirname), self.files)
        self.assertEqual(get_files(self.dirname, fname=self.fname), self.files)
        self.assertEqual(
            get_files(
                self.dirname,
                fname=self.fname,
                endswith=('npz', 'hdf5')
            ),
            self.files
        )

    def tearDown(self):
        shutil.rmtree(self.root)


class TestOutputNumpy(TestCase):
    def setUp(self):
        self.root = mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.root)

    def _get_filename(self, fname):
        return join(self.root, fname) + '.npz'

    def test_dump_and_load_works_by_default(self):
        x = np.linspace(0, 1.0, 10)
        y = x*2.0
        dt = 1.0
        pa = get_particle_array(name='fluid', x=x, y=y)
        fname = self._get_filename('simple')
        dump(fname, [pa], solver_data={'dt': dt})
        data = load(fname)
        solver_data = data['solver_data']
        arrays = data['arrays']
        pa1 = arrays['fluid']
        self.assertListEqual(list(solver_data.keys()), ['dt'])
        self.assertListEqual(list(sorted(pa.properties.keys())),
                             list(sorted(pa1.properties.keys())))
        self.assertTrue(np.allclose(pa.x, pa1.x, atol=1e-14))
        self.assertTrue(np.allclose(pa.y, pa1.y, atol=1e-14))

    def test_dump_and_load_works_with_compress(self):
        x = np.linspace(0, 1.0, 1000)
        y = x*2.0
        dt = 1.0
        pa = get_particle_array(name='fluid', x=x, y=y)
        fname = self._get_filename('simple')
        dump(fname, [pa], solver_data={'dt': dt})
        fnamez = self._get_filename('simplez')
        dump(fnamez, [pa], solver_data={'dt': dt}, compress=True)
        # Check that the file size is indeed smaller
        self.assertTrue(os.stat(fnamez).st_size < os.stat(fname).st_size)

        data = load(fnamez)
        solver_data = data['solver_data']
        arrays = data['arrays']
        pa1 = arrays['fluid']
        self.assertListEqual(list(solver_data.keys()), ['dt'])
        self.assertListEqual(list(sorted(pa.properties.keys())),
                             list(sorted(pa1.properties.keys())))
        self.assertTrue(np.allclose(pa.x, pa1.x, atol=1e-14))
        self.assertTrue(np.allclose(pa.y, pa1.y, atol=1e-14))

    def test_dump_and_load_with_partial_data_dump(self):
        x = np.linspace(0, 1.0, 10)
        y = x*2.0
        pa = get_particle_array_wcsph(name='fluid', x=x, y=y)
        pa.set_output_arrays(['x', 'y'])
        fname = self._get_filename('simple')
        dump(fname, [pa], solver_data={})
        data = load(fname)
        arrays = data['arrays']
        pa1 = arrays['fluid']
        self.assertListEqual(list(sorted(pa.properties.keys())),
                             list(sorted(pa1.properties.keys())))
        self.assertTrue(np.allclose(pa.x, pa1.x, atol=1e-14))
        self.assertTrue(np.allclose(pa.y, pa1.y, atol=1e-14))

    def test_dump_and_load_with_constants(self):
        x = np.linspace(0, 1.0, 10)
        y = x*2.0
        pa = get_particle_array_wcsph(name='fluid', x=x, y=y,
                                      constants={'c1': 1.0, 'c2': [2.0, 3.0]})
        pa.add_property('A', data=2.0, stride=2)
        pa.set_output_arrays(['x', 'y', 'A'])
        fname = self._get_filename('simple')
        dump(fname, [pa], solver_data={})
        data = load(fname)
        arrays = data['arrays']
        pa1 = arrays['fluid']
        self.assertListEqual(list(sorted(pa.properties.keys())),
                             list(sorted(pa1.properties.keys())))
        self.assertListEqual(list(sorted(pa.constants.keys())),
                             list(sorted(pa1.constants.keys())))
        self.assertTrue(np.allclose(pa.x, pa1.x, atol=1e-14))
        self.assertTrue(np.allclose(pa.y, pa1.y, atol=1e-14))
        self.assertTrue(np.allclose(pa.A, pa1.A, atol=1e-14))
        self.assertTrue(np.allclose(pa.c1, pa1.c1, atol=1e-14))
        self.assertTrue(np.allclose(pa.c2, pa1.c2, atol=1e-14))

    def test_that_output_array_information_is_saved(self):
        # Given
        x = np.linspace(0, 1.0, 10)
        y = x*2.0
        pa = get_particle_array(name='fluid', x=x, y=y, u=3*x)

        # When
        output_arrays = ['x', 'y', 'u']
        pa.set_output_arrays(output_arrays)
        fname = self._get_filename('simple')
        dump(fname, [pa], solver_data={})
        data = load(fname)
        pa1 = data['arrays']['fluid']

        # Then.
        self.assertEqual(set(pa.output_property_arrays), set(output_arrays))
        self.assertEqual(set(pa1.output_property_arrays), set(output_arrays))


class TestOutputHdf5(TestOutputNumpy):
    @skipUnless(has_h5py(), "h5py module is not present")
    def setUp(self):
        super(TestOutputHdf5, self).setUp()

    def _get_filename(self, fname):
        return join(self.root, fname) + '.hdf5'


class TestOutputNumpyV1(TestCase):
    def setUp(self):
        self.root = mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.root)

    def _get_filename(self, fname):
        return join(self.root, fname) + '.npz'

    def test_load_works_with_dump_version1(self):
        x = np.linspace(0, 1.0, 10)
        y = x*2.0
        pa = get_particle_array(name='fluid', x=x, y=y)
        fname = self._get_filename('simple')
        dump_v1(fname, [pa], solver_data={})
        data = load(fname)
        arrays = data['arrays']
        pa1 = arrays['fluid']
        self.assertListEqual(list(sorted(pa.properties.keys())),
                             list(sorted(pa1.properties.keys())))
        self.assertTrue(np.allclose(pa.x, pa1.x, atol=1e-14))
        self.assertTrue(np.allclose(pa.y, pa1.y, atol=1e-14))


class TestGetFreePort(TestCase):
    def test_finds_port_not_taken(self):
        # Given
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(('', 9800))
        self.addCleanup(sock.close)

        # When
        port = get_free_port(9800)

        # Then
        self.assertNotEqual(port, 9800)
        self.assertTrue(port > 9800)

        # When
        port1 = get_free_port(port)

        # Then
        # getting a free port should not block that port.
        self.assertEqual(port, port1)

    def test_free_port_skips_given(self):
        # Given
        skip = (9800, 9801)

        # When
        port = get_free_port(9800, skip=skip)

        # Then
        self.assertNotIn(port, skip)
        self.assertTrue(port > 9801)


if __name__ == '__main__':
    main()
