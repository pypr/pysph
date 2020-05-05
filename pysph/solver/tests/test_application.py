# Author: Anshuman Kumar

try:
    # This is for Python-2.6.x
    from unittest2 import TestCase
except ImportError:
    from unittest import TestCase

try:
    from unittest import mock
except ImportError:
    import mock

import os
import shutil
import sys
from tempfile import mkdtemp
import time

from pysph.solver.application import Application
from pysph.solver.solver import Solver
from pysph.solver.utils import get_free_port
from pysph.solver.solver_interfaces import MultiprocessingInterface, XMLRPCInterface


class MockApp(Application):

    @mock.patch('pysph.solver.application.in_parallel', return_value=False)
    def __init__(self, mock_in_parallel, *args, **kw):
        super(MockApp, self).__init__(*args, **kw)

    def add_user_options(self, group):
        group.add_argument(
            "--testarg",
            action="store",
            type=float,
            dest="testarg",
            default=int(10.0),
            help="Test Argument"
        )

    def consume_user_options(self):
        self.testarg = self.options.testarg

    def create_particles(self):
        return []

    def create_equations(self):
        return []

    def create_solver(self):
        solver = Solver()
        solver.particles = []
        solver.solve = mock.Mock()
        solver.setup = mock.Mock()
        return solver

    def create_nnps(self):
        nnps = mock.Mock()
        return nnps


class TestApplication(TestCase):

    def setUp(self):
        self.output_dir = mkdtemp()
        self.app = MockApp(output_dir=self.output_dir)

    def tearDown(self):
        if sys.platform.startswith('win'):
            try:
                shutil.rmtree(self.output_dir)
            except WindowsError:
                pass
        else:
            shutil.rmtree(self.output_dir)

    def test_user_options_when_args_are_not_passed(self):
        # Given
        app = self.app

        # When
        args = []
        app.run(args)

        # Then
        self.assertEqual(app.comm, None)
        expected = 10.0
        error_message = "Expected %f, got %f" % (expected, app.testarg)
        self.assertEqual(expected, app.testarg, error_message)

    def test_user_options_when_args_are_passed(self):
        # Given
        app = self.app

        # When
        args = ['--testarg', '20']
        app.run(args)

        # Then
        expected = 20.0
        error_message = "Expected %f, got %f" % (expected, app.testarg)
        self.assertEqual(expected, app.testarg, error_message)
        # No interfaces should be started by default.
        self.assertEqual(len(app._interfaces), 0)

    def test_output_dir_when_moved_and_read_info_called(self):
        # Given
        app = self.app

        args = ['-d', app.output_dir]
        app.run(args)

        copy_root = mkdtemp()
        copy_dir = os.path.join(copy_root, 'new')
        shutil.copytree(app.output_dir, copy_dir)
        self.addCleanup(shutil.rmtree, copy_root)
        orig_fname = app.fname

        # When
        app = MockApp()
        app.read_info(copy_dir)

        # Then
        realpath = os.path.realpath
        assert realpath(app.output_dir) != realpath(self.output_dir)
        assert realpath(app.output_dir) == realpath(copy_dir)
        assert app.fname == orig_fname

    def test_app_stops_multi_proc_interface_at_end(self):
        # Given
        app = self.app
        port = get_free_port(9000)

        # When
        args = ['--multiproc', 'auto']
        app.run(args)

        # Then
        self.assertEqual(len(app._interfaces), 1)
        self.assertTrue(isinstance(
            app._interfaces[0], MultiprocessingInterface
        ))
        port1 = get_free_port(9000)
        self.assertEqual(port1, port)

    def test_app_stops_xml_rpc_interface_at_end(self):
        # Given
        app = self.app
        port = get_free_port(9000)
        host = '127.0.0.1'

        # When
        args = ['--xml-rpc', '%s:%s' % (host, port)]
        app.run(args)

        # Then
        self.assertEqual(len(app._interfaces), 1)
        self.assertTrue(isinstance(
            app._interfaces[0], XMLRPCInterface
        ))
        port1 = get_free_port(9000)
        count = 0
        while port1 != port and count < 4:
            time.sleep(0.5)
            port1 = get_free_port(9000)
            count += 1
        self.assertEqual(port1, port)
