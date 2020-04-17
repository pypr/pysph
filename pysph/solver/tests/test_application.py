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

from pysph.solver.application import Application
from pysph.solver.solver import Solver


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
