#Author: Anshuman Kumar

try:
    # This is for Python-2.6.x
    from unittest2 import TestCase
except ImportError:
    from unittest import TestCase

try:
    from unittest import mock
except ImportError:
    import mock

from pysph.solver.application import Application
from pysph.solver.solver import Solver

class TestApp(Application):
    def add_user_options(self,group):
        group.add_argument( "--testarg", action="store", type=float,
                dest="testarg", default=int(10.0), help="Test Argument")
        
    def consume_user_options(self):
        self.testarg  = self.options.testarg

    def create_particles(self):
        return []

class TestApplication(TestCase):
    def setUp(self):
        patcher = mock.patch('pysph.base.nnps.BoxSortNNPS', spec=True)
        Nnps = patcher.start()
        self.nnps = Nnps()
        self.addCleanup(patcher.stop)
        self.solver = Solver()
        self.solver.particles =[]
        self.solver.solve = mock.Mock()
        self.solver.setup = mock.Mock()

    # Test When testarg is  notpassed
    def test_user_options_false(self):
        #Given
        self.app = TestApp()

        #When
        args = []
        self.app.args = []
        self.app.setup(solver = self.solver, equations=[],
                particle_factory=self.app.create_particles,
                nnps = self.nnps)
        self.app.run()
        record = self.app.testarg

        #Then
        expected = 10.0
        error_message = "Expected %f, got %f"%(expected, record)
        self.assertEqual(expected,record)

    # Test When testarg is passed
    def test_user_options_true(self):
         #Given
        self.app = TestApp()

        #When
        args = ['--testarg', '20']
        self.app.args = args
        self.app.setup(solver = self.solver, equations=[],
                particle_factory=self.app.create_particles,
                nnps = self.nnps)
        self.app.run()
        record = self.app.testarg

        #Then
        expected = 20.0
        error_message = "Expected %f, got %f"%(expected, record)
        self.assertEqual(expected,record)

