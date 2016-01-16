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

class MockApp(Application):
    def add_user_options(self,group):
        group.add_argument( "--testarg", action="store", type=float,
                dest="testarg", default=int(10.0), help="Test Argument")
        
    def consume_user_options(self):
        self.testarg  = self.options.testarg

    def create_particles(self):
        return []

    def create_equations(self):
        return []

    def create_solver(self):
        solver = Solver()
        solver.particles =[]
        solver.solve = mock.Mock()
        solver.setup = mock.Mock()
        return solver

    def create_nnps(self):
        nnps = mock.Mock()
        return nnps
 
class TestApplication(TestCase):

    # Test When testarg is  notpassed
    def test_user_options_false(self):
        #Given
        app = MockApp()

        #When
        args = []
        app.run(args)
        record = app.testarg

        #Then
        expected = 10.0
        error_message = "Expected %f, got %f"%(expected, app.testarg)
        self.assertEqual(expected,app.testarg,error_message)

    # Test When testarg is passed
    def test_user_options_true(self):
        #Given
        app = MockApp()

        #When
        args = ['--testarg', '20']
        app.run(args)

        #Then
        expected = 20.0
        error_message = "Expected %f, got %f"%(expected, app.testarg)
        self.assertEqual(expected,app.testarg,error_message)

