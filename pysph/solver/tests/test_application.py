
#Author: Anshuman Kumar

try:
    # This is for Python-2.6.x
    from unittest2 import TestCase,i
except ImportError:
    from unittest import TestCase, main
try:
    from unittest import mock
except ImportError:
    import mock

from pysph.solver.application import Application
from pysph.solver.solver import Solver
import numpy

class TestApp(Application):
    def add_user_options(self,group):
        group.add_argument( "--testarg", action="store", type=float,
                dest="testarg", default=int(10.0), help="Test Argument")
        
    def consume_user_options(self):
        self.testarg  = self.options.testarg

class TestApplication(TestCase):
    def setup(self):
        pass

    # Test When testarg is passed
    def test_add_group_true(self):
        self.app = TestApp()
        args =['--testarg', '20']
        self.app.args = args
        self.app._parse_command_line()
        assert(self.app.testarg == 20)

    # Test When testarg is not passed
    def test_add_group_false(self):
        self.app = TestApp()
        args =[]
        self.app.args = args
        self.app._parse_command_line()
        assert(self.app.testarg == 10)

