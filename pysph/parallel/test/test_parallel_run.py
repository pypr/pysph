""" Module to run the example files and report their success/failure results

Add a function to the ExampleTest class corresponding to an example script to
be tested.
This is done till better strategy for parallel testing is implemented

"""

try:
    import mpi4py.MPI as mpi
except ImportError:
    import nose.plugins.skip as skip
    reason = "mpi4py not installed"
    raise skip.SkipTest(reason)

import tempfile
import shutil
import os
import unittest
import numpy

from nose.plugins.attrib import attr

from pysph.solver.utils import load, get_files
from pysph.tools import run_parallel_script


path = run_parallel_script.get_directory(__file__)

class ExampleTestCase(unittest.TestCase):
    """ A script to run an example in serial and parallel and compare results.

    To test an example in parallel, subclass from ExampleTest and
    write a test function like so:

    def test_elliptical_drop(self):
        self.run_example('../../../../examples/elliptical_drop.py',
                         timestep=1e-5, iters=100, nprocs=2, timeout=60)

    """
    def run_example(self, filename, tf=0.01, nprocs=2, load_func=load,
                    timeout=300, ghost_layers=2, dt=1e-4, lb_freq=1):
        """Run an example and compare the results in serial and parallel.

        Parameters:
        -----------

        filename : str
            The name of the file to run

        timestep : double
            The time step argument to pass to the example script

        iters : double
            The number of iteations to evolve the example

        nprocs : int
            Number of processors to use for the example.

        timeout : int
            Time in seconds to wait for execution before an error is raised.

        path : Not used

        """
        prefix = os.path.splitext(os.path.basename(filename))[0]

        try:
            # dir1 is for the serial run
            dir1 = tempfile.mkdtemp()

            # dir2 is for the parallel run
            dir2 = tempfile.mkdtemp()

            args = ['--fname=%s'%prefix,
                    '--directory=%s'%dir1,
                    '--tf=%g'%(tf),
                    '--pfreq=1000000',
                    '--ghost-layers=%g'%ghost_layers,
                    '--timestep=%g'%dt,
                    '--lb-freq=%d'%lb_freq
                    ]

            # run the example script in serial
            run_parallel_script.run(
                filename=filename, args=args, nprocs=1, timeout=timeout, path=path)

            # run the example script in parallel
            args[1] = '--directory=%s'%dir2
            run_parallel_script.run(
                filename=filename, args=args, nprocs=nprocs, timeout=timeout, path=path)

            # get the serial and parallel results
            dir1path = os.path.abspath(dir1)
            dir2path = os.path.abspath(dir2)

            # load the serial output
            file = get_files(dirname=dir1path, fname=prefix)[-1]
            serial = load(file)
            serial = serial['arrays']['fluid']

            # load the parallel output
            file = get_files(dirname=dir2path, fname=prefix)[-1]
            parallel = load(file)
            parallel = parallel['arrays']['fluid']

        finally:
            shutil.rmtree(dir1, True)
            shutil.rmtree(dir2, True)

        # test
        self._test(serial, parallel)

    def _test(self, serial, parallel):
        # make sure the arrays are at the same time
        self.assertAlmostEqual( serial.time, parallel.time, 8)

        # test the results.
        xs, ys, zs, rhos  = serial.get("x","y", "z", "rho")
        xp, yp, zp, rhop, gid = parallel.get("x", "y", "z", "rho", 'gid')

        self.assertTrue( xs.size,xp.size )
        np = xs.size

        places = 14
        for i in range(np):
            self.assertAlmostEqual( xs[ gid[i] ], xp[i], places )
            self.assertAlmostEqual( ys[ gid[i] ], yp[i], places )
            self.assertAlmostEqual( zs[ gid[i] ], zp[i], places )

class ParallelTests(ExampleTestCase):

    @attr(slow=True, very_slow=True, parallel=True)
    def test_3Ddam_break_example(self):
        dt = 1e-5; tf = 100*dt
        self.run_example('./dambreak3D.py',
                         nprocs=4, load_func=load, tf=tf, dt=dt, ghost_layers=1,
                         timeout=900, lb_freq=5)

    @attr(slow=True, parallel=True)
    def test_2Ddam_break_example(self):
        dt = 1e-4; tf = 200*dt
        self.run_example('../../../examples/dam_break.py',
                         nprocs=4, load_func=load, tf=tf, dt=dt, ghost_layers=1,
                         lb_freq=5)

    @attr(slow=True, parallel=True)
    def test_ldcavity_example(self):
        dt=1e-4; tf=200*dt
        self.run_example('../../../examples/TransportVelocity/cavity.py',
                         nprocs=4, load_func=load, tf=tf, dt=dt, ghost_layers=3.0,
                         lb_freq=5)

if __name__ == "__main__":
    unittest.main()
