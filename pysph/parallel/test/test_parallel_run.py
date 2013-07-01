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

import unittest
from subprocess import Popen, PIPE
from threading import Timer
import os
import sys
import tempfile
import shutil
import numpy

from nose.plugins.attrib import attr

from pysph.solver.utils import load, get_files

directory = os.path.dirname(os.path.abspath(__file__))

def kill_process(process):
    print 'KILLING PROCESS ON TIMEOUT'
    process.kill()

def _run_example_script(filename, args=[], nprocs=2, timeout=20.0, path=None):
    """ run a file python script
    
    Parameters:
    -----------
    filename - filename of python script to run under mpi
    nprocs - (2) number of processes of the script to run (0 => serial non-mpi run)
    timeout - (5) time in seconds to wait for the script to finish running,
        else raise a RuntimeError exception
    path - the path under which the script is located
        Defaults to the location of this file (__file__), not curdir
    
    """
    path = directory
    path = os.path.join(path, filename)
    cmd = ['mpiexec','-n', str(nprocs), sys.executable, path] + args

    print 'running test:', cmd

    process = Popen(cmd, stdout=PIPE, stderr=PIPE)
    timer = Timer(timeout, kill_process, [process])
    timer.start()
    out, err = process.communicate()
    timer.cancel()
    retcode = process.returncode
    if retcode:
        msg = 'test ' + filename + ' failed with returncode ' + str(retcode)
        print out
        print err
        print '#'*80
        print msg
        print '#'*80
        raise RuntimeError, msg
    return retcode, out, err

class ExampleTestCase(unittest.TestCase):
    """ A script to run an example in serial and parallel and compare results.

    To test an example in parallel, subclass from ExampleTest and
    write a test function like so:

    def test_elliptical_drop(self):
        self.run_example('../../../../examples/elliptical_drop.py',
                         timestep=1e-5, iters=100, nprocs=2, timeout=60)
    
    """
    def run_example(self, filename, tf=0.01, nprocs=2, load_func=load,
                    timeout=300, ghost_layers=2, dt=1e-4):
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
                    '--timestep=%g'%dt]

            # run the example script in serial
            _run_example_script(filename, args, 1, timeout)

            # run the example script in parallel
            args[1] = '--directory=%s'%dir2
            _run_example_script(filename, args, nprocs, timeout)

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

    @attr(slow=True, parallel=True)
    def test_2Ddam_break_example(self):
        self.run_example('../../../examples/dam_break.py', 
                         nprocs=4, load_func=load, tf=0.01, ghost_layers=1)

    @attr(slow=True, parallel=True)
    def test_ldcavity_example(self):
        self.run_example('../../../examples/TransportVelocity/cavity.py', 
                         nprocs=4, load_func=load, tf=0.025, ghost_layers=3.0)

if __name__ == "__main__":
    unittest.main()
