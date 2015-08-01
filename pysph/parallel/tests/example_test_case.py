
import os
import shutil
import tempfile
import unittest

import numpy as np

from pysph.solver.utils import load, get_files
from pysph.tools import run_parallel_script

MY_DIR = run_parallel_script.get_directory(__file__)

def get_example_script(script):
    """Given a relative posix path to a script located inside the
    pysph.examples package, return the full path to it.

    """
    ex_dir = os.path.join(os.path.dirname(os.path.dirname(MY_DIR)), 'examples')
    return os.path.join(ex_dir, *script.split('/'))


class ExampleTestCase(unittest.TestCase):
    """ A script to run an example in serial and parallel and compare results.

    The comparison is performed in the _test function and currently just checks
    the positions of the particles and tests if they are close.

    To test an example in parallel, subclass from ExampleTest and
    write a test function like so:

    def test_elliptical_drop(self):
        dt = 1e-5; tf = 100*dt
        serial_kwargs = dict(timestep=dt, tf=tf, ghost_layers=1, lb_freq=5)
        self.run_example(
            'dambreak3D.py', nprocs=4, timeout=900,
            serial_kwargs=serial_kwargs
        )

    """
    def _kwargs_to_command_line(self, kwargs):
        """Convert a dictionary of keyword arguments to a list of command-line
        options.
        """
        def _key_to_option(arg):
            return arg.replace('_', '-')

        cmd_line = []
        for key, value in kwargs.items():
            option = _key_to_option(key)
            if value is None:
                arg = "--{option}".format(option=option)
            else:
                arg = "--{option}={value}".format(
                    option=option, value=str(value)
                )

            cmd_line.append(arg)
        return cmd_line

    def run_example(self, filename, nprocs=2, timeout=300, atol=1e-14,
                    serial_kwargs=None, extra_parallel_kwargs=None):
        """Run an example and compare the results in serial and parallel.

        Parameters:
        -----------

        filename : str
            The name of the file to run

        nprocs : int
            Number of processors to use for the parallel run.

        timeout : float
            Time in seconds to wait for execution before an error is raised.

        atol: float
            Absolute tolerance for differences between the runs.

        serial_kwargs : dict
            The options to pass for a serial run.  Note that if the value of a
            particular key is None, the option is simply set and no value
            passed.  For example if `openmp=None`, then `--openmp` is used.

        extra_parallel_kwargs: dict
            The extra options to pass for the parallel run.

        """
        if serial_kwargs is None:
            serial_kwargs = {}
        if extra_parallel_kwargs is None:
            extra_parallel_kwargs = {}

        parallel_kwargs = dict(serial_kwargs)
        parallel_kwargs.update(extra_parallel_kwargs)

        prefix = os.path.splitext(os.path.basename(filename))[0]
        # dir1 is for the serial run
        dir1 = tempfile.mkdtemp()
        serial_kwargs.update(fname=prefix, directory=dir1)

        # dir2 is for the parallel run
        dir2 = tempfile.mkdtemp()
        parallel_kwargs.update(fname=prefix, directory=dir2)

        serial_args = self._kwargs_to_command_line(serial_kwargs)
        parallel_args = self._kwargs_to_command_line(parallel_kwargs)

        try:
            # run the example script in serial
            run_parallel_script.run(
                filename=filename, args=serial_args, nprocs=1,
                timeout=timeout, path=MY_DIR
            )

            # run the example script in parallel
            run_parallel_script.run(
                filename=filename, args=parallel_args, nprocs=nprocs,
                timeout=timeout, path=MY_DIR
            )

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
        self._test(serial, parallel, atol, nprocs)

    def _test(self, serial, parallel, atol, nprocs):
        # make sure the arrays are at the same time
        self.assertAlmostEqual( serial.time, parallel.time, 12)

        # test the results.
        xs, ys, zs, rhos  = serial.get("x","y", "z", "rho")
        xp, yp, zp, rhop, gid = parallel.get("x", "y", "z", "rho", 'gid')

        if nprocs == 1:
            # Not really a parallel run (used for openmp support).
            gid = np.arange(xs.size)

        self.assertTrue( xs.size, xp.size )

        x_err = np.max(xs[gid] - xp)
        y_err = np.max(ys[gid] - yp)
        z_err = np.max(zs[gid] - zp)
        self.assertTrue(np.allclose(xs[gid], xp, atol=atol, rtol=0),
                        "Max x_error %s"%x_err)
        self.assertTrue(np.allclose(ys[gid], yp, atol=atol, rtol=0),
                        "Max y_error %s"%y_err)
        self.assertTrue(np.allclose(zs[gid], zp, atol=atol, rtol=0),
                        "Max z_error %s"%z_err)
