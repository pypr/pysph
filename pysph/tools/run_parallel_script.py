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

def get_directory(file):
    return os.path.dirname(os.path.abspath(file))

def kill_process(process):
    print 'KILLING PROCESS ON TIMEOUT'
    process.kill()

def run(filename, args=[], nprocs=2, timeout=20.0, path=None):
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
