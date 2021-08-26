from __future__ import print_function
from os.path import abspath, dirname, join
from subprocess import Popen, PIPE
import sys
from threading import Timer


def get_directory(file):
    return dirname(abspath(file))


def kill_process(process):
    print('*'*70)
    print('KILLING PROCESS ON TIMEOUT')
    print('*'*70)
    process.kill()


def run(filename, args=None, nprocs=2, timeout=30.0, path=None):
    """Run a python script with MPI or in serial (if nprocs=1).  Kill process
    if it takes longer than the specified timeout.

    Parameters:
    -----------
    filename - filename of python script to run under mpi.
    args - List of arguments to pass to script.
    nprocs - number of processes to run (1 => serial non-mpi run).
    timeout - time in seconds to wait for the script to finish running,
        else raise a RuntimeError exception.
    path - the path under which the script is located
        Defaults to the location of this file (__file__), not curdir.

    """
    if args is None:
        args = []
    file_path = abspath(join(path, filename))
    cmd = [sys.executable, file_path] + args
    if nprocs > 1:
        cmd = ['mpiexec', '-n', str(nprocs), '--oversubscribe'] + cmd

    print('running test:', cmd)

    process = Popen(cmd, stdout=PIPE, stderr=PIPE)
    timer = Timer(timeout, kill_process, [process])
    timer.start()
    out, err = process.communicate()
    timer.cancel()
    retcode = process.returncode
    if retcode:
        msg = 'test ' + filename + ' failed with returncode ' + str(retcode)
        print(out.decode('utf-8'))
        print(err.decode('utf-8'))
        print('#'*80)
        print(msg)
        print('#'*80)
        raise RuntimeError(msg)
    return retcode, out, err
