import multiprocessing
import os
import shlex
import shutil
import sys
import tempfile
import time
import unittest

try:
    from pysph.tools import jobs
except ImportError:
    raise unittest.SkipTest('test_jobs requires psutil')


class TestJob(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.root)

    def test_job_can_handle_string_command(self):
        # Given
        command = '''python -c 'import sys;sys.stdout.write("1");sys.stderr.write("2")' '''
        j = jobs.Job(command=command, output_dir=self.root)

        # When
        j.run()
        j.proc.wait()

        # Then.
        self.assertTrue(isinstance(j.command, list))
        self.assertTrue(j.status(), 'done')
        self.assertEqual(j.get_stdout(), '1')
        self.assertEqual(j.get_stderr(), '2')

    def test_simple_job(self):
        # Given
        command = ['python', '-c', 'import sys;sys.stdout.write("1");sys.stderr.write("2")']
        j = jobs.Job(command=command, output_dir=self.root)

        # When
        j.run()
        j.proc.wait()

        # Then
        self.assertEqual(j.status(), 'done')
        self.assertEqual(j.output_dir, self.root)
        self.assertEqual(j.n_core, 1)
        self.assertEqual(j.n_thread, 1)
        self.assertEqual(j.get_stdout(), '1')
        self.assertEqual(j.get_stderr(), '2')
        state = j.to_dict()
        expect = dict(
            command=command, output_dir=self.root, n_core=1,
            n_thread=1, env=None
        )
        expect['command'][0] = sys.executable
        self.assertDictEqual(state, expect)

    def test_job_status(self):
        # Given/When
        j = jobs.Job(
            [sys.executable, '-c', 'import time; time.sleep(0.25)'],
            output_dir=self.root
        )

        # Then
        self.assertEqual(j.status(), 'not started')

        # When
        j.run()

        # Then
        self.assertEqual(j.status(), 'running')

        # When
        j.proc.wait()
        self.assertEqual(j.status(), 'done')

        # Given
        j = jobs.Job(
            [sys.executable, '-c', 'asdf'],
            output_dir=self.root
        )
        # When
        j.run()
        j.proc.wait()

        # Then
        self.assertEqual(j.status(), 'error')
        self.assertTrue('NameError' in j.get_stderr())

    def test_that_job_sets_env_var(self):
        # Given/When
        j = jobs.Job(
            [sys.executable, '-c', 'import os;print(os.environ.get("FOO"))'],
            output_dir=self.root,
            env=dict(FOO='hello')
        )
        j.run()
        j.proc.wait()

        # Then
        self.assertEqual(j.status(), 'done')
        self.assertEqual(j.get_stdout().strip(), 'hello')

    def test_that_job_sets_omp_var(self):
        j = jobs.Job(
            [sys.executable, '-c',
             'import os;print(os.environ.get("OMP_NUM_THREADS"))'],
            output_dir=self.root,
            n_thread=4,
        )
        j.run()
        j.proc.wait()

        # Then
        self.assertEqual(j.status(), 'done')
        self.assertEqual(j.get_stdout().strip(), '4')

    def test_free_cores(self):
        n = jobs.free_cores()
        self.assertTrue(n > 0)
        self.assertTrue(n <= multiprocessing.cpu_count())


def wait_until(cond, timeout=1, wait=0.1):
    t = 0.0
    while cond():
        time.sleep(wait)
        t += wait
        if t > timeout:
            break

class TestLocalWorker(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.root)

    def test_scheduler_works_with_local_worker(self):
        # Given
        s = jobs.Scheduler(worker_config=[dict(host='localhost')])

        # When
        j = jobs.Job(
            [sys.executable, '-c', 'import time; time.sleep(0.05); print(1)'],
            output_dir=self.root
        )
        proxy = s.submit(j)

        # Then
        wait_until(lambda: proxy.status() != 'done')
        self.assertEqual(proxy.status(), 'done')
        self.assertEqual(proxy.get_stderr(), '')
        self.assertEqual(proxy.get_stdout().strip(), '1')


class TestRemoteWorker(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        try:
            import execnet
        except ImportError:
            raise unittest.SkipTest('This test requires execnet')

    def tearDown(self):
        shutil.rmtree(self.root)

    def test_free_cores(self):
        # Given
        r = jobs.RemoteWorker(host='localhost', python=sys.executable, testing=True)
        # Then.
        n = r.free_cores()
        self.assertTrue(n > 0)
        self.assertTrue(n <= multiprocessing.cpu_count())

    def test_simple(self):
        # Given
        r = jobs.RemoteWorker(host='localhost', python=sys.executable, testing=True)

        # When
        j = jobs.Job(
            [sys.executable, '-c', 'import time; time.sleep(0.05); print(1)'],
            output_dir=self.root
        )
        proxy = r.run(j)

        # Then
        wait_until(lambda: proxy.status() != 'done')
        self.assertEqual(proxy.status(), 'done')
        self.assertEqual(proxy.get_stderr(), '')
        self.assertEqual(proxy.get_stdout().strip(), '1')
