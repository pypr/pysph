import multiprocessing
import shutil
import sys
import os
import tempfile
import time
import unittest
try:
    from unittest import mock
except ImportError:
    import mock

try:
    from pysph.tools import jobs
except ImportError:
    raise unittest.SkipTest('test_jobs requires psutil')


def safe_rmtree(*args, **kw):
    if sys.platform.startswith('win'):
        try:
            shutil.rmtree(*args, **kw)
        except WindowsError:
            pass
    else:
        shutil.rmtree(*args, **kw)


class TestJob(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()

    def tearDown(self):
        safe_rmtree(self.root)

    def test_job_can_handle_string_command(self):
        # Given
        command = '''\
        python -c 'import sys;sys.stdout.write("1");sys.stderr.write("2")' '''
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
        command = ['python', '-c',
                   'import sys;sys.stdout.write("1");sys.stderr.write("2")']
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
        self.assertTrue(n >= 0)
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
        safe_rmtree(self.root)

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
        wait_until(lambda: proxy.status() != 'done', timeout=2)
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
        safe_rmtree(self.root)

    def test_free_cores(self):
        # Given
        r = jobs.RemoteWorker(
            host='localhost', python=sys.executable, testing=True
        )
        # Then.
        n = r.free_cores()
        self.assertTrue(n >= 0)
        self.assertTrue(n <= multiprocessing.cpu_count())

    def test_simple(self):
        # Given
        r = jobs.RemoteWorker(
            host='localhost', python=sys.executable, testing=True
        )

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


class TestScheduler(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.count = 0
        try:
            import execnet
        except ImportError:
            raise unittest.SkipTest('This test requires execnet')

    def tearDown(self):
        safe_rmtree(self.root)

    def _make_dummy_job(self, n_core=1, sleep=0.05):
        output = os.path.join(self.root, 'job%d' % self.count)
        job = jobs.Job(
            [sys.executable, '-c',
             'import time; time.sleep(%f); print(1)' % sleep],
            output_dir=output,
            n_core=n_core
        )
        self.count += 1
        return job

    @mock.patch('pysph.tools.jobs.LocalWorker')
    def test_scheduler_does_not_start_worker_when_created(self, mock_lw):
        # Given
        config = [dict(host='localhost')]

        # When
        s = jobs.Scheduler(worker_config=config)

        # Then
        self.assertEqual(mock_lw.call_count, 0)
        self.assertEqual(len(s.workers), 0)

    @mock.patch('pysph.tools.jobs.LocalWorker')
    def test_scheduler_starts_worker_on_submit(self, mock_lw):
        attrs = {'host': 'localhost', 'free_cores.return_value': 2}
        mock_lw.return_value = mock.MagicMock(**attrs)

        # Given
        config = [dict(host='localhost')]
        s = jobs.Scheduler(worker_config=config)
        j = jobs.Job(
            [sys.executable, '-c', 'print(1)'],
            output_dir=self.root
        )

        # When
        s.submit(j)

        # Then
        self.assertEqual(mock_lw.call_count, 1)
        self.assertEqual(len(s.workers), 1)

    def test_scheduler_only_creates_required_workers(self):
        # Given
        config = [
            dict(host='host1', python=sys.executable, testing=True),
            dict(host='host2', python=sys.executable, testing=True),
        ]
        s = jobs.Scheduler(worker_config=config)
        j = self._make_dummy_job()

        # When
        proxy = s.submit(j)

        # Then
        self.assertEqual(len(s.workers), 1)
        self.assertEqual(proxy.worker.host, 'host1')

        # Wait for this job to end and then see what happens
        # When a new job is submitted.
        self._wait_while_not_done(proxy, 25)

        # When
        j = self._make_dummy_job()
        s.submit(j)

        # Then
        self.assertEqual(len(s.workers), 1)
        self.assertEqual(proxy.worker.host, 'host1')

        # Running two jobs in a row should produce two workers.
        # When
        j = self._make_dummy_job()
        proxy = s.submit(j)

        # Then
        self.assertEqual(len(s.workers), 2)
        self.assertEqual(proxy.worker.host, 'host2')

        # Adding more should work.

        # When
        j = self._make_dummy_job()
        proxy = s.submit(j)
        j = self._make_dummy_job()
        proxy1 = s.submit(j)

        # Then
        self.assertEqual(len(s.workers), 2)
        self._wait_while_not_done(proxy1, 15)

        self.assertEqual(proxy.status(), 'done')
        self.assertEqual(proxy.worker.host, 'host1')
        self.assertEqual(proxy1.worker.host, 'host2')

    def test_scheduler_should_not_overload_worker(self):
        # Given
        n_core = jobs.free_cores()
        config = [dict(host='localhost')]
        s = jobs.Scheduler(worker_config=config)

        j1 = self._make_dummy_job(n_core, sleep=0.5)
        j2 = self._make_dummy_job(n_core, sleep=0.5)
        j3 = self._make_dummy_job(n_core, sleep=0.5)
        j4 = self._make_dummy_job(0, sleep=0.5)

        # When
        proxy1 = s.submit(j1, wait=0.5)
        proxy2 = s.submit(j2, wait=0.5)
        proxy3 = s.submit(j3, wait=0.5)
        proxy4 = s.submit(j4, wait=0.5)

        # Then
        self.assertEqual(len(s.workers), 1)
        # Basically, submit will wait for the existing jobs to complete.
        # Therefore when s.submit(j3) is called it should wait until a
        # free worker is available.
        self.assertEqual(proxy1.status(), 'done')
        self.assertEqual(proxy2.status(), 'done')
        self.assertEqual(proxy3.status(), 'running')

        # Proxy4 will be running since it needs no cores.
        self.assertEqual(proxy4.status(), 'running')
        self._wait_while_not_done(proxy3, 20)
        self.assertEqual(proxy3.status(), 'done')
        self._wait_while_not_done(proxy4, 15)
        self.assertEqual(proxy4.status(), 'done')

    def _wait_while_not_done(self, proxy, n_count, sleep=0.1):
        count = 0
        while proxy.status() != 'done' and count < n_count:
            time.sleep(sleep)
            count += 1
