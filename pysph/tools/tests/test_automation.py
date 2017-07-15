from __future__ import print_function

import os
import shutil
import sys
import tempfile
import unittest

try:
    from unittest import mock
except ImportError:
    import mock

from pysph.tools.automation import (
    CommandTask, Problem, Simulation, SolveProblem, TaskRunner,
    compare_runs
)
try:
    from pysph.tools.jobs import Scheduler, RemoteWorker
except ImportError:
    raise unittest.SkipTest('test_jobs requires psutil')

from pysph.tools.tests.test_jobs import wait_until


class EllipticalDrop(Problem):
    """We define a simple example problem which we will run using the automation
    framework.

    In this case we run two variants of the elliptical drop problem.

    The setup method defines the cases to run which are simply Simulation
    instances.

    The get_commands returns the actual commands to run.

    The run method does the post-processing, after the simulations are done.

    """
    def get_name(self):
        return 'elliptical_drop'

    def setup(self):
        # Two cases, one with update_h and one without.
        cmd = 'python -m pysph.examples.elliptical_drop --max-steps=5'

        # If self.cases is set, the get_commands method will do the right
        # thing.
        self.cases = [
            Simulation(
                root=self.input_path('update_h'),
                base_command=cmd,
                job_info=dict(n_core=1, n_thread=1),
                update_h=None
            ),
            Simulation(
                root=self.input_path('no_update_h'),
                base_command=cmd,
                job_info=dict(n_core=1, n_thread=1),
                no_update_h=None
            ),
        ]

    def run(self):
        self.make_output_dir()
        no_update = self.cases[0].data
        update = self.cases[1].data
        output = open(self.output_path('result.txt'), 'w')
        output.write('no_update_h: %s\n' % no_update['major'])
        output.write('update_h: %s\n' % update['major'])
        output.close()


class TestAutomationBase(unittest.TestCase):
    def setUp(self):
        self.cwd = os.getcwd()
        self.root = tempfile.mkdtemp()
        os.chdir(self.root)
        self.sim_dir = 'sim'
        self.output_dir = 'output'
        patch = mock.patch(
            'pysph.tools.jobs.free_cores', return_value=2
        )
        patch.start()
        self.addCleanup(patch.stop)

    def tearDown(self):
        os.chdir(self.cwd)
        if os.path.exists(self.root):
            shutil.rmtree(self.root)


class TestLocalAutomation(TestAutomationBase):
    def _make_scheduler(self):
        worker = dict(host='localhost')
        s = Scheduler(root='.', worker_config=[worker])
        return s

    def test_automation(self):
        # Given.
        problem = EllipticalDrop(self.sim_dir, self.output_dir)
        s = self._make_scheduler()
        t = TaskRunner(tasks=[SolveProblem(problem=problem)], scheduler=s)

        # When.
        t.run(wait=1)

        # Then.
        sim1 = os.path.join(self.root, self.sim_dir,
                            'elliptical_drop', 'no_update_h')
        self.assertTrue(os.path.exists(sim1))
        sim2 = os.path.join(self.root, self.sim_dir,
                            'elliptical_drop', 'update_h')
        self.assertTrue(os.path.exists(sim2))

        results = os.path.join(self.root, self.output_dir,
                               'elliptical_drop', 'result.txt')
        self.assertTrue(os.path.exists(results))
        data = open(results).read()
        self.assertTrue('no_update_h' in data)
        self.assertTrue('update_h' in data)

        # When.
        problem = EllipticalDrop(self.sim_dir, self.output_dir)
        t = TaskRunner(tasks=[SolveProblem(problem=problem)], scheduler=s)

        # Then.
        self.assertEqual(len(t.todo), 0)

    def test_nothing_is_run_when_output_exists(self):
        # Given.
        s = self._make_scheduler()
        output = os.path.join(self.output_dir, 'elliptical_drop')
        os.makedirs(output)

        # When
        problem = EllipticalDrop(self.sim_dir, self.output_dir)
        t = TaskRunner(tasks=[SolveProblem(problem=problem)], scheduler=s)

        # Then.
        self.assertEqual(len(t.todo), 0)


class TestRemoteAutomation(TestLocalAutomation):
    def setUp(self):
        super(TestRemoteAutomation, self).setUp()
        self.other_dir = tempfile.mkdtemp()
        p = mock.patch.object(
            RemoteWorker, 'free_cores', return_value=2.0
        )
        p.start()
        self.addCleanup(p.stop)

    def tearDown(self):
        super(TestRemoteAutomation, self).tearDown()
        if os.path.exists(self.other_dir):
            if sys.platform.startswith('win'):
                try:
                    shutil.rmtree(self.other_dir)
                except WindowsError:
                    pass
            else:
                shutil.rmtree(self.other_dir)

    def _make_scheduler(self):
        workers = [
            dict(host='localhost'),
            dict(host='test_remote',
                 python=sys.executable, chdir=self.other_dir, testing=True)
        ]
        try:
            import execnet
        except ImportError:
            raise unittest.SkipTest('This test requires execnet')
        return Scheduler(root=self.sim_dir, worker_config=workers)

    def test_job_with_error_is_handled_correctly(self):
        # Given.
        problem = EllipticalDrop(self.sim_dir, self.output_dir)
        problem.cases[0].base_command += ' --xxx'
        s = self._make_scheduler()
        t = TaskRunner(tasks=[SolveProblem(problem=problem)], scheduler=s)

        # When.
        try:
            t.run(wait=1)
        except RuntimeError:
            pass

        # Then.

        # Ensure that the directories are copied over when they have errors.
        sim1 = os.path.join(self.root, self.sim_dir,
                            'elliptical_drop', 'no_update_h')
        self.assertTrue(os.path.exists(sim1))
        sim2 = os.path.join(self.root, self.sim_dir,
                            'elliptical_drop', 'update_h')
        self.assertTrue(os.path.exists(sim2))

        # Ensure that all the correct but already scheduled jobs are completed.
        task_status = t.task_status
        status_values = list(task_status.values())
        self.assertEqual(status_values.count('error'), 1)
        self.assertEqual(status_values.count('done'), 1)
        self.assertEqual(status_values.count('not started'), 1)
        for t, s in task_status.items():
            if s == 'done':
                self.assertTrue(t.complete())
            if s == 'error':
                self.assertFalse(t.complete())


class TestCommandTask(TestAutomationBase):
    def _make_scheduler(self):
        worker = dict(host='localhost')
        s = Scheduler(root='.', worker_config=[worker])
        return s

    def test_command_tasks_executes_simple_command(self):
        # Given
        s = self._make_scheduler()
        cmd = 'python -c "print(1)"'
        t = CommandTask(cmd, output_dir=self.sim_dir)

        self.assertFalse(t.complete())

        # When
        t.run(s)
        wait_until(lambda: not t.complete())

        # Then
        self.assertTrue(t.complete())
        self.assertEqual(t.job_proxy.status(), 'done')
        self.assertEqual(t.job_proxy.get_stdout().strip(), '1')

    def test_command_tasks_converts_dollar_output_dir(self):
        # Given
        s = self._make_scheduler()
        cmd = '''python -c "print('$output_dir')"'''
        t = CommandTask(cmd, output_dir=self.sim_dir)

        self.assertFalse(t.complete())

        # When
        t.run(s)
        wait_until(lambda: not t.complete())

        # Then
        self.assertTrue(t.complete())
        self.assertEqual(t.job_proxy.status(), 'done')
        self.assertEqual(t.job_proxy.get_stdout().strip(), self.sim_dir)

    def test_command_tasks_handles_errors_correctly(self):
        # Given
        s = self._make_scheduler()
        cmd = 'python --junk'
        t = CommandTask(cmd, output_dir=self.sim_dir)

        self.assertFalse(t.complete())

        # When
        t.run(s)
        try:
            wait_until(lambda: not t.complete())
        except RuntimeError:
            pass

        # Then
        self.assertFalse(t.complete())
        self.assertEqual(t.job_proxy.status(), 'error')

        # A new command task should still detect that the run failed, even
        # though the output directory exists.
        # Given
        t = CommandTask(cmd, output_dir=self.sim_dir)
        # When/Then
        self.assertFalse(t.complete())


def test_compare_runs_calls_methods_when_given_names():
    # Given
    sims = [mock.MagicMock(), mock.MagicMock()]
    s0, s1 = sims
    s0.get_labels.return_value = s1.get_labels.return_value = 'label'

    # When
    compare_runs(sims, 'fig', labels=['x'], exact='exact')

    # Then
    s0.exact.assert_called_once_with(color='k', linestyle='-')
    s0.fig.assert_called_once_with(color='k', label='label', linestyle='--')
    s0.get_labels.assert_called_once_with(['x'])
    assert s1.exact.called == False
    s1.fig.assert_called_once_with(color='k', label='label', linestyle='-.')
    s1.get_labels.assert_called_once_with(['x'])


def test_compare_runs_works_when_given_callables():
    # Given
    sims = [mock.MagicMock()]
    s0 = sims[0]
    s0.get_labels.return_value = 'label'

    func = mock.MagicMock()
    exact = mock.MagicMock()

    # When
    compare_runs(sims, func, labels=['x'], exact=exact)

    # Then
    exact.assert_called_once_with(s0, color='k', linestyle='-')
    func.assert_called_once_with(s0, color='k', label='label', linestyle='--')
    s0.get_labels.assert_called_once_with(['x'])
