import os
import shutil
import sys
import tempfile
import time
import unittest

import numpy as np

from pysph.tools.automation import (Problem, PySPHTask, Simulation, Task,
                                    TaskRunner)
try:
    from pysph.tools.jobs import Scheduler
except ImportError:
    raise unittest.SkipTest('test_jobs requires psutil')


class EllipticalDrop(Problem):
    """We define a simple example problem which we will run using the automation
    framework.

    In this case we run two variants of the elliptical drop problem.

    The setup method defines the cases to run which are simply Simulation instances.

    The get_commands returns the actual commands to run.

    The run method does the post-processing, after the simulations are done.

    """
    def get_name(self):
        return 'elliptical_drop'

    def setup(self):
        # Two cases, one with update_h and one without.
        cmd = 'python -m pysph.examples.elliptical_drop --max-steps=5'
        self.cases = [
            Simulation(
                root=self.input_path('no_update_h'),
                base_command=cmd, update_h=None
            ),
            Simulation(
                root=self.input_path('update_h'),
                base_command=cmd, no_update_h=None
            ),
        ]

    def get_commands(self):
        return [(x.name, x.command) for x in self.cases]

    def run(self):
        self.make_output_dir()
        no_update = self.cases[0].data
        update = self.cases[1].data
        output = open(self.output_path('result.txt'), 'w')
        output.write('no_update_h: %s\n'%no_update['major'])
        output.write('update_h: %s\n'%update['major'])
        output.close()


class PlotTask(Task):
    def __init__(self, problem):
        self.problem = problem
        self.reqs = [
            PySPHTask(runner) for name, runner in self.problem.get_requires()
        ]

    def output(self):
        return self.problem.get_outputs()

    def requires(self):
        return self.reqs

    def run(self):
        self.problem.run()



class TestLocalAutomation(unittest.TestCase):
    def setUp(self):
        self.cwd = os.getcwd()
        self.root = tempfile.mkdtemp()
        os.chdir(self.root)
        self.sim_dir = 'sim'
        self.output_dir = 'output'

    def tearDown(self):
        os.chdir(self.cwd)
        if os.path.exists(self.root):
            shutil.rmtree(self.root)

        PySPHTask.scheduler = None

    def _setup_scheduler(self):
        worker = dict(host='localhost')
        s = Scheduler(root='.', worker_config=[worker])
        PySPHTask.scheduler = s

    def test_automation(self):
        # Given.
        self._setup_scheduler()
        problem = EllipticalDrop(self.sim_dir, self.output_dir)
        t = TaskRunner(tasks=[PlotTask(problem=problem)])

        # When.
        t.run(wait=1)

        # Then.
        sim1 = os.path.join(self.root, self.sim_dir, 'elliptical_drop', 'no_update_h')
        self.assertTrue(os.path.exists(sim1))
        sim2 = os.path.join(self.root, self.sim_dir, 'elliptical_drop', 'update_h')
        self.assertTrue(os.path.exists(sim2))

        results = os.path.join(self.root, self.output_dir, 'elliptical_drop', 'result.txt')
        self.assertTrue(os.path.exists(results))
        data = open(results).read()
        self.assertTrue('no_update_h' in data)
        self.assertTrue('update_h' in data)

        # When.
        problem = EllipticalDrop(self.sim_dir, self.output_dir)
        t = TaskRunner(tasks=[PlotTask(problem=problem)])

        # Then.
        self.assertEqual(len(t.todo), 0)

    def test_nothing_is_run_when_output_exists(self):
        # Given.
        self._setup_scheduler()
        output = os.path.join(self.output_dir, 'elliptical_drop')
        os.makedirs(output)

        # When
        problem = EllipticalDrop(self.sim_dir, self.output_dir)
        t = TaskRunner(tasks=[PlotTask(problem=problem)])

        # Then.
        self.assertEqual(len(t.todo), 0)


class TestRemoteAutomation(TestLocalAutomation):
    def setUp(self):
        super(TestRemoteAutomation, self).setUp()
        self.other_dir = tempfile.mkdtemp()
        print self.other_dir, self.sim_dir, self.output_dir

    def tearDown(self):
        super(TestRemoteAutomation, self).tearDown()
        if os.path.exists(self.other_dir):
            shutil.rmtree(self.other_dir)

        PySPHTask.scheduler = None

    def _setup_scheduler(self):
        workers = [
            dict(host='localhost'),
            dict(host='xxx', python=sys.executable, chdir=self.other_dir, testing=True)
        ]
        s = Scheduler(root=self.sim_dir, worker_config=workers)
        PySPHTask.scheduler = s
