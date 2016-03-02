try:
    # This is for Python-2.6.x
    from unittest2 import TestCase, main
except ImportError:
    from unittest import TestCase, main

try:
    from unittest import mock
except ImportError:
    import mock

import numpy as np
import numpy.testing as npt

from pysph.solver.solver import Solver


class TestSolver(TestCase):

    def setUp(self):
        patcher = mock.patch(
            'pysph.sph.acceleration_eval.AccelerationEval', spec=True
        )
        AccelerationEval = patcher.start()
        self.a_eval = AccelerationEval()
        self.addCleanup(patcher.stop)

        patcher = mock.patch(
            'pysph.sph.integrator.PECIntegrator', spec=True
        )
        PECIntegrator = patcher.start()
        self.integrator = PECIntegrator()
        self.addCleanup(patcher.stop)

    def test_solver_dumps_output_given_output_at_times(self):
        # Given
        dt = 0.1
        self.integrator.compute_time_step.return_value = dt
        tf = 10.05
        pfreq = 5
        output_at_times = [0.3, 0.35]
        solver = Solver(
            integrator=self.integrator, tf=tf, dt=dt,
            output_at_times=output_at_times
        )
        solver.set_print_freq(pfreq)
        solver.acceleration_eval = self.a_eval
        solver.particles = []

        # When
        record = []
        record_dt = []
        def _mock_dump_output():
            # Record the time at which the solver dumped anything
            record.append(solver.t)
            # This smells but ...
            sd = solver._get_solver_data()
            record_dt.append(sd['dt'])
        solver.dump_output = mock.Mock(side_effect=_mock_dump_output)

        solver.solve(show_progress=False)

        # Then
        expected = np.asarray(
            [0.0, 0.3, 0.35] + np.arange(0.45, 10.1, 0.5).tolist() + [10.05]
        )
        error_message = "Expected %s, got %s"%(expected, record)
        self.assertEqual(len(expected), len(record), error_message)
        self.assertTrue(
            np.max(np.abs(expected - record)) < 1e-12, error_message
        )
        self.assertEqual(101, solver.count)
        # The final timestep should not be a tiny one due to roundoff.
        self.assertTrue(solver.dt > 0.1*0.25)

        npt.assert_array_almost_equal(
            [0.1]*len(record_dt), record_dt, decimal=12
        )

    def test_solver_honors_set_time_step(self):
        # Given
        dt = 0.1
        tf = 1.0
        pfreq = 1
        solver = Solver(
            integrator=self.integrator, tf=tf, dt=dt, adaptive_timestep=False
        )
        solver.set_print_freq(pfreq)
        solver.acceleration_eval = self.a_eval
        solver.particles = []
        record = []
        def _mock_dump_output():
            # Record the time at which the solver dumped anything
            record.append(solver.t)
        solver.dump_output = mock.Mock(side_effect=_mock_dump_output)

        # When
        solver.set_time_step(0.2)
        solver.solve(show_progress=False)

        # Then
        expected = np.asarray([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        error_message = "Expected %s, got %s"%(expected, record)

        self.assertEqual(len(expected), len(record), error_message)
        self.assertTrue(
            np.max(np.abs(expected - record)) < 1e-12, error_message
        )


if __name__ == '__main__':
    main()
