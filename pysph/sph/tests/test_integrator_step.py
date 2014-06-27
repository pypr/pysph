"""Simple tests for the Integrator steps"""

import numpy
import unittest

from pysph.base.utils import get_particle_array as gpa
from pysph.sph.integrator_step import OneStageRigidBodyStep, TwoStageRigidBodyStep

class RigidBodyMotionTestCase(unittest.TestCase):
    """Tests for linear motion.

    A particle array is subjected to acceleration along one coordinate
    direction and tested for the final acceleration, position and
    velocity.

    """
    def setUp(self):
        # create a single particle particle array
        x = numpy.array( [0.] )
        y = numpy.array( [0.] )

        additional_props = ['ax', 'ay', 'az', 'u0', 'v0', 'w0', 
                            'x0', 'y0', 'z0']
                          
        # create the particle array
        self.pa = pa = gpa(additional_props, name='square', x=x, y=y)
        
        self._set_stepper()

    def _integrate(self, final_time, dt, epec=False):
        """Integrate"""
        pa = self.pa; num_part = pa.get_number_of_particles()
        stepper = self.stepper

        current_time = 0.0
        iteration = 0

        while( current_time < final_time ):

            # initialize
            for i in range(num_part):
                stepper.initialize(
                    i, pa.x, pa.y, pa.z, pa.x0, pa.y0, pa.z0, 
                    pa.u, pa.v, pa.w, pa.u0, pa.v0, pa.w0)
            
            # update accelerations for EPEC
            if epec:
                self._update_accelerations(current_time)

            # stage 1
            for i in range( num_part ):
                stepper.stage1(
                    i, pa.x, pa.y, pa.z, pa.x0, pa.y0, pa.z0, 
                    pa.u, pa.v, pa.w, pa.u0, pa.v0, pa.w0, 
                    pa.ax, pa.ay, pa.az, dt)

            # update time
            current_time = current_time + 0.5 * dt

            # evaluate between stages
            self._update_accelerations(current_time)
            
            # call stage 2
            for i in range( num_part ):
                stepper.stage2(
                    i, pa.x, pa.y, pa.z, pa.x0, pa.y0, pa.z0, 
                    pa.u, pa.v, pa.w, pa.u0, pa.v0, pa.w0, 
                    pa.ax, pa.ay, pa.az, dt)

            # update time and iteration
            current_time = current_time + 0.5 * dt
            iteration += 1

class ConstantAccelerationTestCase(RigidBodyMotionTestCase):
    """ Constnat linear acceleration """
    def _update_accelerations(self, time):
        " Constant accelerations "
        self.pa.ax[0] = 1.0
        self.pa.ay[0] = 1.0
        self.pa.az[0] = 1.0

    def _set_stepper(self):
        # create the integrator stepper class we want to test
        self.stepper = TwoStageRigidBodyStep()

    def _test_motion(self, final_time=1.0, dt=1e-2, epec=True):
        """ Test motion for constant acceleration """
        
        # we simulate a two-stage integration with constant
        # acceleration ax = 1. Initial velocities are zero so we can
        # compare with the elementry formulae: S = 1/2 * a * t * t etc...
        self._integrate( final_time, dt, epec )

        # get the particle arrays to test
        x, y, z, u, v, w = self.pa.get('x', 'y', 'z', 'u', 'v', 'w')

        # positions S = 0 + 1/2 * at^2 = 0.5
        self.assertAlmostEqual( x[0], 0.5, 14 )
        self.assertAlmostEqual( y[0], 0.5, 14 )
        self.assertAlmostEqual( z[0], 0.5, 14 )

        # velocities v = u0 + a*t = 1.0
        self.assertAlmostEqual( u[0], 1.0, 14 )
        self.assertAlmostEqual( v[0], 1.0, 14 )
        self.assertAlmostEqual( w[0], 1.0, 14 )

    def test_one_stage(self):
        self.stepper = OneStageRigidBodyStep()
        self._test_motion()

    def test_two_stage(self):
        self.stepper = TwoStageRigidBodyStep()
        self._test_motion()

if __name__ == '__main__':
    unittest.main()
