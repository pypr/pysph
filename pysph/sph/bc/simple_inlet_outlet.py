"""
Simple Inlet outlet

This class is derived from `InletOutletManager`.
One can simulate EDAC and TVF scheme with this class.
one has to pass all the equations (if any) in the inletinfo/outletinfo.

The manager creates inlet/outlet object which performs the update
of particle when moving from inlet to fluid or fluid to outlet.
"""
# License: BSD

from pysph.sph.integrator_step import IntegratorStep
from pysph.sph.integrator import PECIntegrator
from pysph.sph.scheme import TVFScheme
from pysph.sph.wc.edac import EDACScheme
from pysph.sph.bc.inlet_outlet_manager import InletOutletManager


class SimpleInletOutlet(InletOutletManager):
    def add_io_properties(self, pa, scheme=None):
        DEFAULT_PROPS = ['xn', 'yn', 'zn', 'ioid', 'disp',
                         'wij']
        for prop in DEFAULT_PROPS:
            pa.add_property(prop)

    def get_stepper(self, scheme, cls):
        steppers = {}
        if (cls == PECIntegrator):
            if isinstance(scheme, EDACScheme):
                for inlet in self.inlets:
                    steppers[inlet] = InletOutletStepEDAC()
                for outlet in self.outlets:
                    steppers[outlet] = InletOutletStepEDAC()
                self.active_stages = [2]

            elif isinstance(scheme, TVFScheme):
                for inlet in self.inlets:
                    steppers[inlet] = InletOutletStepTVF()
                for outlet in self.outlets:
                    steppers[outlet] = InletOutletStepTVF()
                self.active_stages = [2]

        return steppers

    def get_equations(self, scheme=None, summation_density=False):
        from pysph.sph.equation import Group

        equations = []
        g00 = []
        for inlet in self.inletinfo:
            for eqn in inlet.equations:
                g00.append(eqn)
        for outlet in self.outletinfo:
            for eqn in outlet.equations:
                g00.append(eqn)

        if g00:
            equations.append(Group(equations=g00, real=False))

        return equations


class InletOutletStepTVF(IntegratorStep):
    def initialize(self, d_x0, d_idx, d_x):
        d_x0[d_idx] = d_x[d_idx]

    def stage1(self, d_idx, d_x, d_u, dt):
        d_x[d_idx] += dt*d_u[d_idx]


class InletOutletStepEDAC(IntegratorStep):
    def initialize(self, d_x0, d_idx, d_x):
        d_x0[d_idx] = d_x[d_idx]

    def stage1(self, d_idx, d_x, d_x0, d_u, dt):
        dtb2 = 0.5 * dt
        d_x[d_idx] = d_x0[d_idx] + dtb2*d_u[d_idx]

    def stage2(self, d_idx, d_x, d_x0, d_u, dt):
        d_x[d_idx] = d_x0[d_idx] + dt*d_u[d_idx]
