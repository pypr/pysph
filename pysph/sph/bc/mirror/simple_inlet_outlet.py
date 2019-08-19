"""Mirroring outlet

Original paper by

    - Tafuni, Angelantonio, et al. "A versatile algorithm for the treatment of
      open boundary conditions in smoothed particle hydrodynamics gpu models."
      Computer Methods in Applied Mechanics and Engineering 342 (2018):
      604-624.

The implementation here is the modification of this by

    - Negi, Pawan, Prabhu Ramachandran, and Asmelash Haftu. "An improved
      non-reflecting outlet boundary condition for weakly-compressible SPH."
      arXiv preprint arXiv:1907.04034 (2019).

"""

from pysph.sph.integrator import PECIntegrator
from pysph.sph.wc.edac import EDACScheme
from pysph.sph.bc.inlet_outlet_manager import InletOutletManager


class SimpleInletOutlet(InletOutletManager):
    def add_io_properties(self, pa, scheme=None):
        DEFAULT_PROPS = [
            'disp', 'ioid', 'Bu', 'Bv', 'Bw', 'Bp', 'xn', 'yn', 'zn', 'A',
            'wij', 'uo', 'vo', 'wo', 'po', 'uho', 'vho', 'who', 'Buh',
            'Bvh', 'Bwh', 'x0', 'y0', 'z0', 'uhat', 'vhat', 'what']
        STRIDE_DATA = {
            'A': 16, 'Bu': 4, 'Bv': 4, 'Bw': 4, 'Bp': 4, 'Brho': 4, 'uo':
            4, 'vo': 4, 'wo': 4, 'po': 4, 'rhoo': 4, 'Buh': 4, 'Bvh': 4,
            'Bwh': 4, 'uho': 4, 'vho': 4, 'who': 4}
        for prop in DEFAULT_PROPS:
            if prop in STRIDE_DATA:
                pa.add_property(prop, stride=STRIDE_DATA[prop])
            else:
                pa.add_property(prop)

        pa.add_constant('uref', 0.0)

    def get_stepper(self, scheme, cls, edactvf=True):
        from pysph.sph.bc.inlet_outlet_manager import (
            InletStep, OutletStepWithUhat)
        steppers = {}
        if (cls == PECIntegrator):
            if isinstance(scheme, EDACScheme):
                for inlet in self.inlets:
                    steppers[inlet] = InletStep()
                for outlet in self.outlets:
                    steppers[outlet] = OutletStepWithUhat()
                for g_inlet in self.ghost_inlets:
                    steppers[g_inlet] = InletStep()
                for g_outlet in self.ghost_outlets:
                    steppers[g_outlet] = OutletStepWithUhat()
                self.active_stages = [2]

        return steppers

    def get_equations(self, scheme=None, summation_density=False,
                      edactvf=True):
        from pysph.sph.equation import Group
        from pysph.sph.bc.interpolate import (
            UpdateMomentMatrix, EvaluateUhat, EvaluateP, EvaluateU,
            ExtrapolateUhat, ExtrapolateP, ExtrapolateU,
            CopyUhatFromGhost, CopyPFromGhost, CopyUFromGhost)
        from pysph.sph.bc.inlet_outlet_manager import (
            UpdateNormalsAndDisplacements, CopyNormalsandDistances)

        all_ghosts = self.ghost_inlets + self.ghost_outlets
        all_info = self.inletinfo + self.outletinfo
        all_pairs = {**self.inlet_pairs, **self.outlet_pairs}

        equations = []
        g00 = []
        i = 0
        for info in all_info:
            g00.append(UpdateNormalsAndDisplacements(
                dest=info.pa_name, sources=None, xn=info.normal[0],
                yn=info.normal[1], zn=info.normal[2], xo=info.refpoint[0],
                yo=info.refpoint[1], zo=info.refpoint[2]
                ))
            g00.append(CopyNormalsandDistances(
                dest=all_pairs[info.pa_name], sources=[info.pa_name]))
            i = i + 1

        equations.append(Group(equations=g00, real=False))

        g02 = []
        for name in all_ghosts:
            g02.append(UpdateMomentMatrix(
                dest=name, sources=self.fluids, dim=self.dim))

        equations.append(Group(equations=g02, real=False))

        g03 = []
        for name in all_ghosts:
            g03.append(EvaluateUhat(dest=name, sources=self.fluids,
                                    dim=self.dim))
            g03.append(EvaluateP(dest=name, sources=self.fluids,
                                 dim=self.dim))
        for name in self.ghost_outlets:
            g03.append(EvaluateU(dest=name, sources=self.fluids,
                                 dim=self.dim))

        equations.append(Group(equations=g03, real=False))

        g04 = []
        for name in all_ghosts:
            g04.append(ExtrapolateUhat(dest=name, sources=None))
            g04.append(ExtrapolateP(dest=name, sources=None))
        for name in self.ghost_outlets:
            g04.append(ExtrapolateU(dest=name, sources=None))

        equations.append(Group(equations=g04, real=False))

        g05 = []
        for io in all_pairs.keys():
            g05.append(CopyUhatFromGhost(
                dest=io, sources=[all_pairs[io]]))
            g05.append(CopyPFromGhost(
                dest=io, sources=[all_pairs[io]]))
        for io in self.outlet_pairs.keys():
            g05.append(CopyUFromGhost(
                dest=io, sources=[all_pairs[io]]))

        equations.append(Group(equations=g05, real=False))

        g06 = []
        for inlet in self.inletinfo:
            for eqn in inlet.equations:
                g06.append(eqn)
        for outlet in self.outletinfo:
            for eqn in outlet.equations:
                g06.append(eqn)

        equations.append(Group(equations=g06, real=False))

        return equations
