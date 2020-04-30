"""do-nothing outlet first used in SPH by

    - Federico, Ivan, et al. "Simulating 2D open-channel flows through an SPH
      model." European Journal of Mechanics-B/Fluids 34 (2012): 35-46.

"""

from pysph.sph.integrator import PECIntegrator
from pysph.sph.bc.inlet_outlet_manager import InletOutletManager
from pysph.sph.wc.edac import EDACScheme


class SimpleInletOutlet(InletOutletManager):
    def add_io_properties(self, pa, scheme=None):
        DEFAULT_PROPS = [
            'disp', 'ioid', 'Bp', 'A', 'wij', 'po', 'uho', 'vho', 'who',
            'Buh', 'Bvh', 'Bwh', 'x0', 'y0', 'z0', 'uhat', 'vhat', 'what',
            'xn', 'yn', 'zn']
        STRIDE_DATA = {
            'A': 16, 'Bu': 4, 'Bv': 4, 'Bw': 4, 'Bp': 4, 'Brho': 4, 'uo':
            4, 'vo': 4, 'wo': 4, 'po': 4, 'rhoo': 4, 'Bau': 4, 'Bav': 4,
            'Baw': 4, 'auo': 4, 'avo': 4, 'awo': 4, 'Buh': 4, 'Bvh': 4,
            'Bwh': 4, 'uho': 4, 'vho': 4, 'who': 4}
        for prop in DEFAULT_PROPS:
            if prop in STRIDE_DATA:
                pa.add_property(prop, stride=STRIDE_DATA[prop])
            else:
                pa.add_property(prop)

        pa.add_constant('uref', 0.0)

    def get_stepper(self, scheme, cls, edactvf=False):

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
                self.active_stages = [2]

        return steppers

    def get_equations(self, scheme=None, summation_density=False,
                      edactvf=False):
        from pysph.sph.equation import Group
        from pysph.sph.bc.interpolate import (
            UpdateMomentMatrix, EvaluateUhat, EvaluateP, ExtrapolateUhat,
            ExtrapolateP, CopyUhatFromGhost, CopyPFromGhost)
        from pysph.sph.bc.inlet_outlet_manager import (
            UpdateNormalsAndDisplacements, CopyNormalsandDistances)

        equations = []
        g00 = []
        i = 0
        for info in self.inletinfo:
            g00.append(UpdateNormalsAndDisplacements(
                dest=info.pa_name, sources=None, xn=info.normal[0],
                yn=info.normal[1], zn=info.normal[2], xo=info.refpoint[0],
                yo=info.refpoint[1], zo=info.refpoint[2]
                ))
            g00.append(CopyNormalsandDistances(
                dest=self.inlet_pairs[info.pa_name], sources=[info.pa_name]))
            i = i + 1

        equations.append(Group(equations=g00, real=False))
        g02 = []
        for name in self.ghost_inlets:
            g02.append(UpdateMomentMatrix(
                dest=name, sources=self.fluids, dim=self.dim))

        equations.append(Group(equations=g02, real=False))

        g03 = []
        for name in self.ghost_inlets:
            g03.append(EvaluateUhat(dest=name, sources=self.fluids,
                                    dim=self.dim))
            g03.append(EvaluateP(dest=name, sources=self.fluids,
                                 dim=self.dim))

        equations.append(Group(equations=g03, real=False))

        g04 = []
        for name in self.ghost_inlets:
            g04.append(ExtrapolateUhat(dest=name, sources=None))
            g04.append(ExtrapolateP(dest=name, sources=None))

        equations.append(Group(equations=g04, real=False))

        g05 = []
        for io in self.inlet_pairs.keys():
            g05.append(CopyUhatFromGhost(
                    dest=io, sources=[self.inlet_pairs[io]]))
            g05.append(CopyPFromGhost(
                dest=io, sources=[self.inlet_pairs[io]]))

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
