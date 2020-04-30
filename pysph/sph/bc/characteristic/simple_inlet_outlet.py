"""Method of characteristic proposed by

    - Lastiwka, Martin, Mihai Basa, and Nathan J. Quinlan. "Permeable and
      non‐reflecting boundary conditions in SPH." International journal for
      numerical methods in fluids 61.7 (2009): 709-724.

"""

from pysph.sph.equation import Equation
from pysph.sph.integrator import PECIntegrator
from pysph.sph.bc.inlet_outlet_manager import InletOutletManager
from pysph.sph.wc.edac import EDACScheme
import numpy


class SimpleInletOutlet(InletOutletManager):
    def add_io_properties(self, pa, scheme=None):
        DEFAULT_PROPS = [
            'disp', 'ioid', 'Bp', 'A', 'wij', 'po', 'uho', 'vho', 'who',
            'Buh', 'Bvh', 'Bwh', 'x0', 'y0', 'z0', 'uhat', 'vhat', 'what',
            'xn', 'yn', 'zn', 'J1', 'J2u']
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
        pa.add_constant('avgj2u', 0.0)
        pa.add_constant('avgj1', 0.0)

    def get_stepper(self, scheme, cls, edactvf=False):
        from pysph.sph.bc.inlet_outlet_manager import (
            InletStep, OutletStep)
        steppers = {}
        if (cls == PECIntegrator):
            if isinstance(scheme, EDACScheme):
                for inlet in self.inlets:
                    steppers[inlet] = InletStep()
                for outlet in self.outlets:
                    steppers[outlet] = OutletStep()
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

        for fluid in self.fluids:
            g00.append(EvalauteCharacterisctics(
                dest=fluid, sources=None, c0=10.0, u0=1.0, v0=0.0,
                p0=1.0, rho0=1000.0))

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
        for name in self.outlets:
            g03.append(EvalauteNumberdensity(dest=name, sources=self.fluids))
            g03.append(ShepardInterpolateCharacteristics(dest=name,
                                                         sources=self.fluids))

        equations.append(Group(equations=g03, real=False))

        g04 = []
        for name in self.ghost_inlets:
            g04.append(ExtrapolateUhat(dest=name, sources=None))
            g04.append(ExtrapolateP(dest=name, sources=None))
        for name in self.outlets:
            g04.append(EvaluatePropertyfromCharacteristics(
                dest=name, sources=None, c0=10.0, u0=1.0, v0=0.0,
                p0=1.0, rho0=1000.0))

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


class EvalauteCharacterisctics(Equation):
    def __init__(self, dest, sources, c0, rho0, u0, p0, v0):
        self.c0 = c0
        self.rho0 = rho0
        self.p0 = p0
        self.u0 = u0
        self.v0 = v0

        super(EvalauteCharacterisctics, self).__init__(dest, sources)

    def initialize(self, d_idx, d_u, d_v, d_p, d_rho, d_J1, d_J2u):
        a = self.c0
        rhoref = self.rho0
        uref = self.u0
        pref = self.p0

        d_J1[d_idx] = -a**2 * (d_rho[d_idx] - rhoref) + (d_p[d_idx] - pref)
        d_J2u[d_idx] = d_rho[d_idx] * a * (d_u[d_idx] - uref) + (d_p[d_idx] -
                                                                 pref)


class EvalauteNumberdensity(Equation):
    def initialize(self, d_idx, d_wij):
        d_wij[d_idx] = 0.0

    def loop(self, d_idx, d_wij, WIJ):
        d_wij[d_idx] += WIJ


class ShepardInterpolateCharacteristics(Equation):
    def initialize(self, d_idx, d_J1, d_J2u):
        d_J1[d_idx] = 0.0
        d_J2u[d_idx] = 0.0

    def loop(self, d_idx, d_J1, d_J2u, s_J1, s_J2u, WIJ, s_idx):
        d_J1[d_idx] += s_J1[s_idx] * WIJ
        d_J2u[d_idx] += s_J2u[s_idx] * WIJ

    def post_loop(self, d_idx, d_J1, d_J2u, d_wij, d_avgj2u, d_avgj1):
        if d_wij[d_idx] > 1e-14:
            d_J1[d_idx] /= d_wij[d_idx]
            d_J2u[d_idx] /= d_wij[d_idx]
        else:
            d_J2u[d_idx] = d_avgj2u[0]
            d_J1[d_idx] = d_avgj1[0]

    def reduce(self, dst, t, dt):
        dst.avgj2u[0] = numpy.average(dst.J2u[dst.wij > 0.0001])
        dst.avgj1[0] = numpy.average(dst.J1[dst.wij > 0.0001])


class EvaluatePropertyfromCharacteristics(Equation):
    def __init__(self, dest, sources, c0, rho0, u0, p0, v0):
        self.c0 = c0
        self.rho0 = rho0
        self.p0 = p0
        self.u0 = u0
        self.v0 = v0

        super(EvaluatePropertyfromCharacteristics, self).__init__(dest,
                                                                  sources)

    def initialize(self, d_idx, d_u, d_v, d_p, d_rho, d_J1, d_J2u):
        j1 = d_J1[d_idx]
        j2u = d_J2u[d_idx]

        d_rho[d_idx] = self.rho0 + (-j1 + 0.5 * j2u) / self.c0**2
        d_u[d_idx] = self.u0 + (j2u) / (2 * d_rho[d_idx] * self.c0)
        d_p[d_idx] = self.p0 + 0.5 * (j2u)
