"""
Hybrid Outlet proposed by

    - Negi, Pawan, Prabhu Ramachandran, and Asmelash Haftu. "An improved
      non-reflecting outlet boundary condition for weakly-compressible SPH."
      arXiv preprint arXiv:1907.04034 (2019).

"""

from pysph.sph.equation import Equation
from pysph.sph.integrator import PECIntegrator
from compyle.api import declare
import numpy

from pysph.sph.wc.edac import EDACScheme
from pysph.sph.bc.inlet_outlet_manager import InletOutletManager


class SimpleInletOutlet(InletOutletManager):
    def add_io_properties(self, pa, scheme=None):
        N = 6
        DEFAULT_PROPS = [
            'disp', 'ioid', 'Bp', 'xn', 'yn', 'zn', 'A', 'wij',
            'po', 'uho', 'vho', 'who', 'Buh', 'Bvh', 'Bwh', 'x0', 'y0', 'z0',
            'uhat', 'vhat', 'what', 'uag', 'vag', 'pag', 'pacu', 'uacu', 'uta',
            'pta', 'Eacu', 'vo', 'uo', 'wo', 'J1', 'J2u']
        STRIDE_DATA = {
            'A': 16, 'Bu': 4, 'Bv': 4, 'Bw': 4, 'Bp': 4, 'Brho': 4, 'uo':
            4, 'vo': 4, 'wo': 4, 'po': 4, 'rhoo': 4, 'Bau': 4, 'Bav': 4,
            'Baw': 4, 'auo': 4, 'avo': 4, 'awo': 4, 'Buh': 4, 'Bvh': 4,
            'Bwh': 4, 'uho': 4, 'vho': 4, 'who': 4, 'Bauh': 4, 'Bavh': 4,
            'Bawh': 4, 'auho': 4, 'avho': 4, 'awho': 4, 'Baz': 4, 'axo': 4,
            'ayo': 4, 'Bay': 4, 'azo': 4, 'Bax': 4, 'uag': N, 'vag': N,
            'pag': N, 'uhag': N, 'vhag': N}
        for prop in DEFAULT_PROPS:
            if prop in STRIDE_DATA:
                pa.add_property(prop, stride=STRIDE_DATA[prop])
            else:
                pa.add_property(prop)

        pa.add_constant('avgj2u', 0.0)
        pa.add_constant('avgj1', 0.0)
        pa.add_constant('uref', 0.0)

    def get_stepper(self, scheme, cls, edactvf=True):
        from pysph.sph.bc.inlet_outlet_manager import InletStep, OutletStep
        steppers = {}
        if (cls == PECIntegrator):
            if isinstance(scheme, EDACScheme):
                for inlet in self.inlets:
                    steppers[inlet] = InletStep()
                for outlet in self.outlets:
                    steppers[outlet] = OutletStep()
                self.active_stages = [2]

        return steppers

    def get_equations(self, scheme=None, summation_density=False,
                      edactvf=True):
        from pysph.sph.equation import Group
        from pysph.sph.bc.interpolate import (
            UpdateMomentMatrix, EvaluateUhat, EvaluateP, ExtrapolateUhat,
            ExtrapolateP, CopyUhatFromGhost, CopyPFromGhost)
        from pysph.sph.bc.inlet_outlet_manager import (
            UpdateNormalsAndDisplacements, CopyNormalsandDistances)

        all_pairs = {**self.inlet_pairs, **self.outlet_pairs}

        umax = []
        for info in self.inletinfo:
            umax.append(info.umax)

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
                dest=all_pairs[info.pa_name], sources=[info.pa_name]))
            i = i + 1

        equations.append(Group(equations=g00, real=False))

        g02 = []
        for name in self.fluids:
            g02.append(CopyTimeValues(dest=name, sources=None, rho=scheme.rho0,
                                      c0=scheme.c0, u0=min(umax)))
            g02.append((EvalauteCharacterisctics(dest=name, sources=None,
                                                 c0=scheme.c0,
                                                 rho0=scheme.rho0)))

        for name in self.ghost_inlets:
            g02.append(UpdateMomentMatrix(
                dest=name, sources=self.fluids, dim=self.dim))

        equations.append(Group(equations=g02, real=False))

        g02a = []
        for name in self.fluids:
            g02a.append(ComputeTimeAverage(dest=name, sources=None))
        for name in self.outlets:
            g02a.append(EvalauteNumberdensity(dest=name, sources=self.fluids))
            g02a.append(ShepardInterpolateCharacteristics(dest=name,
                                                          sources=self.fluids))
        equations.append(Group(equations=g02a, real=False))

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
        for name in self.outlets:
            g04.append(EvaluatePropertyfromCharacteristics(
                dest=name, sources=None, c0=scheme.c0, rho0=scheme.rho0))
        equations.append(Group(equations=g04, real=False))

        g05 = []
        for io in self.inlet_pairs.keys():
            g05.append(CopyUhatFromGhost(
                dest=io, sources=[all_pairs[io]]))
            g05.append(CopyPFromGhost(
                dest=io, sources=[all_pairs[io]]))

        equations.append(Group(equations=g05, real=False))

        g07 = []
        for inlet in self.inletinfo:
            for eqn in inlet.equations:
                g07.append(eqn)
        for outlet in self.outletinfo:
            for eqn in outlet.equations:
                g07.append(eqn)

        equations.append(Group(equations=g07, real=False))

        g08 = []
        for name in self.ghost_inlets:
            g08.append(MoveGhostInlet(dest=name, sources=None))

        equations.append(Group(equations=g08, real=False))

        return equations


class MoveGhostInlet(Equation):
    def loop(self, d_idx, d_u, d_x, dt):
        d_x[d_idx] += d_u[d_idx] * dt


class CopyTimeValues(Equation):
    def __init__(self, dest, sources, rho, c0, u0):
        self.rho = rho
        self.c0 = c0
        self.u0 = u0
        self.Imin = 0.5 * rho * u0**2

        super(CopyTimeValues, self).__init__(dest, sources)

    def initialize(self, d_idx, d_u, d_v, d_p, d_uag, d_pag, d_uta, d_pta,
                   d_Eacu, t, d_uref):
        i6, i, N = declare('int', 3)
        N = 6
        i6 = N * d_idx
        N -= 1

        for i in range(N):
            d_uag[i6+(N-i)] = d_uag[i6+(N-(i+1))]
            d_pag[i6+(N-i)] = d_pag[i6+(N-(i+1))]

        u0 = d_uref[0]
        fac = 1.0 / (2. * self.rho * self.c0)
        Imin = (0.5 * self.rho * u0**2)**2 * fac
        d_Eacu[d_idx] = d_p[d_idx] * d_p[d_idx] * fac

        if d_Eacu[d_idx] < Imin:
            d_uag[i6] = d_u[d_idx]
            d_pag[i6] = d_p[d_idx]


class ComputeTimeAverage(Equation):
    def initialize(self, d_idx, d_uag, d_pag, d_uta, d_pta):
        i6, i, N = declare('int', 3)
        N = 6
        i6 = N * d_idx

        d_uta[d_idx] = 0.0
        d_pta[d_idx] = 0.0

        for i in range(N):
            d_uta[d_idx] += d_uag[i6+i]
            d_pta[d_idx] += d_pag[i6+i]

        d_uta[d_idx] /= N
        d_pta[d_idx] /= N


class EvalauteCharacterisctics(Equation):
    def __init__(self, dest, sources, c0, rho0):
        self.c0 = c0
        self.rho0 = rho0

        super(EvalauteCharacterisctics, self).__init__(dest, sources)

    def initialize(self, d_idx, d_u, d_p, d_J1, d_J2u, d_uta, d_pta):
        a = self.c0
        uref = d_uta[d_idx]
        pref = d_pta[d_idx]

        d_J1[d_idx] = (d_p[d_idx] - pref)
        d_J2u[d_idx] = self.rho0 * a * (d_u[d_idx] - uref) + (d_p[d_idx] -
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
    def __init__(self, dest, sources, c0, rho0):
        self.c0 = c0
        self.rho0 = rho0

        super(EvaluatePropertyfromCharacteristics, self).__init__(dest,
                                                                  sources)

    def initialize(self, d_idx, d_rho, d_J2u, d_uta, d_pta, d_u, d_p, dt, t):
        if t > 20 * dt:
            j2u = d_J2u[d_idx]

            d_u[d_idx] = d_uta[d_idx] + (j2u) / (2 * self.rho0 * self.c0)
            d_p[d_idx] = d_pta[d_idx] + 0.5 * (j2u)
