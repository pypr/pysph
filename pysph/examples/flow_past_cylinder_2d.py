"""
Flow past cylinder
"""
import numpy as np
import os

from pysph.base.kernels import QuinticSpline
from pysph.sph.equation import Equation
from pysph.base.utils import get_particle_array
from pysph.solver.application import Application
from pysph.tools import geometry as G
from pysph.sph.wc.edac import EDACScheme
from pysph.sph.bc.inlet_outlet_manager import (
        InletInfo, OutletInfo)

# Fluid mechanical/numerical parameters
rho = 1000
umax = 1.0
c0 = 10 * umax
p0 = rho * c0 * c0


class SolidWallNoSlipBCReverse(Equation):
    def __init__(self, dest, sources, nu):
        self.nu = nu
        super(SolidWallNoSlipBCReverse, self).__init__(dest, sources)

    def initialize(self, d_idx, d_auf, d_avf, d_awf):
        d_auf[d_idx] = 0.0
        d_avf[d_idx] = 0.0
        d_awf[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_m, d_rho, s_rho, d_V, s_V,
             d_ug, d_vg, d_wg,
             d_auf, d_avf, d_awf,
             s_u, s_v, s_w,
             DWIJ, R2IJ, EPS, XIJ):

        # averaged shear viscosity Eq. (6).
        etai = self.nu * d_rho[d_idx]
        etaj = self.nu * s_rho[s_idx]

        etaij = 2 * (etai * etaj)/(etai + etaj)

        # particle volumes; d_V inverse volume.
        Vi = 1./d_V[d_idx]
        Vj = 1./s_V[s_idx]
        Vi2 = Vi * Vi
        Vj2 = Vj * Vj

        # scalar part of the kernel gradient
        Fij = XIJ[0]*DWIJ[0] + XIJ[1]*DWIJ[1] + XIJ[2]*DWIJ[2]

        # viscous contribution (third term) from Eq. (8), with VIJ
        # defined appropriately using the ghost values
        tmp = 1./d_m[d_idx] * (Vi2 + Vj2) * (etaij * Fij/(R2IJ + EPS))

        d_auf[d_idx] += tmp * (d_ug[d_idx] - s_u[s_idx])
        d_avf[d_idx] += tmp * (d_vg[d_idx] - s_v[s_idx])
        d_awf[d_idx] += tmp * (d_wg[d_idx] - s_w[s_idx])


class ResetInletVelocity(Equation):
    def __init__(self, dest, sources, U, V, W):
        self.U = U
        self.V = V
        self.W = W

        super(ResetInletVelocity, self).__init__(dest, sources)

    def loop(self, d_idx, d_u, d_v, d_w, d_x, d_y, d_z, d_xn, d_yn, d_zn,
             d_uref):
        if d_idx == 0:
            d_uref[0] = self.U
        d_u[d_idx] = self.U
        d_v[d_idx] = self.V
        d_w[d_idx] = self.W


class WindTunnel(Application):
    def initialize(self):
        # Geometric parameters
        self.Lt = 30.0  # length of tunnel
        self.Wt = 15.0  # half width of tunnel
        self.dc = 1.2  # diameter of cylinder
        self.cxy = 10., 0.0  # center of cylinder
        self.nl = 10  # Number of layers for wall/inlet/outlet
        self.io_method = 'donothing'

    def add_user_options(self, group):
        group.add_argument(
            "--re", action="store", type=float, dest="re", default=200,
            help="Reynolds number."
        )
        group.add_argument(
            "--hdx", action="store", type=float, dest="hdx", default=1.2,
            help="Ratio h/dx."
        )
        group.add_argument(
            "--nx", action="store", type=int, dest="nx", default=12,
            help="Number of points in 1D of the cylinder."
        )
        group.add_argument(
            "--lt", action="store", type=float, dest="Lt", default=30,
            help="Length of the WindTunnel."
        )
        group.add_argument(
            "--wt", action="store", type=float, dest="Wt", default=15,
            help="Half width of the WindTunnel."
        )
        group.add_argument(
            "--dc", action="store", type=float, dest="dc", default=1.2,
            help="Diameter of the cylinder."
        )
        group.add_argument(
            "--io-method", action="store", type=str, dest="io_method",
            default='donothing', help="'donothing', 'mirror',"
            "or 'characteristic', 'mod_donothing', hybrid."
        )

    def consume_user_options(self):
        self.dc = dc = self.options.dc
        self.Lt = 15. * dc
        self.Wt = 7.5 * dc
        self.io_method = self.options.io_method
        nx = self.options.nx
        re = self.options.re

        self.nu = nu = umax * self.dc / re

        self.cxy = 5.*self.dc, 0
        self.dx = dx = self.dc / nx
        self.volume = dx * dx
        hdx = self.options.hdx
        self.nl = (int)(6.0*hdx)

        self.h = h = hdx * self.dx
        dt_cfl = 0.25 * h / (c0 + umax)
        dt_viscous = 0.125 * h**2 / nu

        self.dt = min(dt_cfl, dt_viscous)
        self.tf = 100.0

    def _create_fluid(self):
        dx = self.dx
        h0 = self.h
        x, y = np.mgrid[dx / 2:self.Lt:dx, -self.Wt + dx/2:self.Wt:dx]
        x, y = (np.ravel(t) for t in (x, y))
        one = np.ones_like(x)
        volume = dx * dx * one
        m = volume * rho
        fluid = get_particle_array(
            name='fluid', m=m, x=x, y=y, h=h0, V=1.0 / volume, u=umax,
            p=0.0, rho=rho)
        return fluid

    def _create_solid(self):
        dx = self.dx
        h0 = self.h
        x = [0.0]
        y = [0.0]
        r = dx
        nt = 0
        while r - self.dc / 2 < 0.00001:
            nnew = int(np.pi*r**2/dx**2 + 0.5)
            tomake = nnew-nt
            theta = np.linspace(0., 2.*np.pi, tomake + 1)
            for t in theta[:-1]:
                x.append(r*np.cos(t))
                y.append(r*np.sin(t))
            nt = nnew
            r = r + dx
        x = np.array(x)
        y = np.array(y)
        x, y = (t.ravel() for t in (x, y))
        x += self.cxy[0]
        volume = dx*dx
        solid = get_particle_array(
            name='solid', x=x, y=y,
            m=volume*rho, rho=rho, h=h0, V=1.0/volume)
        return solid

    def _create_wall(self):
        dx = self.dx
        h0 = self.h
        x0, y0 = np.mgrid[
            dx/2: self.Lt+self.nl*dx+self.nl*dx: dx, dx/2: self.nl*dx: dx]
        x0 -= self.nl*dx
        y0 -= self.nl*dx+self.Wt
        x0 = np.ravel(x0)
        y0 = np.ravel(y0)

        x1 = np.copy(x0)
        y1 = np.copy(y0)
        y1 += self.nl*dx+2*self.Wt
        x1 = np.ravel(x1)
        y1 = np.ravel(y1)

        x0 = np.concatenate((x0, x1))
        y0 = np.concatenate((y0, y1))
        volume = dx*dx
        wall = get_particle_array(
            name='wall', x=x0, y=y0, m=volume*rho, rho=rho, h=h0,
            V=1.0/volume)
        return wall

    def _set_wall_normal(self, pa):
        props = ['xn', 'yn', 'zn']
        for p in props:
            pa.add_property(p)

        y = pa.y
        cond = y > 0.0
        pa.yn[cond] = 1.0
        cond = y < 0.0
        pa.yn[cond] = -1.0

    def _create_outlet(self):
        dx = self.dx
        h0 = self.h
        x, y = np.mgrid[dx/2:self.nl * dx:dx,  -self.Wt + dx/2:self.Wt:dx]
        x, y = (np.ravel(t) for t in (x, y))
        x += self.Lt
        one = np.ones_like(x)
        volume = dx * dx * one
        m = volume * rho
        outlet = get_particle_array(
            name='outlet', x=x, y=y, m=m, h=h0, V=1.0/volume, u=umax,
            p=0.0, rho=one * rho, uhat=umax)
        return outlet

    def _create_inlet(self):
        dx = self.dx
        h0 = self.h
        x, y = np.mgrid[dx / 2:self.nl*dx:dx, -self.Wt + dx/2:self.Wt:dx]
        x, y = (np.ravel(t) for t in (x, y))
        x = x - self.nl * dx
        one = np.ones_like(x)
        volume = one * dx * dx

        inlet = get_particle_array(
            name='inlet', x=x, y=y, m=volume * rho, h=h0, u=umax, rho=rho,
            V=1.0 / volume, p=0.0)
        return inlet

    def create_particles(self):
        dx = self.dx
        fluid = self._create_fluid()
        solid = self._create_solid()
        G.remove_overlap_particles(fluid, solid, dx, dim=2)
        outlet = self._create_outlet()
        inlet = self._create_inlet()
        wall = self._create_wall()

        ghost_inlet = self.iom.create_ghost(inlet, inlet=True)
        ghost_outlet = self.iom.create_ghost(outlet, inlet=False)

        particles = [fluid, inlet, outlet, solid, wall]
        if ghost_inlet:
            particles.append(ghost_inlet)
        if ghost_outlet:
            particles.append(ghost_outlet)

        self.scheme.setup_properties(particles)
        self._set_wall_normal(wall)

        if self.io_method == 'hybrid':
            fluid.uag[:] = 1.0
            fluid.uta[:] = 1.0
            outlet.uta[:] = 1.0

        return particles

    def create_scheme(self):
        h = nu = None
        s = EDACScheme(
            ['fluid'], ['solid'], dim=2, rho0=rho, c0=c0, h=h, pb=p0,
            nu=nu, inlet_outlet_manager=None,
            inviscid_solids=['wall']
        )
        return s

    def configure_scheme(self):
        scheme = self.scheme
        self.iom = self._create_inlet_outlet_manager()
        scheme.inlet_outlet_manager = self.iom
        pfreq = 100
        kernel = QuinticSpline(dim=2)
        self.iom.update_dx(self.dx)
        scheme.configure(h=self.h, nu=self.nu)

        scheme.configure_solver(kernel=kernel, tf=self.tf, dt=self.dt,
                                pfreq=pfreq, n_damp=0)

    def _get_io_info(self):
        inleteqns = [ResetInletVelocity('ghost_inlet', [], U=-umax, V=0.0,
                     W=0.0),
                     ResetInletVelocity('inlet', [], U=umax, V=0.0,
                     W=0.0)]

        i_update_cls = None
        i_has_ghost = True
        o_update_cls = None
        o_has_ghost = True
        manager = None
        props_to_copy = ['x0', 'y0', 'z0', 'uhat', 'vhat', 'what', 'x', 'y',
                         'z', 'u', 'v', 'w', 'm', 'h', 'rho', 'p', 'ioid']

        if self.io_method == 'donothing':
            from pysph.sph.bc.donothing.inlet import Inlet
            from pysph.sph.bc.donothing.outlet import Outlet
            from pysph.sph.bc.donothing.simple_inlet_outlet import (
                SimpleInletOutlet)
            o_has_ghost = False
            i_update_cls = Inlet
            o_update_cls = Outlet
            manager = SimpleInletOutlet
        elif self.io_method == 'mirror':
            from pysph.sph.bc.mirror.inlet import Inlet
            from pysph.sph.bc.mirror.outlet import Outlet
            from pysph.sph.bc.mirror.simple_inlet_outlet import (
                SimpleInletOutlet)
            i_update_cls = Inlet
            o_update_cls = Outlet
            manager = SimpleInletOutlet
        elif self.io_method == 'hybrid':
            from pysph.sph.bc.hybrid.inlet import Inlet
            from pysph.sph.bc.hybrid.outlet import Outlet
            from pysph.sph.bc.hybrid.simple_inlet_outlet import (
                SimpleInletOutlet)
            i_update_cls = Inlet
            o_update_cls = Outlet
            o_has_ghost = False
            manager = SimpleInletOutlet
            props_to_copy += ['uta', 'pta', 'u0', 'v0', 'w0', 'p0']
        if self.io_method == 'mod_donothing':
            from pysph.sph.bc.mod_donothing.inlet import Inlet
            from pysph.sph.bc.mod_donothing.outlet import Outlet
            from pysph.sph.bc.mod_donothing.simple_inlet_outlet import (
                SimpleInletOutlet)
            o_has_ghost = False
            i_update_cls = Inlet
            o_update_cls = Outlet
            manager = SimpleInletOutlet
        if self.io_method == 'characteristic':
            from pysph.sph.bc.characteristic.inlet import Inlet
            from pysph.sph.bc.characteristic.outlet import Outlet
            from pysph.sph.bc.characteristic.simple_inlet_outlet import (
                SimpleInletOutlet)
            o_has_ghost = False
            i_update_cls = Inlet
            o_update_cls = Outlet
            manager = SimpleInletOutlet

        inlet_info = InletInfo(
            pa_name='inlet', normal=[-1.0, 0.0, 0.0],
            refpoint=[0.0, 0.0, 0.0], equations=inleteqns,
            has_ghost=i_has_ghost, update_cls=i_update_cls,
            umax=umax
            )

        outlet_info = OutletInfo(
            pa_name='outlet', normal=[1.0, 0.0, 0.0],
            refpoint=[self.Lt, 0.0, 0.0], has_ghost=o_has_ghost,
            update_cls=o_update_cls, equations=None,
            props_to_copy=props_to_copy
        )

        return inlet_info, outlet_info, manager

    def _create_inlet_outlet_manager(self):
        inlet_info, outlet_info, manager = self._get_io_info()
        iom = manager(
            fluid_arrays=['fluid'], inletinfo=[inlet_info],
            outletinfo=[outlet_info]
        )
        return iom

    def create_inlet_outlet(self, particle_arrays):
        iom = self.iom
        io = iom.get_inlet_outlet(particle_arrays)
        return io

    def post_process(self, info_fname):
        self.read_info(info_fname)
        if len(self.output_files) == 0:
            return
        t, cd, cl = self._plot_force_vs_t()
        res = os.path.join(self.output_dir, 'results.npz')
        np.savez(res, t=t, cd=cd, cl=cl)

    def _plot_force_vs_t(self):
        from pysph.solver.utils import iter_output, load
        from pysph.tools.sph_evaluator import SPHEvaluator
        from pysph.sph.equation import Group
        from pysph.base.kernels import QuinticSpline
        from pysph.sph.wc.transport_velocity import (
            MomentumEquationPressureGradient,
            SummationDensity, SetWallVelocity
        )

        data = load(self.output_files[0])
        solid = data['arrays']['solid']
        fluid = data['arrays']['fluid']

        prop = ['awhat', 'auhat', 'avhat', 'wg', 'vg', 'ug', 'V', 'uf', 'vf',
                'wf', 'wij', 'vmag', 'pavg', 'nnbr', 'auf', 'avf', 'awf']
        for p in prop:
            solid.add_property(p)
            fluid.add_property(p)

        # We find the force of the solid on the fluid and the opposite of that
        # is the force on the solid. Note that the assumption is that the solid
        # is far from the inlet and outlet so those are ignored.
        print(self.nu, p0, self.dc, rho)
        equations = [
            Group(
                equations=[
                    SummationDensity(dest='fluid', sources=['fluid', 'solid']),
                    SummationDensity(dest='solid', sources=['fluid', 'solid']),
                    SetWallVelocity(dest='solid', sources=['fluid']),
                    ], real=False),
            Group(
                equations=[
                    # Pressure gradient terms
                    MomentumEquationPressureGradient(
                        dest='solid', sources=['fluid'], pb=p0),
                    SolidWallNoSlipBCReverse(
                        dest='solid', sources=['fluid'], nu=self.nu),
                    ], real=True),
        ]
        sph_eval = SPHEvaluator(
            arrays=[solid, fluid], equations=equations, dim=2,
            kernel=QuinticSpline(dim=2)
        )
        t, cd, cl = [], [], []
        import gc
        print(self.dc, self.dx, self.nu)
        print('fxf', 'fxp', 'fyf', 'fyp', 'cd', 'cl', 't')
        for sd, arrays in iter_output(self.output_files[:]):
            fluid = arrays['fluid']
            solid = arrays['solid']
            for p in prop:
                solid.add_property(p)
                fluid.add_property(p)
            t.append(sd['t'])
            sph_eval.update_particle_arrays([solid, fluid])
            sph_eval.evaluate()
            fxp = sum(solid.m*solid.au)
            fyp = sum(solid.m*solid.av)
            fxf = sum(solid.m*solid.auf)
            fyf = sum(solid.m*solid.avf)
            fx = fxf + fxp
            fy = fyf + fyp
            cd.append(fx/(0.5 * rho * umax**2 * self.dc))
            cl.append(fy/(0.5 * rho * umax**2 * self.dc))
            print(fxf, fxp, fyf, fyp, cd[-1], cl[-1], t[-1])
            gc.collect()
        t, cd, cl = list(map(np.asarray, (t, cd, cl)))
        # Now plot the results.
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        plt.figure()
        plt.plot(t, cd, label=r'$C_d$')
        plt.plot(t, cl, label=r'$C_l$')
        plt.xlabel(r'$t$')
        plt.ylabel('cd/cl')
        plt.legend()
        plt.grid()
        fig = os.path.join(self.output_dir, "force_vs_t.png")
        plt.savefig(fig, dpi=300)
        plt.close()
        return t, cd, cl

    def customize_output(self):
        if self.io_method == 'hybrid':
            self._mayavi_config('''
            viewer.scalar = 'u'
            ''')
        elif self.io_method == 'mirror':
            self._mayavi_config('''
            viewer.scalar = 'u'
            parr = ['ghost_outlet', 'ghost_inlet']
            for particle in parr:
                b = particle_arrays[particle]
                b.visible = False
            ''')
        else:
            self._mayavi_config('''
            viewer.scalar = 'u'
            parr = ['ghost_inlet']
            for particle in parr:
                b = particle_arrays[particle]
                b.visible = False
            ''')

    def post_step(self, solver):
        freq = 500
        if solver.count % freq == 0:
            self.nnps.update()
            for i, pa in enumerate(self.particles):
                if pa.name == 'fluid':
                    self.nnps.spatially_order_particles(i)
            self.nnps.update()


if __name__ == '__main__':
    app = WindTunnel()
    app.run()
    app.post_process(app.info_filename)
