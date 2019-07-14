"""
Inlet Outlet Manager
"""

from pysph.sph.equation import Equation
from compyle.api import get_config
from pysph.base.utils import get_particle_array
from pysph.sph.integrator_step import IntegratorStep
import numpy as np
from compyle.api import declare


class InletInfo(object):
    def __init__(self, pa_name, normal, refpoint, has_ghost=True,
                 update_cls=None, equations=None, umax=1.0,
                 props_to_copy=None):
        """Create object with information of inlets, all the others parameters
           which are not passed here get evaluated by `InletOutletManager`once
           the inlet is created.

        Parameters
        ----------

        pa_name : str
            Name of the inlet
        normal : list
            Components of normal (float)
        refpoint : list
            Point at the fluid-inlet interface (float)
        has_ghost : bool
            if True, the ghost particles will be created
        update_cls : class_name
            the class which is to be used to update the inlet/outlet
        equations : list
            List of equations (optional)
        props_to_copy : array
            properties to copy
        """

        self.pa_name = pa_name
        self.normal = normal
        self.refpoint = refpoint
        self.has_ghost = has_ghost
        self.update_cls = InletBase if update_cls is None else update_cls
        self.length = 0.0
        self.dx = 0.1
        self.umax = umax
        self.equations = [] if equations is None else equations
        self.props_to_copy = props_to_copy


class OutletInfo(InletInfo):
    def __init__(self, pa_name, normal, refpoint, has_ghost=False,
                 update_cls=None, equations=None, umax=1.0,
                 props_to_copy=None):
        """Create object with information of outlet

        The name is kept different for distinction only.
        """
        super(OutletInfo, self).__init__(
            pa_name, normal, refpoint, has_ghost, update_cls,
            equations, umax, props_to_copy)
        self.update_cls = OutletBase if update_cls is None else update_cls


class InletOutletManager(object):
    def __init__(self, fluid_arrays, inletinfo, outletinfo,
                 extraeqns=None):
        """Create the object to manage inlet outlet boundary conditions.
           Most of the variables are evaluated after the scheme and particles
           are created.

        Parameters
        -----------

        fluid_arrays : list
            List of fluid particles array names (str)
        inletinfo : list
            List of inlets (InletInfo)
        outletinfo : list
            List of outlet (OutletInfo)
        extraeqns : dict
            Dict of custom equations
        """
        self.fluids = fluid_arrays
        self.dim = None
        self.kernel = None
        self.inlets = [] if inletinfo is None else\
            [x.pa_name for x in inletinfo]
        self.outlets = [] if outletinfo is None else\
            [x.pa_name for x in outletinfo]
        self.inlet_pairs = {}
        self.outlet_pairs = {}
        self.inletinfo = inletinfo
        self.outletinfo = outletinfo
        self.ghost_inlets = []
        self.ghost_outlets = []
        self.inlet_pairs = {}
        self.outlet_pairs = {}
        self.extraeqns = {} if extraeqns is None else extraeqns
        self.active_stages = []
        self._create_ghost_names()

    def create_ghost(self, pa_arr, inlet=True):
        """Creates ghosts for the given inlet/outlet particles

        Parameters
        -----------

        pa_arr : Particle array
            particles array for which ghost is required
        inlet : bool
            if True, inlet info will be used for ghost creation
        """
        xref, yref, zref = 0.0, 0.0, 0.0
        xn, yn, zn = 0.0, 0.0, 0.0
        has_ghost = True
        if inlet:
            for info in self.inletinfo:
                if info.pa_name == pa_arr.name:
                    xref = info.refpoint[0]
                    yref = info.refpoint[1]
                    zref = info.refpoint[2]
                    xn = info.normal[0]
                    yn = info.normal[1]
                    zn = info.normal[2]
                    has_ghost = info.has_ghost
                    break

        if not inlet:
            for info in self.outletinfo:
                if info.pa_name == pa_arr.name:
                    xref = info.refpoint[0]
                    yref = info.refpoint[1]
                    zref = info.refpoint[2]
                    xn = info.normal[0]
                    yn = info.normal[1]
                    zn = info.normal[2]
                    has_ghost = info.has_ghost
                    break

        if not has_ghost:
            return None
        x = pa_arr.x
        y = pa_arr.y
        z = pa_arr.z

        xij = x - xref
        yij = y - yref
        zij = z - zref

        disp = xij * xn + yij * yn + zij * zn
        x = x - 2 * disp * xn
        y = y - 2 * disp * yn
        z = z - 2 * disp * zn

        m = pa_arr.m
        h = pa_arr.h
        rho = pa_arr.rho
        u = pa_arr.u
        name = ''
        if inlet:
            name = self.inlet_pairs[pa_arr.name]
        else:
            name = self.outlet_pairs[pa_arr.name]

        ghost_pa = get_particle_array(
            name=name, m=m, x=x, y=y, h=h, u=u, p=0.0, rho=rho
        )

        return ghost_pa

    def _create_ghost_names(self):
        '''Creates names for ghost for both inlets and outlets if needed'''
        for inlet in self.inletinfo:
            if inlet.has_ghost:
                name = 'ghost_' + inlet.pa_name
                self.inlet_pairs[inlet.pa_name] = name
                self.ghost_inlets.append(name)

        for outlet in self.outletinfo:
            if outlet.has_ghost:
                name = 'ghost_' + outlet.pa_name
                self.outlet_pairs[outlet.pa_name] = name
                self.ghost_outlets.append(name)

    def update_dx(self, dx):
        """Update the discretisation length

        Parameters
        -----------

        dx : float
            The discretisation length
        """
        all_info = self.inletinfo + self.outletinfo
        for info in all_info:
            info.dx = dx

    def _update_inlet_outlet_info(self, pa):
        """Updates refpoint, length and bound

        Parameters
        -----------

        pa : Particle_array
            Particle array of inlet/outlet
        """
        all_info = self.inletinfo + self.outletinfo
        for info in all_info:
            dx = info.dx
            if (info.pa_name == pa.name):
                x = pa.x
                y = pa.y
                z = pa.z
                xmax, xmin = max(x)+dx/2, min(x)-dx/2
                ymax, ymin = max(y)+dx/2, min(y)-dx/2
                zmax, zmin = max(z)+dx/2, min(z)-dx/2
                xdist = xmax - xmin
                ydist = ymax - ymin
                zdist = zmax - zmin
                xn, yn, zn = info.normal[0], info.normal[1], info.normal[2]
                info.length = abs(xdist*xn+ydist*yn+zdist*zn)

    def add_io_properties(self, pa, scheme=None):
        """Add properties to be used in inlet/outlet equations

        Parameters
        ----------

        pa : particle_array
            Particle array of inlet/outlet
        scheme : pysph.sph.scheme
            The insance of scheme class
        """
        pass

    def get_io_names(self, ghost=False):
        """return all the names of inlets and outlets
        Parameters
        ----------

        ghost : bool
            if True, return the names of ghost also
        """
        if ghost:
            return (self.inlets + self.outlets + self.ghost_inlets +
                    self.ghost_outlets)
        else:
            return self.inlets + self.outlets

    def get_stepper(self, scheme, integrator, **kw):
        """Returns the steppers for inlet/outlet

        Parameters
        ----------

        scheme : pysph.sph.scheme
            The instance of the scheme class
        intergrator : pysph.sph.integrator
            The parent class of the integrator
        **kw : extra arguments
            Extra arguments depending upon the scheme used
        """
        raise NotImplementedError()

    def setup_iom(self, dim, kernel):
        """Essential data passed

        Parameters
        ----------

        dim : int
            dimension of the problem
        kernel : pysph.base.kernel
            the kernel instance
        """
        self.dim = dim
        self.kernel = kernel

    def get_equations(self, scheme, **kw):
        """Returns the equations for inlet/outlet

        Parameters
        ----------

        scheme : pysph.sph.scheme
            The instance of the scheme class
        **kw : extra arguments
            Extra arguments depending upon the scheme used
        """
        return []

    def get_equations_post_compute_acceleration(self):
        """Returns the equations for inlet/outlet used post acceleration
           computation"""
        return []

    def get_inlet_outlet(self, particle_array):
        """ Returns list of `Inlet` and `Outlet` instances which
            performs the change in inlet particles to outlet
            particles.

        Parameters
        -----------

        particle_array : list
            List of all particle_arrays
        """
        result = []
        for inlet in self.inletinfo:
            i_name = inlet.pa_name
            self._update_inlet_outlet_info(particle_array[i_name])
            ghost_pa = None if i_name not in self.inlet_pairs\
                else particle_array[self.inlet_pairs[i_name]]
            for fluid in self.fluids:
                in1 = inlet.update_cls(
                        particle_array[i_name],
                        particle_array[fluid],
                        inlet,
                        self.kernel,
                        self.dim,
                        self.active_stages,
                        ghost_pa=ghost_pa
                        )
            result.append(in1)

        for outlet in self.outletinfo:
            o_name = outlet.pa_name
            self._update_inlet_outlet_info(particle_array[o_name])
            ghost_pa = None if o_name not in self.outlet_pairs else\
                particle_array[self.outlet_pairs[o_name]]
            for fluid in self.fluids:
                out1 = outlet.update_cls(
                        particle_array[o_name],
                        particle_array[fluid],
                        outlet,
                        self.kernel,
                        self.dim,
                        self.active_stages,
                        ghost_pa=ghost_pa
                        )
            result.append(out1)

        return result


class IOEvaluate(Equation):
    def __init__(self, dest, sources, x, y, z, xn, yn, zn,
                 maxdist=1000.0):
        """Compute ioid for the particles
           0 : particle is in fluid
           1 : particle is inside the inlet/outlet
           2 : particle is out of inlet/outlet

        parameters:
        ----------

        dest : str
            destination particle array name
        sources : list
            List of source particle arrays
        x : float
            x coordinate of interface point
        y : float
            y coordinate of interface point
        z : float
            z coordinate of interface point
        xn : float
            x component of interface outward normal
        yn : float
            y component of interface outward normal
        zn : float
            z component of interface outward normal
        maxdist : float
            Maximum length of inlet/outlet

        """
        self.x = x
        self.y = y
        self.z = z
        self.xn = xn
        self.yn = yn
        self.zn = zn
        self.maxdist = maxdist

        super(IOEvaluate, self).__init__(dest, sources)

    def initialize(self, d_ioid, d_idx):
        d_ioid[d_idx] = 1

    def loop(self, d_idx, d_x, d_y, d_z, d_ioid, d_disp):
        delx = d_x[d_idx] - self.x
        dely = d_y[d_idx] - self.y
        delz = d_z[d_idx] - self.z

        d_disp[d_idx] = delx * self.xn + dely * self.yn + delz * self.zn

        if ((d_disp[d_idx] > 0.000001) and
           (d_disp[d_idx]-self.maxdist < 0.000001)):
            d_ioid[d_idx] = 1
        elif (d_disp[d_idx] - self.maxdist > 0.000001):
            d_ioid[d_idx] = 2
        else:
            d_ioid[d_idx] = 0


class UpdateNormalsAndDisplacements(Equation):
    def __init__(self, dest, sources, xn, yn, zn, xo, yo, zo):
        """Update normal and perpendicular distance from the interface
        for the inlet/outlet particles

        parameters:
        ----------

        dest : str
            destination particle array name
        sources : list
            List of source particle arrays
        xn : float
            x component of interface outward normal
        yn : float
            y component of interface outward normal
        zn : float
            z component of interface outward normal
        xo : float
            x coordinate of interface point
        yo : float
            y coordinate of interface point
        zo : float
            z coordinate of interface point
        """
        self.xn = xn
        self.yn = yn
        self.zn = zn
        self.xo = xo
        self.yo = yo
        self.zo = zo

        super(UpdateNormalsAndDisplacements, self).__init__(dest, sources)

    def loop(self, d_idx, d_xn, d_yn, d_zn, d_x, d_y, d_z, d_disp):
        d_xn[d_idx] = self.xn
        d_yn[d_idx] = self.yn
        d_zn[d_idx] = self.zn

        xij = declare('matrix(3)')
        xij[0] = d_x[d_idx] - self.xo
        xij[1] = d_y[d_idx] - self.yo
        xij[2] = d_z[d_idx] - self.zo

        d_disp[d_idx] = abs(xij[0]*d_xn[d_idx] + xij[1]*d_yn[d_idx] +
                            xij[2]*d_yn[d_idx])


class CopyNormalsandDistances(Equation):
    """Copy normals and distances from outlet/inlet particles to ghosts"""

    def initialize_pair(self, d_idx, d_xn, d_yn, d_zn, s_xn, s_yn, s_zn,
                        d_disp, s_disp):
        d_xn[d_idx] = s_xn[d_idx]
        d_yn[d_idx] = s_yn[d_idx]
        d_zn[d_idx] = s_zn[d_idx]

        d_disp[d_idx] = s_disp[d_idx]


class InletStep(IntegratorStep):
    def initialize(self, d_x0, d_idx, d_x):
        d_x0[d_idx] = d_x[d_idx]

    def stage1(self, d_idx, d_x, d_x0, d_u, dt):
        dtb2 = 0.5 * dt
        d_x[d_idx] = d_x0[d_idx] + dtb2*d_u[d_idx]

    def stage2(self, d_idx, d_x, d_x0, d_u, dt):
        d_x[d_idx] = d_x0[d_idx] + dt*d_u[d_idx]


class OutletStepWithUhat(IntegratorStep):
    def initialize(self, d_x0, d_idx, d_x):
        d_x0[d_idx] = d_x[d_idx]

    def stage1(self, d_idx, d_x, d_x0, d_uhat, dt):
        dtb2 = 0.5 * dt
        d_x[d_idx] = d_x0[d_idx] + dtb2*d_uhat[d_idx]

    def stage2(self, d_idx, d_x, d_x0, d_uhat, dt):
        d_x[d_idx] = d_x0[d_idx] + dt*d_uhat[d_idx]


class OutletStep(InletStep):
    pass


class InletBase(object):
    def __init__(self, inlet_pa, dest_pa, inletinfo, kernel, dim,
                 active_stages=[1], callback=None, ghost_pa=None):
        """An API to add/delete particle when moving between inlet-fluid

        Parameters
        ----------

        inlet_pa : particle_array
            particle array for inlet
        dest_pa : particle_array
            particle_array of the fluid
        inletinfo : InletInfo instance
            contains information fo inlet
        kernel : Kernel instance
            Kernel to be used for computations
        dim : int
            dimension of the problem
        active_stages : list
            stages of integrator at which update should be active
        callback : function
            callback after the update function
        ghost_pa : particle_array
            particle_array of the ghost_inlet
        """
        self.inlet_pa = inlet_pa
        self.dest_pa = dest_pa
        self.ghost_pa = ghost_pa
        self.callback = callback
        self.dim = dim
        self.kernel = kernel
        self.inletinfo = inletinfo
        self.x = self.y = self.z = 0.0
        self.xn = self.yn = self.zn = 0.0
        self.length = 0.0
        self.dx = 0.0
        self.active_stages = active_stages
        self.io_eval = None
        self._init = False
        cfg = get_config()
        self.gpu = cfg.use_opencl or cfg.use_cuda

    def initialize(self):
        """Function to initialize the class variables after
        evaluation in SimpleInletOutlet class"""
        inletinfo = self.inletinfo
        self.x = inletinfo.refpoint[0]
        self.y = inletinfo.refpoint[1]
        self.z = inletinfo.refpoint[2]
        self.xn = inletinfo.normal[0]
        self.yn = inletinfo.normal[1]
        self.zn = inletinfo.normal[2]
        self.length = inletinfo.length
        self.dx = inletinfo.dx

    def _create_io_eval(self):
        """Evaluator to assign ioid to particles leaving a domain"""
        if self.io_eval is None:
            from pysph.sph.equation import Group
            from pysph.tools.sph_evaluator import SPHEvaluator
            i_name = self.inlet_pa.name
            f_name = self.dest_pa.name
            eqns = []
            eqns.append(Group(equations=[
                IOEvaluate(
                    i_name, [], x=self.x, y=self.y, z=self.z,
                    xn=self.xn, yn=self.yn, zn=self.zn,
                    maxdist=self.length)],
                real=False, update_nnps=False))

            eqns.append(Group(equations=[
                IOEvaluate(
                    f_name, [], x=self.x, y=self.y, z=self.z, xn=self.xn,
                    yn=self.yn, zn=self.zn)],
                real=False, update_nnps=False))

            if self.gpu:
                from pysph.base.gpu_nnps import ZOrderGPUNNPS as NNPS
            else:
                from pysph.base.nnps import LinkedListNNPS as NNPS

            arrays = [self.inlet_pa] + [self.dest_pa]
            io_eval = SPHEvaluator(
                arrays=arrays, equations=eqns, dim=self.dim,
                kernel=self.kernel, nnps_factory=NNPS)
            return io_eval
        else:
            return self.io_eval

    def update(self, time, dt, stage):
        """ Update function called after each stage"""
        if not self._init:
            self.initialize()
            self._init = True
        if stage in self.active_stages:

            dest_pa = self.dest_pa
            inlet_pa = self.inlet_pa
            ghost_pa = self.ghost_pa

            self.io_eval = self._create_io_eval()
            self.io_eval.update()
            self.io_eval.evaluate()

            if self.gpu:
                inlet_pa.gpu.pull(*'ioid x y z'.split())
                ghost_pa.gpu.pull(*'x y z'.split())
                dest_pa.gpu.pull('ioid')
            io_id = inlet_pa.ioid
            cond = (io_id == 0)
            all_idx = np.where(cond)[0]
            inlet_pa.extract_particles(all_idx, dest_pa)

            # moving the moved particles back to the array beginning.
            inlet_pa.x[all_idx] += self.length * self.xn
            inlet_pa.y[all_idx] += self.length * self.yn
            inlet_pa.z[all_idx] += self.length * self.zn

            if ghost_pa:
                ghost_pa.x[all_idx] -= self.length * self.xn
                ghost_pa.y[all_idx] -= self.length * self.yn
                ghost_pa.z[all_idx] -= self.length * self.zn

            if self.callback is not None:
                self.callback(dest_pa, inlet_pa)


class OutletBase(object):
    def __init__(self, outlet_pa, source_pa, outletinfo, kernel,
                 dim, active_stages=[1], callback=None, ghost_pa=None):
        """An API to add/delete particle when moving between fluid-outlet

        Parameters
        ----------

        outlet_pa : particle_array
            particle array for outlet
        source_pa : particle_array
            particle_array of the fluid
        ghost_pa : particle_array
            particle_array of the outlet ghost
        outletinfo : OutletInfo instance
            contains information fo outlet
        kernel : Kernel instance
            Kernel to be used for computations
        dim : int
            dimnesion of the problem
        active_stages : list
            stages of integrator at which update should be active
        callback : function
            callback after the update function
        """
        self.outlet_pa = outlet_pa
        self.source_pa = source_pa
        self.ghost_pa = ghost_pa
        self.dim = dim
        self.kernel = kernel
        self.outletinfo = outletinfo
        self.x = self.y = self.z = 0.0
        self.xn = self.yn = self.zn = 0.0
        self.length = 0.0
        self.callback = callback
        self.active_stages = active_stages
        self.io_eval = None
        self._init = False
        self.props_to_copy = None
        cfg = get_config()
        self.gpu = cfg.use_opencl or cfg.use_cuda

    def initialize(self):
        """Function to initialize the class variables after
        evaluation in SimpleInletOutlet class"""
        outletinfo = self.outletinfo
        self.x = outletinfo.refpoint[0]
        self.y = outletinfo.refpoint[1]
        self.z = outletinfo.refpoint[2]
        self.xn = outletinfo.normal[0]
        self.yn = outletinfo.normal[1]
        self.zn = outletinfo.normal[2]
        self.length = outletinfo.length
        self.props_to_copy = outletinfo.props_to_copy

    def _create_io_eval(self):
        """Evaluator to assign ioid to particles leaving a domain"""
        if self.io_eval is None:
            from pysph.sph.equation import Group
            from pysph.tools.sph_evaluator import SPHEvaluator
            o_name = self.outlet_pa.name
            f_name = self.source_pa.name
            eqns = []
            if self.gpu:
                from pysph.base.gpu_nnps import ZOrderGPUNNPS as NNPS
            else:
                from pysph.base.nnps import LinkedListNNPS as NNPS

            eqns.append(Group(equations=[
                IOEvaluate(
                    o_name, [], x=self.x, y=self.y, z=self.z, xn=self.xn,
                    yn=self.yn, zn=self.zn, maxdist=self.length)],
                real=False, update_nnps=False))

            eqns.append(Group(equations=[
                IOEvaluate(
                    f_name, [], x=self.x, y=self.y, z=self.z, xn=self.xn,
                    yn=self.yn, zn=self.zn,)], real=False, update_nnps=False))

            arrays = [self.outlet_pa] + [self.source_pa]
            io_eval = SPHEvaluator(arrays=arrays, equations=eqns, dim=self.dim,
                                   kernel=self.kernel, nnps_factory=NNPS)
            return io_eval
        else:
            return self.io_eval

    def update(self, time, dt, stage):
        """Update function called after each stage"""
        if not self._init:
            self.initialize()
            self._init = True
        if stage in self.active_stages:
            props_to_copy = self.props_to_copy

            outlet_pa = self.outlet_pa
            source_pa = self.source_pa

            self.io_eval = self._create_io_eval()
            self.io_eval.update()
            self.io_eval.evaluate()

            # adding particles to the destination array.
            if self.gpu:
                source_pa.gpu.pull('ioid')
            io_id = source_pa.ioid
            cond = (io_id == 1)
            all_idx = np.where(cond)[0]
            source_pa.extract_particles(
                all_idx, dest_array=outlet_pa, props=props_to_copy)
            source_pa.remove_particles(all_idx)

            if self.gpu:
                outlet_pa.gpu.pull('ioid')
            io_id = outlet_pa.ioid
            cond = (io_id == 2)
            all_idx = np.where(cond)[0]
            outlet_pa.remove_particles(all_idx)

            if self.callback is not None:
                self.callback(source_pa, outlet_pa)
