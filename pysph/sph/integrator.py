"""Basic code for the templated integrators.

Currently we only support two-step integrators.

These classes are used to generate the code for the actual integrators
from the `sph_eval` module.
"""

import inspect

# Local imports.
from pysph.sph.equation import get_array_names
from pysph.base.cython_generator import CythonGenerator

from numpy import sqrt

###############################################################################
# `IntegratorStep` class
###############################################################################
class IntegratorStep(object):
    """Subclass this and implement the methods ``predictor`` and ``corrector``.
    Use the same conventions as the equations.
    """
    def initialize(self):
        pass
    def predictor(self):
        pass
    def corrector(self):
        pass


###############################################################################
# `EulerStep` class
###############################################################################
class EulerStep(IntegratorStep):
    """Fast but inaccurate integrator. Use this for testing"""
    def initialize(self):
        pass
    def predictor(self):
        pass
    def corrector(self, d_idx, d_u, d_v, d_w, d_au, d_av, d_aw, d_x, d_y,
                  d_z, d_rho, d_arho, dt=0.0):
        d_u[d_idx] += dt*d_au[d_idx]
        d_v[d_idx] += dt*d_av[d_idx]
        d_w[d_idx] += dt*d_aw[d_idx]

        d_x[d_idx] += dt*d_u[d_idx]
        d_y[d_idx] += dt*d_v[d_idx]
        d_z[d_idx] += dt*d_w[d_idx]

        d_rho[d_idx] += dt*d_arho[d_idx]

###############################################################################
# `WCSPHStep` class
###############################################################################
class WCSPHStep(IntegratorStep):
    """Standard Predictor Corrector integrator for the WCSPH formulation
    
    Use this integrator for WCSPH formulations. In the predictor step,
    the particles are advanced to `t + dt/2`. The particles are then
    advanced with the new force computed at this position.

    This integrator can be used in PEC or EPEC mode.

    The same integrator can be used for other problems. Like for
    example solid mechanics (see SolidMechStep)

    """
    def initialize(self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z,
                   d_u0, d_v0, d_w0, d_u, d_v, d_w, d_rho0, d_rho):
        d_x0[d_idx] = d_x[d_idx]
        d_y0[d_idx] = d_y[d_idx]
        d_z0[d_idx] = d_z[d_idx]

        d_u0[d_idx] = d_u[d_idx]
        d_v0[d_idx] = d_v[d_idx]
        d_w0[d_idx] = d_w[d_idx]

        d_rho0[d_idx] = d_rho[d_idx]

    def predictor(self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z,
                   d_u0, d_v0, d_w0, d_u, d_v, d_w, d_rho0, d_rho, d_au, d_av,
                   d_aw, d_ax, d_ay, d_az, d_arho, dt=0.0):
        dtb2 = 0.5*dt
        d_u[d_idx] = d_u0[d_idx] + dtb2*d_au[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dtb2*d_av[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dtb2*d_aw[d_idx]

        d_x[d_idx] = d_x0[d_idx] + dtb2 * d_ax[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dtb2 * d_ay[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dtb2 * d_az[d_idx]

        # Update densities and smoothing lengths from the accelerations
        d_rho[d_idx] = d_rho0[d_idx] + dtb2 * d_arho[d_idx]

    def corrector(self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z,
                   d_u0, d_v0, d_w0, d_u, d_v, d_w, d_rho0, d_rho, d_au, d_av,
                   d_aw, d_ax, d_ay, d_az, d_arho, dt=0.0):

        d_u[d_idx] = d_u0[d_idx] + dt*d_au[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dt*d_av[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dt*d_aw[d_idx]

        d_x[d_idx] = d_x0[d_idx] + dt * d_ax[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dt * d_ay[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dt * d_az[d_idx]

        # Update densities and smoothing lengths from the accelerations
        d_rho[d_idx] = d_rho0[d_idx] + dt * d_arho[d_idx]

###############################################################################
# `SolidMechStep` class
###############################################################################
class SolidMechStep(IntegratorStep):
    """Predictor corrector Integrator for solid mechanics problems"""
    def initialize(self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z,
                   d_u0, d_v0, d_w0, d_u, d_v, d_w, d_rho0, d_rho,
                   d_s00, d_s01, d_s02, d_s11, d_s12, d_s22,
                   d_s000, d_s010, d_s020, d_s110, d_s120, d_s220,
                   d_e0, d_e):
        d_x0[d_idx] = d_x[d_idx]
        d_y0[d_idx] = d_y[d_idx]
        d_z0[d_idx] = d_z[d_idx]

        d_u0[d_idx] = d_u[d_idx]
        d_v0[d_idx] = d_v[d_idx]
        d_w0[d_idx] = d_w[d_idx]

        d_rho0[d_idx] = d_rho[d_idx]
        d_e0[d_idx] = d_e[d_idx]

        d_s000[d_idx] = d_s00[d_idx]
        d_s010[d_idx] = d_s01[d_idx]
        d_s020[d_idx] = d_s02[d_idx]
        d_s110[d_idx] = d_s11[d_idx]
        d_s120[d_idx] = d_s12[d_idx]
        d_s220[d_idx] = d_s22[d_idx]

    def predictor(self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z,
                  d_u0, d_v0, d_w0, d_u, d_v, d_w, d_rho0, d_rho, d_au, d_av,
                  d_aw, d_ax, d_ay, d_az, d_arho, d_e, d_e0, d_ae,
                  d_s00, d_s01, d_s02, d_s11, d_s12, d_s22,
                  d_s000, d_s010, d_s020, d_s110, d_s120, d_s220,
                  d_as00, d_as01, d_as02, d_as11, d_as12, d_as22,
                  dt=0.0):
        dtb2 = 0.5*dt
        d_u[d_idx] = d_u0[d_idx] + dtb2*d_au[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dtb2*d_av[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dtb2*d_aw[d_idx]

        d_x[d_idx] = d_x0[d_idx] + dtb2 * d_ax[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dtb2 * d_ay[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dtb2 * d_az[d_idx]

        # Update densities and smoothing lengths from the accelerations
        d_rho[d_idx] = d_rho0[d_idx] + dtb2 * d_arho[d_idx]
        d_e[d_idx] = d_e0[d_idx] + dtb2 * d_ae[d_idx]

        # update deviatoric stress components
        d_s00[d_idx] = d_s000[d_idx] + dtb2 * d_as00[d_idx]
        d_s01[d_idx] = d_s010[d_idx] + dtb2 * d_as01[d_idx]
        d_s02[d_idx] = d_s020[d_idx] + dtb2 * d_as02[d_idx]
        d_s11[d_idx] = d_s110[d_idx] + dtb2 * d_as11[d_idx]
        d_s12[d_idx] = d_s120[d_idx] + dtb2 * d_as12[d_idx]
        d_s22[d_idx] = d_s220[d_idx] + dtb2 * d_as22[d_idx]

    def corrector(self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z,
                  d_u0, d_v0, d_w0, d_u, d_v, d_w, d_rho0, d_rho, d_au, d_av,
                  d_aw, d_ax, d_ay, d_az, d_arho, d_e, d_ae, d_e0,
                  d_s00, d_s01, d_s02, d_s11, d_s12, d_s22,
                  d_s000, d_s010, d_s020, d_s110, d_s120, d_s220,
                  d_as00, d_as01, d_as02, d_as11, d_as12, d_as22,
                  dt=0.0):

        d_u[d_idx] = d_u0[d_idx] + dt*d_au[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dt*d_av[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dt*d_aw[d_idx]

        d_x[d_idx] = d_x0[d_idx] + dt * d_ax[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dt * d_ay[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dt * d_az[d_idx]

        # Update densities and smoothing lengths from the accelerations
        d_rho[d_idx] = d_rho0[d_idx] + dt * d_arho[d_idx]
        d_e[d_idx] = d_e0[d_idx] + dt * d_ae[d_idx]

        # update deviatoric stress components
        d_s00[d_idx] = d_s000[d_idx] + dt * d_as00[d_idx]
        d_s01[d_idx] = d_s010[d_idx] + dt * d_as01[d_idx]
        d_s02[d_idx] = d_s020[d_idx] + dt * d_as02[d_idx]
        d_s11[d_idx] = d_s110[d_idx] + dt * d_as11[d_idx]
        d_s12[d_idx] = d_s120[d_idx] + dt * d_as12[d_idx]
        d_s22[d_idx] = d_s220[d_idx] + dt * d_as22[d_idx]

###############################################################################
# `TransportVelocityStep` class
###############################################################################
class TransportVelocityStep(IntegratorStep):
    """Integrator defined in 'A transport velocity formulation for
    smoothed particle hydrodynamics', 2013, JCP, 241, pp 292--307

    For a predictor-corrector style of integrator, this integrator
    should operate only in PEC mode.
    
    """
    def initialize(self):
        pass

    def predictor(self, d_idx, d_u, d_v, d_au, d_av, d_uhat, d_auhat, d_vhat,
                  d_avhat, d_x, d_y, dt=0.0):
        dtb2 = 0.5*dt

        # velocity update eqn (14)
        d_u[d_idx] += dtb2*d_au[d_idx]
        d_v[d_idx] += dtb2*d_av[d_idx]

        # advection velocity update eqn (15)
        d_uhat[d_idx] = d_u[d_idx] + dtb2*d_auhat[d_idx]
        d_vhat[d_idx] = d_v[d_idx] + dtb2*d_avhat[d_idx]

        # position update eqn (16)
        d_x[d_idx] += dt*d_uhat[d_idx]
        d_y[d_idx] += dt*d_vhat[d_idx]

    def corrector(self, d_idx, d_u, d_v, d_au, d_av, d_vmag, dt=0.0):
        dtb2 = 0.5*dt

        # corrector update eqn (17)
        d_u[d_idx] += dtb2*d_au[d_idx]
        d_v[d_idx] += dtb2*d_av[d_idx]

        # magnitude of velocity squared
        d_vmag[d_idx] = d_u[d_idx]*d_u[d_idx] + d_v[d_idx]*d_v[d_idx]

###############################################################################
# `AdamiVerletStep` class
###############################################################################
class AdamiVerletStep(IntegratorStep):
    """Verlet time integration described in `A generalized wall
    boundary condition for smoothed particle hydrodynamics` 2012, JCP,
    231, pp 7057--7075

    This integrator can operate in either PEC mode or in EPEC mode as
    described in the paper.

    """
    def initialize(self):
        pass

    def predictor(self, d_idx, d_u, d_v, d_au, d_av, d_x, d_y, dt=0.0):
        dtb2 = 0.5*dt

        # velocity predictor eqn (14)
        d_u[d_idx] += dtb2*d_au[d_idx]
        d_v[d_idx] += dtb2*d_av[d_idx]

        # position predictor eqn (15)
        d_x[d_idx] += dtb2*d_u[d_idx]
        d_y[d_idx] += dtb2*d_v[d_idx]

    def corrector(self, d_idx, d_u, d_v, d_au, d_av, d_x, d_y, d_rho, d_arho,
                  d_vmag, dt=0.0):
        dtb2 = 0.5*dt

        # velocity corrector eqn (18)
        d_u[d_idx] += dtb2*d_au[d_idx]
        d_v[d_idx] += dtb2*d_av[d_idx]

        # position corrector eqn (17)
        d_x[d_idx] += dtb2*d_u[d_idx]
        d_y[d_idx] += dtb2*d_v[d_idx]

        # density corrector eqn (16)
        d_rho[d_idx] += dt * d_arho[d_idx]

        # magnitude of velocity squared
        d_vmag[d_idx] = d_u[d_idx]*d_u[d_idx] + d_v[d_idx]*d_v[d_idx]

###############################################################################
# `Integrator` class
###############################################################################
class Integrator(object):
    """Generic class for Predictor Corrector integrators in PySPH

    Predictor corrector integrators can have two modes of
    operation. Consider the ODE system `\frac{dy}{dt} = F(y)`. 

    In the Predict-Evaluate-Correct (PEC) mode, the system is advanced
    using:
    
    y^{n+\frac{1}{2}} = y^n + \frac{\Delta t}{2}F(y^{n-\frac{1}{2}}) --> Predict
    
    F(y^{n+\frac{1}{2}}) --> Evaluate

    y^{n + 1} = y^n + \Delta t F(y^{n+\frac{1}{2}})

    In the Evaluate-Predict-Evaluate-Correct (EPEC) mode, the system
    is advanced using:
    
    F(y^n) --> Evaluate
    
    y^{n+\frac{1}{2}} = y^n + F(y^n) --> Predict

    F(y^{n+\frac{1}{2}}) --> Evaluate

    y^{n+1} = y^n + \Delta t F(y^{n+\frac{1}{2}}) --> Correct

    Notes:
    
    The Evaluate stage of the integrator forces a function
    evaluation. Therefore, the PEC mode is much faster but relies on
    old accelertions for the Prediction stage.

    In the EPEC mode, the final corrector can be modified to

    y^{n+1} = y^n + \frac{\Delta t}{2}\left( F(y^n) + F(y^{n+\frac{1}{2}}) \right)

    This would require additional storage for the accelerations.

    """

    def __init__(self, epec=False, **kw):
        """Pass fluid names and suitable `IntegratorStep` instances.

        For example::

            >>> integrator = Integrator(fluid=WCSPHStep(), solid=WCSPHStep())

        where "fluid" and "solid" are the names of the particle arrays.
        """
        self.epec = epec

        for array_name, integrator_step in kw.iteritems():
            if not isinstance(integrator_step, IntegratorStep):
                msg='Stepper %s must be an instance of IntegratorStep'%(integrator_step)
                raise ValueError(msg)

        self.steppers = kw
        self.parallel_manager = None
        # This is set later when the underlying compiled integrator is created
        # by the SPHEval.
        self.integrator = None

    def set_fixed_h(self, fixed_h):
        # compute h_minimum once for constant smoothing lengths
        if fixed_h:
            self.compute_h_minimum()

        self.fixed_h=fixed_h

    def compute_h_minimum(self):
        calc = self.integrator.sph_calc

        hmin = 1.0
        for pa in calc.particle_arrays:
            h = pa.get_carray('h')
            h.update_min_max()

            if h.minimum < hmin:
                hmin = h.minimum

        self.h_minimum = hmin

    def compute_time_step(self, dt, cfl):
        calc = self.integrator.sph_calc

        # different time step controls
        dt_cfl_factor = calc.dt_cfl
        dt_visc_factor = calc.dt_viscous

        # force factor is acceleration squared
        dt_force_factor = sqrt(calc.dt_force)

        # iterate over particles and find hmin if using vatialbe h
        if not self.fixed_h:
            self.compute_h_minimum()

        hmin = self.h_minimum

        # default time steps set to some large value
        dt_cfl = dt_force = dt_viscous = 1e20

        # stable time step based on courant condition
        if dt_cfl_factor > 0:
            dt_cfl = hmin/dt_cfl_factor

        # stable time step based on force criterion
        if dt_force_factor > 0:
            dt_force = sqrt( hmin/dt_force_factor )

        # stable time step based on viscous condition
        if dt_visc_factor > 0:
            dt_viscous = hmin/dt_visc_factor

        # minimum of all three
        dt_min = min( dt_cfl, dt_force, dt_viscous )

        # return the computed time steps. If dt factors aren't
        # defined, the default dt is returned
        if dt_min <= 0.0:
            return dt
        else:
            return cfl*dt_min

    def get_stepper_code(self):
        classes = {}
        for dest, stepper in self.steppers.iteritems():
            cls = stepper.__class__.__name__
            classes[cls] = stepper

        wrappers = []
        code_gen = CythonGenerator()
        for cls in sorted(classes.keys()):
            code_gen.parse(classes[cls])
            wrappers.append(code_gen.get_code())
        return '\n'.join(wrappers)

    def get_stepper_defs(self):
        lines = []
        for dest, stepper in self.steppers.iteritems():
            cls_name = stepper.__class__.__name__
            code = 'cdef public {cls} {name}'.format(cls=cls_name,
                                                     name=dest+'_stepper')
            lines.append(code)
        return '\n'.join(lines)

    def get_stepper_init(self):
        lines = []
        for dest, stepper in self.steppers.iteritems():
            cls_name = stepper.__class__.__name__
            code = 'self.{name} = {cls}(**steppers["{dest}"].__dict__)'\
                        .format(name=dest+'_stepper', cls=cls_name,
                                dest=dest)
            lines.append(code)
        return '\n'.join(lines)

    def get_args(self, dest, method):
        stepper = self.steppers[dest]
        meth = getattr(stepper, method)
        return inspect.getargspec(meth).args

    def get_array_declarations(self, method):
        arrays = set()
        for dest in self.steppers:
            s, d = get_array_names(self.get_args(dest, method))
            arrays.update(s | d)

        decl = []
        for arr in sorted(arrays):
            decl.append('cdef double* %s'%arr)
        return '\n'.join(decl)

    def get_array_setup(self, dest, method):
        s, d = get_array_names(self.get_args(dest, method))
        lines = ['%s = dst.%s.data'%(n, n[2:]) for n in s|d]
        return '\n'.join(lines)

    def get_stepper_loop(self, dest, method):
        args = self.get_args(dest, method)
        if 'self' in args:
            args.remove('self')
        call_args = ', '.join(args)
        c = 'self.{obj}.{method}({args})'\
                .format(obj=dest+'_stepper', method=method, args=call_args)
        return c

    def integrate(self, time, dt, count):
        self.integrator.integrate(time, dt, count)

    def set_parallel_manager(self, pm):
        self.integrator.set_parallel_manager(pm)

    def set_integrator(self, integrator):
        self.integrator = integrator
        self.integrator.set_predictor_corrector_mode(self.epec)
