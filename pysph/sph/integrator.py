"""Basic code for the templated integrators.

Currently we only support two-step integrators.

These classes are used to generate the code for the actual integrators
from the `sph_eval` module.
"""

from numpy import sqrt
import numpy as np

# Local imports.
from .integrator_step import IntegratorStep


###############################################################################
# `Integrator` class
###############################################################################
class Integrator(object):
    r"""Generic class for multi-step integrators in PySPH for a system of
    ODES of the form :math:`\frac{dy}{dt} = F(y)`.
    """

    def __init__(self, **kw):
        """Pass fluid names and suitable `IntegratorStep` instances.

        For example::

            >>> integrator = Integrator(fluid=WCSPHStep(), solid=WCSPHStep())

        where "fluid" and "solid" are the names of the particle arrays.
        """
        for array_name, integrator_step in kw.items():
            if not isinstance(integrator_step, IntegratorStep):
                msg = ('Stepper %s must be an instance of '
                       'IntegratorStep' % (integrator_step))
                raise ValueError(msg)

        self.steppers = kw
        self.parallel_manager = None
        self.nnps = None
        self.acceleration_evals = None
        # This is set later when the underlying compiled integrator is created
        # by the SPHCompiler.
        self.c_integrator = None
        self._has_dt_adapt = None
        self.fixed_h = False

    def __repr__(self):
        name = self.__class__.__name__
        s = self.steppers
        args = ', '.join(['%s=%s' % (k, s[k]) for k in s])
        return '%s(%s)' % (name, args)

    def _get_dt_adapt_factors(self):
        a_eval = self.acceleration_evals[0]
        factors = [-1.0, -1.0, -1.0]
        for pa in a_eval.particle_arrays:
            prop_names = []
            for i, name in enumerate(('dt_cfl', 'dt_force', 'dt_visc')):
                if name in pa.properties:
                    if pa.gpu:
                        prop_names.append(name)
                    else:
                        max_val = np.max(pa.get(name))
                        factors[i] = max(factors[i], max_val)
            if pa.gpu:
                pa.gpu.update_minmax_cl(prop_names, only_max=True)
                for i, name in enumerate(('dt_cfl', 'dt_force', 'dt_visc')):
                    if name in pa.properties:
                        max_val = getattr(pa.gpu, name).maximum
                        factors[i] = max(factors[i], max_val)
        cfl_f, force_f, visc_f = factors
        return cfl_f, force_f, visc_f

    def _get_explicit_dt_adapt(self):
        """Checks if the user is defining a 'dt_adapt' property where the
        timestep is directly specified.

        This returns None if no such parameter is found, else it returns the
        allowed timestep.
        """
        a_eval = self.acceleration_evals[0]
        if self._has_dt_adapt is None:
            self._has_dt_adapt = any(
                'dt_adapt' in pa.properties for pa in a_eval.particle_arrays
            )
        if self._has_dt_adapt:
            dt_min = 1e20
            for pa in a_eval.particle_arrays:
                if 'dt_adapt' in pa.properties:
                    if pa.gpu is not None:
                        from compyle.array import minimum
                        min_val = minimum(pa.gpu.dt_adapt)
                    else:
                        min_val = np.min(pa.dt_adapt)
                    dt_min = min(dt_min, min_val)
            if dt_min > 0.0:
                return dt_min
            else:
                return None
        else:
            return None

    ##########################################################################
    # Public interface.
    ##########################################################################
    def set_acceleration_evals(self, a_evals):
        '''Set the acceleration evaluators.

        This must be done before the integrator is used.

        If you are using the SPHCompiler, it automatically calls this method.

        '''
        if isinstance(a_evals, (list, tuple)):
            self.acceleration_evals = a_evals
        else:
            self.acceleration_evals = [a_evals]

    def set_fixed_h(self, fixed_h):
        # compute h_minimum once for constant smoothing lengths
        if fixed_h:
            self.compute_h_minimum()

        self.fixed_h = fixed_h

    def set_nnps(self, nnps):
        self.nnps = nnps
        self.c_integrator.set_nnps(nnps)

    def compute_h_minimum(self):
        a_eval = self.acceleration_evals[0]

        hmin = 1.0
        for pa in a_eval.particle_arrays:
            if pa.gpu:
                h = pa.gpu.get_device_array('h')
            else:
                h = pa.get_carray('h')

            if h.minimum < hmin:
                hmin = h.minimum

        self.h_minimum = hmin

    def compute_time_step(self, dt, cfl):
        """If there are any adaptive timestep constraints, the appropriate
        timestep is returned, else None is returned.
        """
        dt_adapt = self._get_explicit_dt_adapt()
        if dt_adapt is not None:
            return dt_adapt

        dt_cfl_fac, dt_force_fac, dt_visc_fac = self._get_dt_adapt_factors()

        # iterate over particles and find hmin if using variable h
        if not self.fixed_h:
            self.compute_h_minimum()

        hmin = self.h_minimum

        # default time steps set to some large value
        dt_cfl = dt_force = dt_viscous = 1e10

        # stable time step based on courant condition
        if dt_cfl_fac > 0:
            dt_cfl = hmin/dt_cfl_fac

        # stable time step based on force criterion
        if dt_force_fac > 0:
            dt_force = sqrt(hmin/sqrt(dt_force_fac))

        # stable time step based on viscous condition
        if dt_visc_fac > 0:
            dt_viscous = hmin/dt_visc_fac

        # minimum of all three
        dt_min = min(dt_cfl, dt_force, dt_viscous)

        # return the computed time steps. If dt factors aren't
        # defined, the default dt is returned
        if dt_min <= 0.0 or abs(dt_min - 1e10) < 1:
            return None
        else:
            return cfl*dt_min

    def one_timestep(self, t, dt):
        """User written function that actually does one timestep.

        This function is used in the high-performance Cython implementation.
        The assumptions one may make are the following:

            - t and dt are passed.

            - the following methods are available:

                - self.initialize()

                - self.stage1(), self.stage2() etc. depending on the number of
                  stages available.

                - self.compute_accelerations(index=0, update_nnps=True)
                - self.do_post_stage(stage_dt, stage_count_from_1)
                - self.update_domain()

        Please see any of the concrete implementations of the Integrator class
        to study.  By default the Integrator implements a
        predict-evaluate-correct method, the same as PECIntegrator.

        """
        self.initialize()

        # Predict
        self.stage1()
        self.update_domain()

        # Call any post-stage functions.
        self.do_post_stage(0.5*dt, 1)

        self.compute_accelerations()

        # Correct
        self.stage2()
        self.update_domain()

        # Call any post-stage functions.
        self.do_post_stage(dt, 2)

    def set_compiled_object(self, c_integrator):
        """Set the high-performance compiled object to call internally.
        """
        self.c_integrator = c_integrator

    def set_parallel_manager(self, pm):
        self.parallel_manager = pm
        self.c_integrator.set_parallel_manager(pm)

    def set_post_stage_callback(self, callback):
        """This callback is called when the particles are moved, i.e
        one stage of the integration is done.

        This callback is passed the current time value, the timestep and the
        stage.

        The current time value is  t + stage_dt, for example this would be
        0.5*dt for a two stage predictor corrector integrator.

        """
        self.c_integrator.set_post_stage_callback(callback)

    def step(self, time, dt):
        """This function is called by the solver.

        To implement the integration step please override the
        ``one_timestep`` method.
        """
        self.c_integrator.step(time, dt)

    def compute_accelerations(self, index=0, update_nnps=True):
        if update_nnps:
            # update NNPS since particles have moved
            if self.parallel_manager:
                self.parallel_manager.update()
            self.nnps.update()

        # Evaluate
        c_integrator = self.c_integrator
        a_eval = self.acceleration_evals[index]
        a_eval.compute(c_integrator.t, c_integrator.dt)

    def initial_acceleration(self, t, dt):
        """Compute the initial accelerations if needed before the iterations start.

        The default implementation only does this for the first acceleration
        evaluator. So if you have multiple evaluators, you must override this
        method in a subclass.

        """
        self.acceleration_evals[0].compute(t, dt)

    def update_domain(self):
        """Update the domain of the simulation.

        This is to be called when particles move so the ghost particles
        (periodicity, mirror boundary conditions) can be reset. Further, this
        also recalculates the appropriate cell size based on the particle
        kernel radius, `h`. This should be called explicitly when desired but
        usually this is done when the particles are moved or the `h` is
        changed.

        The integrator should explicitly call this when needed in the
        `one_timestep` method.
        """
        self.nnps.update_domain()


###############################################################################
# `EulerIntegrator` class
###############################################################################
class EulerIntegrator(Integrator):
    def one_timestep(self, t, dt):
        self.compute_accelerations()
        self.stage1()
        self.update_domain()
        self.do_post_stage(dt, 1)


###############################################################################
# `PECIntegrator` class
###############################################################################
class PECIntegrator(Integrator):
    r"""
    In the Predict-Evaluate-Correct (PEC) mode, the system is advanced using:

    .. math::

        y^{n+\frac{1}{2}} = y^n + \frac{\Delta t}{2}F(y^{n-\frac{1}{2}})
        --> Predict

        F(y^{n+\frac{1}{2}}) --> Evaluate

        y^{n + 1} = y^n + \Delta t F(y^{n+\frac{1}{2}})

    """
    def one_timestep(self, t, dt):
        self.initialize()

        # Predict
        self.stage1()
        self.update_domain()

        # Call any post-stage functions.
        self.do_post_stage(0.5*dt, 1)

        self.compute_accelerations()

        # Correct
        self.stage2()
        self.update_domain()

        # Call any post-stage functions.
        self.do_post_stage(dt, 2)


###############################################################################
# `EPECIntegrator` class
###############################################################################
class EPECIntegrator(Integrator):
    r"""
    Predictor corrector integrators can have two modes of
    operation.

    In the Evaluate-Predict-Evaluate-Correct (EPEC) mode, the
    system is advanced using:

    .. math::

        F(y^n) --> Evaluate

        y^{n+\frac{1}{2}} = y^n + F(y^n) --> Predict

        F(y^{n+\frac{1}{2}}) --> Evaluate

        y^{n+1} = y^n + \Delta t F(y^{n+\frac{1}{2}}) --> Correct

    Notes:

    The Evaluate stage of the integrator forces a function
    evaluation. Therefore, the PEC mode is much faster but relies on
    old accelertions for the Prediction stage.

    In the EPEC mode, the final corrector can be modified to:

    :math:`y^{n+1} = y^n + \frac{\Delta t}{2}\left( F(y^n) +
                                F(y^{n+\frac{1}{2}}) \right)`

    This would require additional storage for the accelerations.

    """
    def one_timestep(self, t, dt):
        self.initialize()

        self.compute_accelerations()

        # Predict
        self.stage1()
        self.update_domain()

        # Call any post-stage functions.
        self.do_post_stage(0.5*dt, 1)

        self.compute_accelerations()

        # Correct
        self.stage2()
        self.update_domain()

        # Call any post-stage functions.
        self.do_post_stage(dt, 2)


###############################################################################
# `TVDRK3Integrator` class
###############################################################################
class TVDRK3Integrator(Integrator):
    r"""
    In the TVD-RK3 integrator, the system is advanced using:

    .. math::

        y^{n + \frac{1}{3}} = y^n + \Delta t F( y^n )

        y^{n + \frac{2}{3}} = \frac{3}{4}y^n +
        \frac{1}{4}(y^{n + \frac{1}{3}} + \Delta t F(y^{n + \frac{1}{3}}))

        y^{n + 1} = \frac{1}{3}y^n + \frac{2}{3}(y^{n + \frac{2}{3}}
        + \Delta t F(y^{n + \frac{2}{3}}))

    """
    def one_timestep(self, t, dt):
        self.initialize()

        # stage 1
        self.compute_accelerations()
        self.stage1()
        self.update_domain()
        self.do_post_stage(1./3*dt, 1)

        # stage 2
        self.compute_accelerations()
        self.stage2()
        self.update_domain()
        self.do_post_stage(2./3*dt, 2)

        # stage 3 and end
        self.compute_accelerations()
        self.stage3()
        self.update_domain()
        self.do_post_stage(dt, 3)


###############################################################################
class LeapFrogIntegrator(PECIntegrator):
    r"""A leap-frog integrator.
    """

    def one_timestep(self, t, dt):

        self.stage1()
        self.update_domain()
        self.do_post_stage(0.5*dt, 1)

        self.compute_accelerations()
        self.stage2()
        self.update_domain()
        self.do_post_stage(dt, 2)


###############################################################################
class PEFRLIntegrator(Integrator):
    r"""A Position-Extended Forest-Ruth-Like integrator [Omeylan2002]_

    References
    ----------
    .. [Omeylan2002] I.M. Omelyan, I.M. Mryglod and R. Folk, "Optimized
       Forest-Ruth- and Suzuki-like algorithms for integration of motion
       in many-body systems", Computer Physics Communications 146, 188 (2002)
       http://arxiv.org/abs/cond-mat/0110585

    """

    def one_timestep(self, t, dt):

        self.stage1()
        self.update_domain()
        self.do_post_stage(0.1786178958448091*dt, 1)

        self.compute_accelerations()
        self.stage2()
        self.update_domain()
        self.do_post_stage(0.1123533131749906*dt, 2)

        self.compute_accelerations()
        self.stage3()
        self.update_domain()
        self.do_post_stage(0.8876466868250094*dt, 3)

        self.compute_accelerations()
        self.stage4()
        self.update_domain()
        self.do_post_stage(0.8213821041551909*dt, 4)

        self.compute_accelerations()
        self.stage5()
        self.update_domain()
        self.do_post_stage(dt, 5)
