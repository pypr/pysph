# Automatically generated, do not edit.
#cython: cdivision=True
<%def name="indent(text, level=0)" buffered="True">
% for l in text.splitlines():
${' '*4*level}${l}
% endfor
</%def>

from libc.math cimport *

from cython import address
from pysph.base.nnps_base cimport NNPS


${helper.get_helper_code()}

${helper.get_stepper_code()}


# #############################################################################
cdef class Integrator:
    cdef public ParticleArrayWrapper ${helper.get_particle_array_names()}
    cdef public AccelerationEval acceleration_eval
    cdef public object integrator
    cdef public double dt, t, orig_t
    cdef object _post_stage_callback
    cdef object steppers

    ${indent(helper.get_stepper_defs(), 1)}

    def __init__(self, integrator, acceleration_eval, steppers):
        self.integrator = integrator
        self.acceleration_eval = acceleration_eval
        self.steppers = steppers
        self._post_stage_callback = None
        % for name in sorted(helper.object.steppers.keys()):
        self.${name} = acceleration_eval.${name}
        % endfor
        ${indent(helper.get_stepper_init(), 2)}

    def set_nnps(self, NNPS nnps):
        pass

    def set_parallel_manager(self, object pm):
        pass

    def set_post_stage_callback(self, object callback):
        self._post_stage_callback = callback

    cpdef compute_accelerations(self, int index=0, update_nnps=True):
        self.integrator.compute_accelerations(index, update_nnps)

    cpdef update_domain(self):
        self.integrator.update_domain()

    cpdef do_post_stage(self, double stage_dt, int stage):
        """This is called after every stage of the integrator.

        Internally, this calls any post_stage_callback function that has
        been given to take suitable action.

        Parameters
        ----------

         - stage_dt : double: the timestep taken at this stage.

         - stage : int: the stage completed (starting from 1).
        """
        self.t = self.orig_t + stage_dt
        if self._post_stage_callback is not None:
            self._post_stage_callback(self.t, self.dt, stage)

    cpdef step(self, double t, double dt):
        """Main step routine.
        """
        self.orig_t = t
        self.t = t
        self.dt = dt
        self.one_timestep(t, dt)

    cdef one_timestep(self, double t, double dt):
        ${indent(helper.get_timestep_code(), 2)}

    % for method in helper.get_stepper_method_wrapper_names():
    cdef ${method}(self):
        cdef long NP_DEST
        cdef long d_idx
        cdef ParticleArrayWrapper dst
        cdef double dt = self.dt
        cdef double t = self.t
        ${indent(helper.get_array_declarations(method), 2)}

        % for dest in sorted(helper.object.steppers.keys()):
        # ---------------------------------------------------------------------
        # Destination ${dest}.
        dst = self.${dest}
        ##
        ${indent(helper.get_py_stage_code(dest, method), 2)}
        ##
        % if helper.has_stepper_loop(dest, method):
        # Only iterate over real particles.
        NP_DEST = dst.size(real=True)
        ${indent(helper.get_array_setup(dest, method), 2)}
        for d_idx in range(NP_DEST):
            ${indent(helper.get_stepper_loop(dest, method), 3)}
        % endif
        % endfor
    % endfor
