""" An implementation of a general solver base class """

# System library imports.
import os
import numpy

# PySPH imports
from pysph.base.kernels import CubicSpline
from pysph.sph.acceleration_eval import AccelerationEval
from pysph.sph.sph_compiler import SPHCompiler

from pysph.solver.utils import FloatPBar, load, dump

import logging
logger = logging.getLogger(__name__)

EPSILON = numpy.finfo(float).eps*2

class Solver(object):
    """Base class for all PySPH Solvers
    """
    def __init__(self, dim=2, integrator=None, kernel=None,
                 n_damp=0, tf=1.0, dt=1e-3,
                 adaptive_timestep=False, cfl=0.3,
                 output_at_times=(),
                 fixed_h=False, **kwargs):
        """**Constructor**

        Any additional keyword args are used to set the values of any
        of the attributes.

        Parameters
        ----------

        dim : int
            Dimension of the problem

        integrator : pysph.sph.integrator.Integrator
            Integrator to use

        kernel : pysph.base.kernels.Kernel
            SPH kernel to use

        n_damp : int
            Number of timesteps for which the initial damping is required.
            This is used to improve stability for problems with strong
            discontinuity in initial condition.
            Setting it to zero will disable damping of the timesteps.

        dt : double
            Suggested initial time step for integration

        tf : double
            Final time for integration

        adaptive_timestep : bint
            Flag to use adaptive time steps

        cfl : double
            CFL number for adaptive time stepping

        pfreq : int
            Output files dumping frequency.

        output_at_times : list/array
            Optional list of output times to force dump the output file

        fixed_h : bint
            Flag for constant smoothing lengths `h`

        Example
        -------

        >>> integrator = PECIntegrator(fluid=WCSPHStep())
        >>> kernel = CubicSpline(dim=2)
        >>> solver = Solver(dim=2, integrator=integrator, kernel=kernel,
        ...                 n_damp=50, tf=1.0, dt=1e-3, adaptive_timestep=True,
        ...                 pfreq=100, cfl=0.5, output_at_times=[1e-1, 1.0])

        """

        self.integrator = integrator
        self.dim = dim
        if kernel is not None:
            self.kernel = kernel
        else:
            self.kernel = CubicSpline(dim)

        # set the particles to None
        self.particles = None

        # Set the AccelerationEval instance to None.
        self.acceleration_eval = None

        # solver time and iteration count
        self.t = 0
        self.count = 0

        self.execute_commands = None

        # list of functions to be called before and after an integration step
        self.pre_step_callbacks = []
        self.post_step_callbacks = []

        # List of functions to be called after each stage of the integrator.
        self.post_stage_callbacks = []

        # default output printing frequency
        self.pfreq = 100

        # Compress generated files.
        self.compress_output = False
        self.disable_output = False

        # the process id for parallel runs
        self.pid = None

        # set the default rank to 0
        self.rank = 0

        # set the default comm to None.
        self.comm = None

        # set the default mode to serial
        self.in_parallel = False

        # arrays to print output
        self.arrays_to_print = []

        # the default parallel output mode
        self.parallel_output_mode = "collected"

        # flag to print all arrays
        self.detailed_output = False

        # flag to save Remote arrays
        self.output_only_real = True

        # output filename
        self.fname = self.__class__.__name__

        # output drectory
        self.output_directory = self.fname+'_output'

        # solution damping to avoid impulsive starts
        self.n_damp = n_damp

        # Use adaptive time steps and cfl number
        self.adaptive_timestep = adaptive_timestep
        self.cfl = cfl

        # list of output times
        self.output_at_times = numpy.asarray(output_at_times)
        self.force_output = False

        # default time step constants
        self.tf = tf
        self.dt = dt
        self.max_steps = 1 << 31
        self._prev_dt = None
        self._damping_factor = 1.0
        self._epsilon = EPSILON*tf

        # flag for constant smoothing lengths
        self.fixed_h = fixed_h

        # Set all extra keyword arguments
        for attr, value in kwargs.items():
            if hasattr(self, attr):
                setattr(self, attr, value)
            else:
                msg = 'Unknown keyword arg "%s" passed to constructor'%attr
                raise TypeError(msg)


    ##########################################################################
    # Public interface.
    ##########################################################################
    def setup(self, particles, equations, nnps, kernel=None, fixed_h=False):
        """ Setup the solver.

        The solver's processor id is set if the in_parallel flag is set
        to true.

        The order of the integrating calcs is determined by the solver's
        order attribute.

        This is usually called at the start of a PySPH simulation.

        """

        self.particles = particles
        if kernel is not None:
            self.kernel = kernel

        mode = 'mpi' if self.in_parallel else 'serial'
        self.acceleration_eval = AccelerationEval(
            particles, equations, self.kernel, mode
        )

        sep = '-'*70
        eqn_info = '[\n' + ',\n'.join([str(e) for e in equations]) + '\n]'
        logger.info('Using equations:\n%s\n%s\n%s'%(sep, eqn_info, sep))
        logger.info(
            'Using integrator:\n%s\n  %s\n%s'%(sep, self.integrator, sep)
        )

        sph_compiler = SPHCompiler(
            self.acceleration_eval, self.integrator
        )
        sph_compiler.compile()

        # Set the nnps for all concerned objects.
        self.acceleration_eval.set_nnps(nnps)
        self.integrator.set_nnps(nnps)

        # set the parallel manager for the integrator
        self.integrator.set_parallel_manager(self.pm)

        # Set the post_stage_callback.
        self.integrator.set_post_stage_callback(self._post_stage_callback)

        # set integrator option for constant smoothing length
        self.fixed_h = fixed_h
        self.integrator.set_fixed_h( fixed_h )

        logger.debug("Solver setup complete.")

    def add_post_stage_callback(self, callback):
        """These callbacks are called *after* each integrator stage.

        The callbacks are passed (current_time, dt, stage).  See the the
        `Integrator.one_timestep` methods for examples of how this is called.

        Example
        -------

        >>> def post_stage_callback_function(t, dt, stage):
        >>>     # This function is called after every stage of integrator.
        >>>     print t, dt, stage
        >>>     # Do something
        >>> solver.add_post_stage_callback(post_stage_callback_function)
        """
        self.post_stage_callbacks.append(callback)

    def add_post_step_callback(self, callback):
        """These callbacks are called *after* each timestep is performed.

        The callbacks are passed the solver instance (i.e. self).

        Example
        -------

        >>> def post_step_callback_function(solver):
        >>>     # This function is called after every time step.
        >>>     print solver.t, solver.dt
        >>>     # Do something
        >>> solver.add_post_step_callback(post_step_callback_function)
        """
        self.post_step_callbacks.append(callback)

    def add_pre_step_callback(self, callback):
        """These callbacks are called *before* each timestep is performed.

        The callbacks are passed the solver instance (i.e. self).

        Example
        -------

        >>> def pre_step_callback_function(solver):
        >>>     # This function is called before every time step.
        >>>     print solver.t, solver.dt
        >>>     # Do something
        >>> solver.add_pre_step_callback(pre_step_callback_function)
        """
        self.pre_step_callbacks.append(callback)

    def append_particle_arrrays(self, arrays):
        """ Append the particle arrays to the existing particle arrays
        """
        if not self.particles:
            print('Warning! Particles not defined.')
            return

        for array in self.particles:
            array_name = array.name
            for arr in arrays:
                if array_name == arr.name:
                    array.append_parray(arr)

        self.setup(self.particles)

    def set_adaptive_timestep(self, value):
        """Set it to True to use adaptive timestepping based on
        cfl, viscous and force factor.

        Look at pysph.sph.integrator.compute_time_step for more details.
        """
        self.adaptive_timestep = value

    def set_cfl(self, value):
        'Set the CFL number for adaptive time stepping'
        self.cfl = value

    def set_final_time(self, tf):
        """ Set the final time for the simulation """
        self.tf = tf
        self._epsilon = EPSILON*tf

    def set_n_damp(self, ndamp):
        """Set the number of timesteps for which the timestep should be
        initially damped.
        """
        self.n_damp = ndamp

    def set_time_step(self, dt):
        """ Set the time step to use """
        self.dt = dt

    def set_print_freq(self, n):
        """ Set the output print frequency """
        self.pfreq = n

    def set_disable_output(self, value):
        """Disable file output.
        """
        self.disable_output = value

    def set_arrays_to_print(self, array_names=None):
        """Only print the arrays with the given names.
        """

        available_arrays = [array.name for array in self.particles]

        if array_names:
            for name in array_names:
                if not name in available_arrays:
                    raise RuntimeError("Array %s not availabe"%(name))

                for arr in self.particles:
                    if arr.name == name:
                        array = arr
                        break
                self.arrays_to_print.append(array)
        else:
            self.arrays_to_print = self.particles

    def set_output_fname(self, fname):
        """ Set the output file name """
        self.fname = fname

    def set_output_printing_level(self, detailed_output):
        """ Set the output printing level """
        self.detailed_output = detailed_output

    def set_output_only_real(self, output_only_real):
        """ Set the flag to save out only real particles """
        self.output_only_real = output_only_real

    def set_output_directory(self, path):
        """ Set the output directory """
        self.output_directory = path

    def set_output_at_times(self, output_at_times):
        """ Set a list of output times """
        self.output_at_times = numpy.asarray(output_at_times)

    def set_max_steps(self, max_steps):
        """Set the maximum number of iterations to perform.
        """
        self.max_steps = max_steps

    def set_compress_output(self, compress):
        """Compress the dumped output files.
        """
        self.compress_output = compress

    def set_parallel_output_mode(self, mode="collected"):
        """Set the default solver dump mode in parallel.

        The available modes are:

        collected : Collect array data from all processors on root and
                    dump a single file.


        distributed : Each processor dumps a file locally.

        """
        assert mode in ("collected", "distributed")
        self.parallel_output_mode = mode

    def set_command_handler(self, callable, command_interval=1):
        """ set the `callable` to be called at every `command_interval` iteration

        the `callable` is called with the solver instance as an argument
        """
        self.execute_commands = callable
        self.command_interval = command_interval

    def set_parallel_manager(self, pm):
        self.pm = pm

    def barrier(self):
        if self.comm:
            self.comm.barrier()

    def solve(self, show_progress=True):
        """ Solve the system

        Notes
        -----
        Pre-stepping functions are those that need to be called before
        the integrator is called.

        Similarly, post step functions are those that are called after
        the stepping within the integrator.

        """
        if self.in_parallel:
            show = False
        else:
            show = show_progress
        bar = FloatPBar(self.t, self.tf, show=show)
        self._epsilon = EPSILON*self.tf

        # Initial solution
        self.dump_output()
        self.barrier() # everybody waits for this to complete

        # Compute the accelerations once for the predictor corrector
        # integrator to work correctly at the first time step.
        self.acceleration_eval.compute(self.t, self.dt)

        # Now get a suitable adaptive (if requested) and damped timestep to
        # integrate with.
        self.dt = self._get_timestep()

        while (self.tf - self.t) > self._epsilon and \
              (self.count < self.max_steps):

            # perform any pre step functions
            for callback in self.pre_step_callbacks:
                callback(self)

            if self.rank == 0:
                logger.debug(
                    "Iteration=%d, time=%f, timestep=%f" % \
                        (self.count, self.t, self.dt)
                )
            # perform the integration and update the time.
            #print 'Solver Iteration', self.count, self.dt, self.t
            self.integrator.step(self.t, self.dt)

            # perform any post step functions
            for callback in self.post_step_callbacks:
                callback(self)

            # update time and iteration counters if successfully
            # integrated
            self.t += self.dt
            self.count += 1
            self._epsilon = EPSILON*self.tf*self.count

            # Compute the next timestep.
            self.dt = self._get_timestep()

            # Note: this may adjust dt to land at a desired time.
            self._dump_output_if_needed()

            # update progress bar
            bar.update(self.t)

            # update the time for all arrays
            self.update_particle_time()

            if self.execute_commands is not None:
                if self.count % self.command_interval == 0:
                    self.execute_commands(self)

        # close the progress bar
        bar.finish()

        # final output save
        self.dump_output()

    def update_particle_time(self):
        for array in self.particles:
            array.set_time(self.t)

    def dump_output(self):
        """Dump the simulation results to file

        The arrays used for printing are determined by the particle
        array's `output_property_arrays` data attribute. For debugging
        it is sometimes nice to have all the arrays (including
        accelerations) saved. This can be chosen from using the
        command line option `--detailed-output`

        Output data Format:

        A single file named as: <fname>_<rank>_<iteration_count>.npz

        The data is saved as a Python dictionary with two keys:

        `solver_data` : Solver meta data like time, dt and iteration number

        `arrays` : A dictionary keyed on particle array names and with
                   particle properties as value.

         Example:

         You can load the data output by PySPH like so:

         >>> from pysph.solver.utils import load
         >>> data = load('output_directory/filename_x_xxx.npz')
         >>> solver_data = data['solver_data']
         >>> arrays = data['arrays']
         >>> fluid = arrays['fluid']
         >>> ...

         In the above example, it is assumed that the output file
         contained an array named fluid.

        """
        if self.disable_output:
            return

        if self.rank == 0:
            msg = 'Writing output at time %g, iteration %d, dt = %g'%(
                self.t, self.count, self.dt)
            logger.info(msg)

        fname = os.path.join(self.output_directory,
                             self.fname  + '_' + str(self.count))

        comm = None
        if self.parallel_output_mode == "collected" and self.in_parallel:
            comm = self.comm

        dump(fname, self.particles, self._get_solver_data(),
             detailed_output=self.detailed_output,
             only_real=self.output_only_real, mpi_comm=comm,
             compress=self.compress_output)

    def load_output(self, count):
        """Load particle data from dumped output file.

        Parameters
        ----------
        count : str
            The iteration time from which to load the data. If time is '?' then
            list of available data files is returned else the latest available
            data file is used

        Notes
        -----
        Data is loaded from the :py:attr:`output_directory` using the same format
        as stored by the :py:meth:`dump_output` method.
        Proper functioning required that all the relevant properties of arrays be
        dumped.

        """
        # get the list of available files
        available_files = [i.rsplit('_',1)[1][:-4]
                            for i in os.listdir(self.output_directory)
                             if i.startswith(self.fname) and i.endswith('.npz')]

        if count == '?':
            return sorted(set(available_files), key=int)

        else:
            if not count in available_files:
                msg = "File with iteration count `%s` does not exist"%(count)
                msg += "\nValid iteration counts are %s"%(sorted(set(available_files), key=int))
                #print msg
                raise IOError(msg)

        array_names = [pa.name for pa in self.particles]

        # load the output file
        data = load(os.path.join(self.output_directory,
                                 self.fname+'_'+str(count)+'.npz'))

        arrays = [ data["arrays"][i] for i in array_names ]

        # set the Particle's arrays
        self.particles = arrays

        solver_data = data['solver_data']

        self.t = float(solver_data['t'])
        self.dt = float(solver_data['dt'])
        self.count = int(solver_data['count'])

    def get_options(self, arg_parser):
        """ Implement this to add additional options for the application """
        pass

    def setup_solver(self, options=None):
        """ Implement the basic solvers here

        All subclasses of Solver may implement this function to add the
        necessary operations for the problem at hand.

        Parameters
        ----------
        options : dict
            options set by the user using commandline (there is no guarantee
            of existence of any key)
        """
        pass

    ##########################################################################
    # Non-public interface.
    ##########################################################################
    def _compute_timestep(self):
        undamped_dt = self._get_undamped_timestep()
        if self.adaptive_timestep:
            # locally stable time step
            dt = self.integrator.compute_time_step(undamped_dt, self.cfl)

            # set the globally stable time step across all processors
            if self.in_parallel:
                if dt is None:
                    # For some reason this processor does not have an adaptive
                    # timestep constraint so we set it to a large number so the
                    # timestep is determined by the other processors.
                    dt = 1e20
                dt = self.pm.update_time_steps(dt)
            else:
                if dt is None:
                    dt = undamped_dt
        else:
            dt = undamped_dt

        return dt

    def _damp_timestep(self, dt):
        """Damp the timestep initially to prevent transient errors at startup.

        This basically damps the initial timesteps by the factor

        0.5  (sin(pi*(-0.5 + count/n_damp)) + 1)

        Where n_damp is the number of iterations to damp the timestep for and
        count is the number of iterations.

        """
        n_damp = self.n_damp
        if self.count < n_damp and n_damp > 0:
            iter_fraction = (self.count+1)/float(n_damp)
            fac = 0.5*(numpy.sin(numpy.pi*(-0.5 + iter_fraction)) + 1.0)
            self._damping_factor = fac
        else:
            self._damping_factor = 1.0

        return dt*self._damping_factor

    def _dump_output_if_needed(self):
        """Dump output if needed while solve is running.

        This is called by `solve`.

        Warning
        -------

        This will adjust `dt` if the user has asked for output at a
        non-integral multiple of dt.
        """
        if abs(self.t - self.tf) < self._epsilon:
            return

        # dump output if the iteration number is a multiple of the printing
        # frequency.
        dump = self.count % self.pfreq == 0

        # Consider the other cases if user has requested output at a specified
        # time.

        output_at_times = self.output_at_times
        dt = self.dt

        # adjust dt to land on specific output times or dump output if we have
        # reached a desired time.
        if len(output_at_times) > 0:
            tdiff = output_at_times - self.t

            if numpy.any(numpy.abs(tdiff) < self._epsilon):
                dump = True

            # Our next step may exceed a required timestep so we adjust the
            # timestep.
            timestep_too_big = (tdiff > 0.0) & (tdiff < dt)
            if numpy.any(timestep_too_big):
                index = numpy.where(timestep_too_big)[0]
                output_time = output_at_times[index]
                if abs(output_time - self.t) > self._epsilon:
                    # It sometimes happens that the current time is just
                    # shy of the requested output time which results in a
                    # ridiculously small dt so we skip that case.

                    # Compute the new time-step to fall on the specified output
                    # time instant and save the previous dt value.
                    self._prev_dt = dt
                    self.dt = float(output_time - self.t)

        if dump:
            self.dump_output()
            self.barrier()

    def _get_solver_data(self):
        if self._prev_dt is not None:
            dt = self._prev_dt/self._damping_factor
        else:
            dt = self._get_undamped_timestep()

        return {'dt': dt, 't': self.t, 'count': self.count}

    def _get_timestep(self):
        if abs(self.tf - self.t) < self._epsilon:
            # We have reached the end, so no need to adjust the timestep
            # anymore.
            return self.dt

        if self._prev_dt is not None and \
           abs(self._prev_dt - self.dt) > self._epsilon:
            # if the _prev_dt was set then we need to use it as the current dt
            # was set to print at an intermediate time.
            self.dt = self._prev_dt
            self._prev_dt = None

        dt = self._compute_timestep()
        dt = self._damp_timestep(dt)

        # adjust dt to land exactly on final time
        if (self.t + dt) > (self.tf - self._epsilon):
            dt = self.tf - self.t

        return dt

    def _get_undamped_timestep(self):
        return self.dt/self._damping_factor

    def _post_stage_callback(self, time, dt, stage):
        for callback in self.post_stage_callbacks:
            callback(time, dt, stage)


############################################################################
