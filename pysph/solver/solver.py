""" An implementation of a general solver base class """

# System library imports.
import os
import numpy

# PySPH imports
from pysph.base.kernels import CubicSpline
from pysph.sph.sph_eval import SPHEval

from utils import FloatPBar, savez, load

import logging
logger = logging.getLogger(__name__)

class Solver(object):
    """ Base class for all PySPH Solvers

    **Attributes**

    - particles -- the particle arrays to operate on

    - integrator_type -- the class of the integrator. This may be one of any
      defined in solver/integrator.py

    - kernel -- the kernel to be used throughout the calculations. This may
      need to be modified to handle several kernels.

    - t -- the internal time step counter

    - pre_step_functions -- a list of functions to be performed before stepping

    - post_step_functions -- a list of functions to execute after stepping

    - pfreq -- the output print frequency

    - dim -- the dimension of the problem

    - pid -- the processor id if running in parallel

    - cell_iteration :bool: -- should we use cell or particle iteration.

    """

    def __init__(self, dim=2, integrator=None, kernel=None,
                 tdamp=0.001, tf=1.0, dt=1e-3,
                 adaptive_timestep=False, cfl=0.3,
                 output_at_times = [],
                 fixed_h=False, **kwargs):
        """Constructor

        Any additional keyword args are used to set the values of any
        of the attributes.

        Parameters
        -----------

        dim : int
            Problem dimensionality

        integrator_type : integrator.Integrator
            The integrator to use

        kernel : base.kernels.Kernel
            SPH kernel to use

        tdamp : double
            Time upto which damping of the initial solution is required

        tf, dt : double
            Final time and suggested initial time-step

        adaptive_timestep : bint
            Flag to use adaptive time-steps

        cfl : double
            CFL number for adaptive time stepping

        output_at_times : list
            Optional list of output times to force output

        fixed_h : bint
            Flag for constant smoothing lengths

        """

        self.integrator = integrator
        self.dim = dim
        if kernel is not None:
            self.kernel = kernel
        else:
            self.kernel = CubicSpline(dim)

        # set the particles to None
        self.particles = None

        # Set the SPHEval instance to None.
        self.sph_eval = None

        # solver time and iteration count
        self.t = 0
        self.count = 0

        self.execute_commands = None

        # list of functions to be called before and after an integration step
        self.pre_step_functions = []
        self.post_step_functions = []

        # List of functions to be called after each stage of the integrator.
        self.post_stage_callbacks = []

        # default output printing frequency
        self.pfreq = 100

        self.disable_output = False

        # the process id for parallel runs
        self.pid = None

        # set the default rank to 0
        self.rank = 0

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
        self.tdamp = tdamp

        # Use adaptive time steps and cfl number
        self.adaptive_timestep = adaptive_timestep
        self.cfl = cfl

        # list of output times
        self.output_at_times = output_at_times
        self.force_output = False

        # Use cell iterations or not.
        self.cell_iteration = False

        # Set all extra keyword arguments
        for attr, value in kwargs.iteritems():
            if hasattr(self, attr):
                setattr(self, attr, value)
            else:
                msg = 'Unknown keyword arg "%s" passed to constructor'%attr
                raise TypeError(msg)

        # default time step constants
        self.tf = tf
        self.dt = dt

        # flag for constant smoothing lengths
        self.fixed_h = fixed_h

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

        self.sph_eval = SPHEval(particles, equations, self.kernel,
                                self.integrator,
                                cell_iteration=self.cell_iteration)
        self.sph_eval.set_nnps(nnps)

        # set the parallel manager for the integrator
        self.integrator.set_parallel_manager(self.pm)

        # Set the post_stage_callback.
        self.integrator.set_post_stage_callback(self._post_stage_callback)

        # set integrator option for constant smoothing length
        self.fixed_h = fixed_h
        self.integrator.set_fixed_h( fixed_h )

        logger.debug("Solver setup complete.")

    def add_post_stage_callback(self, callback):
        """These callbacks are called after each integrator stage.

        The callbacks are passed (current_time, dt, stage).  See the the
        `Integrator.one_timestep` methods for examples of how this is called.
        """
        self.post_stage_callbacks.append(callback)

    def append_particle_arrrays(self, arrays):
        """ Append the particle arrays to the existing particle arrays
        """
        if not self.particles:
            print 'Warning! Particles not defined.'
            return

        for array in self.particles:
            array_name = array.name
            for arr in arrays:
                if array_name == arr.name:
                    array.append_parray(arr)

        self.setup(self.particles)

    def set_adaptive_timestep(self, value):
        """Set if we should use adaptive timesteps or not.
        """
        self.adaptive_timestep = value

    def set_cfl(self, value):
        'Set the CFL number for adaptive time stepping'
        self.cfl = value

    def set_cell_iteration(self, value):
        """Set if we should use cell_iteration or not.
        """
        self.cell_iteration = value

    def set_final_time(self, tf):
        """ Set the final time for the simulation """
        self.tf = tf

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
        self.output_at_times = output_at_times

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
        dt = self.dt

        if self.in_parallel:
            show = False
        else:
            show = show_progress
        bar = FloatPBar(self.t, self.tf, show=show)

        self.dump_output()
        self.barrier() # everybody waits for this to complete

        # initial solution damping time
        tdamp = self.tdamp

        # Compute the accelerations once for the predictor corrector
        # integrator to work correctly at the first time step.
        self.sph_eval.compute(self.t, dt)

        # solution output times
        output_at_times = numpy.array( self.output_at_times )

        while self.t < self.tf:

            # perform any pre step functions
            for func in self.pre_step_functions:
                func.eval(self)

            if self.rank == 0:
                logger.debug(
                    "Iteration=%d, time=%f, timestep=%f" % \
                        (self.count, self.t, dt)
                )
            # perform the integration and update the time.
            #print 'Solver Iteration', self.count, dt, self.t, tdamp
            self.integrator.step(self.t, dt)

            # perform any post step functions
            for func in self.post_step_functions:
                func.eval(self)

            # update time and iteration counters if successfully
            # integrated
            self.t += dt
            self.count += 1

            # dump output if the iteration number is a multiple of the
            # printing frequency
            if self.count % self.pfreq == 0:
                self.dump_output()
                self.barrier()

            # dump output if forced
            if self.force_output:
                self.dump_output()
                self.barrier()

                self.force_output = False

                if self.rank == 0:
                    msg = 'Writing output at time %g, iteration %d, dt = %g'%(
                        self.t, self.count, self.dt)
                    logger.info(msg)

            # update progress bar
            bar.update(self.t)

            # update the time for all arrays
            self.update_particle_time()

            # compute the new time step across all processors
            if self.adaptive_timestep:
                # locally stable time step
                dt = self.integrator.compute_time_step(
                    self.dt, self.cfl)

                # damp the initial solution
                if self.t < tdamp:
                    dt *= 0.5 * (numpy.sin(numpy.pi*(-0.5+self.t/tdamp)) + 1.0)

                # set the globally stable time step
                if self.in_parallel:
                    dt = self.pm.update_time_steps(dt)

            # adjust dt to land on final time
            if self.t + dt > self.tf:
                dt = self.tf - self.t
                self.dt = dt

            # adjust dt to land on specified output time
            tdiff = output_at_times - self.t
            condition = (tdiff > 0) & (tdiff < dt)
            if numpy.any( condition ):
                output_time = output_at_times[ numpy.where(condition) ]
                dt = output_time - self.t
                self.dt = dt

                self.force_output = True

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

        fname = self.fname + '_'
        output_data = {"arrays":{}, "solver_data":{}}

        _fname = os.path.join(self.output_directory,
                              fname  + str(self.count) +'.npz')

        # Array data
        for array in self.particles:
            output_data["arrays"][array.name] = array.get_property_arrays(
                all=self.detailed_output, only_real=self.output_only_real)

        # Add the solver data
        output_data["solver_data"]["dt"] = self.dt
        output_data["solver_data"]["t"] = self.t
        output_data["solver_data"]["count"] = self.count

        # Gather particle data on root
        if self.parallel_output_mode == "collected" and self.in_parallel:
            comm = self.comm

            arrays = output_data["arrays"]
            array_names = arrays.keys()

            # gather the data from all processors
            collected_data = comm.gather(arrays, root=0)

            if self.rank == 0:
                output_data["arrays"] = {}
                size = comm.Get_size()

                # concatenate the arrays
                for array_name in array_names:
                    output_data["arrays"][array_name] = {}

                    _props = collected_data[0][array_name].keys()
                    for prop in _props:
                        data = [collected_data[pid][array_name][prop]
                                        for pid in range(size)]
                        prop_arr = numpy.concatenate(data)
                        output_data["arrays"][array_name][prop] = prop_arr

                savez(_fname, version=1, **output_data)

        else:
            savez(_fname, version=1, **output_data)

    def load_output(self, count):
        """ Load particle data from dumped output file.

        Parameters
        ----------
        count : string
            The iteration time from which to load the data. If time is
            '?' then list of available data files is returned else
             the latest available data file is used

        Notes
        -----
        Data is loaded from the :py:attr:`output_directory` using the same format
        as stored by the :py:meth:`dump_output` method.
        Proper functioning required that all the relevant properties of arrays be
        dumped

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

        self.t = float(data["solver_data"]['t'])
        self.count = int(data["solver_data"]['count'])

    def get_options(self, opt_parser):
        """ Implement this to add additional options for the application """
        pass

    def setup_solver(self, options=None):
        """ Implement the basic solvers here

        All subclasses of Solver may implement this function to add the
        necessary operations for the problem at hand.

        Look at solver/fluid_solver.py for an example.

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
    def _post_stage_callback(self, time, dt, stage):
        for callback in self.post_stage_callbacks:
            callback(time, dt, stage)

############################################################################
