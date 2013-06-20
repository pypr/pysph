""" An implementation of a general solver base class """

# System library imports.
import os
import numpy

# PySPH imports
from pysph.base.kernels import CubicSpline
from pysph.sph.integrator import WCSPHRK2Integrator
from pysph.sph.sph_eval import SPHEval

from utils import PBar, savez, load

import logging
logger = logging.getLogger()

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

    """
    
    def __init__(self, integrator_type=WCSPHRK2Integrator, kernel=None, 
                 dim=2):
        """Constructor
        
        Parameters
        -----------
        
        integrator_type : The integrator to use.
        """

        self.integrator_type = integrator_type
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

        # default output printing frequency
        self.pfreq = 100

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

        # default particle properties to print
        self.print_properties = ['x','u','m','h','p','e','rho',]
        if self.dim > 1:
            self.print_properties.extend(['y','v'])

        if self.dim > 2:
            self.print_properties.extend(['z','w'])

        # flag to print all arrays 
        self.detailed_output = False

        # output filename
        self.fname = self.__class__.__name__

        # output drectory
        self.output_directory = self.fname+'_output'

    def setup(self, particles, equations, nnps, kernel=None, 
              integrator_type=None):
        """ Setup the solver.

        The solver's processor id is set if the in_parallel flag is set 
        to true.

        The order of the integrating calcs is determined by the solver's 
        order attribute.

        This is usually called at the start of a PySPH simulation.

        """
        
        self.particles = particles
        if integrator_type is not None:
            self.integrator_type = integrator_type
        if kernel is not None:
            self.kernel = kernel
            
        self.sph_eval = SPHEval(particles, equations, None, self.kernel)
        self.sph_eval.set_nnps(nnps)

        # instantiate the Integrator
        self.integrator = integrator = self.integrator_type(evaluator=self.sph_eval, 
                                                            particles=particles)
        # set the parallel manager for the integrator
        integrator.set_parallel_manager(self.pm)
        integrator.set_solver(self)

    def add_print_properties(self, props):
        """ Add a list of properties to print """
        for prop in props:
            if not prop in self.print_properties:
                self.print_properties.append(prop)            

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

    def set_final_time(self, tf):
        """ Set the final time for the simulation """
        self.tf = tf

    def set_time_step(self, dt):
        """ Set the time step to use """
        self.dt = dt

    def set_print_freq(self, n):
        """ Set the output print frequency """
        self.pfreq = n

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

    def set_output_directory(self, path):
        """ Set the output directory """
        self.output_directory = path

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

    def solve(self, show_progress=False):
        """ Solve the system

        Notes
        -----
        Pre-stepping functions are those that need to be called before
        the integrator is called. 

        Similarly, post step functions are those that are called after
        the stepping within the integrator.

        """
        dt = self.dt

        bt = (self.tf - self.t)/1000.0
        bcount = 0.0
        bar = PBar(1000 + int(dt/bt), show=show_progress)

        self.dump_output(dt, *self.print_properties)
        if self.comm:
            self.comm.barrier() # everybody waits for this to complete

        # the parallel manager
        pm = self.pm

        # set the time for the integrator
        #self.integrator.time = self.t

        while self.t < self.tf:
            self.t += dt
            self.count += 1

            # perform any pre step functions
            for func in self.pre_step_functions:
                func.eval(self)

            logger.info("Time %f, time step %f, rank  %d"%(self.t, dt,
                                                           self.rank))
            # perform the integration and update the time.
            self.integrator.integrate(self.t, dt, self.count)

            # update the time for all arrays
            self.update_particle_time()

            # perform any post step functions            
            for func in self.post_step_functions:
                func.eval(self)

            # dump output
            if self.count % self.pfreq == 0:
                self.dump_output(dt, *self.print_properties)
                if self.comm:
                    self.comm.barrier()                

            bcount += int(self.dt/bt)
            while bcount > 0:
                bar.update()
                bcount -= 1
        
            if self.execute_commands is not None:
                if self.count % self.command_interval == 0:
                    self.execute_commands(self)

        bar.finish()

    def update_particle_time(self):
        for array in self.particles:
            array.set_time(self.t)

    def dump_output(self, dt, *print_properties):
        """ Print output based on level of detail required
        
        The default detail level (low) is the integrator's calc's update 
        property for each named particle array.
        
        The higher detail level dumps all particle array properties.

        Format:
        -------

        A single file named as: <fname>_<rank>_<count>.npz

        The output file contains the following fields:

        solver_data : Solver related data like time step, time and
        iteration count. These are used to resume a simulation.

        arrays : A dictionary keyed on particle array names and with
        particle properties as value.

        version : The version number for this format of file
        output. The current version number is 1

        Example:
        --------

        data = load('foo.npz')

        version = data['version']

        dt = data['solver_data']['dt']
        t = data['solver_data']['t']
        
        array = data['arrays'][array_name].astype(object)
        array['x']

        """
        fname = self.fname + '_' 
        props = {"arrays":{}, "solver_data":{}}

        _fname = os.path.join(self.output_directory,
                              fname  + str(self.count) +'.npz')

        # save the cell partitions
        #if self.in_parallel:
        #    self.pm.save_partition(self.output_directory, self.count)

        if self.detailed_output:
            for array in self.particles:
                props["arrays"][array.name]=array.get_property_arrays(all=True)
        else:
            for array in self.particles:
                props["arrays"][array.name]=array.get_property_arrays(all=False)

        # Add the solver data
        props["solver_data"]["dt"] = dt
        props["solver_data"]["t"] = self.t
        props["solver_data"]["count"] = self.count

        if self.parallel_output_mode == "collected" and self.in_parallel:

            comm = self.comm
            
            arrays = props["arrays"]
            array_names = arrays.keys()

            # gather the data from all processors
            collected_data = comm.gather(arrays, root=0)
            
            if self.rank == 0:
                props["arrays"] = {}
                size = comm.Get_size()

                # concatenate the arrays
                for array_name in array_names:
                    props["arrays"][array_name] = {}

                    _props = collected_data[0][array_name].keys()
                    for prop in _props:
                        data = [collected_data[pid][array_name][prop] 
                                        for pid in range(size)]
                        prop_arr = numpy.concatenate(data)
                        props["arrays"][array_name][prop] = prop_arr

                savez(_fname, version=1, **props)

        else:
            savez(_fname, version=1, **props)

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

    def set_parallel_manager(self, pm):
        self.pm = pm

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

############################################################################
