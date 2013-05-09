# Standard imports.
import os
import logging
from optparse import OptionParser, OptionGroup, Option
from os.path import basename, splitext, abspath
import sys

# PySPH imports.
from pysph.base.particle_array import ParticleArray
from pysph.base.nnps import NNPS
from pysph.solver.controller import CommandManager
import pysph.base.kernels as kernels
from utils import mkdir

integration_methods = ['RK2']
kernel_names = ['CubicSpline']


# MPI conditional imports
HAS_MPI = True
try:
    from mpi4py import MPI
except ImportError:
    HAS_MPI = False
else:
    # Add parallel module imports.
    pass

def list_option_callback(option, opt, value, parser):
    val = value.split(',')
    val.extend( parser.rargs )
    setattr( parser.values, option.dest, val )

##############################################################################
# `Application` class.
############################################################################## 
class Application(object):
    """ Class used by any SPH application.
    """

    def __init__(self, fname=None):
        """ Constructor

        Parameters
        ----------
        fname : file name to use.

        """
        self._solver = None
        self._parallel_manager = None

        if fname == None:
            fname = sys.argv[0].split('.')[0]

        self.fname = fname

        self.args = sys.argv[1:]

        # MPI related vars.
        self.comm = None
        self.num_procs = 1
        self.rank = 0
        if HAS_MPI:
            self.comm = comm = MPI.COMM_WORLD
            self.num_procs = comm.Get_size()
            self.rank = comm.Get_rank()
        
        self._log_levels = {'debug': logging.DEBUG,
                           'info': logging.INFO,
                           'warning': logging.WARNING,
                           'error': logging.ERROR,
                           'critical': logging.CRITICAL,
                           'none': None}

        self._setup_optparse()

        self.path = None
    
    def _setup_optparse(self):
        usage = """
        %prog [options] 

        Note that you may run this program via MPI and the run will be
        automatically parallelized.  To do this run::

         $ mpirun -n 4 /path/to/your/python %prog [options]
   
        Replace '4' above with the number of processors you have.
        Below are the options you may pass.

        """
        parser = OptionParser(usage)
        self.opt_parse = parser

        # Add some default options.
        # -v
        valid_vals = "Valid values: %s"%self._log_levels.keys()
        parser.add_option("-v", "--loglevel", action="store",
                          type="string",
                          dest="loglevel",
                          default='warning',
                          help="Log-level to use for log messages. " +
                               valid_vals)
        # --logfile
        parser.add_option("--logfile", action="store",
                          type="string",
                          dest="logfile",
                          default=None,
                          help="Log file to use for logging, set to "+
                               "empty ('') for no file logging.")
        # -l 
        parser.add_option("-l", "--print-log", action="store_true",
                          dest="print_log", default=False,
                          help="Print log messages to stderr.")
        # --final-time
        parser.add_option("--final-time", action="store",
                          type="float",
                          dest="final_time",
                          default=None,
                          help="Total time for the simulation.")
        # --timestep
        parser.add_option("--timestep", action="store",
                          type="float",
                          dest="time_step",
                          default=None,
                          help="Timestep to use for the simulation.")
        # -q/--quiet.
        parser.add_option("-q", "--quiet", action="store_true",
                         dest="quiet", default=False,
                         help="Do not print any progress information.")

        # -o/ --output
        parser.add_option("-o", "--output", action="store",
                          dest="output", default=self.fname,
                          help="File name to use for output")

        # --output-freq.
        parser.add_option("--freq", action="store",
                          dest="freq", default=20, type="int",
                          help="Printing frequency for the output")
        
        # -d/ --detailed-output.
        parser.add_option("-d", "--detailed-output", action="store_true",
                         dest="detailed_output", default=False,
                         help="Dump detailed output.")

        # --directory
        parser.add_option("--directory", action="store",
                         dest="output_dir", default=self.fname+'_output',
                         help="Dump output in the specified directory.")

        # --kernel
        parser.add_option("--kernel", action="store",
                          dest="kernel", type="int",
                          help="%-55s"%"The kernel function to use:"+
                          ''.join(['%d - %-51s'%(d,s) for d,s in
                                     enumerate(kernel_names)]))

        # --integration
        parser.add_option("--integration", action="store",
                          dest="integration", type="int",
                          help="%-55s"%"The integration method to use:"+
                          ''.join(['%d - %-51s'%(d,s) for d,s in
                                     enumerate(integration_methods)]))

        # --cl
        parser.add_option("--cl", action="store_true", dest="with_cl",
                          default=False, help=""" Use OpenCL to run the
                          simulation on an appropriate device """)

        # --parallel-mode
        parser.add_option("--parallel-mode", action="store",
                          dest="parallel_mode", default="auto",
                          help = """Use specified parallel mode.""")

        # --parallel-output-mode
        parser.add_option("--parallel-output-mode", action="store",
                          dest="parallel_output_mode", default="collected",
                          help="""Use 'collected' to dump one output at
                          root or 'distributed' for every processor. """)


        # solver interfaces
        interfaces = OptionGroup(parser, "Interfaces",
                                 "Add interfaces to the solver")

        interfaces.add_option("--interactive", action="store_true",
                              dest="cmd_line", default=False,
                              help=("Add an interactive commandline interface "
                                    "to the solver"))
        
        interfaces.add_option("--xml-rpc", action="store",
                              dest="xml_rpc", metavar='[HOST:]PORT',
                              help=("Add an XML-RPC interface to the solver; "
                                    "HOST=0.0.0.0 by default"))
        
        interfaces.add_option("--multiproc", action="store",
                              dest="multiproc", metavar='[[AUTHKEY@]HOST:]PORT[+]',
                              default="pysph@0.0.0.0:8800+",
                              help=("Add a python multiprocessing interface "
                                    "to the solver; "
                                    "AUTHKEY=pysph, HOST=0.0.0.0, PORT=8800+ by"
                                    " default (8800+ means first available port "
                                    "number 8800 onwards)"))
        
        interfaces.add_option("--no-multiproc", action="store_const",
                              dest="multiproc", const=None,
                              help=("Disable multiprocessing interface "
                                    "to the solver"))
        
        parser.add_option_group(interfaces)
        
        # solver job resume support
        parser.add_option('--resume', action='store', dest='resume',
                          metavar='COUNT|count|?',
                          help=('Resume solver from specified time (as stored '
                                'in the data in output directory); count chooses '
                                'a particular file; ? lists all '
                                'available files')
                          )

    def _process_command_line(self):
        """ Parse any command line arguments.

        Add any new options before this is called.  This also sets up
        the logging automatically.

        """
        (options, args) = self.opt_parse.parse_args(self.args)
        self.options = options
        
        # Setup logging based on command line options.
        level = self._log_levels[options.loglevel]

        #save the path where we want to dump output
        self.path = abspath(options.output_dir)
        mkdir(self.path)

        if level is not None:
            self._setup_logging(options.logfile, level,
                                options.print_log)

    def _setup_logging(self, filename=None, loglevel=logging.WARNING,
                       stream=True):
        """ Setup logging for the application.
        
        Parameters
        ----------
        filename : The filename to log messages to.  If this is None
                   a filename is automatically chosen and if it is an
                   empty string, no file is used

        loglevel : The logging level

        stream : Boolean indicating if logging is also printed on
                    stderr
        """
        # logging setup
        self.logger = logger = logging.getLogger()
        logger.setLevel(loglevel)

        # Setup the log file.
        if filename is None:
            filename = splitext(basename(sys.argv[0]))[0] + '.log'

        if len(filename) > 0:
            lfn = os.path.join(self.path,filename)
            if self.num_procs > 1:
                logging.basicConfig(level=loglevel, filename=lfn,
                                    filemode='w')
        if stream:
            logger.addHandler(logging.StreamHandler())

    def _create_particles(self, particle_factory, *args, **kw):
                          
        """ Create particles given a callable `particle_factory` and any
        arguments to it.

        This will also automatically distribute the particles among
        processors if this is a parallel run.  Returns a list of particle 
        arrays that are created.
        """

        num_procs = self.num_procs
        rank = self.rank
        if rank == 0:
            # Only master creates the particles.
            pa = particle_factory(*args, **kw)
        if num_procs > 1:
            # FIXME: this needs to do the right thing for multiple processors.
            pass

        self.particles = pa

        return self.particles

    ######################################################################
    # Public interface.
    ###################################################################### 
    def set_args(self, args):
        self.args = args

    def add_option(self, opt):
        """ Add an Option/OptionGroup or their list to OptionParser """
        if isinstance(opt, OptionGroup):
            self.opt_parse.add_option_group(opt)
        elif isinstance(opt, Option):
            self.opt_parse.add_option(opt)
        else:
            # assume a list of Option/OptionGroup
            for o in opt:
                self.add_option(o)

    def setup(self, solver, equations, nnps=None, particle_factory=None, 
              *args, **kwargs):
        """Set the application's solver.  This will call the solver's
        `setup` method.

        The following solver options are set:

        dt -- the time step for the solver

        tf -- the final time for the simulationl

        fname -- the file name for output file printing

        freq -- the output print frequency

        level -- the output detail level

        dir -- the output directory

        integration_type -- The integration method

        default_kernel -- the default kernel to use for operations

        Parameters
        ----------
        particle_factory : callable or None
            If supplied, particles will be created for the solver using the
            particle arrays returned by the callable. Else particles for the
            solver need to be set before calling this method

        """
        self._solver = solver
        solver_opts = solver.get_options(self.opt_parse)
        if solver_opts is not None:
            self.add_option(solver_opts)
        self._process_command_line()

        options = self.options

        if particle_factory:
            self._create_particles(particle_factory, *args, **kwargs)
        
        self._solver.setup_solver(options.__dict__)

        if nnps is None:
            if self.num_procs == 1:
                nnps = NNPS(dim=solver.dim, particles=self.particles, 
                            radius_scale=2.0)
            else:
                nnps = None
                # FIXME. add parallel stuff here.
                pass
        self.nnps = nnps

        dt = options.time_step
        if dt is not None:
            solver.set_time_step(dt)

        tf = options.final_time
        if tf is not None:
            solver.set_final_time(tf)

        # Setup the solver output file name
        fname = options.output
        
        if HAS_MPI:
            rank = self.rank
            if self.num_procs > 1:
                fname += '_' + str(rank)

        # set the rank for the solver
        solver.rank = self.rank
        solver.pid = self.rank
        solver.comm = self.comm

        # set the in parallel flag for the solver
        if self.num_procs > 1:
            solver.in_parallel = True

        # output file name
        solver.set_output_fname(fname)

        # output print frequency
        solver.set_print_freq(options.freq)

        # output printing level (default is not detailed)
        solver.set_output_printing_level(options.detailed_output)

        # output directory
        solver.set_output_directory(abspath(options.output_dir))

        # set parallel output mode
        solver.set_parallel_output_mode(options.parallel_output_mode)

        # default kernel
        if options.kernel is not None:
            solver.kernel = getattr(kernels,
                                kernel_names[options.kernel])(dim=solver.dim)
        
        if options.resume is not None:
            solver.particles = self.particles # needed to be able to load particles
            r = solver.load_output(options.resume)
            if r is not None:
                print 'available files for resume:'
                print r
                sys.exit(0)

        if options.integration is not None:
            # FIXME, this is bogus
            #solver.integrator_type = integration_methods[options.integration]
            pass
        
        # setup the solver
        solver.setup(particles=self.particles, equations=equations, nnps=nnps)
        
        # add solver interfaces
        self.command_manager = CommandManager(solver, self.comm)
        solver.set_command_handler(self.command_manager.execute_commands)
        
        if self.rank == 0:
            # commandline interface
            if options.cmd_line:
                from pysph.solver.solver_interfaces import CommandlineInterface
                self.command_manager.add_interface(CommandlineInterface().start)
        
            # XML-RPC interface
            if options.xml_rpc:
                from pysph.solver.solver_interfaces import XMLRPCInterface
                addr = options.xml_rpc
                idx = addr.find(':')
                host = "0.0.0.0" if idx == -1 else addr[:idx]
                port = int(addr[idx+1:])
                self.command_manager.add_interface(XMLRPCInterface((host,port)).start)
        
            # python MultiProcessing interface
            if options.multiproc:
                from pysph.solver.solver_interfaces import MultiprocessingInterface
                addr = options.multiproc
                idx = addr.find('@')
                authkey = "pysph" if idx == -1 else addr[:idx]
                addr = addr[idx+1:]
                idx = addr.find(':')
                host = "0.0.0.0" if idx == -1 else addr[:idx]
                port = addr[idx+1:]
                if port[-1] == '+':
                    try_next_port = True
                    port = port[:-1]
                else:
                    try_next_port = False
                port = int(port)

                interface = MultiprocessingInterface((host,port), authkey,
                                                     try_next_port)

                self.command_manager.add_interface(interface.start)

                self.logger.info('started multiprocessing interface on %s'%(
                        interface.address,))

    def run(self):
        """Run the application.
        """
        self._solver.solve(not self.options.quiet)

    def dump_code(self, file):
        """Dump the generated code to given file.
        """
        file.write(self._solver.sph_eval.ext_mod.code)