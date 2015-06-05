# Standard imports.
import os
import logging
from optparse import OptionParser, OptionGroup, Option
from os.path import abspath, basename, splitext
import sys
import time

# PySPH imports.
from pysph.base.config import get_config
from pysph.base import utils
from pysph.base.nnps import BoxSortNNPS, LinkedListNNPS
from pysph.solver.controller import CommandManager
from pysph.solver.utils import mkdir, load

# conditional parallel imports
from pysph import Has_MPI, Has_Zoltan
if (Has_MPI and Has_Zoltan):
    from pysph.parallel.parallel_manager import ZoltanParallelManagerGeometric
    import mpi4py.MPI as mpi


def list_option_callback(option, opt, value, parser):
    val = value.split(',')
    val.extend( parser.rargs )
    setattr( parser.values, option.dest, val )

logger = logging.getLogger(__name__)

##############################################################################
# `Application` class.
##############################################################################
class Application(object):
    """ Class used by any SPH application.
    """

    def __init__(self, fname=None, domain=None):
        """ Constructor

        Parameters
        ----------
        fname : str
            file name to use.
        domain : pysph.nnps.DomainManager
            A domain manager to use. This is used for periodic domains etc.
        """
        self.is_periodic = False
        self.domain = domain
        if domain is not None:
            self.is_periodic = domain.is_periodic

        self._solver = None
        self._parallel_manager = None

        if fname == None:
            fname = splitext(basename(abspath(sys.argv[0])))[0]

        self.fname = fname

        self.args = sys.argv[1:]

        # MPI related vars.
        self.comm = None
        self.num_procs = 1
        self.rank = 0
        if Has_MPI:
            self.comm = comm = mpi.COMM_WORLD
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
        self.particles = []
        self.inlet_outlet = []

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
                          default='info',
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
        parser.add_option("--tf", action="store",
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

        # --adaptive-timestep
        parser.add_option("--adaptive-timestep", action="store_true",
                          dest="adaptive_timestep", default=None,
                          help="Use adaptive time stepping.")
        parser.add_option("--no-adaptive-timestep", action="store_false",
                          dest="adaptive_timestep", default=None,
                          help="Do not use adaptive time stepping.")

        # --cfl
        parser.add_option("--cfl", action="store", dest="cfl", type='float',
                          default=0.3,
                          help="CFL number for adaptive time steps")

        # -q/--quiet.
        parser.add_option("-q", "--quiet", action="store_true",
                         dest="quiet", default=False,
                         help="Do not print any progress information.")

        # --disable-output
        parser.add_option("--disable-output", action="store_true",
                         dest="disable_output", default=False,
                         help="Do not dump any output files.")

        # -o/ --fname
        parser.add_option("-o", "--fname", action="store",
                          dest="output", default=self.fname,
                          help="File name to use for output")

        # --pfreq.
        parser.add_option("--pfreq", action="store",
                          dest="freq", default=None, type="int",
                          help="Printing frequency for the output")

        # -d/ --detailed-output.
        parser.add_option("-d", "--detailed-output", action="store_true",
                         dest="detailed_output", default=None,
                         help="Dump detailed output.")

        # --output-remote
        parser.add_option("--output-dump-remote", action="store_true",
                          dest="output_dump_remote", default=False,
                          help="Save Remote particles in parallel")
        # --directory
        parser.add_option("--directory", action="store",
                         dest="output_dir", default=self.fname+'_output',
                         help="Dump output in the specified directory.")

        # --openmp
        parser.add_option("--openmp", action="store_true", dest="with_openmp",
                          default=None, help="Use OpenMP to run the "\
                            "simulation using multiple cores.")
        parser.add_option("--no-openmp", action="store_false", dest="with_openmp",
                          default=None, help="Do not use OpenMP to run the "\
                            "simulation using multiple cores.")

        # Restart options
        restart = OptionGroup(parser, "Restart options",
                              "Restart options for PySPH")

        restart.add_option("--restart-file", action="store", dest="restart_file",
                           default=None,
                           help=("""Restart a PySPH simulation using a specified file """)),

        restart.add_option("--rescale-dt", action="store", dest="rescale_dt",
                           default=1.0, type="float",
                           help=("Scale dt upon restarting by a numerical constant"))

        parser.add_option_group( restart )

        # NNPS options
        nnps_options = OptionGroup(parser, "NNPS", "Nearest Neighbor searching")

        # --nnps
        nnps_options.add_option("--nnps", dest="nnps",
                                choices=['box', 'll'],
                                default='ll',
                                help="Use one of box-sort ('box') or "\
                                     "the linked list algorithm ('ll'). "
                                )

        # --fixed-h
        nnps_options.add_option("--fixed-h", dest="fixed_h",
                                action="store_true", default=False,
                                help="Option for fixed smoothing lengths")

        nnps_options.add_option("--cache-nnps", dest="cache_nnps",
                                action="store_true", default=False,
                        help="Option to enable the use of neighbor caching.")

        nnps_options.add_option(
            "--sort-gids", dest="sort_gids", action="store_true",
            default=False, help="Sort neighbors by the GIDs to get "\
            "consistent results in serial and parallel (slows down a bit)."
        )

        parser.add_option_group( nnps_options )

        # Zoltan Options
        zoltan = OptionGroup(parser, "PyZoltan",
                             "Zoltan load balancing options")

        zoltan.add_option("--with-zoltan", action="store_true",
                          dest="with_zoltan", default=True,
                          help=("""Use PyZoltan for dynamic load balancing """))

        zoltan.add_option("--zoltan-lb-method", action="store",
                          dest="zoltan_lb_method", default="RCB",
                          help=("""Choose the Zoltan load balancnig method"""))

        # --rcb-lock
        zoltan.add_option("--rcb-lock", action="store_true", dest="zoltan_rcb_lock_directions",
                          default=False,
                          help=("Lock the directions of the RCB cuts"))

        # rcb--reuse
        zoltan.add_option("--rcb-reuse", action='store_true', dest="zoltan_rcb_reuse",
                          default=False,
                          help=("Reuse previous RCB cuts"))

        # rcb-rectilinear
        zoltan.add_option("--rcb-rectilinear", action="store_true", dest='zoltan_rcb_rectilinear',
                          default=False,
                          help=("Produce nice rectilinear blocks without projections"))

        # rcb-set-direction
        zoltan.add_option("--rcb-set-direction", action='store', dest="zoltan_rcb_set_direction",
                          default=0, type="int",
                          help=("Set the order of the RCB cuts"))

        zoltan.add_option("--zoltan-weights", action="store_false",
                          dest="zoltan_weights", default=True,
                          help=("""Switch between using weights for input to Zoltan.
                          defaults to True"""))

        zoltan.add_option("--ghost-layers", action='store', dest='ghost_layers',
                          default=3.0, type='float',
                          help=('Number of ghost cells to share for remote neighbors'))

        zoltan.add_option("--lb-freq", action='store', dest='lb_freq',
                          default=10, type='int',
                          help=('The frequency for load balancing'))

        zoltan.add_option("--zoltan-debug-level", action="store",
                          dest="zoltan_debug_level", default="0",
                          help=("""Zoltan debugging level"""))

        parser.add_option_group( zoltan )

        # Options to control parallel execution
        parallel_options=OptionGroup(parser, "Parallel Options")

        # --update-cell-sizes
        parallel_options.add_option("--update-cell-sizes", action='store_true',
                                    dest='update_cell_sizes', default=False,
                                    help=("Recompute cell sizes for binning in parallel"))

        # --parallel-scale-factor
        parallel_options.add_option("--parallel-scale-factor", action="store",
                                    dest="parallel_scale_factor", default=2.0, type='float',
                                    help=("""Kernel scale factor for the parallel update"""))

        # --parallel-output-mode
        parallel_options.add_option("--parallel-output-mode", action="store",
                            dest="parallel_output_mode", default=None,
                            help="""Use 'collected' to dump one output at
                          root or 'distributed' for every processor. """)

        parser.add_option_group( parallel_options )


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
        try:
            # If this is being run inside an IPython console or notebook
            # then this is defined and we should not parse the command line
            # arguments.
            __IPYTHON__
        except NameError:
            (options, args) = self.opt_parse.parse_args(self.args)
        else:
            (options, args) = self.opt_parse.parse_args([])
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
        logger.setLevel(loglevel)

        # Setup the log file.
        if filename is None:
            filename = splitext(basename(sys.argv[0]))[0] + '.log'

        if len(filename) > 0:
            lfn = os.path.join(self.path,filename)
            format = '%(levelname)s|%(asctime)s|%(name)s|%(message)s'
            logging.basicConfig(level=loglevel, format=format,
                                filename=lfn, filemode='a')
        if stream:
            logger.addHandler(logging.StreamHandler())

    def _create_inlet_outlet(self, inlet_outlet_factory):
        """Create the inlets and outlets if needed.

        This method requires that the particles be already created.

        The `inlet_outlet_factory` is passed a dictionary of the particle
        arrays.  The factory should return a list of inlets and outlets.
        """
        if inlet_outlet_factory is not None:
            solver = self._solver
            particle_arrays = dict([(p.name, p) for p in self.particles])
            self.inlet_outlet = inlet_outlet_factory(particle_arrays)
            # Hook up the inlet/outlet's update method to be called after
            # each stage.
            for obj in self.inlet_outlet:
                solver.add_post_step_callback(obj.update)

    def _create_particles(self, particle_factory, *args, **kw):

        """ Create particles given a callable `particle_factory` and any
        arguments to it.
        """
        solver = self._solver
        options = self.options
        rank = self.rank

        # particle array info that is used to create dummy particles
        # on non-root processors
        particles_info = {}

        # Only master actually calls the particle factory, the rest create
        # dummy particle arrays.
        if rank == 0:
            if options.restart_file is not None:
                data = load(options.restart_file)

                arrays = data['arrays']
                solver_data = data['solver_data']

                # arrays and particles
                particles = []
                for array_name in arrays:
                    particles.append( arrays[array_name] )

                # save the particles list
                self.particles = particles

                # time, timestep and solver iteration count at restart
                t, dt, count = solver_data['t'], solver_data['dt'], solver_data['count']

                # rescale dt at restart
                dt *= options.rescale_dt
                solver.t, solver.dt, solver.count  = t, dt, count

            else:
                self.particles = particle_factory(*args, **kw)

            # get the array info which will be b'casted to other procs
            particles_info = utils.get_particles_info(self.particles)

        # Broadcast the particles_info to other processors for parallel runs
        if self.num_procs > 1:
            particles_info = self.comm.bcast(particles_info, root=0)

        # now all processors other than root create dummy particle arrays
        if rank != 0:
            self.particles = utils.create_dummy_particles(particles_info)

    def _do_initial_load_balancing(self):
        """ This will automatically distribute the particles among processors
        if this is a parallel run.
        """
        # Instantiate the Parallel Manager here and do an initial LB
        num_procs = self.num_procs
        options = self.options
        solver = self._solver
        comm = self.comm

        self.pm = None
        if num_procs > 1:
            options = self.options

            if options.with_zoltan:
                if not (Has_Zoltan and Has_MPI):
                    raise RuntimeError("Cannot run in parallel!")

            else:
                raise ValueError("""Sorry. You're stuck with Zoltan for now

                use the option '--with_zoltan' for parallel runs

                """)

            # create the parallel manager
            obj_weight_dim = "0"
            if options.zoltan_weights:
                obj_weight_dim = "1"

            zoltan_lb_method = options.zoltan_lb_method
            zoltan_debug_level = options.zoltan_debug_level
            zoltan_obj_wgt_dim = obj_weight_dim

            # ghost layers
            ghost_layers = options.ghost_layers

            # radius scale for the parallel update
            radius_scale = options.parallel_scale_factor*solver.kernel.radius_scale

            self.pm = pm = ZoltanParallelManagerGeometric(
                dim=solver.dim, particles=self.particles, comm=comm,
                lb_method=zoltan_lb_method,
                obj_weight_dim=obj_weight_dim,
                ghost_layers=ghost_layers,
                update_cell_sizes=options.update_cell_sizes,
                radius_scale=radius_scale,
                )

            ### ADDITIONAL LOAD BALANCING FUNCTIONS FOR ZOLTAN ###

            # RCB lock directions
            if options.zoltan_rcb_lock_directions:
                pm.set_zoltan_rcb_lock_directions()

            if options.zoltan_rcb_reuse:
                pm.set_zoltan_rcb_reuse()

            if options.zoltan_rcb_rectilinear:
                pm.set_zoltan_rcb_rectilinear_blocks()

            if options.zoltan_rcb_set_direction > 0:
                pm.set_zoltan_rcb_directions( str(options.zoltan_rcb_set_direction) )

            # set zoltan options
            pm.pz.Zoltan_Set_Param("DEBUG_LEVEL", options.zoltan_debug_level)
            pm.pz.Zoltan_Set_Param("DEBUG_MEMORY", "0")

            # do an initial load balance
            pm.update()
            pm.initial_update = False

            # set subsequent load balancing frequency
            lb_freq = options.lb_freq
            if lb_freq < 1 : raise ValueError("Invalid lb_freq %d"%lb_freq)
            pm.set_lb_freq( lb_freq )

            # wait till the initial partition is done
            comm.barrier()

        # set the solver's parallel manager
        solver.set_parallel_manager(self.pm)

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

    def setup(self, solver, equations, nnps=None, inlet_outlet_factory=None,
              particle_factory=None, *args, **kwargs):
        """Setup the application's solver.

        This will parse the command line arguments (if this is not called from
        within an IPython notebook or shell) and then using those parameters
        and any additional parameters and call the solver's setup method.

        Parameters
        ----------
        solver: pysph.solver.solver.Solver
            The solver instance.

        equations: list
            A list of Groups/Equations.

        nnps: pysph.base.nnps.NNPS
            Optional NNPS instance. If None is given a default NNPS is created.

        inlet_outlet_factory: callable or None
            The `inlet_outlet_factory` is passed a dictionary of the particle
            arrays.  The factory should return a list of inlets and outlets.

        particle_factory : callable or None
            If supplied, particles will be created for the solver using the
            particle arrays returned by the callable. Else particles for the
            solver need to be set before calling this method

        args:
            extra positional arguments passed on to the `particle_factory`.

        kwargs:
            extra keyword arguments passed to the `particle_factory`.


        Examples
        --------

        >>> def create_particles():
        ...    ...
        ...
        >>> solver = Solver(...)
        >>> equations = [...]
        >>> app = Application()
        >>> app.setup(solver=solver, equations=equations,
        ...           particle_factory=create_particles)
        >>> app.run()
        """
        start_time = time.time()
        self._solver = solver
        solver_opts = solver.get_options(self.opt_parse)
        if solver_opts is not None:
            self.add_option(solver_opts)
        self._process_command_line()

        options = self.options

        # Setup configuration options.
        if options.with_openmp is not None:
            get_config().use_openmp = options.with_openmp

        # Create particles either from scratch or restart
        self._create_particles(particle_factory, *args, **kwargs)

        # This must be done before the initial load balancing
        # as the inlets will create new particles.
        self._create_inlet_outlet(inlet_outlet_factory)

        self._do_initial_load_balancing()

        # setup the solver using any options
        self._solver.setup_solver(options.__dict__)

        # fixed smoothing lengths
        fixed_h = solver.fixed_h or options.fixed_h

        if nnps is None:
            kernel = self._solver.kernel
            cache = options.cache_nnps

            # create the NNPS object
            if options.nnps == 'box':
                nnps = BoxSortNNPS(
                    dim=solver.dim, particles=self.particles,
                    radius_scale=kernel.radius_scale, domain=self.domain,
                    cache=cache, sort_gids=options.sort_gids
                )

            elif options.nnps == 'll':
                nnps = LinkedListNNPS(
                    dim=solver.dim, particles=self.particles,
                    radius_scale=kernel.radius_scale, domain=self.domain,
                    fixed_h=fixed_h, cache=cache,
                    sort_gids=options.sort_gids
                )

        # once the NNPS has been set-up, we set the default Solver
        # post-stage callback to the DomainManager.setup_domain
        # method. This method is responsible to computing the new cell
        # size and doing any periodicity checks if needed.
        solver.add_post_stage_callback( nnps.update_domain )

        # inform NNPS if it's working in parallel
        if self.num_procs > 1:
            nnps.set_in_parallel(True)

        # save the NNPS with the application
        self.nnps = nnps

        dt = options.time_step
        if dt is not None:
            solver.set_time_step(dt)

        tf = options.final_time
        if tf is not None:
            solver.set_final_time(tf)

        # Setup the solver output file name
        fname = options.output

        if Has_MPI:
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

        # disable_output
        solver.set_disable_output(options.disable_output)

        # output print frequency
        if options.freq is not None:
            solver.set_print_freq(options.freq)

        # output printing level (default is not detailed)
        if options.detailed_output is not None:
            solver.set_output_printing_level(options.detailed_output)

        # solver output behaviour in parallel
        if options.output_dump_remote:
            solver.set_output_only_real( False )

        # output directory
        solver.set_output_directory(abspath(options.output_dir))

        # set parallel output mode
        if options.parallel_output_mode is not None:
            solver.set_parallel_output_mode(options.parallel_output_mode)

        # Set the adaptive timestep
        if options.adaptive_timestep is not None:
            solver.set_adaptive_timestep(options.adaptive_timestep)

            # set solver cfl number
            solver.set_cfl(options.cfl)

        # setup the solver. This is where the code is compiled
        solver.setup(particles=self.particles, equations=equations, nnps=nnps, fixed_h=fixed_h)

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

                interface = MultiprocessingInterface(
                    (host,port), authkey.encode(), try_next_port)

                self.command_manager.add_interface(interface.start)

                logger.info('started multiprocessing interface on %s'%(
                             interface.address,))
        end_time = time.time()
        self._message("Setup took: %.5f secs"%(end_time - start_time))

    def run(self):
        """Run the application.
        """
        start_time = time.time()
        self._solver.solve(not self.options.quiet)
        end_time = time.time()
        self._message("Run took: %.5f secs"%(end_time - start_time))

    def dump_code(self, file):
        """Dump the generated code to given file.
        """
        file.write(self._solver.sph_eval.ext_mod.code)

    def _message(self, msg):
        if self.options.quiet:
            return
        if self.num_procs == 1:
            logger.info(msg)
            print(msg)
        elif (self.num_procs > 1 and self.rank in (0,1)):
            s = "Rank %d: %s"%(self.rank, msg)
            logger.info(s)
            print(s)
