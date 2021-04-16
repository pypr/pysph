# Standard imports.
from argparse import ArgumentDefaultsHelpFormatter
import atexit
from compyle.utils import ArgumentParser
import glob
import inspect
import json
import logging
import os
from os.path import (abspath, basename, dirname, isdir, join, realpath,
                     splitext)
import socket
import sys
from textwrap import dedent
import time
import numpy as np
import warnings

# PySPH imports.
from pysph.base import utils
from pysph.base.utils import is_overloaded_method

from pysph.base.nnps import LinkedListNNPS, BoxSortNNPS, SpatialHashNNPS, \
    ExtendedSpatialHashNNPS, CellIndexingNNPS, StratifiedHashNNPS, \
    StratifiedSFCNNPS, OctreeNNPS, CompressedOctreeNNPS, ZOrderNNPS

from pysph.base import kernels
from compyle.config import get_config
from compyle.profile import print_profile, profile2csv, get_profile_info
from .controller import CommandManager
from .utils import mkdir, load, get_files, get_free_port, is_using_ipython

# conditional parallel imports
from pysph import has_mpi, has_zoltan, in_parallel

if in_parallel():
    from pysph.parallel.parallel_manager import ZoltanParallelManagerGeometric
    import mpi4py.MPI as mpi

logger = logging.getLogger(__name__)


def list_all_kernels():
    """Return list of available kernels.
    """
    return [n for n in dir(kernels) if inspect.isclass(getattr(kernels, n))]


##############################################################################
# `Application` class.
##############################################################################
class Application(object):
    """Subclass this to run any SPH simulation. There are several important
    methods that this class provides. The application is typically used as
    follows::

        class EllipticalDrop(Application):
            def create_particles(self):
                # ...
            def create_scheme(self):
                # ...
            ...
        app = EllipticalDrop()
        app.run()
        app.post_process(app.info_filename)

    .. py:currentmodule:: pysph.solver.application

    The :py:meth:`post_process` method is entirely optional and typically
    performs the post-processing. It is important to understand the correct
    sequence of the method calls. When the ``Application`` instance is created,
    the following methods are invoked by the :py:meth:`__init__` method:

    1. :py:meth:`initialize()`: use this to setup any constants etc.

    2. :py:meth:`create_scheme()`: this needs to be overridden if one wishes to
       use a :py:class:`pysph.sph.scheme.Scheme`. If one does not want to use a
       scheme, the :py:meth:`create_equations` and :py:meth:`create_solver`
       methods must be overridden.

    3. ``self.scheme.add_user_options()``: i.e. the scheme's command line
       options are added, if there is a scheme.

    4. :py:meth:`add_user_options()`: add any user specified command line
       options.

    When ``app.run()`` is called, the following methods are called in order:

    1. ``_parse_command_line()``: this is a private method but it is important
       to note that the command line arguments are first parsed.

    2. :py:meth:`consume_user_options()`: this is called right after the
       command line args are parsed.

    3. :py:meth:`configure_scheme()`: This is where one may configure the
       scheme according to the passed command line arguments.

    4. :py:meth:`create_solver()`: Create the solver, note that this is needed
       only if one has not used a scheme, otherwise, this will by default
       return the solver created by the scheme chosen.

    5. :py:meth:`create_equations()`: Create any equations. Defaults to letting
       the scheme generate and return the desired equations.

    6. :py:meth:`create_particles()`

    7. :py:meth:`create_inlet_outlet()`

    8. :py:meth:`create_domain()`:  Not needed for non-periodic domains.

    9. :py:meth:`create_nnps()`: Not needed unless one wishes to override the
       default NNPS.

    10. :py:meth:`create_tools()`: Add any ``pysph.solver.tools.Tool``
        instances.

    11. :py:meth:`customize_output()`: Customize the output visualization.

    Additionally, as the application runs there are several convenient optional
    callbacks setup:

    1. :py:meth:`pre_step`: Called before each time step.

    2. :py:meth:`post_stage`: Called after every stage of the integration.

    3. :py:meth:`post_step`: Called after each time step.

    Finally, it is a good idea to overload the :py:meth:`post_process` method
    to perform any post processing for the generated data.

    The application instance also has several important attributes, some of
    these are as follows:

    - ``args``: command line arguments, typically ``sys.argv[1:]``.

    - ``domain``: optional :py:class:`pysph.base.nnps_base.DomainManager`
      instance.

    - ``fname``: filename pattern to use when dumping output.

    - ``inlet_outlet``: list of inlet/outlets.

    - ``nnps``: instance of :py:class:`pysph.base.nnps_base.NNPS`.

    - ``num_procs``: total number of processes running.

    - ``output_dir``: Output directory.

    - ``parallel_manager``: in parallel, an instance of
      :py:class:`pysph.parallel.parallel_manager.ParallelManager`.

    - ``particles``: list of
      :py:class:`pysph.base.particle_array.ParticleArray`.

    - ``rank``: Rank of this process.

    - ``scheme``: the optional :py:class:`pysph.sph.scheme.Scheme` instance.

    - ``solver``: the solver instance, :py:class:`pysph.solver.solver.Solver`.

    - ``tools``: a list of possible :py:class:`pysph.solver.tools.Tool`.

    """

    def __init__(self, fname=None, output_dir=None, domain=None):
        """ Constructor

        Parameters
        ----------
        fname : str
            file name to use for the output files.
        output_dir : str
            output directory name.
        domain : pysph.base.nnps_base.DomainManager
            A domain manager to use. This is used for periodic domains etc.
        """
        self.domain = domain

        self.solver = None
        self.nnps = None
        self.scheme = None
        self.tools = []
        self.parallel_manager = None

        if fname is None:
            fname = self._guess_output_filename()

        self.fname = fname

        self.args = sys.argv[1:]

        # MPI related vars.
        self.comm = None
        self.num_procs = 1
        self.rank = 0
        if in_parallel():
            if not mpi.Is_initialized():
                mpi.Init()
            self.comm = comm = mpi.COMM_WORLD
            self.num_procs = comm.Get_size()
            self.rank = comm.Get_rank()

        self._log_levels = {
            'debug': logging.DEBUG,
            'info': logging.INFO,
            'warning': logging.WARNING,
            'error': logging.ERROR,
            'critical': logging.CRITICAL,
            'none': None
        }

        if output_dir is None:
            self.output_dir = abspath(self._get_output_dir_from_fname())
        else:
            self.output_dir = output_dir
        self.particles = []
        self.inlet_outlet = []
        # The default value that is overridden by the command line
        # options passed or in initialize.
        self.cache_nnps = False
        self.iom = None

        # The solver interface in use.
        self._interfaces = []
        self.initialize()
        self.scheme = self.create_scheme()
        self._setup_argparse()

    def _get_output_dir_from_fname(self):
        return self.fname + '_output'

    def _guess_output_filename(self):
        """Try to guess the output filename to use.
        """
        module = self.__module__.rsplit('.', 1)[-1]
        if is_using_ipython():
            return module
        else:
            if len(sys.argv[0]) == 0:
                return module
            else:
                return splitext(basename(abspath(sys.argv[0])))[0]

    def _setup_argparse(self):
        usage = '%(prog)s [options]'
        description = """
        Note that you may run this program via MPI and the run will be
        automatically parallelized.  To do this run::

         $ mpirun -n 4 /path/to/your/python %prog [options]

        Replace '4' above with the number of processors you have.
        Below are the options you may pass.

        """
        parser = ArgumentParser(
            usage=usage,
            description=description,
            formatter_class=ArgumentDefaultsHelpFormatter)
        self.arg_parse = parser

        # Add some default options.
        # -v
        valid_vals = "Valid values: %s" % self._log_levels.keys()
        parser.add_argument(
            "-v",
            "--loglevel",
            action="store",
            dest="loglevel",
            default='info',
            help="Log-level to use for log messages. " + valid_vals)
        # --logfile

        parser.add_argument(
            "--logfile",
            action="store",
            dest="logfile",
            default=None,
            help="Log file to use for logging, set to " +
            "empty ('') for no file logging.")
        # -l
        parser.add_argument(
            "-l",
            "--print-log",
            action="store_true",
            dest="print_log",
            default=False,
            help="Print log messages to stderr.")
        # --final-time
        parser.add_argument(
            "--tf",
            action="store",
            type=float,
            dest="final_time",
            default=None,
            help="Total time for the simulation.")
        # --timestep
        parser.add_argument(
            "--timestep",
            action="store",
            type=float,
            dest="time_step",
            default=None,
            help="Timestep to use for the simulation.")
        # --max-steps
        parser.add_argument(
            "--max-steps",
            action="store",
            type=int,
            dest="max_steps",
            default=1 << 31,
            help="Maximum number of iteration steps to take (defaults to a "
            "very large value).")

        # --n-damp
        parser.add_argument(
            "--n-damp",
            action="store",
            type=int,
            dest="n_damp",
            default=None,
            help="Number of iterations to damp timesteps initially.")

        # --adaptive-timestep
        parser.add_argument(
            "--adaptive-timestep",
            action="store_true",
            dest="adaptive_timestep",
            default=None,
            help="Use adaptive time stepping.")
        parser.add_argument(
            "--no-adaptive-timestep",
            action="store_false",
            dest="adaptive_timestep",
            default=None,
            help="Do not use adaptive time stepping.")

        # --cfl
        parser.add_argument(
            "--cfl",
            action="store",
            dest="cfl",
            type=float,
            default=0.3,
            help="CFL number for adaptive time steps")

        # -q/--quiet.
        parser.add_argument(
            "-q",
            "--quiet",
            action="store_true",
            dest="quiet",
            default=False,
            help="Do not print any progress information.")

        # --disable-output
        parser.add_argument(
            "--disable-output",
            action="store_true",
            dest="disable_output",
            default=False,
            help="Do not dump any output files.")

        # -o/ --fname
        parser.add_argument(
            "-o",
            "--fname",
            action="store",
            dest="fname",
            default=self.fname,
            help="File name to use for output")

        # --pfreq.
        parser.add_argument(
            "--pfreq",
            action="store",
            dest="freq",
            default=None,
            type=int,
            help="Printing frequency for the output")

        parser.add_argument(
            '--reorder-freq', action="store", dest="reorder_freq",
            default=None, type=int,
            help="Frequency between spatially reordering particles."
        )

        # --detailed-output.
        parser.add_argument(
            "--detailed-output",
            action="store_true",
            dest="detailed_output",
            default=None,
            help="Dump detailed output.")

        # -z/--compress-output
        parser.add_argument(
            "-z",
            "--compress-output",
            action="store_true",
            dest="compress_output",
            default=False,
            help="Compress generated output files.")

        # --output-remote
        parser.add_argument(
            "--output-dump-remote",
            action="store_true",
            dest="output_dump_remote",
            default=False,
            help="Save Remote particles in parallel")
        # -d/--directory
        parser.add_argument(
            "-d",
            "--directory",
            action="store",
            dest="output_dir",
            default=self.output_dir,
            help="Dump output in the specified directory.")

        # --openmp
        parser.add_argument(
            "--no-openmp",
            action="store_false",
            dest="with_openmp",
            default=None,
            help="Do not use OpenMP to run the "
            "simulation using multiple cores.")

        # --omp-schedule
        parser.add_argument(
            "--omp-schedule",
            action="store",
            dest="omp_schedule",
            default="dynamic,64",
            help="""Schedule how loop iterations
            are divided amongst multiple threads""")

        # --opencl
        parser.add_argument(
            "--opencl",
            action="store_true",
            dest="with_opencl",
            default=False,
            help="Use OpenCL to run the simulation.")

        # --cuda
        parser.add_argument(
            "--cuda",
            action="store_true",
            dest="with_cuda",
            default=False,
            help="Use CUDA to run the simulation."
        )

        # --use-local-memory
        parser.add_argument(
            "--use-local-memory",
            action="store_true",
            dest="with_local_memory",
            default=False,
            help="Use local memory with OpenCL (Experimental)"
        )
        # --kernel
        all_kernels = list_all_kernels()
        parser.add_argument(
            "--kernel",
            action="store",
            dest="kernel",
            default=None,
            choices=all_kernels,
            help="Use specified kernel from %s" % all_kernels)

        parser.add_argument(
            '--post-process', action="store",
            dest="post_process", default=None,
            help="Only perform post-processing and exit."
        )

        # Restart options
        restart = parser.add_argument_group("Restart options",
                                            "Restart options for PySPH")

        restart.add_argument(
            "--restart-file",
            action="store",
            dest="restart_file",
            default=None,
            help=("""Restart a PySPH simulation using a specified file """))

        restart.add_argument(
            "--rescale-dt",
            action="store",
            dest="rescale_dt",
            default=1.0,
            type=float,
            help=("Scale dt upon restarting by a numerical constant"))

        # NNPS options
        nnps_options = parser.add_argument_group("NNPS",
                                                 "Nearest Neighbor searching")

        # --nnps
        nnps_options.add_argument(
            "--nnps",
            dest="nnps",
            choices=[
                'box', 'll', 'sh', 'esh', 'ci', 'sfc', 'comp_tree',
                'strat_hash', 'strat_sfc', 'tree', 'gpu_octree'
            ],
            default='ll',
            help="Use one of box-sort ('box') or "
            "the linked list algorithm ('ll') or "
            "the spatial hash algorithm ('sh') or "
            "the extended spatial hash algorithm ('esh') or "
            "the cell indexing algorithm ('ci') or "
            "the z-order space filling curve based algorithm ('sfc') or "
            "the stratified hash algorithm ('strat_hash') or "
            "the stratified sfc algorithm ('strat_sfc') or "
            "the octree algorithm ('tree') or "
            "the compressed octree algorithm ('comp_tree') or "
            "the gpu octree algorithm ('gpu_octree')")

        nnps_options.add_argument(
            "--spatial-hash-sub-factor",
            dest="H",
            type=int,
            default=3,
            help="Sub division factor for ExtendedSpatialHashNNPS")

        nnps_options.add_argument(
            "--approximate-nnps",
            dest="approximate_nnps",
            action="store_true",
            default=False,
            help="Use for approximate NNPS")

        nnps_options.add_argument(
            "--spatial-hash-table-size",
            dest="table_size",
            type=int,
            default=131072,
            help="Table size for SpatialHashNNPS and ExtendedSpatialHashNNPS")

        nnps_options.add_argument(
            "--stratified-grid-num-levels",
            dest="num_levels",
            type=int,
            default=1,
            help="Number of levels for StratifiedHashNNPS and \
            StratifiedSFCNNPS")

        nnps_options.add_argument(
            "--tree-leaf-max-particles",
            dest="leaf_max_particles",
            type=int,
            default=10,
            help="Maximum number of particles in leaf of octree")

        # --fixed-h
        nnps_options.add_argument(
            "--fixed-h",
            dest="fixed_h",
            action="store_true",
            default=False,
            help="Option for fixed smoothing lengths")

        nnps_options.add_argument(
            "--cache-nnps",
            dest="cache_nnps",
            action="store_true",
            default=self.cache_nnps,
            help="Option to enable the use of neighbor caching.")

        nnps_options.add_argument(
            "--sort-gids",
            dest="sort_gids",
            action="store_true",
            default=False,
            help="Sort neighbors by the GIDs to get " +
            "consistent results in serial and parallel (slows down a bit).")

        # Zoltan Options
        zoltan = parser.add_argument_group("PyZoltan",
                                           "Zoltan load balancing options")

        zoltan.add_argument(
            "--with-zoltan",
            action="store_true",
            dest="with_zoltan",
            default=True,
            help="Use PyZoltan for dynamic load balancing")

        zoltan.add_argument(
            "--zoltan-lb-method",
            action="store",
            dest="zoltan_lb_method",
            default='HSFC',
            choices=['RCB', 'RIB', 'HSFC'],
            help="Choose the Zoltan load balancing method")

        zoltan.add_argument(
            "--rcb-lock",
            action="store_true",
            dest="zoltan_rcb_lock_directions",
            default=False,
            help="Lock the directions of the RCB cuts")

        zoltan.add_argument(
            "--rcb-reuse",
            action='store_true',
            dest="zoltan_rcb_reuse",
            default=False,
            help="Reuse previous RCB cuts")

        zoltan.add_argument(
            "--rcb-rectilinear",
            action="store_true",
            dest='zoltan_rcb_rectilinear',
            default=False,
            help="Produce nice rectilinear blocks without projections")

        zoltan.add_argument(
            "--rcb-set-direction",
            action='store',
            dest="zoltan_rcb_set_direction",
            default=0,
            choices=[0, 1, 2, 3, 4, 5, 6],
            type=int,
            help="Set the order of the RCB cuts (0: no order, 1:'xyz', "
                 "2:'xzy', 3:'yzx', 4:'yxz', 5:'zxy', 6:'zyx')")

        zoltan.add_argument(
            "--zoltan-weights",
            action="store_false",
            dest="zoltan_weights",
            default=True,
            help=("""Switch between using weights for input to Zoltan.
            defaults to True"""))

        zoltan.add_argument(
            "--ghost-layers",
            action='store',
            dest='ghost_layers',
            default=1.0,
            type=float,
            help=('Number of ghost cells to share for remote neighbors'))

        zoltan.add_argument(
            "--lb-freq",
            action='store',
            dest='lb_freq',
            default=10,
            type=int,
            help=('The frequency for load balancing'))

        zoltan.add_argument(
            "--zoltan-debug-level",
            action="store",
            dest="zoltan_debug_level",
            default="0",
            help=("""Zoltan debugging level"""))

        # Options to control parallel execution
        parallel_options = parser.add_argument_group("Parallel Options")

        # --update-cell-sizes
        parallel_options.add_argument(
            "--update-cell-sizes",
            action='store_true',
            dest='update_cell_sizes',
            default=False,
            help=("Recompute cell sizes for binning in parallel"))

        # --parallel-scale-factor
        parallel_options.add_argument(
            "--parallel-scale-factor",
            action="store",
            dest="parallel_scale_factor",
            default=2.0,
            type=float,
            help=("""Kernel scale factor for the parallel update"""))

        # --parallel-output-mode
        parallel_options.add_argument(
            "--parallel-output-mode",
            action="store",
            dest="parallel_output_mode",
            default='collected',
            choices=['collected', 'distributed'],
            help="""Use 'collected' to dump one output at
            root or 'distributed' for every processor. """)

        # solver interfaces
        interfaces = parser.add_argument_group("Interfaces",
                                               "Add interfaces to the solver")

        interfaces.add_argument(
            "--interactive",
            action="store_true",
            dest="cmd_line",
            default=False,
            help=("Add an interactive commandline interface to the solver"))

        interfaces.add_argument(
            "--xml-rpc",
            action="store",
            dest="xml_rpc",
            metavar="[HOST:] PORT",
            help=("Add an XML-RPC interface to the solver;"
                  "HOST=0.0.0.0 by default"))

        interfaces.add_argument(
            "--multiproc",
            action="store",
            dest="multiproc",
            metavar='[[AUTHKEY@] HOST:] PORT[+] ',
            default=None,
            help=("Add a python multiprocessing interface "
                  "to the solver; "
                  "AUTHKEY=pysph, HOST=0.0.0.0, PORT=8800+ when"
                  " given 'auto' (8800+ means first available port "
                  "number 8800 onwards);"))

        interfaces.add_argument(
            "--octree-leaf-size",
            dest="octree_leaf_size",
            default=32,
            help=("Specify leaf size of octree. "
                  "Must be multiples of 32 (Experimental)")
        )

        interfaces.add_argument(
            "--octree-elementwise-nnps",
            action="store_const",
            dest="octree_elementwise",
            default=False,
            const=True,
            help=("Run NNPS for different particles "
                  "on different threads (Experimental)")
        )

        # Scheme options.
        if self.scheme is not None:
            scheme_options = parser.add_argument_group(
                "SPH Scheme options",
                "Scheme related command line arguments",
                conflict_handler="resolve")
            self.scheme.add_user_options(scheme_options)
        # User options.
        user_options = parser.add_argument_group(
            "User", "User defined command line arguments")
        self.add_user_options(user_options)

    def _parse_command_line(self, force=False):
        """If force is True, it will parse the arguments regardless of whether
        it is running in IPython or not.  This is handy when you want to parse
        the command line for a previously run case.
        """
        if is_using_ipython() and not force:
            # Don't parse the command line args.
            options = self.arg_parse.parse_args([])
        else:
            options = self.arg_parse.parse_args(self.args)

        if options.profile:
            # Remove the default callback from compyle.
            atexit.unregister(print_profile)

        self.options = options

    def _process_command_line(self):
        """Process the parsed command line arguments.

        This method calls the scheme's ``consume_user_options`` and
        :py:meth:`consume_user_options` as well as the
        :py:meth:`configure_scheme`.

        """
        options = self.options
        if options.post_process:
            self._message('-'*70)
            self._message('Performing post processing alone.')
            self.post_process(options.post_process)
            # Exit right after this so even if the user
            # has an app.post_process call, it doesn't call it.
            sys.exit(0)
        # save the path where we want to dump output
        self.output_dir = abspath(options.output_dir)
        mkdir(self.output_dir)
        if self.scheme is not None:
            self.scheme.consume_user_options(options)
        self.consume_user_options()
        if self.scheme is not None:
            self.configure_scheme()

    def _setup_logging(self):
        """Setup logging for the application.
        """
        options = self.options
        # Setup logging based on command line options.
        level = self._log_levels[options.loglevel]

        if level is None:
            return

        # logging setup
        logger.setLevel(level)

        filename = options.logfile
        # Setup the log file.
        if filename is None:
            filename = self.fname + '.log'

        if len(filename) > 0:
            # This is needed if the application is launched twice,
            # as in that case, the old handlers must be removed.
            for handler in logging.root.handlers[:]:
                handler.close()
                logging.root.removeHandler(handler)
            lfn = os.path.join(self.output_dir, filename)
            mkdir(self.output_dir)
            format = '%(levelname)s|%(asctime)s|%(name)s|%(message)s'
            logging.basicConfig(
                level=level, format=format, filename=lfn, filemode='a')
        if options.print_log:
            logger.addHandler(logging.StreamHandler())

        host = socket.gethostname()
        try:
            ip = socket.gethostbyname(host)
        except socket.gaierror:
            ip = host
        cmd = ' '.join(sys.argv)
        logger.info(
            'Started as:\n{command}'.format(command=cmd)
        )
        logger.info(
            'Running on {host} with address {ip}'.format(host=host, ip=ip)
        )

    def _create_inlet_outlet(self, inlet_outlet_factory):
        """Create the inlets and outlets if needed.

        This method requires that the particles be already created.

        The `inlet_outlet_factory` is passed a dictionary of the particle
        arrays.  The factory should return a list of inlets and outlets.
        """
        if inlet_outlet_factory is not None:
            solver = self.solver
            particle_arrays = dict([(p.name, p) for p in self.particles])
            self.inlet_outlet = inlet_outlet_factory(particle_arrays)
            # Hook up the inlet/outlet's update method to be called after
            # each stage.
            for obj in self.inlet_outlet:
                solver.add_post_stage_callback(obj.update)

    def _create_particles(self, particle_factory, *args, **kw):
        """ Create particles given a callable `particle_factory` and any
        arguments to it.
        """
        options = self.options
        rank = self.rank

        # particle array info that is used to create dummy particles
        # on non-root processors
        particles_info = {}

        # Only master actually calls the particle factory, the rest create
        # dummy particle arrays.
        if rank == 0:
            if options.restart_file is not None:
                # FIXME: not tested, probably does not work!
                solver = self.solver
                data = load(options.restart_file)

                arrays = data['arrays']
                solver_data = data['solver_data']

                # arrays and particles
                particles = []
                for array_name in arrays:
                    particles.append(arrays[array_name])

                # save the particles list
                self.particles = particles

                # time, timestep and solver iteration count at restart
                t, dt, count = solver_data['t'], solver_data[
                    'dt'], solver_data['count']

                # rescale dt at restart
                dt *= options.rescale_dt
                solver.t, solver.dt, solver.count = t, dt, count

            else:
                self.particles = particle_factory(*args, **kw)

            for pa in self.particles:
                if len(pa.x) > 0:
                    if np.max(pa.h) < 1e-12:
                        warnings.warn(
                            "'h' for particle array '{}' is 0.0".format(
                                pa.name), UserWarning)
                    if np.max(pa.m) < 1e-12:
                        warnings.warn(
                            "Mass 'm' for particle array '{}' is 0.0".format(
                                pa.name), UserWarning)

            # get the array info which will be b'casted to other procs
            particles_info = utils.get_particles_info(self.particles)

        # Broadcast the particles_info to other processors for parallel runs
        if self.num_procs > 1:
            particles_info = self.comm.bcast(particles_info, root=0)

        # now all processors other than root create dummy particle arrays
        if rank != 0:
            self.particles = utils.create_dummy_particles(particles_info)

    def _configure_global_config(self):
        options = self.options
        # Setup configuration options.
        config = get_config()
        if options.with_openmp is not None:
            config.use_openmp = options.with_openmp
            logger.info('Using OpenMP')
        if options.omp_schedule is not None:
            config.set_omp_schedule(options.omp_schedule)
            logger.info('Using OpenMP schedule %s', options.omp_schedule)

        if options.with_opencl:
            config.use_opencl = True
            logger.info('Using OpenCL')
        elif options.with_cuda:
            config.use_cuda = True
            logger.info('Using CUDA')

        if options.with_local_memory:
            leaf_size = int(options.octree_leaf_size)
            config.wgs = leaf_size
            config.use_local_memory = True
        if options.use_double:
            config.use_double = options.use_double
            logger.info('Using double precision')
        if options.profile:
            config.profile = options.profile

    def _configure_solver(self):
        """Configures the application using the options from the
        command-line.
        """
        options = self.options
        # setup the solver using any options
        self.solver.setup_solver(options.__dict__)

        solver = self.solver

        # fixed smoothing lengths
        fixed_h = solver.fixed_h or options.fixed_h

        kernel = solver.kernel
        if options.kernel is not None:
            kernel = getattr(kernels, options.kernel)(dim=solver.dim)
            solver.kernel = kernel

        # This should be called before an NNPS is created as the particles are
        # changed after the initial load-balancing.
        self._setup_parallel_manager_and_initial_load_balance()

        if self.nnps is None:
            cache = options.cache_nnps

            # create the NNPS object
            if options.with_opencl or options.with_cuda:
                if options.nnps == 'gpu_octree':
                    leaf_size = int(options.octree_leaf_size)
                    # if leaf_size % 32 != 0:
                    #     raise ValueError("GPU Octree leaf size must "
                    #                      "be a multiple of 32")

                    from pysph.base.octree_gpu_nnps import OctreeGPUNNPS
                    # Sorting enabled by default
                    print("Using elementwise: ", options.octree_elementwise)
                    nnps = OctreeGPUNNPS(
                        dim=solver.dim,
                        particles=self.particles,
                        radius_scale=kernel.radius_scale,
                        domain=self.domain,
                        cache=True,
                        sort_gids=options.sort_gids,
                        allow_sort=True,
                        leaf_size=leaf_size,
                        use_elementwise=options.octree_elementwise,
                    )
                else:
                    from pysph.base.gpu_nnps import ZOrderGPUNNPS
                    nnps = ZOrderGPUNNPS(
                        dim=solver.dim,
                        particles=self.particles,
                        radius_scale=kernel.radius_scale,
                        domain=self.domain,
                        cache=True,
                        sort_gids=options.sort_gids)

            elif options.nnps == 'box':
                nnps = BoxSortNNPS(
                    dim=solver.dim,
                    particles=self.particles,
                    radius_scale=kernel.radius_scale,
                    domain=self.domain,
                    cache=cache,
                    sort_gids=options.sort_gids)

            elif options.nnps == 'll':
                nnps = LinkedListNNPS(
                    dim=solver.dim,
                    particles=self.particles,
                    radius_scale=kernel.radius_scale,
                    domain=self.domain,
                    fixed_h=fixed_h,
                    cache=cache,
                    sort_gids=options.sort_gids)

            elif options.nnps == 'sh':
                nnps = SpatialHashNNPS(
                    dim=solver.dim,
                    particles=self.particles,
                    radius_scale=kernel.radius_scale,
                    domain=self.domain,
                    fixed_h=fixed_h,
                    cache=cache,
                    table_size=options.table_size,
                    sort_gids=options.sort_gids)

            elif options.nnps == 'esh':
                nnps = ExtendedSpatialHashNNPS(
                    dim=solver.dim,
                    particles=self.particles,
                    radius_scale=kernel.radius_scale,
                    domain=self.domain,
                    fixed_h=fixed_h,
                    cache=cache,
                    H=options.H,
                    table_size=options.table_size,
                    sort_gids=options.sort_gids,
                    approximate=options.approximate_nnps)

            elif options.nnps == 'strat_hash':
                nnps = StratifiedHashNNPS(
                    dim=solver.dim,
                    particles=self.particles,
                    radius_scale=kernel.radius_scale,
                    domain=self.domain,
                    fixed_h=fixed_h,
                    cache=cache,
                    table_size=options.table_size,
                    sort_gids=options.sort_gids,
                    num_levels=options.num_levels)

            elif options.nnps == 'strat_sfc':
                nnps = StratifiedSFCNNPS(
                    dim=solver.dim,
                    particles=self.particles,
                    radius_scale=kernel.radius_scale,
                    domain=self.domain,
                    fixed_h=fixed_h,
                    cache=cache,
                    sort_gids=options.sort_gids,
                    num_levels=options.num_levels)

            elif options.nnps == 'tree':
                nnps = OctreeNNPS(
                    dim=solver.dim,
                    particles=self.particles,
                    radius_scale=kernel.radius_scale,
                    domain=self.domain,
                    fixed_h=fixed_h,
                    cache=cache,
                    leaf_max_particles=options.leaf_max_particles,
                    sort_gids=options.sort_gids)

            elif options.nnps == 'ci':
                nnps = CellIndexingNNPS(
                    dim=solver.dim,
                    particles=self.particles,
                    radius_scale=kernel.radius_scale,
                    domain=self.domain,
                    fixed_h=fixed_h,
                    cache=cache,
                    sort_gids=options.sort_gids)

            elif options.nnps == 'sfc':
                nnps = ZOrderNNPS(
                    dim=solver.dim,
                    particles=self.particles,
                    radius_scale=kernel.radius_scale,
                    domain=self.domain,
                    fixed_h=fixed_h,
                    cache=cache,
                    sort_gids=options.sort_gids)

            elif options.nnps == 'comp_tree':
                nnps = CompressedOctreeNNPS(
                    dim=solver.dim,
                    particles=self.particles,
                    radius_scale=kernel.radius_scale,
                    domain=self.domain,
                    fixed_h=fixed_h,
                    cache=cache,
                    leaf_max_particles=options.leaf_max_particles,
                    sort_gids=options.sort_gids)

            self.nnps = nnps

        nnps = self.nnps

        # inform NNPS if it's working in parallel
        if self.num_procs > 1:
            nnps.set_in_parallel(True)

        dt = options.time_step
        if dt is not None:
            solver.set_time_step(dt)

        tf = options.final_time
        if tf is not None:
            solver.set_final_time(tf)

        solver.set_max_steps(self.options.max_steps)

        # Setup the solver output file name
        fname = options.fname

        if in_parallel():
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

        solver.set_compress_output(options.compress_output)
        # disable_output
        solver.set_disable_output(options.disable_output)

        if options.reorder_freq is None:
            if options.with_opencl or options.with_cuda:
                solver.set_reorder_freq(50)
        else:
            solver.set_reorder_freq(options.reorder_freq)

        # output print frequency
        if options.freq is not None:
            solver.set_print_freq(options.freq)

        # output printing level (default is not detailed)
        if options.detailed_output is not None:
            solver.set_output_printing_level(options.detailed_output)

        # solver output behaviour in parallel
        if options.output_dump_remote:
            solver.set_output_only_real(False)

        # output directory
        solver.set_output_directory(abspath(options.output_dir))
        self._message("Generating output in %s" % self.output_dir)

        # set parallel output mode
        solver.set_parallel_output_mode(options.parallel_output_mode)

        # Set the adaptive timestep
        if options.adaptive_timestep is not None:
            solver.set_adaptive_timestep(options.adaptive_timestep)

            # set solver cfl number
            solver.set_cfl(options.cfl)

        if options.n_damp is not None:
            solver.set_n_damp(options.n_damp)

        # setup the solver. This is where the code is compiled
        solver.setup(
            particles=self.particles,
            equations=self.equations,
            nnps=nnps,
            kernel=kernel,
            fixed_h=fixed_h)

        # add solver interfaces
        self.command_manager = CommandManager(solver, self.comm)
        solver.set_command_handler(self.command_manager.execute_commands)

        _used_ports = []
        if self._interfaces:
            self._stop_interfaces()
            self._interfaces = []

        if self.rank == 0:
            if sys.platform == 'win32':
                auto = "pysph@127.0.0.1:8800+"
                default_host = "127.0.0.1"
            else:
                auto = "pysph@0.0.0.0:8800+"
                default_host = "0.0.0.0"
            # commandline interface
            if options.cmd_line:
                from pysph.solver.solver_interfaces import CommandlineInterface
                interface = CommandlineInterface()
                self._interfaces.append(interface)
                self.command_manager.add_interface(interface.start)
                logger.info('Started Commandline interface')

            # XML-RPC interface
            if options.xml_rpc:
                from pysph.solver.solver_interfaces import XMLRPCInterface
                addr = options.xml_rpc
                idx = addr.find(':')
                host = default_host if idx == -1 else addr[:idx]
                port = int(addr[idx + 1:])
                interface = XMLRPCInterface((host, port))
                self._interfaces.append(interface)
                self.command_manager.add_interface(interface.start)
                _used_ports.append(port)
                logger.info(
                    'Started XML-RPC interface on %s:%d' % (host, port)
                )

            # python MultiProcessing interface
            if options.multiproc:
                from pysph.solver.solver_interfaces import (
                    MultiprocessingInterface
                )
                if options.multiproc == 'auto':
                    addr = auto
                else:
                    addr = options.multiproc
                idx = addr.find('@')
                authkey = "pysph" if idx == -1 else addr[:idx]
                addr = addr[idx + 1:]
                idx = addr.find(':')
                host = default_host if idx == -1 else addr[:idx]
                port = addr[idx + 1:]
                if port[-1] == '+':
                    port = get_free_port(int(port[:-1]), skip=_used_ports)
                else:
                    port = int(port)

                interface = MultiprocessingInterface(
                    (host, port), authkey.encode()
                )
                self._interfaces.append(interface)
                self.command_manager.add_interface(interface.start)

                logger.info('Started multiprocessing interface on %s:%d' %
                            (host, port))

    def _configure(self):
        """Configures the application using the options from the
        command-line.
        """
        self._configure_global_config()
        self._configure_solver()

    def _setup_parallel_manager_and_initial_load_balance(self):
        """This will automatically distribute the particles among processors
        if this is a parallel run.
        """
        # Instantiate the Parallel Manager here and do an initial LB
        num_procs = self.num_procs
        options = self.options
        solver = self.solver
        comm = self.comm

        self.parallel_manager = None
        if num_procs > 1:
            options = self.options

            if options.with_zoltan:
                if not (has_zoltan() and has_mpi()):
                    raise RuntimeError("Cannot run in parallel!")

            else:
                raise ValueError("""Sorry. You're stuck with Zoltan for now
                use the option '--with-zoltan' for parallel runs
                """)

            # create the parallel manager
            obj_weight_dim = "0"
            if options.zoltan_weights:
                obj_weight_dim = "1"

            zoltan_lb_method = options.zoltan_lb_method

            # ghost layers
            ghost_layers = options.ghost_layers

            # radius scale for the parallel update
            radius_scale = (options.parallel_scale_factor *
                            solver.kernel.radius_scale)

            self.parallel_manager = pm = ZoltanParallelManagerGeometric(
                dim=solver.dim,
                particles=self.particles,
                comm=comm,
                lb_method=zoltan_lb_method,
                obj_weight_dim=obj_weight_dim,
                ghost_layers=ghost_layers,
                update_cell_sizes=options.update_cell_sizes,
                radius_scale=radius_scale
            )

            # ## ADDITIONAL LOAD BALANCING FUNCTIONS FOR ZOLTAN ###

            # RCB lock directions
            if options.zoltan_rcb_lock_directions:
                pm.set_zoltan_rcb_lock_directions()

            if options.zoltan_rcb_reuse:
                pm.set_zoltan_rcb_reuse()

            if options.zoltan_rcb_rectilinear:
                pm.set_zoltan_rcb_rectilinear_blocks()

            if options.zoltan_rcb_set_direction > 0:
                pm.set_zoltan_rcb_directions(
                    str(options.zoltan_rcb_set_direction))

            # set zoltan options
            pm.pz.Zoltan_Set_Param("DEBUG_LEVEL", options.zoltan_debug_level)
            pm.pz.Zoltan_Set_Param("DEBUG_MEMORY", "0")

            # do an initial load balance
            pm.update()
            pm.initial_update = False

            # set subsequent load balancing frequency
            lb_freq = options.lb_freq
            if lb_freq < 1:
                raise ValueError("Invalid lb_freq %d" % lb_freq)
            pm.set_lb_freq(lb_freq)

            # wait till the initial partition is done
            comm.barrier()

        # set the solver's parallel manager
        solver.set_parallel_manager(self.parallel_manager)

    def _setup_solver_callbacks(self, obj):
        """Setup any solver callbacks given an object with any of `pre_step`,
        `post_step' and `post_stage`
        """
        if is_overloaded_method(obj.pre_step):
            self.solver.add_pre_step_callback(obj.pre_step)

        if is_overloaded_method(obj.post_stage):
            self.solver.add_post_stage_callback(obj.post_stage)

        if is_overloaded_method(obj.post_step):
            self.solver.add_post_step_callback(obj.post_step)

    def _stop_interfaces(self):
        for interface in self._interfaces:
            interface.stop()

    def _message(self, msg):
        if self.num_procs == 1:
            logger.info(msg)
            if not self.options.quiet:
                print(msg)
        elif (self.num_procs > 1 and self.rank in (0, 1)):
            s = "Rank %d: %s" % (self.rank, msg)
            logger.info(s)
            if not self.options.quiet:
                print(s)

    def _write_info(self, filename, **kw):
        """Write the information dictionary to given filename. Any extra
        keyword arguments are written to the file.
        """
        if self.rank == 0:
            info = dict(
                fname=self.fname, output_dir=self.output_dir, args=self.args)
            info.update(kw)
            with open(filename, 'w') as f:
                json.dump(info, f)

    def _write_profile_info(self):
        # Note that this is called when the run method ends and NOT
        # at exit, so any post-processing will be after this is dumped.
        fname = join(self.output_dir, 'profile_info.csv')
        data = None
        if self.num_procs > 1:
            data = dict(get_profile_info())
            data = self.comm.gather(data, root=0)
        if self.rank == 0:
            profile2csv(fname, info=data)
        if self.options.profile and self.rank == 0:
            print_profile()

    def _log_solver_info(self, solver):
        sep = '-'*70

        pa_info = {p.name: p.get_number_of_particles()
                   for p in solver.particles}
        particle_info = '\n  '.join(
            ['%s: %d' % (k, v) for k, v in pa_info.items()]
        )
        total = sum(pa_info.values())
        if len(pa_info) > 1:
            particle_info += '\n  Total: %d' % total
        p_msg = '%s\nNo of particles:\n  %s\n%s' % (sep, particle_info, sep)
        self._message(p_msg)

        kernel_name = solver.kernel.__class__.__name__
        kernel_info = '%s(dim=%s)' % (kernel_name, solver.dim)
        logger.info('Using kernel:\n%s\n  %s\n%s', sep, kernel_info, sep)

        nnps_name = self.nnps.__class__.__name__
        nnps_info = '%s(dim=%s)' % (nnps_name, solver.dim)
        logger.info('Using nnps:\n%s\n  %s\n%s', sep, nnps_info, sep)

        logger.info(
            'Using integrator:\n%s\n  %s\n%s', sep, solver.integrator, sep
        )

        equations = self.equations
        if isinstance(equations, list):
            eqn_info = '[\n' + ',\n'.join([str(e) for e in equations]) + '\n]'
        else:
            eqn_info = equations
        logger.info('Using equations:\n%s\n%s\n%s', sep, eqn_info, sep)

        logger.info("Callbacks:\n%s\n", sep)
        logger.info(
            "Pre-step callbacks:\n%s\n", repr(self.solver.pre_step_callbacks)
        )
        logger.info(
            "Post-step callbacks:\n%s\n", repr(self.solver.post_step_callbacks)
        )
        logger.info(
            "Post-stage callbacks:\n%s\n%s\n",
            repr(self.solver.post_stage_callbacks), sep
        )

    def _mayavi_config(self, code):
        """Write out the given code to a `mayavi_config.py` in the output
        directory.

        Note that this will call `textwrap.dedent` on the code.
        """
        cfg = os.path.join(self.output_dir, 'mayavi_config.py')
        if not os.path.exists(cfg):
            with open(cfg, 'w') as fp:
                fp.write(dedent(code))

    ######################################################################
    # Public interface.
    ######################################################################
    def add_tool(self, tool):
        """Add a :py:class:`pysph.solver.tools.Tool` instance to the
        application.
        """
        self._setup_solver_callbacks(tool)
        self.tools.append(tool)

    def dump_code(self, file):
        """Dump the generated code to given file.
        """
        file.write(self.solver.sph_eval.ext_mod.code)

    @property
    def info_filename(self):
        return abspath(join(self.output_dir, self.fname + '.info'))

    def initialize(self):
        """Called on the constructor, set constants etc. up here if needed.
        """
        pass

    @property
    def output_files(self):
        return get_files(self.output_dir, self.fname)

    def read_info(self, fname_or_dir):
        """Read the information from the given info file (or directory
        containing the info file, the first found info file will be used).
        """
        if isdir(fname_or_dir):
            fname_or_dir = glob.glob(join(fname_or_dir, "*.info"))[0]
        info_dir = dirname(fname_or_dir)
        with open(fname_or_dir, 'r') as f:
            info = json.load(f)

        self.args = info.get('args', self.args)
        self._parse_command_line(force=True)

        self.fname = info.get('fname', self.fname)

        output_dir = info.get('output_dir', self.output_dir)
        if realpath(info_dir) != realpath(output_dir):
            # Happens if someone moved the directory!
            self.output_dir = info_dir
            info['output_dir'] = info_dir
        else:
            self.output_dir = output_dir

        # Set the output directory of the options so it is corrected as per the
        # info file.
        self.options.output_dir = self.output_dir
        self._process_command_line()

        return info

    def run(self, argv=None):
        """Run the application.

        This basically calls ``setup()`` and then ``solve()``.

        Parameters
        ----------

        argv: list
            Optional command line arguments.  Handy when running
            interactively.
        """
        self.setup(argv)
        self.solve()

    def set_args(self, args):
        self.args = args

    def setup(self, argv=None):
        """Setup the application.

        This may be used to setup the various pieces of infrastructure to run
        an SPH simulation, for example, this will parse the command line
        arguments passed, setup the scheme, solver, equations etc. It will not
        call the solver's solve method though. This can be useful if you wish
        to manually run the solver.

        Parameters
        ----------

        argv: list
            Optional command line arguments.  Handy when running
            interactively.

        """
        if argv is not None:
            self.set_args(argv)

        if self.solver is None:
            start_time = time.time()

            self._parse_command_line(force=argv is not None)
            self._process_command_line()
            self._setup_logging()
            self._configure_global_config()

            self.solver = self.create_solver()
            msg = "Solver is None, you may have forgotten to return it!"
            assert self.solver is not None, msg
            self.equations = self.create_equations()

            self._create_particles(self.create_particles)

            # This must be done before the initial load balancing
            # as the inlets will create new particles.
            if is_overloaded_method(self.create_inlet_outlet):
                self._create_inlet_outlet(self.create_inlet_outlet)

            if self.domain is None:
                self.domain = self.create_domain()

            self.nnps = self.create_nnps()

            self._configure_solver()

            self._setup_solver_callbacks(self)
            for tool in self.create_tools():
                self.add_tool(tool)

            self._log_solver_info(self.solver)

            end_time = time.time()
            setup_duration = end_time - start_time
            self._message("Setup took: %.5f secs" % (setup_duration))
            self._write_info(self.info_filename, completed=False, cpu_time=0)

        self.customize_output()

    def solve(self):
        """This runs the solver.

        Note that this method expects that ``setup`` has already been called.

        Don't use this method unless you really know what you are doing.
        """
        start_time = time.time()
        self.solver.solve(not self.options.quiet)
        end_time = time.time()
        run_duration = end_time - start_time
        self._message("Run took: %.5f secs" % (run_duration))
        self._write_info(
            self.info_filename, completed=True, cpu_time=run_duration
        )

        self._stop_interfaces()
        self._write_profile_info()

    ######################################################################
    # User methods that could be overloaded.
    ######################################################################
    def add_user_options(self, group):
        """Add any user-defined options to the given option group.

        Note
        ----

        This uses the `argparse` module.
        """
        pass

    def configure_scheme(self):
        """This is called after :py:meth:`consume_user_options` is called.
        One can configure the SPH scheme here as at this point all the command
        line options are known.
        """
        pass

    def consume_user_options(self):
        """This is called right after the command line arguments are parsed.

        All the parsed options are available in ``self.options`` and can be
        used in this method.

        This is meant to be overridden by users to setup any internal variables
        etc. that depend on the command line arguments passed.  Note that this
        method is called well before the solver or particles are created.
        """
        pass

    def create_domain(self):
        """Create a `pysph.base.nnps_base.DomainManager` and return it if
        needed.

        This is used for periodic domains etc.  Note that if the domain
        is passed to :py:meth:`__init__`, then this method is not called.

        """
        return None

    def create_inlet_outlet(self, particle_arrays):
        """Create inlet and outlet objects and return them as a list.

        The method is passed a dictionary of particle arrays keyed on the name
        of the particle array.
        """
        pass

    def create_equations(self):
        """Create the equations to be used and return them.
        """
        if self.scheme is not None:
            return self.scheme.get_equations()
        else:
            msg = "Application.create_equations method must be overloaded."
            raise NotImplementedError(msg)

    def create_nnps(self):
        """Create any NNPS if desired and return it, else a default NNPS will
        be created automatically.
        """
        return None

    def create_particles(self):
        """Create particle arrays and return a list of them.
        """
        message = "Application.create_particles method must be overloaded."
        raise NotImplementedError(message)

    def create_scheme(self):
        """Create a suitable SPH scheme and return it.

        Note that this method is called after the arguments are all
        processed and after :py:meth:`consume_user_options` is called.
        """
        return None

    def create_solver(self):
        """Create the solver and return it.
        """
        if self.scheme is not None:
            return self.scheme.get_solver()
        else:
            msg = "Application.create_solver method must be overloaded."
            raise NotImplementedError(msg)

    def create_tools(self):
        """Create any tools and return a sequence of them.  This method is
        called after particles/inlets etc. are all setup, configured etc.
        """
        return []

    def customize_output(self):
        """Customize the output file visualization by adding any files.

        For example, the pysph view command will look for a
        ``mayavi_config.py`` file that can be used to script the viewer. You
        can use self._mayavi_config('code') to add a default customization
        here.

        Note that this is executed before the simulation starts.
        """
        pass

    def pre_step(self, solver):
        """If overloaded, this is called automatically before each integrator
        step.  The method is passed the solver instance.
        """
        pass

    def post_stage(self, current_time, dt, stage):
        """If overloaded, this is called automatically after each integrator
        stage, i.e. if the integrator is a two stage integrator it will be
        called after the first and second stages.

        The method is passed (current_time, dt, stage). See the the
        :py:meth:`pysph.sph.integrator.Integrator.one_timestep` methods for
        examples of how this is called.

        """
        pass

    def post_step(self, solver):
        """If overloaded, this is called automatically after each integrator
        step.  The method is passed the solver instance.
        """
        pass

    def post_process(self, info_fname_or_directory):
        """Given an info filename or a directory containing the info file, read
        the information and do any post-processing of the results.  Please
        overload the method to perform any processing.

        The info file has a few useful attributes and can be read using the
        :py:meth:`read_info` method.

        The `output_files` property should provide the output files
        generated.
        """
        print('Overload this method to post-process the results.')
