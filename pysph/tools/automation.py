import itertools
import glob
import json
import os
import shutil
import subprocess
import numpy as np


class PySPHRunner(object):
    """Convenience class to run a PySPH simulation via an automation
    framework.  The class provides a method to run the simulation and
    also check if the simulation is completed.
    """
    def __init__(self, command, output_dir):
        if isinstance(command, str):
            self.command = command.split()
        else:
            self.command = command
        self.output_dir = output_dir

    ###### Public protocol ###########################################

    def is_done(self, *ignored, **kw_ignored):
        """Returns True if the simulation completed.

        The extra arguments are not used but exist for compatibility
        with automation toolkits.
        """
        if not os.path.exists(self.output_dir):
            return False
        info_fname = self._get_info_filename()
        if not info_fname or not os.path.exists(info_fname):
            return False
        d = json.load(open(info_fname))
        return d.get('completed')

    def run(self):
        """Actually run the command.
        """
        cmd = self._full_command()
        print("Running: %s"%' '.join(cmd))
        subprocess.call(cmd)

    def clean(self):
        """Clean out any generated results.

        This completely removes the output directory.

        """
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    ###### Private protocol ###########################################

    def _full_command(self):
        return self.command + ['-d', self.output_dir]

    def _get_info_filename(self):
        files = glob.glob(os.path.join(self.output_dir, '*.info'))
        if len(files) > 0:
            return files[0]
        else:
            return None


class Problem(object):
    """This class represents a numerical problem or computational
    problem of interest that needs to be solved.

    The class helps one run a variety of commands (or simulations),
    and then assemble/compare the results from those in the `run`
    method.  This is perhaps easily understood with an example.  Let
    us say one wishes to run the elliptical drop example problem with
    the standard SPH and TVF and compare the results and their
    convergence properties while also keep track of the computational
    time.  To do this one will have to run several simulations, then
    collect and process the results.  This is achieved by subclassing
    this class and implementing the following methods:

     - `get_name(self)`: returns a string of the name of the problem.  All
       results and simulations are collected inside a directory with
       this name.
     - `get_commands(self)`: returns a sequence of (directory_name,
       command_string) pairs.  These are to be exeuted before the
       `run` method is called.
     - `run(self)`: Processes the completed simulations to make plots etc.

    See the `EllipticalDrop` example class below to see a full implementation.

    """
    def __init__(self, simulation_dir, output_dir):
        """Constructor.

        Parameters
        ----------

        simulation_dir : str : directory where simulation output goes.
        output_dir : str : directory where outputs from `run` go.
        """
        self.out_dir = output_dir
        self.sim_dir = simulation_dir
        self.setup()

    ###### Public protocol ###########################################

    def input_path(self, *args):
        """Given any arguments, relative to the simulation dir, return
        the absolute path.
        """
        return os.path.join(self.sim_dir, self.get_name(), *args)

    def output_path(self, *args):
        """Given any arguments relative to the output_dir return the
        absolute path.
        """
        return os.path.join(self.out_dir, self.get_name(), *args)

    def setup(self):
        """Called by init, so add any initialization here.
        """
        pass

    def get_requires(self):
        """Used by task runners like doit/luigi to run required
        commands.
        """
        base = self.get_name()
        result = []
        for name, cmd in self.get_commands():
            sim_output_dir = self.input_path(name)
            runner = PySPHRunner(cmd, sim_output_dir)
            task_name = '%s.%s'%(base, name)
            result.append((task_name, runner))
        return result

    def make_output_dir(self):
        """Convenience to make the output directory if needed.
        """
        base = self.output_path()
        if not os.path.exists(base):
            os.makedirs(base)

    def get_name(self):
        """Return the name of this problem, this name is used as a
        directory for the simulation and the outputs.
        """
        raise NotImplementedError()

    def get_commands(self):
        """Return a sequence of (name, command_string), where name
        represents the command being run.
        """
        return []

    def get_outputs(self):
        """Get a list of outputs generated by this problem.  By default it
        returns the output directory (as a single element of a list).
        """
        return [self.output_path()]

    def run(self):
        """Run any analysis code for the simulations completed.  This
        is usually run after the simulation commands are completed.
        """
        raise NotImplementedError()

    def clean(self):
        """Cleanup any generated output from the analysis code.  This does not
        clean the output of any nested commands.
        """
        for path in self.get_outputs():
            if os.path.exists(path):
                if os.path.isdir(path):
                    shutil.rmtree(path)
                elif os.path.isfile(path):
                    os.remove(path)


def key_to_option(key):
    """Convert a dictionary key to a valid command line option.  This simply
    replaces underscores with dashes.
    """
    return key.replace('_', '-')

def kwargs_to_command_line(kwargs):
    """Convert a dictionary of keyword arguments to a list of command-line
    options.  If the value of the key is None, no value is passed.

    Examples
    --------

    >>> sorted(kwargs_to_command_line(dict(some_arg=1, something_else=None)))
    ['--some-arg=1', '--something-else']
    """
    cmd_line = []
    for key, value in kwargs.items():
        option = key_to_option(key)
        if value is None:
            arg = "--{option}".format(option=option)
        else:
            arg = "--{option}={value}".format(
                option=option, value=str(value)
            )

        cmd_line.append(arg)
    return cmd_line


def linestyles():
    """Cycles over a set of possible linestyles to use for plotting.
    """
    ls = [dict(color=x[0], linestyle=x[1]) for x in
          itertools.product("kbgr", ["-", "--", "-.", ":"])]
    return itertools.cycle(ls)


class Simulation(object):
    """A convenient class to abstract code for a particular simulation.
    Simulation objects are typically created by ``Problem`` instances in order
    to abstract and simulate repetitive code for a particular simulation.

    For example if one were comparing the elliptical_drop example, one could
    instantiate a Simulation object as follows::

        >>> s = Simlation('outputs/sph', 'pysph run elliptical_drop')

    One can pass any additional command line arguments as follows::

        >>> s = Simlation(
        ...     'outputs/sph', 'pysph run elliptical_drop', timestep=0.005
        ... )
        >>> s.command
        'pysph run elliptical_drop --timestep=0.001'
        >>> s.input_path('results.npz')
        'outputs/sph/results.npz'

    The extra parameters can be used to filter and compare different
    simulations.  One can define additional plot methods for a particular
    subclass and use these to easily plot results for different cases.

    The object has other methods that are convenient when comparing plots.
    Along with the ``compare_cases``, ``filter_cases`` and ``filter_by_name``
    this is an extremely powerful way to automate and compare results.

    """
    def __init__(self, root, base_command, **kw):
        """Constructor

        Parameters
        ----------

        root: str
            Path to simulation output directory.
        base_command: str
            Base command to run.
        **kw: dict
            Additional parameters to pass to command.
        """
        self.root = root
        self.name = os.path.basename(root)
        self.base_command = base_command
        self.params = dict(kw)
        self._results = None

    def input_path(self, *args):
        """Given any arguments, relative to the simulation dir, return
        the absolute path.
        """
        return os.path.join(self.root, *args)

    @property
    def command(self):
        return self.base_command + ' ' + self.get_command_line_args()

    @property
    def data(self):
        if self._results is None:
            self._results = np.load(self.input_path('results.npz'))
        return self._results

    def get_labels(self, labels):
        render = self.render_parameter
        if isinstance(labels, str):
            return render(labels)
        else:
            s = [render(x) for x in labels]
            s = [x for x in s if len(x) > 0]
            return r', '.join(s)

    def kwargs_to_command_line(self, kwargs):
        return kwargs_to_command_line(kwargs)

    def get_command_line_args(self):
        return ' '.join(self.kwargs_to_command_line(self.params))

    def render_parameter(self, param):
        """Return string to be used for labels for given parameter.
        """
        if param not in self.params:
            return ''
        value = self.params[param]
        if value is None:
            return r'%s'%param
        else:
            return r'%s=%s'%(param, self.params[param])


def compare_runs(sims, method, labels, exact=None):
    """Given a sequence of Simulation instances, a method name, the labels to
    compare and an optional method name for an exact solution, this calls the
    methods with the appropriate parameters for each simulation.

    Parameters
    ----------

    sims: sequence
        Sequence of `Simulation` objects.
    method: str
        Name of a method on each simulation method to call for plotting.
    labels: sequence
        Sequence of parameters to use as labels for the plot.
    exact: str
        Name of a method that produces an exact solution plot.
    """
    ls = linestyles()
    if exact is not None:
        getattr(sims[0], exact)(**next(ls))
    for s in sims:
        m = getattr(s, method)
        m(label=s.get_labels(labels), **next(ls))

def filter_cases(runs, **params):
    """Given a sequence of simulations and any additional parameters, filter
    out all the cases having exactly those parameters and return a list of
    them.
    """
    def _check_match(run):
        for param, expected in params.items():
            if param not in run.params or run.params[param] != expected:
                return False
        return True

    return list(filter(_check_match, runs))

def filter_by_name(cases, names):
    """Filter a sequence of Simulations by their names.  That is, if the case
    has a name contained in the given `names`, it will be selected.
    """
    return sorted(
        [x for x in cases if x.name in names],
        key=lambda x: names.index(x.name)
    )
