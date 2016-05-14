"""A particle viewer using Mayavi.

This code uses the :py:class:`MultiprocessingClient` solver interface to
communicate with a running solver and displays the particles using
Mayavi.  It can also display a list of supplied files or a directory.
"""

from functools import reduce
import glob
import sys
import math
import numpy
import os
import os.path

if not os.environ.get('ETS_TOOLKIT'):
    # Set the default toolkit to qt4 unless the user has explicitly
    # set the default manually via the env var.
    try:
        from traits.etsconfig.api import ETSConfig
    except ImportError:
        from enthought.etsconfig.api import ETSConfig
    ETSConfig.toolkit = 'qt4'

try:
    from traits.api import (Any, Array, Dict, HasTraits, Instance,
        on_trait_change, List, Str, Int, Range, Float, Bool, Button,
        Directory, Event, Password, Property, cached_property)
    from traitsui.api import (View, Item, Group, Handler, HSplit, ListEditor,
        EnumEditor, TitleEditor, HGroup, ShellEditor, VSplit)
    from mayavi.core.api import PipelineBase
    from mayavi.core.ui.api import (MayaviScene, SceneEditor, MlabSceneModel)
    from pyface.timer.api import Timer, do_later
    from tvtk.api import tvtk
    from tvtk.array_handler import array2vtk
except ImportError:
    from enthought.traits.api import (Any, Array, Dict, HasTraits,
        Instance, on_trait_change, List, Str, Int, Range, Float, Bool, Button,
        Directory, Event, Password, Property, cached_property)
    from enthought.traits.ui.api import (View, Item, Group, Handler,
        HSplit, ListEditor, EnumEditor, TitleEditor, HGroup, ShellEditor,
        VSplit)
    from enthought.mayavi.core.api import PipelineBase
    from enthought.mayavi.core.ui.api import (MayaviScene, SceneEditor,
                                              MlabSceneModel)
    from enthought.pyface.timer.api import Timer, do_later
    from enthought.tvtk.api import tvtk
    from enthought.tvtk.array_handler import array2vtk

from pysph.base.particle_array import ParticleArray
from pysph.solver.solver_interfaces import MultiprocessingClient
from pysph.solver.utils import load, dump, output_formats
from pysph.solver.utils import remove_irrelevant_files, _sort_key
from pysph.tools.interpolator import (get_bounding_box, get_nx_ny_nz,
    Interpolator)


import logging
logger = logging.getLogger()

def set_arrays(dataset, particle_array):
    """ Code to add all the arrays to a dataset given a particle array."""
    props = set(particle_array.properties.keys())
    # Add the vector data.
    vec = numpy.empty((len(particle_array.x), 3), dtype=float)
    vec[:,0] = particle_array.u
    vec[:,1] = particle_array.v
    vec[:,2] = particle_array.w
    va = tvtk.to_tvtk(array2vtk(vec))
    va.name = 'velocity'
    dataset.data.point_data.add_array(vec)
    # Now add the scalar data.
    scalars = props - set(('u', 'v', 'w'))
    for sc in scalars:
        arr = particle_array.get(sc)
        va = tvtk.to_tvtk(array2vtk(arr))
        va.name = sc
        dataset.data.point_data.add_array(va)
    dataset._update_data()

def glob_files(fname):
    """Glob for all similar files given one of them.

    This assumes that the files are of the form *_[0-9]*.*.
    """
    fbase = fname[:fname.rfind('_')+1]
    ext = fname[fname.rfind('.'):]
    return glob.glob("%s*%s"%(fbase, ext))

def sort_file_list(files):
    """Given a list of input files, sort them in serial order, in-place.
    """
    files[:] = remove_irrelevant_files(files)
    files.sort(key=_sort_key)
    return files


##############################################################################
# `InterpolatorView` class.
##############################################################################
class InterpolatorView(HasTraits):

    # The bounds on which to interpolate.
    bounds = Array(cols=3, dtype=float,
                   desc='spatial bounds for the interpolation '\
                        '(xmin, xmax, ymin, ymax, zmin, zmax)')

    # The number of points to interpolate onto.
    num_points = Int(100000, enter_set=True, auto_set=False,
                     desc='number of points on which to interpolate')

    # The particle arrays to interpolate from.
    particle_arrays = List

    # The scalar to interpolate.
    scalar = Str('rho', desc='name of the active scalar to view')

    # Sync'd trait with the scalar lut manager.
    show_legend = Bool(False, desc='if the scalar legend is to be displayed')

    # Enable/disable the interpolation
    visible = Bool(False, desc='if the interpolation is to be displayed')

    # A button to use the set bounds.
    set_bounds = Button('Set Bounds')

    # A button to recompute the bounds.
    recompute_bounds = Button('Recompute Bounds')

    #### Private traits. ######################################################

    # The interpolator we are a view for.
    interpolator = Instance(Interpolator)

    # The mlab plot for this particle array.
    plot = Instance(PipelineBase)

    scalar_list = List

    scene = Instance(MlabSceneModel)

    source = Instance(PipelineBase)

    _arrays_changed = Bool(False)

    #### View definition ######################################################
    view = View(Item(name='visible'),
                Item(name='scalar',
                     editor=EnumEditor(name='scalar_list')
                    ),
                Item(name='num_points'),
                Item(name='bounds'),
                Item(name='set_bounds', show_label=False),
                Item(name='recompute_bounds', show_label=False),
                Item(name='show_legend'),
                )

    #### Private protocol  ###################################################
    def _change_bounds(self):
        interp = self.interpolator
        if interp is not None:
            interp.set_domain(self.bounds, self.interpolator.shape)
            self._update_plot()

    def _setup_interpolator(self):
        if self.interpolator is None:
            interpolator = Interpolator(
                self.particle_arrays, num_points=self.num_points
            )
            self.bounds = interpolator.bounds
            self.interpolator = interpolator
        else:
            if self._arrays_changed:
                self.interpolator.update_particle_arrays(self.particle_arrays)
                self._arrays_changed = False

    #### Trait handlers  #####################################################
    def _particle_arrays_changed(self, pas):
        if len(pas) > 0:
            all_props = reduce(set.union,
                               [set(x.properties.keys()) for x in pas])
        else:
            all_props = set()
        self.scalar_list = list(all_props)
        self._arrays_changed = True
        self._update_plot()

    def _num_points_changed(self, value):
        interp = self.interpolator
        if interp is not None:
            bounds = self.interpolator.bounds
            shape = get_nx_ny_nz(value, bounds)
            interp.set_domain(bounds, shape)
            self._update_plot()

    def _recompute_bounds_fired(self):
        bounds = get_bounding_box(self.particle_arrays)
        self.bounds = bounds
        self._change_bounds()

    def _set_bounds_fired(self):
        self._change_bounds()

    def _bounds_default(self):
        return [0, 1, 0, 1, 0, 1]

    @on_trait_change('scalar, visible')
    def _update_plot(self):
        if self.visible:
            mlab = self.scene.mlab
            self._setup_interpolator()
            interp = self.interpolator
            prop = interp.interpolate(self.scalar)
            if self.source is None:
                src = mlab.pipeline.scalar_field(
                    interp.x, interp.y, interp.z, prop
                )
                self.source = src
            else:
                self.source.mlab_source.reset(
                    x=interp.x, y=interp.y, z=interp.z, scalars=prop
                )
            src = self.source

            if self.plot is None:
                if interp.dim == 3:
                    plot = mlab.pipeline.scalar_cut_plane(src)
                else:
                    plot = mlab.pipeline.surface(src)
                self.plot = plot
                scm = plot.module_manager.scalar_lut_manager
                scm.set(show_legend=self.show_legend,
                        use_default_name=False,
                        data_name=self.scalar)
                self.sync_trait('show_legend', scm, mutual=True)
            else:
                self.plot.visible = True
                scm = self.plot.module_manager.scalar_lut_manager
                scm.data_name = self.scalar
        else:
            if self.plot is not None:
                self.plot.visible = False


##############################################################################
# `ParticleArrayHelper` class.
##############################################################################
class ParticleArrayHelper(HasTraits):
    """
    This class manages a particle array and sets up the necessary
    plotting related information for it.
    """

    # The particle array we manage.
    particle_array = Instance(ParticleArray)

    # The name of the particle array.
    name = Str

    # Current time.
    time = Float(0.0)

    # The active scalar to view.
    scalar = Str('rho', desc='name of the active scalar to view')

    # The mlab plot for this particle array.
    plot = Instance(PipelineBase)

    # List of available scalars in the particle array.
    scalar_list = List(Str)

    scene = Instance(MlabSceneModel)

    # Sync'd trait with the scalar lut manager.
    show_legend = Bool(False, desc='if the scalar legend is to be displayed')

    # Sync'd trait with the dataset to turn on/off visibility.
    visible = Bool(True, desc='if the particle array is to be displayed')

    # Show the time of the simulation on screen.
    show_time = Bool(False, desc='if the current time is displayed')

    # Do we show the hidden arrays?
    show_hidden_arrays = Bool(False,
                              desc='if hidden arrays are to be listed')

    # Private attribute to store the Text module.
    _text = Instance(PipelineBase)

    # Extra scalars to show.  These will be added and saved to the data if
    # needed.
    extra_scalars = List(Str)

    # Set to True when the particle array is updated with a new property say.
    updated = Event

    ########################################
    # View related code.
    view = View(Item(name='name',
                     show_label=False,
                     editor=TitleEditor()),
                Group(
                      Item(name='visible'),
                      Item(name='show_hidden_arrays'),
                      Item(name='scalar',
                           editor=EnumEditor(name='scalar_list')
                          ),
                      Item(name='show_legend'),
                      Item(name='show_time'),
                      ),
                )

    #######  Private protocol ############################################
    def _add_vmag(self, pa):
        if 'vmag' not in pa.properties:
            if 'vmag2' in pa.output_property_arrays:
                vmag = numpy.sqrt(pa.vmag2)
            else:
                vmag = numpy.sqrt(pa.u**2 + pa.v**2 + pa.w**2)
            pa.add_property(name='vmag', data=vmag)
            if len(pa.output_property_arrays) > 0:
                # We do not call add_output_arrays when the default is empty
                # as if it is empty, all arrays are saved anyway. However,
                # adding just vmag in this case will mean that when the
                # particle array is saved it will only save vmag!  This is
                # not what we want, hence we add vmag *only* if the
                # output_property_arrays is non-zero length.
                pa.add_output_arrays(['vmag'])
            self.updated = True

    def _get_scalar(self, pa, scalar):
        """Return the requested scalar from the given particle array.
        """
        if scalar in self.extra_scalars:
            method_name = '_add_' + scalar
            method = getattr(self, method_name)
            method(pa)
        return getattr(pa, scalar)

    #######  Traits handlers #############################################
    def _particle_array_changed(self, pa):
        self.name = pa.name
        # Setup the scalars.
        self._show_hidden_arrays_changed(self.show_hidden_arrays)

        # Update the plot.
        x, y, z = pa.x, pa.y, pa.z
        s = self._get_scalar(pa, self.scalar)
        p = self.plot
        mlab = self.scene.mlab
        if p is None:
            src = mlab.pipeline.scalar_scatter(x, y, z, s)
            p = mlab.pipeline.glyph(src, mode='point', scale_mode='none')
            p.actor.property.point_size = 3
            scm = p.module_manager.scalar_lut_manager
            scm.set(show_legend=self.show_legend,
                    use_default_name=False,
                    data_name=self.scalar)
            self.sync_trait('visible', p.mlab_source.m_data,
                             mutual=True)
            self.sync_trait('show_legend', scm, mutual=True)
            #set_arrays(p.mlab_source.m_data, pa)
            self.plot = p
        else:
            if len(x) == len(p.mlab_source.x):
                p.mlab_source.set(x=x, y=y, z=z, scalars=s)
            else:
                p.mlab_source.reset(x=x, y=y, z=z, scalars=s)

        # Setup the time.
        self._show_time_changed(self.show_time)

    def _scalar_changed(self, value):
        p = self.plot
        if p is not None:
            p.mlab_source.scalars = self._get_scalar(
                self.particle_array, value
            )
            p.module_manager.scalar_lut_manager.data_name = value

    def _show_hidden_arrays_changed(self, value):
        pa = self.particle_array
        sc_list = pa.properties.keys()
        if value:
            self.scalar_list = sorted(set(sc_list + self.extra_scalars))
        else:
            self.scalar_list = sorted(
                set(
                    [x for x in sc_list if not x.startswith('_')] +
                    self.extra_scalars
                )
            )

    def _show_time_changed(self, value):
        txt = self._text
        mlab = self.scene.mlab
        if value:
            if txt is not None:
                txt.visible = True
            elif self.plot is not None:
                mlab.get_engine().current_object = self.plot
                txt = mlab.text(0.01, 0.01, 'Time = 0.0',
                                width=0.35)
                self._text = txt
                self._time_changed(self.time)
        else:
            if txt is not None:
                txt.visible = False

    def _time_changed(self, value):
        txt = self._text
        if txt is not None:
            txt.text = 'Time = %.3e'%(value)

    def _extra_scalars_default(self):
        return ['vmag']


class PythonShellView(HasTraits):
    ns = Dict()
    view = View(Item('ns', editor=ShellEditor(), show_label=False))


class ViewerHandler(Handler):

    def closed(self, info, is_ok):
        """Call the viewer's on_close method when the UI is closed.
        """
        info.object.on_close()


##############################################################################
# `MayaviViewer` class.
##############################################################################
class MayaviViewer(HasTraits):
    """
    This class represents a Mayavi based viewer for the particles.  They
    are queried from a running solver.
    """

    particle_arrays = List(Instance(ParticleArrayHelper), [])
    pa_names = List(Str, [])

    interpolator = Instance(InterpolatorView)

    # The default scalar to load up when running the viewer.
    scalar = Str("rho")

    scene = Instance(MlabSceneModel, ())

    ########################################
    # Traits to pull data from a live solver.
    live_mode = Bool(False, desc='if data is obtained from a running solver '\
                                 'or from saved files')

    shell = Button('Launch Python Shell')
    host = Str('localhost', desc='machine to connect to')
    port = Int(8800, desc='port to use to connect to solver')
    authkey = Password('pysph', desc='authorization key')
    host_changed = Bool(True)
    client = Instance(MultiprocessingClient)
    controller = Property(depends_on='live_mode, host_changed')

    ########################################
    # Traits to view saved solver output.
    files = List(Str, [])
    directory = Directory()
    current_file = Str('', desc='the file being viewed currently')
    update_files = Button('Refresh')
    file_count = Range(low='_low', high='_n_files', value=0,
                       desc='the file counter')
    play = Bool(False, desc='if all files are played automatically')
    play_delay = Float(0.2, desc='the delay between loading files')
    loop = Bool(False, desc='if the animation is looped')
    # This is len(files) - 1.
    _n_files = Int(0)
    _low = Int(0)

    ########################################
    # Timer traits.
    timer = Instance(Timer)
    interval = Range(0.5, 20.0, 2.0,
                     desc='frequency in seconds with which plot is updated')

    ########################################
    # Solver info/control.
    current_time = Float(0.0, desc='the current time in the simulation')
    time_step = Float(0.0, desc='the time-step of the solver')
    iteration = Int(0, desc='the current iteration number')
    pause_solver = Bool(False, desc='if the solver should be paused')

    ########################################
    # Movie.
    record = Bool(False, desc='if PNG files are to be saved for animation')
    frame_interval = Range(1, 100, 5, desc='the interval between screenshots')
    movie_directory = Str
    # internal counters.
    _count = Int(0)
    _frame_count = Int(0)
    _last_time = Float
    _solver_data = Any
    _file_name = Str
    _particle_array_updated = Bool

    ########################################
    # The layout of the dialog created
    view = View(HSplit(
                  Group(
                    Item(name='live_mode'),
                    Group(
                        Group(
                            Item(name='directory'),
                            Item(name='current_file'),
                            Item(name='file_count'),
                            HGroup(Item(name='play'),
                                   Item(name='play_delay',
                                        label='Delay',
                                        resizable=True),
                                   Item(name='loop'),
                                   Item(name='update_files',
                                        show_label=False),
                                   padding=0,
                               ),
                            label='Saved Data',
                            selected=True,
                            enabled_when='not live_mode',
                            ),
                        Group(
                            Item(name='host'),
                            Item(name='port'),
                            Item(name='authkey'),
                            label='Connection',
                            enabled_when='live_mode',
                            ),
                        layout='tabbed'
                        ),
                    Group(
                        Group(
                              Item(name='current_time'),
                              Item(name='time_step'),
                              Item(name='iteration'),
                              Item(name='pause_solver',
                                   enabled_when='live_mode'
                                   ),
                              Item(name='interval',
                                   enabled_when='not live_mode'
                                   ),
                              label='Solver',
                             ),
                        Group(
                              Item(name='record'),
                              Item(name='frame_interval'),
                              Item(name='movie_directory'),
                              label='Movie',
                            ),
                        layout='tabbed',

                        ),
                    Group(
                          Item(name='particle_arrays',
                               style='custom',
                               show_label=False,
                               editor=ListEditor(use_notebook=True,
                                                 deletable=False,
                                                 page_name='.name'
                                                 )
                               ),
                          Item(name='interpolator',
                               style='custom',
                               show_label=False),
                          layout='tabbed'
                         ),
                    Item(name='shell', show_label=False),
                  ),
                  Group(
                    Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                            height=400, width=600, show_label=False),
                  )
                ),
                resizable=True,
                title='PySPH Particle Viewer',
                height=640,
                width=1024,
                handler=ViewerHandler
                )

    ######################################################################
    # `MayaviViewer` interface.
    ######################################################################
    def on_close(self):
        self._handle_particle_array_updates()

    @on_trait_change('scene:activated')
    def start_timer(self):
        if not self.live_mode:
            # No need for the timer if we are rendering files.
            return

        # Just accessing the timer will start it.
        t = self.timer
        if not t.IsRunning():
            t.Start(int(self.interval*1000))

    @on_trait_change('scene:activated')
    def update_plot(self):

        # No need to do this if files are being used.
        if not self.live_mode:
            return

        # do not update if solver is paused
        if self.pause_solver:
            return

        if self.client is None:
            self.host_changed = True

        controller = self.controller
        if controller is None:
            return

        self.current_time = t = controller.get_t()
        self.time_step = controller.get_dt()
        self.iteration = controller.get_count()

        arrays = []
        for idx, name in enumerate(self.pa_names):
            pa = controller.get_named_particle_array(name)
            arrays.append(pa)
            pah = self.particle_arrays[idx]
            pah.set(particle_array=pa, time=t)

        self.interpolator.particle_arrays = arrays

        if self.record:
            self._do_snap()

    ######################################################################
    # Private interface.
    ######################################################################
    def _do_snap(self):
        """Generate the animation."""
        p_arrays = self.particle_arrays
        if len(p_arrays) == 0:
            return
        if self.current_time == self._last_time:
            return

        if len(self.movie_directory) == 0:
            controller = self.controller
            output_dir = controller.get_output_directory()
            movie_dir = os.path.join(output_dir, 'movie')
            self.movie_directory = movie_dir
        else:
            movie_dir = self.movie_directory
        if not os.path.exists(movie_dir):
            os.mkdir(movie_dir)

        interval = self.frame_interval
        count = self._count
        if count%interval == 0:
            fname = 'frame%06d.png'%(self._frame_count)
            p_arrays[0].scene.save_png(os.path.join(movie_dir, fname))
            self._frame_count += 1
            self._last_time = self.current_time
        self._count += 1

    @on_trait_change('host,port,authkey')
    def _mark_reconnect(self):
        if self.live_mode:
            self.host_changed = True

    @cached_property
    def _get_controller(self):
        ''' get the controller, also sets the iteration count '''
        if not self.live_mode:
            return None

        reconnect = self.host_changed
        if not reconnect:
            try:
                c = self.client.controller
            except Exception as e:
                logger.info('Error: no connection or connection closed: '\
                        'reconnecting: %s'%e)
                reconnect = True
                self.client = None
            else:
                try:
                    self.client.controller.get_count()
                except IOError:
                    self.client = None
                    reconnect = True

        if reconnect:
            self.host_changed = False
            try:
                if MultiprocessingClient.is_available((self.host, self.port)):
                    self.client = MultiprocessingClient(
                        address=(self.host, self.port),
                        authkey=self.authkey
                    )
                else:
                    logger.info(
                        'Could not connect: Multiprocessing Interface'\
                        ' not available on %s:%s'%(self.host,self.port)
                    )
                    return None
            except Exception as e:
                logger.info('Could not connect: check if solver is '\
                            'running:%s'%e)
                return None
            c = self.client.controller
            self.iteration = c.get_count()

        if self.client is None:
            return None
        else:
            return self.client.controller

    def _client_changed(self, old, new):
        if not self.live_mode:
            return

        self._clear()
        if new is None:
            return
        else:
            self.pa_names = self.client.controller.get_particle_array_names()

        self.particle_arrays = [
            self._make_particle_array_helper(self.scene, x)
            for x in self.pa_names
        ]
        self.interpolator = InterpolatorView(scene=self.scene)
        # Turn on the legend for the first particle array.
        if len(self.particle_arrays) > 0:
            self.particle_arrays[0].set(show_legend=True, show_time=True)

    def _timer_event(self):
        # catch all Exceptions else timer will stop
        try:
            self.update_plot()
        except Exception as e:
            logger.info('Exception: %s caught in timer_event'%e)

    def _interval_changed(self, value):
        t = self.timer
        if t is None:
            return
        if t.IsRunning():
            t.Stop()
            t.Start(int(value*1000))

    def _timer_default(self):
        return Timer(int(self.interval*1000), self._timer_event)

    def _pause_solver_changed(self, value):
        if self.live_mode:
            c = self.controller
            if c is None:
                return
            if value:
                c.pause_on_next()
            else:
                c.cont()

    def _record_changed(self, value):
        if value:
            self._do_snap()

    def _files_changed(self, value):
        if len(value) == 0:
            return
        else:
            d = os.path.dirname(os.path.abspath(value[0]))
            self.movie_directory = os.path.join(d, 'movie')
            self.set(directory=d, trait_change_notify=False)
        self._n_files = len(value) - 1
        self.frame_interval = 1
        fc = self.file_count
        self.file_count = 0
        if fc == 0:
            # Force an update when our original file count is 0.
            self._file_count_changed(fc)
        t = self.timer
        if not self.live_mode:
            if t.IsRunning():
                t.Stop()
        else:
            if not t.IsRunning():
                t.Stop()
                t.Start(self.interval*1000)

    def _file_count_changed(self, value):
        # Save out any updates for the previous file if needed.
        self._handle_particle_array_updates()
        # Load the new file.
        fname = self.files[value]
        self._file_name = fname
        self.current_file = os.path.basename(fname)
        # Code to read the file, create particle array and setup the helper.
        data = load(fname)
        solver_data = data["solver_data"]
        arrays = data["arrays"]
        self._solver_data = solver_data
        self.current_time = t = float(solver_data['t'])
        self.time_step = float(solver_data['dt'])
        self.iteration = int(solver_data['count'])
        names = list(arrays.keys())
        pa_names = self.pa_names

        if len(pa_names) == 0:
            self.interpolator = InterpolatorView(scene=self.scene)
            self.pa_names = names
            pas = []
            for name in names:
                pa = arrays[name]
                pah = self._make_particle_array_helper(self.scene, name)
                # Must set this after setting the scene.
                pah.set(particle_array=pa, time=t)
                pas.append(pah)
            self.particle_arrays = pas
        else:
            for idx, name in enumerate(pa_names):
                pa = arrays[name]
                pah = self.particle_arrays[idx]
                pah.set(particle_array=pa, time=t)

        self.interpolator.particle_arrays = list(arrays.values())

        if self.record:
            self._do_snap()

    def _loop_changed(self, value):
        if value and self.play:
            self._play_changed(self.play)

    def _play_changed(self, value):
        t = self.timer
        if value:
            t.Stop()
            t.callable = self._play_event
            t.Start(1000*self.play_delay)
        else:
            t.Stop()
            t.callable = self._timer_event

    def _clear(self):
        self.pa_names = []
        self.scene.mayavi_scene.children[:] = []

    def _play_event(self):
        nf = self._n_files
        pc = self.file_count
        pc += 1
        if pc > nf:
            if self.loop:
                pc = 0
            else:
                self.timer.Stop()
                pc = nf
        self.file_count = pc
        self._handle_particle_array_updates()

    def _play_delay_changed(self):
        if self.play:
            self._play_changed(self.play)

    def _scalar_changed(self, value):
        for pa in self.particle_arrays:
            pa.scalar = value

    def _update_files_fired(self):
        fc = self.file_count
        files = glob_files(self.files[fc])
        sort_file_list(files)
        self.files = files
        self.file_count = fc
        if self.play:
            self._play_changed(self.play)

    def _shell_fired(self):
        ns = dict(viewer=self, particle_arrays=self.particle_arrays,
                  interpolator=self.interpolator, scene=self.scene,
                  mlab=self.scene.mlab)
        obj = PythonShellView(ns=ns)
        obj.edit_traits()

    def _directory_changed(self, d):
        ext = os.path.splitext(self.files[-1])[1]
        files = glob.glob(os.path.join(d, '*' + ext))
        if len(files) > 0:
            self._clear()
            sort_file_list(files)
            self.files = files
            self.file_count = min(self.file_count, len(files))
        else:
            pass

    def _live_mode_changed(self, value):
        if value:
            self._file_name = ''
            self.client = None
            self._clear()
            self._mark_reconnect()
            self.start_timer()
        else:
            self.client = None
            self._clear()
            self.timer.Stop()

    def _particle_array_helper_updated(self, value):
        self._particle_array_updated = True

    def _handle_particle_array_updates(self):
        # Called when the particle array helper fires an updated event.
        if self._particle_array_updated and self._file_name:
            sd = self._solver_data
            arrays = [x.particle_array for x in self.particle_arrays]
            dump(self._file_name, arrays, sd)
            self._particle_array_updated = False

    def _make_particle_array_helper(self, scene, name):
        pah = ParticleArrayHelper(scene=scene, name=name, scalar=self.scalar)
        pah.on_trait_change(self._particle_array_helper_updated, 'updated')
        return pah


######################################################################
def usage():
    print("""Usage:
pysph view [-v] <trait1=value> <trait2=value> [directory or files.npz]

If a directory or *.npz files are not supplied it will connect to a running
solver, if not it will display the given files.

The arguments <trait1=value> are optional settings like host, port and authkey
etc.  The following traits are available:

  scalar        -- the default scalar to display on the view.

  host          -- hostname/IP address to connect to.
  port          -- Port to connect to
  authkey       -- authorization key to use.
  interval      -- time interval to refresh display
  pause_solver  -- Set True/False, will pause running solver

  movie_directory -- directory to dump movie files (automatically set if not
                       supplied)
  record        -- True/False: record movie, i.e. store screenshots of display.

  play          -- True/False: Play all stored data files.
  loop          -- True/False: Loop over data files.

Options:
--------

  -h/--help   prints this message.

  -v          sets verbose mode which will print solver connection
              status failures on stdout.

Examples::
----------

  $ pysph view scalar=u play=True loop=True elliptical_drop_output/
  $ pysph view ellptical_drop_100.npz
  $ pysph view interval=10 host=localhost port=8900

""")

def error(msg):
    print(msg)
    sys.exit()

def main(args=None):
    if args is None:
        args = sys.argv[1:]

    if '-h' in args or '--help' in args:
        usage()
        sys.exit(0)

    if '-v' in args:
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.INFO)
        args.remove('-v')

    kw = {}
    files = []
    for arg in args:
        if '=' not in arg:
            if arg.endswith(output_formats):
                files.extend(glob.glob(arg))
                continue
            elif os.path.isdir(arg):
                _files = glob.glob(os.path.join(arg, '*.hdf5'))
                if len(_files) == 0:
                    _files = glob.glob(os.path.join(arg, '*.npz'))
                files.extend(_files)
                continue
            else:
                usage()
                sys.exit(1)
        key, arg = [x.strip() for x in arg.split('=')]
        try:
            val = eval(arg, math.__dict__)
            # this will fail if arg is a string.
        except NameError:
            val = arg
        kw[key] = val

    sort_file_list(files)
    live_mode = len(files) == 0

    # If we set the particle arrays before the scene is activated, the arrays
    # are not displayed on screen so we use do_later to set the  files.
    m = MayaviViewer(live_mode=live_mode)
    do_later(m.set, files=files, **kw)
    m.configure_traits()

if __name__ == '__main__':
    main()
