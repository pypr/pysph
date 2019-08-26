import json
import glob
from pysph.solver.utils import load, get_files, mkdir
from IPython.display import display, Image, clear_output, HTML
import ipywidgets as widgets
import numpy as np
import matplotlib as mpl
from decimal import Decimal

mpl.use('module://ipympl.backend_nbagg')
# Now the user does not have to use the IPython magic command
# '%matplotlib ipympl' in the notebook, this takes care of it.
# The matplotlib backend needs to be set before matplotlib.pyplot
# is imported and this ends up violating the PEP 8 style guide.

import matplotlib.pyplot as plt


class Viewer(object):

    '''
    Base class for viewers.
    '''

    def __init__(self, path, cache=True):

        self.path = path
        self.paths_list = get_files(path)

        # Caching #
        # Note : Caching is only used by get_frame and widget handlers.
        if cache:
            self.cache = {}
        else:
            self.cache = None

    def get_frame(self, frame):
        '''Return particle arrays for a given frame number with caching.


        Parameters
        ----------

        frame : int

        Returns
        -------

        A dictionary.

        Examples
        --------

        >>> sample = Viewer2D('/home/deep/pysph/trivial_inlet_outlet_output/')
        >>> sample.get_frame(12)
        {
        'arrays': {
        'fluid': <pysph.base.particle_array.ParticleArray at 0x7f3f7d144d60>,
        'inlet': <pysph.base.particle_array.ParticleArray at 0x7f3f7d144b98>,
        'outlet': <pysph.base.particle_array.ParticleArray at 0x7f3f7d144c30>
                },
        'solver_data': {'count': 240, 'dt': 0.01, 't': 2.399999999999993}
        }


        '''

        if self.cache is not None:
            if frame in self.cache:
                temp_data = self.cache[frame]
            else:
                self.cache[frame] = temp_data = load(self.paths_list[frame])
        else:
            temp_data = load(self.paths_list[frame])

        return temp_data

    def show_log(self):
        '''
        Prints the content of log file.
        '''

        print("Printing log : \n\n")
        path = self.path + "/*.log"
        with open(glob.glob(path)[0], 'r') as logfile:
            for lines in logfile:
                print(lines)

    def show_results(self):
        '''
        Show if there are any png, jpeg, jpg, or bmp images.
        '''

        imgs = tuple()
        for extension in ['png', 'jpg', 'jpeg', 'bmp']:
            temppath = self.path + "/*." + extension
            for paths in glob.glob(temppath):
                imgs += (Image(paths),)
        if len(imgs) != 0:
            display(*imgs)
        else:
            print("No results to show.")

    def show_info(self):
        '''
        Print contents of the .info file present in the output directory,
        keys present in results.npz,  number of files and
        information about paricle arrays.
        '''

        # General Info #

        path = self.path + "/*.info"
        with open(glob.glob(path)[0], 'r') as infofile:
            data = json.load(infofile)

            print('Printing info : \n')
            for key in data.keys():
                if key == 'cpu_time':
                    print(key + " : " + str(data[key]) + " seconds")
                else:
                    print(key + " : " + str(data[key]))

            print('Number of files : {}'.format(len(self.paths_list)))

        # Particle Info #

        temp_data = load(self.paths_list[0])['arrays']

        for key in temp_data:
            print("  {} :".format(key))
            print("    Number of particles : {}".format(
                temp_data[key].get_number_of_particles())
            )
            print("    Output Property Arrays : {}".format(
                temp_data[key].output_property_arrays)
            )

        # keys in results.npz

        from numpy import load as npl

        path = self.path + "*results*"
        files = glob.glob(path)
        if len(files) != 0:
            data = npl(files[0])
            print("\nKeys in results.npz :")
            print(data.keys())

    def show_all(self):
        self.show_info()
        self.show_results()
        self.show_log()

    def _cmap_helper(self, data, array_name, for_plot_vectors=False):
        '''
        Helper Function:
        Takes in a numpy array and returns its maximum,
        minimum , subject to the constraints provided by the user
        in the legend_lower_lim and legend_upper_lim text boxes.
        Also returns the input array normalized by the maximum.
        '''

        pa_widgets = self._widgets.particles[array_name]
        if for_plot_vectors is False:
            ulim = pa_widgets.legend_upper_lim.value
            llim = pa_widgets.legend_lower_lim.value
        elif for_plot_vectors is True:
            ulim = ''
            llim = ''

        if llim == '' and ulim == '':
            pass
        elif llim != '' and ulim == '':
            for i in range(len(data)):
                if data[i] < float(llim):
                    data[i] = float(llim)
        elif llim == '' and ulim != '':
            for i in range(len(data)):
                if data[i] > float(ulim):
                    data[i] = float(ulim)
        elif llim != '' and ulim != '':
            for i in range(len(data)):
                if data[i] > float(ulim):
                    data[i] = float(ulim)
                elif data[i] < float(llim):
                    data[i] = float(llim)

        actual_minm = np.min(data)
        if llim != '' and actual_minm > float(llim):
            actual_minm = float(llim)

        actual_maxm = np.max(data)
        if ulim != '' and actual_maxm < float(ulim):
            actual_maxm = float(ulim)

        if len(set(data)) == 1:
            # This takes care of the case when all the values are the same.
            # Use case is the initialization of some scalars (like density).
            if ulim == '' and llim == '':
                if actual_maxm != 0:
                    return actual_minm, actual_maxm, np.ones_like(data)
                else:
                    return actual_minm, actual_maxm, np.zeros_like(data)
            else:
                data_norm = (data-actual_minm)/(actual_maxm-actual_minm)
                return actual_minm, actual_maxm, data_norm
        else:
            data_norm = (data-actual_minm)/(actual_maxm-actual_minm)
            return actual_minm, actual_maxm, data_norm

    def _create_widgets(self):

        if self.viewer_type == 'Viewer1D':
            self._widgets = Viewer1DWidgets(
                file_name=self.paths_list[0],
                file_count=len(self.paths_list) - 1,
            )
        elif self.viewer_type == 'Viewer2D':
            self._widgets = Viewer2DWidgets(
                file_name=self.paths_list[0],
                file_count=len(self.paths_list) - 1,
            )
        elif self.viewer_type == 'Viewer3D':
            self._widgets = Viewer3DWidgets(
                file_name=self.paths_list[0],
                file_count=len(self.paths_list) - 1,
            )

        if 'general_properties' in self.config.keys():
            gen_prop = self.config['general_properties']
            for widget_name in gen_prop.keys():
                try:
                    widget = getattr(
                        self._widgets,
                        widget_name
                    )
                    widget.value = gen_prop[widget_name]
                except AttributeError:
                    continue
            if 'cull_factor' in gen_prop.keys():
                self.cull_factor = gen_prop['cull_factor']
                if self.cull_factor > 0:
                    self._widgets.frame.step = self.cull_factor
                    self._widgets.play_button.step = self.cull_factor
                else:
                    print('cull_factor must be a positive integer.')

        self._widgets.frame.observe(self._frame_handler, 'value')
        self._widgets.save_figure.on_submit(self._save_figure_handler)
        self._widgets.delay_box.observe(self._delay_box_handler, 'value')
        self._widgets.save_all_plots.observe(
            self._save_all_plots_handler,
            'value'
        )
        self._widgets.print_config.on_click(
            self._print_present_config_dictionary
        )
        if self.viewer_type == 'Viewer2D' or self.viewer_type == 'Viewer1D':
            self._widgets.show_solver_time.observe(
                self._show_solver_time_handler,
                'value'
            )

        # PLEASE NOTE:
        # All widget handlers take in 'change' as an argument. This is usually
        # a dictionary conatining information about the widget and the change
        # in state. However, these functions are also used outside of the use
        # case of a user-triggered-event, and in these scenarios None should
        # be passed as the argument. This is of particular significance
        # because in some of these functions plt.figure.show() gets called
        # only if the argument passed is not None.

        for array_name in self._widgets.particles.keys():
            pa_widgets = self._widgets.particles[array_name]
            # Changing the properties as per the configuration dictionary.
            if array_name in self.config.keys():
                pa_config = self.config[array_name]
                for widget_name in pa_config.keys():
                    try:
                        widget = getattr(
                            pa_widgets,
                            widget_name
                        )
                        widget.value = pa_config[widget_name]
                    except AttributeError:
                        continue
            for widget_name in list(pa_widgets.__dict__.keys())[1:]:
                widget = getattr(
                    pa_widgets,
                    widget_name
                )
                if (widget_name == 'legend_lower_lim' or
                        widget_name == 'legend_upper_lim'):
                    widget_handler = self._legend_lim_handler
                elif (widget_name == 'right_spine_lower_lim' or
                        widget_name == 'right_spine_upper_lim'):
                    widget_handler = self._right_spine_lim_handler
                else:
                    widget_handler = getattr(
                        self,
                        '_' + widget_name + '_handler'
                    )
                widget.observe(widget_handler, 'value')

    def _legend_lim_handler(self, change):

        array_name = change['owner'].owner
        pa_widgets = self._widgets.particles[array_name]
        if pa_widgets.scalar.value != 'None':
            temp_data = self.get_frame(
                self._widgets.frame.value
            )['arrays']
            sct = self._scatters[array_name]
            n = pa_widgets.masking_factor.value
            stride, component = self._stride_and_component(
                temp_data[array_name], pa_widgets
            )
            c = self._get_c(
                pa_widgets,
                temp_data[array_name],
                component,
                stride
            )
            colormap = getattr(
                plt.cm,
                pa_widgets.scalar_cmap.value
            )
            min_c, max_c, c_norm = self._cmap_helper(
                c,
                array_name
            )
            if self.viewer_type == 'Viewer2D':
                sct.set_facecolors(colormap(c_norm[::n]))
                self._legend_handler(None)
                self.figure.show()
            elif self.viewer_type == 'Viewer3D':
                sct.color = colormap(c_norm[::n])
                self._legend_handler(None)

    def _delay_box_handler(self, change):

        self._widgets.play_button.interval = change['new']*1000

    def _save_all_plots_handler(self, change):

        if self.viewer_type == 'Viewer3D':
            import ipyvolume.pylab as p3

        if change['new'] is True:
            mkdir('all_plots')
            self._widgets.frame.disabled = True
            self._widgets.play_button.disabled = True
            self._widgets.delay_box.disabled = True
            self._widgets.save_figure.disabled = True
            self._widgets.save_all_plots.disabled = True
            self._widgets.print_config.disabled = True
            if self.viewer_type == 'Viewer2D':
                self._widgets.show_solver_time.disabled = True

            for array_name in self._widgets.particles.keys():
                pa_widgets = self._widgets.particles[array_name]
                for widget_name in list(pa_widgets.__dict__.keys())[1:]:
                    widget = getattr(
                        pa_widgets,
                        widget_name
                    )
                    widget.disabled = True

            file_count = len(self.paths_list) - 1

            for i in np.arange(0, file_count + 1, self.cull_factor):
                self._widgets.frame.value = i
                self._frame_handler(None)
                if self.viewer_type == 'Viewer2D':
                    self.figure.savefig(
                        'all_plots/frame_%s.png' % i,
                        dpi=300
                    )
                elif self.viewer_type == 'Viewer3D':
                    p3.savefig(
                        'all_plots/frame_%s.png' % i,
                        width=600,
                        height=600,
                        fig=self.plot
                    )

            print(
                "Saved the plots in the folder 'all_plots'" +
                " in the present working directory"
            )

            self._widgets.frame.disabled = False
            self._widgets.play_button.disabled = False
            self._widgets.delay_box.disabled = False
            self._widgets.save_figure.disabled = False
            self._widgets.save_all_plots.disabled = False
            self._widgets.print_config.disabled = False
            if self.viewer_type == 'Viewer2D':
                self._widgets.show_solver_time.disabled = False

            self._widgets.save_all_plots.value = False

            for array_name in self._widgets.particles.keys():
                pa_widgets = self._widgets.particles[array_name]
                for widget_name in list(pa_widgets.__dict__.keys())[1:]:
                    widget = getattr(
                        pa_widgets,
                        widget_name
                    )
                    widget.disabled = False

    def _print_present_config_dictionary(self, change):

        _widgets = self._widgets
        config = {'general_properties': {}}

        gen_prop = config['general_properties']
        gen_prop['frame'] = _widgets.frame.value
        gen_prop['delay_box'] = _widgets.delay_box.value
        gen_prop['cull_factor'] = _widgets.frame.step
        if self.viewer_type == 'Viewer2D':
            gen_prop[
                'show_solver_time'
            ] = _widgets.show_solver_time.value

        for array_name in self._widgets.particles.keys():
            pa_widgets = self._widgets.particles[array_name]
            config[array_name] = {}
            pa_config = config[array_name]
            for widget_name in list(pa_widgets.__dict__.keys())[1:]:
                widget = getattr(
                    pa_widgets,
                    widget_name
                )
                pa_config[widget_name] = widget.value

        print(config)

    def _masking_factor_handler(self, change):

        array_name = change['owner'].owner
        pa_widgets = self._widgets.particles[array_name]
        if pa_widgets.is_visible.value is True:
            n = pa_widgets.masking_factor.value
            if n > 0:
                temp_data = self.get_frame(self._widgets.frame.value)['arrays']
                stride, component = self._stride_and_component(
                    temp_data[array_name], pa_widgets
                )
                c = self._get_c(
                    pa_widgets,
                    temp_data[array_name],
                    component,
                    stride
                )
                colormap = getattr(
                    plt.cm,
                    pa_widgets.scalar_cmap.value
                )
                min_c, max_c, c_norm = self._cmap_helper(
                    c,
                    array_name
                )
                if self.viewer_type == 'Viewer2D':
                    self._scatters[array_name].remove()
                    del self._scatters[array_name]
                    self._scatters[array_name] = self._scatter_ax.scatter(
                        temp_data[array_name].x[component::stride][::n],
                        temp_data[array_name].y[component::stride][::n],
                        s=pa_widgets.scalar_size.value,
                    )
                    self._scatters[array_name].set_facecolors(
                        colormap(c_norm[::n])
                    )
                    self.figure.show()
                elif self.viewer_type == 'Viewer3D':
                    import ipyvolume.pylab as p3
                    copy = self.plot.scatters.copy()
                    copy.remove(self._scatters[array_name])
                    del self._scatters[array_name]
                    if array_name in self._vectors.keys():
                        copy.remove(self._vectors[array_name])
                        del self._vectors[array_name]
                    self.plot.scatters = copy
                    self._scatters[array_name] = p3.scatter(
                        temp_data[array_name].x[component::stride][::n],
                        temp_data[array_name].y[component::stride][::n],
                        temp_data[array_name].z[component::stride][::n],
                        color=colormap(c_norm[::n]),
                        size=pa_widgets.scalar_size.value,
                        marker=pa_widgets.scatter_plot_marker.value,
                    )
                    self._plot_vectors(
                        pa_widgets,
                        temp_data[array_name],
                        array_name
                    )
                else:
                    print('Masking factor must be a positive integer.')

    def _get_c(self, pa_widgets, data, component=0, stride=1, need_vmag=False):

        c = [0]
        if pa_widgets.scalar.value == 'vmag' or need_vmag is True:
            u = getattr(data, 'u')[component::stride]
            v = getattr(data, 'v')[component::stride]
            c = u**2 + v**2
            if self.viewer_type == 'Viewer3D':
                w = getattr(data, 'w')[component::stride]
                c += w**2
            c = c**0.5
        elif pa_widgets.scalar.value != 'None':
            c = getattr(
                data,
                pa_widgets.scalar.value
            )[component::stride]
        return c

    def _opacity_handler(self, change):

        array_name = change['owner'].owner
        pa_widgets = self._widgets.particles[array_name]
        if pa_widgets.is_visible.value is True:
            alpha = pa_widgets.opacity.value
            n = pa_widgets.masking_factor.value
            temp_data = self.get_frame(
                self._widgets.frame.value
            )['arrays']
            stride, component = self._stride_and_component(
                    temp_data[array_name], pa_widgets
            )
            c = self._get_c(
                pa_widgets,
                temp_data[array_name],
                component,
                stride
            )
            colormap = getattr(
                plt.cm,
                pa_widgets.scalar_cmap.value
            )
            sct = self._scatters[array_name]
            min_c, max_c, c_norm = self._cmap_helper(
                c,
                array_name
            )
            cm = colormap(c_norm[::n])
            cm[0:, 3] *= alpha
            if self.viewer_type == 'Viewer2D':
                sct.set_facecolors(cm)
                self._legend_handler(None)
                self.figure.show()

    def _components_handler(self, change, array_name=None):

        if array_name is None:
            array_name = change['owner'].owner
        pa_widgets = self._widgets.particles[array_name]
        scalar = pa_widgets.scalar.value
        if pa_widgets.is_visible.value is True:
            temp_data = self.get_frame(self._widgets.frame.value)['arrays']
            if scalar in temp_data[array_name].stride.keys():
                stride = temp_data[array_name].stride[array_name]
                if pa_widgets.components.value > stride:
                    pa_widgets.components.value = stride
                elif pa_widgets.components.value < 1:
                    pa_widgets.components.value = 1
            self._scalar_handler({'owner': pa_widgets.scalar})

    def _stride_and_component(self, data, pa_widgets):

        if pa_widgets.scalar.value in data.stride.keys():
            pa_widgets.components.disabled = False
            component = pa_widgets.components.value - 1
            stride = data.stride[pa_widgets.scalar.value]
        else:
            pa_widgets.components.disabled = True
            stride = 1
            component = 0
        return stride, component


class ParticleArrayWidgets1D(object):

    def __init__(self, particlearray):

        self.array_name = particlearray.name
        self.scalar = widgets.Dropdown(
            options=['None'] +
            particlearray.output_property_arrays +
            ['vmag'],
            value='rho',
            description="scalar",
            disabled=False,
            layout=widgets.Layout(width='240px', display='flex')
        )
        self.scalar.owner = self.array_name
        self.is_visible = widgets.Checkbox(
            value=True,
            description='visible',
            disabled=False,
            layout=widgets.Layout(width='170px', display='flex')
        )
        self.is_visible.owner = self.array_name
        self.masking_factor = widgets.IntText(
            value=1,
            description='masking',
            disabled=False,
            layout=widgets.Layout(width='160px', display='flex'),
        )
        self.masking_factor.owner = self.array_name
        self.components = widgets.IntText(
            value=1,
            description='component',
            disabled=True,
            layout=widgets.Layout(width='160px', display='flex'),
        )
        self.components.owner = self.array_name
        self.right_spine = widgets.Checkbox(
            value=False,
            description='scale',
            disabled=False,
            layout=widgets.Layout(width='170px', display='flex')
        )
        self.right_spine.owner = self.array_name
        self.right_spine_lower_lim = widgets.Text(
            value='',
            placeholder='min',
            description='lower limit',
            disabled=False,
            layout=widgets.Layout(width='160px', display='flex'),
            continuous_update=False
        )
        self.right_spine_lower_lim.owner = self.array_name
        self.right_spine_upper_lim = widgets.Text(
            value='',
            placeholder='max',
            description='upper limit',
            disabled=False,
            layout=widgets.Layout(width='160px', display='flex'),
            continuous_update=False
        )
        self.right_spine_upper_lim.owner = self.array_name

    def _tab_config(self):

        VBox1 = widgets.VBox(
            [
                self.scalar,
                self.components,
            ]
        )
        VBox2 = widgets.VBox(
            [
                self.masking_factor,
                self.is_visible,
            ]
        )
        VBox3 = widgets.VBox(
            [
                self.right_spine,
                widgets.HBox(
                    [
                        self.right_spine_upper_lim,
                        self.right_spine_lower_lim,
                    ]
                ),

            ]
        )
        hbox = widgets.HBox([VBox1, VBox2, VBox3])
        return hbox


class Viewer1DWidgets(object):

    def __init__(self, file_name, file_count):

        self.temp_data = load(file_name)
        self.time = str(self.temp_data['solver_data']['t'])
        self.temp_data = self.temp_data['arrays']
        self.frame = widgets.IntSlider(
            min=0,
            max=file_count,
            step=1,
            value=0,
            description='frame',
            layout=widgets.Layout(width='500px'),
            continuous_update=False,
        )
        self.play_button = widgets.Play(
            min=0,
            max=file_count,
            step=1,
            disabled=False,
        )
        self.link = widgets.jslink(
            (self.frame, 'value'),
            (self.play_button, 'value'),
        )
        self.delay_box = widgets.FloatText(
            value=0.2,
            description='Delay',
            disabled=False,
            layout=widgets.Layout(width='160px', display='flex'),
        )
        self.save_figure = widgets.Text(
                value='',
                placeholder='example.pdf',
                description='Save figure',
                disabled=False,
                layout=widgets.Layout(width='240px', display='flex'),
        )
        self.save_all_plots = widgets.ToggleButton(
                value=False,
                description='Save all plots!',
                disabled=False,
                tooltip='Saves the corresponding plots for all the' +
                        ' frames in the presently set styling.',
                icon='',
                layout=widgets.Layout(display='flex'),
        )
        self.solver_time = widgets.HTML(
            value=self.time,
            description='Solver time:'
        )
        self.show_solver_time = widgets.Checkbox(
            value=False,
            description="Show solver time",
            disabled=False,
            layout=widgets.Layout(display='flex'),
        )
        self.print_config = widgets.Button(
            description='print present config.',
            tooltip='Prints the configuration dictionary ' +
                    'for the current viewer state',
            disabled=False,
            layout=widgets.Layout(display='flex'),
        )
        self.particles = {}
        for array_name in self.temp_data.keys():
            self.particles[array_name] = ParticleArrayWidgets1D(
                self.temp_data[array_name],
            )

    def _create_tabs(self):

        children = []
        for array_name in self.particles.keys():
            children.append(self.particles[array_name]._tab_config())

        tab = widgets.Tab(children=children)
        for i in range(len(children)):
            tab.set_title(i, list(self.particles.keys())[i])

        return widgets.VBox(
                [
                    tab,
                    widgets.HBox(
                     [
                        self.play_button,
                        self.frame
                     ]
                    ),
                    widgets.HBox(
                     [
                        self.delay_box,
                        self.save_figure,
                        self.save_all_plots,
                     ]
                    ),
                    widgets.HBox(
                     [
                        self.print_config,
                        self.show_solver_time,
                        self.solver_time,
                     ]
                    )
                ]
            )


class Viewer1D(Viewer):

    '''
    Example
    -------

    >>> from pysph.tools.ipy_viewer import Viewer1D
    >>> sample = Viewer1D(
        '/home/uname/pysph_files/blastwave_output'
        )
    >>> sample.interactive_plot()
    >>> sample.show_log()
    >>> sample.show_info()
    '''

    def _configure_plot(self):

        '''
        Set attributes for plotting.
        '''

        self.figure, temp = plt.subplots()
        self.add_axes = False
        self._scatters_ax = {'host': temp}
        self._scatters = {}
        self._solver_time_ax = {}

        self.figure.show()

    def interactive_plot(self, config={}):
        '''
        Set plotting attributes, create widgets and display them
        along with the interactive plot.
        '''

        self.config = config  # The configuration dictionary.
        self.viewer_type = 'Viewer1D'
        self._configure_plot()
        self._create_widgets()
        display(self._widgets._create_tabs())
        temp_data = self.get_frame(0)
        self.time = str(temp_data['solver_data']['t'])
        temp_data = temp_data['arrays']

        self.xmin = None
        self.xmax = None
        for array_name in self._widgets.particles.keys():
            pa_widgets = self._widgets.particles[array_name]
            if (pa_widgets.scalar.value != 'None' and
                    pa_widgets.is_visible.value is True):

                n = pa_widgets.masking_factor.value
                stride, component = self._stride_and_component(
                    temp_data[array_name], pa_widgets
                )
                c = self._get_c(
                    pa_widgets,
                    temp_data[array_name],
                    component,
                    stride
                )
                if self.xmin is None:
                    self.xmin = min(temp_data[array_name].x)
                elif min(temp_data[array_name].x) < self.xmin:
                    self.xmin = min(temp_data[array_name].x)
                if self.xmax is None:
                    self.xmax = max(temp_data[array_name].x)
                elif max(temp_data[array_name].x) > self.xmax:
                    self.xmax = max(temp_data[array_name].x)
                min_c = min(c)
                max_c = max(c)
                llim = pa_widgets.right_spine_lower_lim.value
                ulim = pa_widgets.right_spine_upper_lim.value
                if llim != '':
                    min_c = float(llim)
                if ulim != '':
                    max_c = float(ulim)
                if max_c-min_c != 0:
                    c_norm = (c - min_c)/(max_c - min_c)
                elif max_c != 0:
                    c_norm = c/max_c
                else:
                    c_norm = c

                color = 'C' + str(
                    list(
                        self._widgets.particles.keys()
                    ).index(array_name)
                )
                ax = self._scatters_ax['host'].twinx()
                self._scatters[array_name] = ax.scatter(
                    temp_data[array_name].x[component::stride][::n],
                    c_norm[::n],
                    color=color,
                    label=array_name,
                )
                ax.set_ylim(-0.1, 1.1)
                self._make_patch_spines_invisible(ax)
                self._scatters_ax[array_name] = ax

        self._scatters_ax['host'].set_xlim(self.xmin-0.1, self.xmax+0.1)
        self._make_patch_spines_invisible(self._scatters_ax['host'])

        self.solver_time_textbox = None
        # So that _show_solver_time_handler does not glitch at intialization.
        self._frame_handler(None)
        self._show_solver_time_handler(None)
        self._right_spine_handler(None)

    def _make_patch_spines_invisible(self, ax, val=True):
        '''
        Helper function for making individual y-axes
        for different particle arrays
        '''
        ax.set_frame_on(val)
        ax.patch.set_visible(False)
        for sp in ax.spines.values():
            sp.set_visible(False)

    def _frame_handler(self, change):

        temp_data = self.get_frame(self._widgets.frame.value)
        self.time = str(temp_data['solver_data']['t'])
        self._widgets.solver_time.value = self.time
        temp_data = temp_data['arrays']

        for array_name in self._widgets.particles.keys():
            pa_widgets = self._widgets.particles[array_name]
            if (pa_widgets.scalar.value != 'None' and
                    pa_widgets.is_visible.value is True):
                n = pa_widgets.masking_factor.value
                stride, component = self._stride_and_component(
                    temp_data[array_name], pa_widgets
                )
                sct = self._scatters[array_name]
                c = self._get_c(
                    pa_widgets,
                    temp_data[array_name],
                    component,
                    stride
                )
                if self.xmin is None:
                    self.xmin = min(temp_data[array_name].x)
                elif min(temp_data[array_name].x) < self.xmin:
                    self.xmin = min(temp_data[array_name].x)
                if self.xmax is None:
                    self.xmax = max(temp_data[array_name].x)
                if max(temp_data[array_name].x) > self.xmax:
                    self.xmax = max(temp_data[array_name].x)
                min_c = min(c)
                max_c = max(c)
                llim = pa_widgets.right_spine_lower_lim.value
                ulim = pa_widgets.right_spine_upper_lim.value
                if llim != '':
                    min_c = float(llim)
                if ulim != '':
                    max_c = float(ulim)
                if max_c-min_c != 0:
                    c_norm = (c - min_c)/(max_c - min_c)
                elif max_c != 0:
                    c_norm = c/max_c
                else:
                    c_norm = c
                sct.set_offsets(
                    np.vstack(
                        (
                            temp_data[array_name].x[component::stride][::n],
                            c_norm[::n]
                        )
                    ).T
                )
                self._scatters_ax[array_name].set_ylim(-0.1, 1.1)

        self._scatters_ax['host'].set_xlim(self.xmin-0.1, self.xmax+0.1)
        self._right_spine_handler(None)
        self._show_solver_time_handler(None)
        self.figure.show()

    def _scalar_handler(self, change):

        array_name = change['owner'].owner
        pa_widgets = self._widgets.particles[array_name]
        if pa_widgets.is_visible.value is True:
            n = pa_widgets.masking_factor.value
            temp_data = self.get_frame(
                self._widgets.frame.value
            )['arrays']
            stride, component = self._stride_and_component(
                    temp_data[array_name], pa_widgets
            )
            c = self._get_c(
                pa_widgets,
                temp_data[array_name],
                component,
                stride
            )
            if self.xmin is None:
                self.xmin = min(temp_data[array_name].x)
            elif min(temp_data[array_name].x) < self.xmin:
                self.xmin = min(temp_data[array_name].x)
            if self.xmax is None:
                self.xmax = max(temp_data[array_name].x)
            elif max(temp_data[array_name].x) > self.xmax:
                self.xmax = max(temp_data[array_name].x)
            min_c = min(c)
            max_c = max(c)
            llim = pa_widgets.right_spine_lower_lim.value
            ulim = pa_widgets.right_spine_upper_lim.value
            if llim != '':
                min_c = float(llim)
            if ulim != '':
                max_c = float(ulim)
            if max_c-min_c != 0:
                c_norm = (c - min_c)/(max_c - min_c)
            elif max_c != 0:
                c_norm = c/max_c
            else:
                c_norm = c

            new = change['new']
            old = change['old']

            if (new == 'None' and old == 'None'):
                pass
            elif (new == 'None' and old != 'None'):
                sct = self._scatters[array_name]
                copy = self._scatters_ax[array_name].collections.copy()
                copy.remove(sct)
                self._scatters_ax[array_name].collections = copy
                del self._scatters[array_name]
            elif (new != 'None' and old == 'None'):
                color = 'C' + str(
                    list(
                        self._widgets.particles.keys()
                    ).index(array_name)
                )
                self._scatters[array_name] = self._scatters_ax[
                    array_name
                ].scatter(
                    temp_data[array_name].x[component::stride][::n],
                    c_norm[::n],
                    color=color
                )
                self._scatters_ax[array_name].set_ylim(-0.1, 1.1)
            elif (new != 'None' and old != 'None'):
                sct = self._scatters[array_name]
                sct.set_offsets(
                    np.vstack(
                        (
                            temp_data[array_name].x[component::stride][::n],
                            c_norm[::n]
                        )
                    ).T
                )
                self._scatters_ax[array_name].set_ylim(-0.1, 1.1)

            self._scatters_ax['host'].set_xlim(self.xmin-0.1, self.xmax+0.1)
            self._right_spine_handler(None)
            self.figure.show()

    def _save_figure_handler(self, change):

        file_was_saved = False
        for extension in [
            '.eps', '.pdf', '.pgf',
            '.png', '.ps', '.raw',
            '.rgba', '.svg', '.svgz'
        ]:
            if self._widgets.save_figure.value.endswith(extension):
                self.figure.savefig(self._widgets.save_figure.value)
                print(
                    "Saved figure as {} in the present working directory"
                    .format(
                        self._widgets.save_figure.value
                    )
                )
                file_was_saved = True
                break
        self._widgets.save_figure.value = ""
        if file_was_saved is False:
            print(
                "Please use a valid extension, that is, one of the following" +
                ": '.eps', '.pdf', '.pgf', '.png', '.ps', '.raw', '.rgba'," +
                " '.svg' or '.svgz'."
            )

    def _is_visible_handler(self, change):

        array_name = change['owner'].owner
        pa_widgets = self._widgets.particles[array_name]
        temp_data = self.get_frame(self._widgets.frame.value)['arrays']

        if pa_widgets.scalar.value != 'None':
            if change['new'] is False:
                sct = self._scatters[array_name]
                copy = self._scatters_ax[array_name].collections.copy()
                copy.remove(sct)
                self._scatters_ax[array_name].collections = copy
                del self._scatters[array_name]
            elif change['new'] is True:
                n = pa_widgets.masking_factor.value
                stride, component = self._stride_and_component(
                    temp_data[array_name], pa_widgets
                )
                c = self._get_c(
                    pa_widgets,
                    temp_data[array_name],
                    component,
                    stride
                )
                if self.xmin is None:
                    self.xmin = min(temp_data[array_name].x)
                elif min(temp_data[array_name].x) < self.xmin:
                    self.xmin = min(temp_data[array_name].x)
                if self.xmax is None:
                    self.xmax = max(temp_data[array_name].x)
                elif max(temp_data[array_name].x) > self.xmax:
                    self.xmax = max(temp_data[array_name].x)
                min_c = min(c)
                max_c = max(c)
                llim = pa_widgets.right_spine_lower_lim.value
                ulim = pa_widgets.right_spine_upper_lim.value
                if llim != '':
                    min_c = float(llim)
                if ulim != '':
                    max_c = float(ulim)
                if max_c-min_c != 0:
                    c_norm = (c - min_c)/(max_c - min_c)
                elif max_c != 0:
                    c_norm = c/max_c
                else:
                    c_norm = c
                color = 'C' + str(
                    list(
                        self._widgets.particles.keys()
                    ).index(array_name)
                )
                self._scatters[array_name] = self._scatters_ax[
                    array_name
                ].scatter(
                    temp_data[array_name].x[component::stride][::n],
                    c_norm[::n],
                    color=color
                )
                self._scatters_ax[array_name].set_ylim(-0.1, 1.1)
        self._scatters_ax['host'].set_xlim(self.xmin-0.1, self.xmax+0.1)
        self._right_spine_handler(None)
        self.figure.show()

    def _show_solver_time_handler(self, change):

        if self._widgets.show_solver_time.value is True:
            if self.solver_time_textbox is not None:
                self.solver_time_textbox.remove()
            self.solver_time_textbox = self._scatters_ax['host'].text(
                x=0.02,
                y=0.02,
                s='Solver time: ' + self.time,
                verticalalignment='bottom',
                horizontalalignment='left',
                transform=self._scatters_ax['host'].transAxes,
                fontsize=12,
                bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 3},
            )
        elif self._widgets.show_solver_time.value is False:
            if self.solver_time_textbox is not None:
                self.solver_time_textbox.remove()
            self.solver_time_textbox = None
        if change is not None:
            self.figure.show()

    def _right_spine_handler(self, change):

        temp_data = self.get_frame(
            self._widgets.frame.value
        )['arrays']

        number_of_spines = 0

        self._make_patch_spines_invisible(
            self._scatters_ax['host'],
            False
        )

        for array_name in self._widgets.particles.keys():
            pa_widgets = self._widgets.particles[array_name]
            ax = self._scatters_ax[array_name]
            self._make_patch_spines_invisible(ax, False)
            if (pa_widgets.right_spine.value is True and
                    pa_widgets.is_visible.value is True and
                    pa_widgets.scalar.value != 'None'):

                number_of_spines += 1
                stride, component = self._stride_and_component(
                    temp_data[array_name], pa_widgets
                )
                c = self._get_c(
                    pa_widgets,
                    temp_data[array_name],
                    component,
                    stride
                )
                self._scatters_ax['host'].set_position(
                    [0, 0, 1. - 0.2*number_of_spines, 1]
                )

                min_c = min(c)
                max_c = max(c)
                llim = pa_widgets.right_spine_lower_lim.value
                ulim = pa_widgets.right_spine_upper_lim.value
                if llim != '':
                    min_c = float(llim)
                if ulim != '':
                    max_c = float(ulim)
                locs = np.linspace(0, 1, 20)

                if min_c - max_c != 0:
                    labels = [
                        '%.2E' % Decimal(str(val)) for val in np.linspace(
                            min_c,
                            max_c,
                            20
                        )
                    ]
                elif max_c == 0:
                    labels = ['%.2E' % Decimal(str(val)) for val in locs]
                else:
                    labels = [
                        '%.2E' % Decimal(str(val)) for val in np.linspace(
                            0,
                            max_c,
                            20
                        )
                    ]

                color = 'C' + str(
                    list(
                        self._widgets.particles.keys()
                    ).index(array_name)
                )

                ax.set_frame_on(True)

                ax.spines["right"].set_position(
                    (
                        "axes",
                        1.+0.2*(number_of_spines-1)/(1.-0.2*number_of_spines)
                    )
                )
                ax.spines["right"].set_visible(True)

                ax.set_ylabel(
                    array_name + " : " +
                    pa_widgets.scalar.value
                )
                ax.set_yticks(ticks=locs)
                ax.set_yticklabels(labels=labels)
                ax.tick_params(axis='y', colors=color)
                ax.yaxis.label.set_color(color)
                ax.spines["right"].set_color(color)
            else:
                ax.spines["right"].set_color('none')
                ax.yaxis.set_ticks([])
                ax.set_ylabel('')

        if number_of_spines == 0:
            self._scatters_ax['host'].set_frame_on(True)
            self._scatters_ax['host'].set_position(
                [0, 0, 1, 1]
            )
        if change is not None:
            self.figure.show()

    def _right_spine_lim_handler(self, change):
        self._frame_handler(None)


class ParticleArrayWidgets2D(object):

    def __init__(self, particlearray):

        self.array_name = particlearray.name
        self.scalar = widgets.Dropdown(
            options=['None'] +
            particlearray.output_property_arrays +
            ['vmag'],
            value='rho',
            description="scalar",
            disabled=False,
            layout=widgets.Layout(width='240px', display='flex')
        )
        self.scalar.owner = self.array_name
        self.scalar_cmap = widgets.Dropdown(
            options=list(map(str, plt.colormaps())),
            value='viridis',
            description="Colormap",
            disabled=False,
            layout=widgets.Layout(width='240px', display='flex')
        )
        self.scalar_cmap.owner = self.array_name
        self.legend = widgets.Checkbox(
            value=False,
            description='legend',
            disabled=False,
            layout=widgets.Layout(width='170px', display='flex')
        )
        self.legend.owner = self.array_name
        self.legend_lower_lim = widgets.Text(
            value='',
            placeholder='min',
            description='lower limit',
            disabled=False,
            layout=widgets.Layout(width='160px', display='flex'),
            continuous_update=False
        )
        self.legend_lower_lim.owner = self.array_name
        self.legend_upper_lim = widgets.Text(
            value='',
            placeholder='max',
            description='upper limit',
            disabled=False,
            layout=widgets.Layout(width='160px', display='flex'),
            continuous_update=False
        )
        self.legend_upper_lim.owner = self.array_name
        self.vector = widgets.Text(
            value='',
            placeholder='variable1,variable2',
            description='vector',
            disabled=False,
            layout=widgets.Layout(width='240px', display='flex'),
            continuous_update=False
        )
        self.vector.owner = self.array_name
        self.vector_width = widgets.FloatSlider(
            min=1,
            max=100,
            step=1,
            value=25,
            description='vector width',
            layout=widgets.Layout(width='300px'),
            continuous_update=False,
        )
        self.vector_width.owner = self.array_name
        self.vector_scale = widgets.FloatSlider(
            min=1,
            max=100,
            step=1,
            value=55,
            description='vector scale',
            layout=widgets.Layout(width='300px'),
            continuous_update=False,
        )
        self.vector_scale.owner = self.array_name
        self.scalar_size = widgets.FloatSlider(
            min=0,
            max=50,
            step=1,
            value=10,
            description='scalar size',
            layout=widgets.Layout(width='300px'),
            continuous_update=False,
        )
        self.scalar_size.owner = self.array_name
        self.is_visible = widgets.Checkbox(
            value=True,
            description='visible',
            disabled=False,
            layout=widgets.Layout(width='170px', display='flex')
        )
        self.is_visible.owner = self.array_name
        self.masking_factor = widgets.IntText(
            value=1,
            description='masking',
            disabled=False,
            layout=widgets.Layout(width='160px', display='flex'),
        )
        self.masking_factor.owner = self.array_name
        self.opacity = widgets.FloatSlider(
            min=0,
            max=1,
            step=0.01,
            value=1,
            description='opacity',
            layout=widgets.Layout(width='300px'),
            continuous_update=False,
        )
        self.opacity.owner = self.array_name
        self.components = widgets.IntText(
            value=1,
            description='component',
            disabled=True,
            layout=widgets.Layout(width='160px', display='flex'),
        )
        self.components.owner = self.array_name

    def _tab_config(self):

        VBox1 = widgets.VBox(
            [
                self.scalar,
                self.components,
                self.scalar_size,
                self.scalar_cmap,
            ]
        )
        VBox2 = widgets.VBox(
            [
                self.vector,
                self.vector_scale,
                self.vector_width,
                self.opacity,
            ]
        )
        VBox3 = widgets.VBox(
            [
                self.legend,
                widgets.HBox(
                    [
                        self.legend_upper_lim,
                        self.legend_lower_lim,
                    ]
                ),
                self.masking_factor,
                self.is_visible,
            ]
        )
        hbox = widgets.HBox([VBox1, VBox2, VBox3])
        return hbox


class Viewer2DWidgets(object):

    def __init__(self, file_name, file_count):

        self.temp_data = load(file_name)
        self.time = str(self.temp_data['solver_data']['t'])
        self.temp_data = self.temp_data['arrays']
        self.frame = widgets.IntSlider(
            min=0,
            max=file_count,
            step=1,
            value=0,
            description='frame',
            layout=widgets.Layout(width='500px'),
            continuous_update=False,
        )
        self.play_button = widgets.Play(
            min=0,
            max=file_count,
            step=1,
            disabled=False,
        )
        self.link = widgets.jslink(
            (self.frame, 'value'),
            (self.play_button, 'value'),
        )
        self.delay_box = widgets.FloatText(
            value=0.2,
            description='Delay',
            disabled=False,
            layout=widgets.Layout(width='160px', display='flex'),
        )
        self.save_figure = widgets.Text(
                value='',
                placeholder='example.pdf',
                description='Save figure',
                disabled=False,
                layout=widgets.Layout(width='240px', display='flex'),
        )
        self.save_all_plots = widgets.ToggleButton(
                value=False,
                description='Save all plots!',
                disabled=False,
                tooltip='Saves the corresponding plots for all the' +
                        ' frames in the presently set styling.',
                icon='',
                layout=widgets.Layout(display='flex'),
        )
        self.solver_time = widgets.HTML(
            value=self.time,
            description='Solver time:'
        )
        self.show_solver_time = widgets.Checkbox(
            value=False,
            description="Show solver time",
            disabled=False,
            layout=widgets.Layout(display='flex'),
        )
        self.print_config = widgets.Button(
            description='print present config.',
            tooltip='Prints the configuration dictionary ' +
                    'for the current viewer state',
            disabled=False,
            layout=widgets.Layout(display='flex'),
        )
        self.particles = {}
        for array_name in self.temp_data.keys():
            self.particles[array_name] = ParticleArrayWidgets2D(
                self.temp_data[array_name],
            )

    def _create_tabs(self):

        children = []
        for array_name in self.particles.keys():
            children.append(self.particles[array_name]._tab_config())

        tab = widgets.Tab(children=children)
        for i in range(len(children)):
            tab.set_title(i, list(self.particles.keys())[i])

        return widgets.VBox(
                [
                    tab,
                    widgets.HBox(
                     [
                        self.play_button,
                        self.frame
                     ]
                    ),
                    widgets.HBox(
                     [
                        self.delay_box,
                        self.save_figure,
                        self.save_all_plots,
                     ]
                    ),
                    widgets.HBox(
                     [
                        self.print_config,
                        self.show_solver_time,
                        self.solver_time,
                     ]
                    )
                ]
            )


class Viewer2D(Viewer):

    '''
    Example
    -------

    >>> from pysph.tools.ipy_viewer import Viewer2D
    >>> sample = Viewer2D(
        '/home/uname/pysph_files/dam_Break_2d_output'
        )
    >>> sample.interactive_plot()
    >>> sample.show_log()
    >>> sample.show_info()
    '''

    def _configure_plot(self):

        '''
        Set attributes for plotting.
        '''

        self.figure = plt.figure()
        self._scatter_ax = self.figure.add_axes([0, 0, 1, 1])
        self._vector_ax = self.figure.add_axes(
            self._scatter_ax.get_position(),
            frameon=False
            )
        self._vector_ax.get_xaxis().set_visible(False)
        self._vector_ax.get_yaxis().set_visible(False)
        self._scatters = {}
        self._cbar_ax = {}
        self._cbars = {}
        self._vectors = {}
        self._solver_time_ax = {}

        self.figure.show()

    def interactive_plot(self, config={}):

        '''
        Set plotting attributes, create widgets and display them
        along with the interactive plot.
        '''

        self.config = config  # The configuration dictionary.
        self.viewer_type = 'Viewer2D'
        self._configure_plot()
        self._create_widgets()
        display(self._widgets._create_tabs())
        temp_data = self.get_frame(0)
        self.time = str(temp_data['solver_data']['t'])
        temp_data = temp_data['arrays']

        for sct in self._scatters.values():
            if sct in self._scatter_ax.collections:
                self._scatter_ax.collections.remove(sct)

        self._scatters = {}
        for array_name in self._widgets.particles.keys():
            pa_widgets = self._widgets.particles[array_name]
            if pa_widgets.scalar.value != 'None':
                n = pa_widgets.masking_factor.value
                alpha = pa_widgets.opacity.value
                stride, component = self._stride_and_component(
                    temp_data[array_name], pa_widgets
                )
                sct = self._scatters[array_name] = self._scatter_ax.scatter(
                    temp_data[array_name].x[component::stride][::n],
                    temp_data[array_name].y[component::stride][::n],
                    s=pa_widgets.scalar_size.value,
                )
                c = self._get_c(
                    pa_widgets,
                    temp_data[array_name],
                    component,
                    stride
                )
                colormap = getattr(
                    plt.cm,
                    pa_widgets.scalar_cmap.value
                )
                min_c, max_c, c_norm = self._cmap_helper(
                    c,
                    array_name
                )
                cm = colormap(c_norm[::n])
                cm[0:, 3] *= alpha
                sct.set_facecolors(cm)
                if pa_widgets.is_visible.value is False:
                    sct.set_offsets(None)

        self._scatter_ax.axis('equal')

        self.solver_time_textbox = None
        # So that _show_solver_time_handler does not glitch at intialization.
        self._frame_handler(None)
        self._show_solver_time_handler(None)
        self._legend_handler(None)
        self._plot_vectors()

    def _plot_vectors(self):

        temp_data = self.get_frame(self._widgets.frame.value)
        temp_data = temp_data['arrays']
        self.figure.delaxes(self._vector_ax)
        self._vector_ax = self.figure.add_axes(
            self._scatter_ax.get_position(),
            frameon=False
            )
        self._vector_ax.get_xaxis().set_visible(False)
        self._vector_ax.get_yaxis().set_visible(False)
        self._vectors = {}

        for array_name in self._widgets.particles.keys():
            pa_widgets = self._widgets.particles[array_name]
            if (pa_widgets.vector.value != '' and
                    pa_widgets.is_visible.value is True):
                n = pa_widgets.masking_factor.value
                stride, component = self._stride_and_component(
                    temp_data[array_name], pa_widgets
                )
                temp_data_arr = temp_data[array_name]
                x = temp_data_arr.x[component::stride][::n]
                y = temp_data_arr.y[component::stride][::n]

                try:
                    v1 = getattr(
                        temp_data_arr,
                        pa_widgets.vector.value.split(",")[0]
                        )[component::stride][::n]
                    v2 = getattr(
                        temp_data_arr,
                        pa_widgets.vector.value.split(",")[1]
                        )[component::stride][::n]
                except AttributeError:
                    continue

                vmag = (v1**2 + v2**2)**0.5
                self._vectors[array_name] = self._vector_ax.quiver(
                    x,
                    y,
                    v1,
                    v2,
                    vmag,
                    scale=pa_widgets.vector_scale.value,
                    width=(pa_widgets.vector_width.value)/10000,
                    units='xy'
                )
        self._vector_ax.set_xlim(self._scatter_ax.get_xlim())
        self._vector_ax.set_ylim(self._scatter_ax.get_ylim())

    def _frame_handler(self, change):

        temp_data = self.get_frame(self._widgets.frame.value)
        self.time = str(temp_data['solver_data']['t'])
        self._widgets.solver_time.value = self.time
        temp_data = temp_data['arrays']

        for array_name in self._widgets.particles.keys():
            pa_widgets = self._widgets.particles[array_name]
            if (pa_widgets.scalar.value != 'None' and
                    pa_widgets.is_visible.value is True):
                n = pa_widgets.masking_factor.value
                alpha = pa_widgets.opacity.value
                stride, component = self._stride_and_component(
                    temp_data[array_name], pa_widgets
                )
                sct = self._scatters[array_name]
                sct.set_offsets(
                    np.vstack(
                        (
                            temp_data[array_name].x[component::stride][::n],
                            temp_data[array_name].y[component::stride][::n]
                        )
                    ).T
                )
                c = self._get_c(
                    pa_widgets,
                    temp_data[array_name],
                    component,
                    stride
                )
                colormap = getattr(
                    plt.cm,
                    pa_widgets.scalar_cmap.value
                )
                min_c, max_c, c_norm = self._cmap_helper(
                    c,
                    array_name
                )
                cm = colormap(c_norm[::n])
                cm[0:, 3] *= alpha
                sct.set_facecolors(cm)

        self._legend_handler(None)
        self._vector_handler(None)
        self._show_solver_time_handler(None)
        self._adjust_axes()
        self.figure.show()

    def _scalar_handler(self, change):

        array_name = change['owner'].owner
        pa_widgets = self._widgets.particles[array_name]
        if pa_widgets.is_visible.value is True:
            n = pa_widgets.masking_factor.value
            alpha = pa_widgets.opacity.value
            temp_data = self.get_frame(
                self._widgets.frame.value
            )['arrays']
            stride, component = self._stride_and_component(
                    temp_data[array_name], pa_widgets
            )
            sct = self._scatters[array_name]
            c = self._get_c(
                pa_widgets,
                temp_data[array_name],
                component,
                stride
            )
            colormap = getattr(
                plt.cm,
                pa_widgets.scalar_cmap.value
            )
            min_c, max_c, c_norm = self._cmap_helper(
                c,
                array_name
            )
            cm = colormap(c_norm[::n])
            cm[0:, 3] *= alpha

            new = change['new']
            old = change['old']

            if (new == 'None' and old == 'None'):
                pass
            elif (new == 'None' and old != 'None'):
                sct.set_offsets(None)
            elif (new != 'None' and old == 'None'):
                sct.set_offsets(
                    np.vstack(
                        (
                            temp_data[array_name].x[component::stride][::n],
                            temp_data[array_name].y[component::stride][::n]
                        )
                    ).T
                )
                sct.set_facecolors(cm)
            elif (new != 'None' and old != 'None'):
                sct.set_facecolors(cm)

            self._plot_vectors()
            self._legend_handler(None)
            self.figure.show()

    def _vector_handler(self, change):
        '''
        Bug : Arrows go out of the figure
        '''

        self._plot_vectors()
        if change is not None:
            pa_widgets = self._widgets.particles[change['owner'].owner]
            if pa_widgets.is_visible.value is True:
                self.figure.show()

    def _vector_scale_handler(self, change):

        self._plot_vectors()
        pa_widgets = self._widgets.particles[change['owner'].owner]
        if pa_widgets.is_visible.value is True:
            self.figure.show()

    def _adjust_axes(self):

        if hasattr(self, '_vector_ax'):
            self._vector_ax.set_xlim(self._scatter_ax.get_xlim())
            self._vector_ax.set_ylim(self._scatter_ax.get_ylim())
        else:
            pass

    def _scalar_size_handler(self, change):

        array_name = change['owner'].owner
        pa_widgets = self._widgets.particles[array_name]
        if pa_widgets.is_visible.value is True:
            self._scatters[array_name].set_sizes([change['new']])
            self.figure.show()

    def _vector_width_handler(self, change):

        self._plot_vectors()
        pa_widgets = self._widgets.particles[change['owner'].owner]
        if pa_widgets.is_visible.value is True:
            self.figure.show()

    def _scalar_cmap_handler(self, change):

        array_name = change['owner'].owner
        pa_widgets = self._widgets.particles[array_name]
        if pa_widgets.is_visible.value is True:
            n = pa_widgets.masking_factor.value
            alpha = pa_widgets.opacity.value
            temp_data = self.get_frame(
                self._widgets.frame.value
            )['arrays']
            stride, component = self._stride_and_component(
                    temp_data[array_name], pa_widgets
            )
            c = self._get_c(
                pa_widgets,
                temp_data[array_name],
                component,
                stride
            )
            colormap = getattr(
                plt.cm,
                pa_widgets.scalar_cmap.value
            )
            sct = self._scatters[array_name]
            min_c, max_c, c_norm = self._cmap_helper(
                c,
                array_name
            )
            cm = colormap(c_norm[::n])
            cm[0:, 3] *= alpha
            sct.set_facecolors(cm)
            self._legend_handler(None)
            self.figure.show()

    def _legend_handler(self, change):

        temp_data = self.get_frame(
            self._widgets.frame.value
        )['arrays']
        for _cbar_ax in self._cbar_ax.values():
            self.figure.delaxes(_cbar_ax)
        self._cbar_ax = {}
        self._cbars = {}

        for array_name in self._widgets.particles.keys():
            pa_widgets = self._widgets.particles[array_name]
            if (pa_widgets.legend.value is True and
                    pa_widgets.is_visible.value is True):
                if pa_widgets.scalar.value != 'None':
                    stride, component = self._stride_and_component(
                        temp_data[array_name], pa_widgets
                    )
                    c = self._get_c(
                        pa_widgets,
                        temp_data[array_name],
                        component,
                        stride
                    )
                    cmap = pa_widgets.scalar_cmap.value
                    colormap = getattr(mpl.cm, cmap)
                    self._scatter_ax.set_position(
                        [0, 0, 0.84 - 0.15*len(self._cbars.keys()), 1]
                    )
                    self._cbar_ax[array_name] = self.figure.add_axes(
                            [
                                0.85 - 0.15*len(self._cbars.keys()),
                                0.02,
                                0.02,
                                0.82
                            ]
                    )

                    min_c, max_c, c_norm = self._cmap_helper(
                        c,
                        array_name
                    )

                    if max_c == 0:
                        ticks = np.linspace(0, 1, 26)
                        norm = mpl.colors.Normalize(vmin=0, vmax=1)
                    elif max_c == min_c:
                        # this occurs at initialization for some properties
                        # like pressure, and stays true throughout for
                        # others like mass of the particles
                        ticks = np.linspace(0, max_c, 26)
                        norm = mpl.colors.Normalize(vmin=0, vmax=max_c)
                    else:
                        ticks = np.linspace(min_c, max_c, 26)
                        norm = mpl.colors.Normalize(vmin=min_c, vmax=max_c)

                    self._cbars[array_name] = mpl.colorbar.ColorbarBase(
                        ax=self._cbar_ax[array_name],
                        cmap=colormap,
                        norm=norm,
                        ticks=ticks,
                    )
                    self._cbars[array_name].set_label(
                        array_name + " : " +
                        pa_widgets.scalar.value
                    )
        if len(self._cbars.keys()) == 0:
            self._scatter_ax.set_position(
                [0, 0, 1, 1]
            )
        self._plot_vectors()
        if change is not None:
            self.figure.show()

    def _save_figure_handler(self, change):

        file_was_saved = False
        for extension in [
            '.eps', '.pdf', '.pgf',
            '.png', '.ps', '.raw',
            '.rgba', '.svg', '.svgz'
        ]:
            if self._widgets.save_figure.value.endswith(extension):
                self.figure.savefig(self._widgets.save_figure.value)
                print(
                    "Saved figure as {} in the present working directory"
                    .format(
                        self._widgets.save_figure.value
                    )
                )
                file_was_saved = True
                break
        self._widgets.save_figure.value = ""
        if file_was_saved is False:
            print(
                "Please use a valid extension, that is, one of the following" +
                ": '.eps', '.pdf', '.pgf', '.png', '.ps', '.raw', '.rgba'," +
                " '.svg' or '.svgz'."
            )

    def _is_visible_handler(self, change):

        array_name = change['owner'].owner
        pa_widgets = self._widgets.particles[array_name]
        temp_data = self.get_frame(self._widgets.frame.value)['arrays']
        sct = self._scatters[array_name]

        if pa_widgets.scalar.value != 'None':
            if change['new'] is False:
                sct.set_offsets(None)
            elif change['new'] is True:
                n = pa_widgets.masking_factor.value
                alpha = pa_widgets.opacity.value
                stride, component = self._stride_and_component(
                    temp_data[array_name], pa_widgets
                )
                sct.set_offsets(
                    np.vstack(
                        (
                            temp_data[array_name].x[component::stride][::n],
                            temp_data[array_name].y[component::stride][::n]
                        )
                    ).T
                )
                c = self._get_c(
                    pa_widgets,
                    temp_data[array_name],
                    component,
                    stride
                )
                colormap = getattr(
                    plt.cm,
                    pa_widgets.scalar_cmap.value
                )
                min_c, max_c, c_norm = self._cmap_helper(
                    c,
                    array_name
                )
                cm = colormap(c_norm[::n])
                cm[0:, 3] *= alpha
                sct.set_facecolors()

        self._legend_handler(None)
        self._plot_vectors()
        self.figure.show()

    def _show_solver_time_handler(self, change):

        if self._widgets.show_solver_time.value is True:
            if self.solver_time_textbox is not None:
                self.solver_time_textbox.remove()
            self.solver_time_textbox = self._scatter_ax.text(
                x=0.02,
                y=0.02,
                s='Solver time: ' + self.time,
                verticalalignment='bottom',
                horizontalalignment='left',
                transform=self._scatter_ax.transAxes,
                fontsize=12,
                bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 3},
            )
        elif self._widgets.show_solver_time.value is False:
            if self.solver_time_textbox is not None:
                self.solver_time_textbox.remove()
            self.solver_time_textbox = None
        if change is not None:
            self.figure.show()


class ParticleArrayWidgets3D(object):

    def __init__(self, particlearray):
        self.array_name = particlearray.name
        self.scalar = widgets.Dropdown(
            options=['None'] +
            particlearray.output_property_arrays +
            ['vmag'],
            value='rho',
            description="scalar",
            disabled=False,
            layout=widgets.Layout(width='240px', display='flex')
        )
        self.scalar.owner = self.array_name
        self.scalar_cmap = widgets.Dropdown(
            options=list(map(str, plt.colormaps())),
            value='viridis',
            description="Colormap",
            disabled=False,
            layout=widgets.Layout(width='240px', display='flex')
        )
        self.scalar_cmap.owner = self.array_name
        self.legend = widgets.Checkbox(
            value=False,
            description="legend",
            disabled=False,
            layout=widgets.Layout(width='200px', display='flex')
        )
        self.legend.owner = self.array_name
        self.legend_lower_lim = widgets.Text(
            value='',
            description='lower limit',
            placeholder='min',
            disabled=False,
            continuous_update=False,
            layout=widgets.Layout(width='160px', display='flex'),
        )
        self.legend_lower_lim.owner = self.array_name
        self.legend_upper_lim = widgets.Text(
            value='',
            description='upper limit',
            placeholder='max',
            disabled=False,
            continuous_update=False,
            layout=widgets.Layout(width='160px', display='flex')
        )
        self.legend_upper_lim.owner = self.array_name
        self.velocity_vectors = widgets.Checkbox(
            value=False,
            description="Velocity Vectors",
            disabled=False,
            layout=widgets.Layout(width='300px', display='flex')
        )
        self.velocity_vectors.owner = self.array_name
        self.vector_size = widgets.FloatSlider(
            min=1,
            max=10,
            step=0.01,
            value=5.5,
            description='vector size',
            layout=widgets.Layout(width='300px'),
        )
        self.vector_size.owner = self.array_name
        self.scalar_size = widgets.FloatSlider(
            min=0,
            max=3,
            step=0.02,
            value=1,
            description='scalar size',
            layout=widgets.Layout(width='300px'),
        )
        self.scalar_size.owner = self.array_name
        self.is_visible = widgets.Checkbox(
            value=True,
            description="visible",
            disabled=False,
            layout=widgets.Layout(width='200px', display='flex')
        )
        self.is_visible.owner = self.array_name
        self.masking_factor = widgets.IntText(
            value=1,
            description='masking',
            disabled=False,
            layout=widgets.Layout(width='160px', display='flex')
        )
        self.masking_factor.owner = self.array_name
        self.scatter_plot_marker = widgets.Dropdown(
            options=[
                        'arrow',
                        'box',
                        'diamond',
                        'sphere',
                        'point_2d',
                        'square_2d',
                        'triangle_2d',
                        'circle_2d'
            ],
            value='circle_2d',
            description="Marker",
            disabled=False,
            layout=widgets.Layout(width='240px', display='flex')
        )
        self.scatter_plot_marker.owner = self.array_name
        self.components = widgets.IntText(
            value=1,
            description='component',
            disabled=True,
            layout=widgets.Layout(width='160px', display='flex'),
        )
        self.components.owner = self.array_name

    def _tab_config(self):

        VBox1 = widgets.VBox(
            [
                self.scalar,
                self.components,
                self.scalar_size,
                self.scalar_cmap,
            ]
        )
        VBox2 = widgets.VBox(
            [
                self.velocity_vectors,
                self.vector_size,
                self.is_visible,
            ]
        )
        VBox3 = widgets.VBox(
            [
                self.legend,
                widgets.HBox(
                    [
                        self.legend_upper_lim,
                        self.legend_lower_lim,
                    ]
                ),
                self.masking_factor,
                self.scatter_plot_marker,
            ]
        )
        hbox = widgets.HBox([VBox1, VBox2, VBox3])
        return hbox


class Viewer3DWidgets(object):

    def __init__(self, file_name, file_count):

        self.temp_data = load(file_name)
        self.time = str(self.temp_data['solver_data']['t'])
        self.temp_data = self.temp_data['arrays']
        self.frame = widgets.IntSlider(
            min=0,
            max=file_count,
            step=1,
            value=0,
            description='frame',
            layout=widgets.Layout(width='500px'),
            continuous_update=False
        )
        self.play_button = widgets.Play(
            min=0,
            max=file_count,
            step=1,
            disabled=False,
            interval=1000,
        )
        self.link = widgets.jslink(
            (self.frame, 'value'),
            (self.play_button, 'value')
        )
        self.delay_box = widgets.FloatText(
            value=1.0,
            description='Delay',
            disabled=False,
            layout=widgets.Layout(width='160px', display='flex')
        )
        self.save_figure = widgets.Text(
                value='',
                placeholder='example.png',
                description='Save figure',
                disabled=False,
                layout=widgets.Layout(width='240px', display='flex')
        )
        self.save_all_plots = widgets.ToggleButton(
                value=False,
                description='Save all plots!',
                disabled=False,
                tooltip='Saves the corresponding plots for all the' +
                        ' frames in the presently set styling.',
                icon='',
                layout=widgets.Layout(display='flex'),
        )
        self.solver_time = widgets.HTML(
                value=self.time,
                description='Solver time:',
        )
        self.print_config = widgets.Button(
            description='print present config.',
            tooltip='Prints the configuration dictionary ' +
                    'for the current viewer state',
            disabled=False,
            layout=widgets.Layout(display='flex'),
        )
        self.particles = {}
        for array_name in self.temp_data.keys():
            self.particles[array_name] = ParticleArrayWidgets3D(
                self.temp_data[array_name],
            )

    def _create_tabs(self):

        children = []
        for array_name in self.particles.keys():
            children.append(self.particles[array_name]._tab_config())

        tab = widgets.Tab(children=children)
        for i in range(len(children)):
            tab.set_title(i, list(self.particles.keys())[i])

        return widgets.VBox(
                [
                    tab,
                    widgets.HBox(
                     [
                        self.play_button,
                        self.frame
                     ]
                    ),
                    widgets.HBox(
                     [
                        self.delay_box,
                        self.save_figure,
                        self.save_all_plots,
                     ]
                    ),
                    widgets.HBox(
                     [
                        self.print_config,
                        self.solver_time,
                     ]
                    )
                ]
            )


class Viewer3D(Viewer):

    '''
    Example
    -------

    >>> from pysph.tools.ipy_viewer import Viewer3D
    >>> sample = Viewer3D(
        '/home/uname/pysph_files/dam_Break_3d_output'
        )
    >>> sample.interactive_plot()
    >>> sample.show_log()
    >>> sample.show_info()
    '''

    def interactive_plot(self, config={}):

        import ipyvolume.pylab as p3

        self.config = config  # The configuration dictionary.
        self.viewer_type = 'Viewer3D'
        self._create_widgets()
        self._scatters = {}
        self._vectors = {}
        self._cbars = {}
        self._cbar_ax = {}
        self._cbar_labels = {}

        self.pltfigure = plt.figure(figsize=(9, 1), dpi=100)
        self._initial_ax = self.pltfigure.add_axes([0, 0, 1, 1])
        self._initial_ax.axis('off')
        # Creating a dummy axes element, that prevents the plot
        # from glitching and showing random noise when no legends are
        # being displayed.

        self.plot = p3.figure(width=800)
        temp_data = self.get_frame(0)['arrays']
        for array_name in self._widgets.particles.keys():
            pa_widgets = self._widgets.particles[array_name]
            if pa_widgets.scalar.value != 'None':
                n = pa_widgets.masking_factor.value
                colormap = getattr(mpl.cm, pa_widgets.scalar_cmap.value)
                stride, component = self._stride_and_component(
                    temp_data[array_name], pa_widgets
                )
                c = self._get_c(
                    pa_widgets,
                    temp_data[array_name],
                    component,
                    stride
                )
                min_c, max_c, c_norm = self._cmap_helper(
                    c,
                    array_name
                )
                cm = colormap(c_norm[::n])
                self._scatters[array_name] = p3.scatter(
                    temp_data[array_name].x[component::stride][::n],
                    temp_data[array_name].y[component::stride][::n],
                    temp_data[array_name].z[component::stride][::n],
                    color=cm,
                    size=pa_widgets.scalar_size.value,
                    marker=pa_widgets.scatter_plot_marker.value,
                )
                self._plot_vectors(
                    pa_widgets,
                    temp_data[array_name],
                    array_name
                )
                if pa_widgets.is_visible.value is False:
                    self._scatters[array_name].visible = False
                    if array_name in self._vectors.keys():
                        copy = self.plot.scatters.copy()
                        copy.remove(self._vectors[array_name])
                        self.plot.scatters = copy
                        del self._vectors[array_name]

        p3.squarelim()  # Makes sure the figure doesn't appear distorted.

        self._frame_handler(None)
        self._legend_handler(None)

        self.pltfigure.show()
        display(self.plot)
        display(self._widgets._create_tabs())

    def _plot_vectors(self, pa_widgets, data, array_name):

        if pa_widgets.velocity_vectors.value is True:
            n = pa_widgets.masking_factor.value
            colormap = getattr(mpl.cm, pa_widgets.scalar_cmap.value)
            stride, component = self._stride_and_component(
                data, pa_widgets
            )
            c = self._get_c(
                pa_widgets,
                data,
                component,
                stride,
                need_vmag=True
            )
            min_c, max_c, c_norm = self._cmap_helper(
                c,
                array_name,
                for_plot_vectors=True
            )
            if array_name in self._vectors.keys():
                vectors = self._vectors[array_name]
                vectors.x = data.x[component::stride][::n]
                vectors.y = data.y[component::stride][::n]
                vectors.z = data.z[component::stride][::n]
                vectors.vx = getattr(data, 'u')[component::stride][::n]
                vectors.vy = getattr(data, 'v')[component::stride][::n]
                vectors.vz = getattr(data, 'w')[component::stride][::n]
                vectors.size = pa_widgets.vector_size.value
                vectors.color = colormap(c_norm)[::n]
            else:
                import ipyvolume.pylab as p3
                self._vectors[array_name] = p3.quiver(
                    x=data.x[component::stride][::n],
                    y=data.y[component::stride][::n],
                    z=data.z[component::stride][::n],
                    u=getattr(data, 'u')[component::stride][::n],
                    v=getattr(data, 'v')[component::stride][::n],
                    w=getattr(data, 'w')[component::stride][::n],
                    size=pa_widgets.vector_size.value,
                    color=colormap(c_norm)[::n]
                )
        else:
            pass

    def _frame_handler(self, change):

        temp_data = self.get_frame(self._widgets.frame.value)
        self.time = str(temp_data['solver_data']['t'])
        self._widgets.solver_time.value = self.time
        temp_data = temp_data['arrays']
        for array_name in self._widgets.particles.keys():
            pa_widgets = self._widgets.particles[array_name]
            if pa_widgets.is_visible.value is True:
                if pa_widgets.scalar.value != 'None':
                    n = pa_widgets.masking_factor.value
                    colormap = getattr(mpl.cm, pa_widgets.scalar_cmap.value)
                    stride, comp = self._stride_and_component(
                        temp_data[array_name], pa_widgets
                    )
                    c = self._get_c(
                        pa_widgets,
                        temp_data[array_name],
                        comp,
                        stride
                    )
                    scatters = self._scatters[array_name]
                    min_c, max_c, c_norm = self._cmap_helper(
                        c,
                        array_name
                    )
                    cm = colormap(c_norm[::n])
                    scatters.x = temp_data[array_name].x[comp::stride][::n]
                    scatters.y = temp_data[array_name].y[comp::stride][::n]
                    scatters.z = temp_data[array_name].z[comp::stride][::n]
                    scatters.color = cm
                self._plot_vectors(
                    pa_widgets,
                    temp_data[array_name],
                    array_name
                )
        self._legend_handler(None)

    def _scalar_handler(self, change):

        import ipyvolume.pylab as p3

        array_name = change['owner'].owner
        pa_widgets = self._widgets.particles[array_name]
        new = change['new']
        old = change['old']
        if pa_widgets.is_visible.value is True:
            if old == 'None' and new == 'None':
                pass
            elif old != 'None' and new == 'None':
                copy = self.plot.scatters.copy()
                copy.remove(self._scatters[array_name])
                self.plot.scatters = copy
                del self._scatters[array_name]
            else:
                n = pa_widgets.masking_factor.value
                colormap = getattr(mpl.cm, pa_widgets.scalar_cmap.value)
                temp_data = self.get_frame(self._widgets.frame.value)['arrays']
                stride, component = self._stride_and_component(
                    temp_data[array_name], pa_widgets
                )
                c = self._get_c(
                    pa_widgets,
                    temp_data[array_name],
                    component,
                    stride
                )
                min_c, max_c, c_norm = self._cmap_helper(
                    c,
                    array_name
                )
                cm = colormap(c_norm[::n])
                if old != 'None' and new != 'None':
                    self._scatters[array_name].color = cm
                else:
                    self._scatters[array_name] = p3.scatter(
                        temp_data[array_name].x[component::stride][::n],
                        temp_data[array_name].y[component::stride][::n],
                        temp_data[array_name].z[component::stride][::n],
                        color=cm,
                        size=pa_widgets.scalar_size.value,
                        marker=pa_widgets.scatter_plot_marker.value,
                    )
        self._legend_handler(None)

    def _velocity_vectors_handler(self, change):

        import ipyvolume.pylab as p3

        temp_data = self.get_frame(self._widgets.frame.value)['arrays']
        array_name = change['owner'].owner
        pa_widgets = self._widgets.particles[array_name]
        if pa_widgets.is_visible.value is True:
            if change['new'] is False:
                copy = self.plot.scatters.copy()
                copy.remove(self._vectors[array_name])
                self.plot.scatters = copy
                del self._vectors[array_name]
            else:
                self._plot_vectors(
                    pa_widgets,
                    temp_data[array_name],
                    array_name
                )

    def _scalar_size_handler(self, change):

        array_name = change['owner'].owner
        if (array_name in self._scatters.keys() and
                self._widgets.particles[array_name].is_visible.value is True):
            self._scatters[array_name].size = change['new']

    def _vector_size_handler(self, change):

        array_name = change['owner'].owner
        pa_widgets = self._widgets.particles[array_name]
        if array_name in self._vectors.keys():
            if (pa_widgets.velocity_vectors.value is True and
                    pa_widgets.is_visible.value is True):
                self._vectors[array_name].size = change['new']

    def _scalar_cmap_handler(self, change):

        array_name = change['owner'].owner
        pa_widgets = self._widgets.particles[array_name]
        temp_data = self.get_frame(self._widgets.frame.value)['arrays']

        if pa_widgets.is_visible.value is True:
            n = pa_widgets.masking_factor.value
            colormap = getattr(mpl.cm, change['new'])
            stride, component = self._stride_and_component(
                temp_data[array_name], pa_widgets
            )
            c = self._get_c(
                pa_widgets,
                temp_data[array_name],
                component,
                stride
            )
            min_c, max_c, c_norm = self._cmap_helper(
                c,
                array_name
            )
            cm = colormap(c_norm[::n])
            self._scatters[array_name].color = cm
            self._legend_handler(None)

    def _legend_handler(self, change):

        if len(self._cbar_ax) == 4:
            print(
                'Four legends are already being displayed. This is the ' +
                'maximum number of legends that can be displayed at once. ' +
                'Please deactivate one of them if you wish to display another.'
            )
        else:
            temp_data = self.get_frame(self._widgets.frame.value)['arrays']
            for array_name in self._cbar_ax.keys():
                self.pltfigure.delaxes(self._cbar_ax[array_name])
                self._cbar_labels[array_name].remove()
            self._cbar_labels = {}
            self._cbar_ax = {}
            self._cbars = {}
            for array_name in self._widgets.particles.keys():
                pa_widgets = self._widgets.particles[array_name]
                if (pa_widgets.scalar.value != 'None' and
                        pa_widgets.legend.value is True):
                    if pa_widgets.is_visible.value is True:
                        cmap = getattr(mpl.cm, pa_widgets.scalar_cmap.value)
                        stride, component = self._stride_and_component(
                            temp_data[array_name], pa_widgets
                        )
                        c = self._get_c(
                            pa_widgets,
                            temp_data[array_name],
                            component,
                            stride
                        )
                        min_c, max_c, c_norm = self._cmap_helper(
                            c,
                            array_name
                        )

                        if max_c == 0:
                            ticks = np.linspace(0, 1, 11)
                            norm = mpl.colors.Normalize(vmin=0, vmax=1)
                        elif min_c == max_c:
                            # This occurs at initialization for some properties
                            # like pressure, and stays true throughout for
                            # others like mass of the particles.
                            ticks = np.linspace(0, max_c, 11)
                            norm = mpl.colors.Normalize(vmin=0, vmax=max_c)
                        else:
                            ticks = np.linspace(min_c, max_c, 11)
                            norm = mpl.colors.Normalize(vmin=min_c, vmax=max_c)

                        self._cbar_ax[array_name] = self.pltfigure.add_axes(
                            [
                                0.05,
                                0.75 - 0.25*len(self._cbars.keys()),
                                0.75,
                                0.09
                            ]
                        )
                        self._cbars[array_name] = mpl.colorbar.ColorbarBase(
                            ax=self._cbar_ax[array_name],
                            cmap=cmap,
                            norm=norm,
                            ticks=ticks,
                            orientation='horizontal'
                        )
                        self._cbar_ax[array_name].tick_params(
                            direction='in',
                            pad=0,
                            bottom=False,
                            top=True,
                            labelbottom=False,
                            labeltop=True,
                        )
                        self._cbar_labels[array_name] = self._initial_ax.text(
                            x=0.82,
                            y=1 - 0.25*len(self._cbars.keys()),
                            s=array_name + " : " + pa_widgets.scalar.value,
                        )
            self.pltfigure.show()

    def _save_figure_handler(self, change):

        import ipyvolume.pylab as p3

        file_was_saved = False
        for extension in [
            '.jpg', '.jpeg', '.png', '.svg'
        ]:
            if self._widgets.save_figure.value.endswith(extension):
                p3.savefig(
                            self._widgets.save_figure.value,
                            width=600,
                            height=600,
                            fig=self.plot
                        )
                print(
                    "Saved figure as {} in the present working directory"
                    .format(
                        self._widgets.save_figure.value
                    )
                )
                flag = True
                break
        if file_was_saved is False:
            print(
                "Please use '.jpg', '.jpeg', '.png' or" +
                "'.svg' as the file extension."
                )
        self._widgets.save_figure.value = ""

    def _is_visible_handler(self, change):

        import ipyvolume.pylab as p3

        array_name = change['owner'].owner
        pa_widgets = self._widgets.particles[array_name]

        if pa_widgets.scalar.value != 'None':
            if change['new'] is False:
                copy = self.plot.scatters.copy()
                copy.remove(self._scatters[array_name])
                self.plot.scatters = copy
                del self._scatters[array_name]
                if array_name in self._vectors.keys():
                    copy = self.plot.scatters.copy()
                    copy.remove(self._vectors[array_name])
                    self.plot.scatters = copy
                    del self._vectors[array_name]
            elif change['new'] is True:
                n = pa_widgets.masking_factor.value
                colormap = getattr(mpl.cm, pa_widgets.scalar_cmap.value)
                temp_data = self.get_frame(self._widgets.frame.value)['arrays']
                stride, component = self._stride_and_component(
                            temp_data[array_name], pa_widgets
                )
                c = self._get_c(
                    pa_widgets,
                    temp_data[array_name],
                    component,
                    stride
                )
                min_c, max_c, c_norm = self._cmap_helper(
                    c,
                    array_name
                )
                cm = colormap(c_norm[::n])
                self._scatters[array_name] = p3.scatter(
                    temp_data[array_name].x[component::stride][::n],
                    temp_data[array_name].y[component::stride][::n],
                    temp_data[array_name].z[component::stride][::n],
                    color=cm,
                    size=pa_widgets.scalar_size.value,
                    marker=pa_widgets.scatter_plot_marker.value,
                )
                self._velocity_vectors_handler(change)

            self._legend_handler(None)

    def _scatter_plot_marker_handler(self, change):

        change['new'] = False
        self._is_visible_handler(change)
        change['new'] = True
        self._is_visible_handler(change)
