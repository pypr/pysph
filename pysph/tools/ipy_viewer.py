import json
import glob
from pysph.solver.utils import load, get_files, mkdir
from IPython.display import display, Image, clear_output, HTML
import ipywidgets as widgets
import numpy as np
import matplotlib as mpl

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
            temppath = self.path + "*." + extension
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

    def _cmap_helper(self, data, array_name):
        '''
        Helper Function:
        Takes in a numpy array and returns its maximum,
        minimum and absolute maximum values, subject to the constraints
        provided by the user in the legend_lower_lim and legend_upper_lim
        text boxes. Also returns the input array normalized by the maximum.
        '''

        pa_widgets = self._widgets.particles[array_name]
        ulim = pa_widgets.legend_upper_lim.value
        llim = pa_widgets.legend_lower_lim.value
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
                return actual_minm, actual_maxm, np.ones_like(data)
            else:
                return actual_minm, actual_maxm, (data-actual_minm)/(actual_maxm-actual_minm)
        else:
            return actual_minm, actual_maxm, (data - actual_minm)/(actual_maxm - actual_minm)


class ParticleArrayWidgets(object):

    def __init__(self, particlearray):

        self.array_name = particlearray.name
        self.scalar = widgets.Dropdown(
            options=[
                'None'
            ] + particlearray.output_property_arrays,
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

    def _tab_config(self):
        
        VBox1 = widgets.VBox(
            [
                self.scalar,
                self.scalar_size,
                self.scalar_cmap,
            ]
        )
        VBox2 = widgets.VBox(
            [
                self.vector,
                self.vector_scale,
                self.vector_width,
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
        )
        self.solver_time = widgets.HTML(
            value=self.time,
            description='Solver time:'
        )
        self.show_solver_time = widgets.Checkbox(
            value=False,
            description="Show solver time",
            disabled=False,
            layout=widgets.Layout(width='240px', display='flex')
        )
        self.particles = {}
        for array_name in self.temp_data.keys():
            self.particles[array_name] = ParticleArrayWidgets(
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

    def _create_widgets(self):

        self._widgets = Viewer2DWidgets(
            file_name=self.paths_list[0],
            file_count=len(self.paths_list) - 1,
        )
        self._widgets.frame.observe(self._frame_handler, 'value')
        self._widgets.save_figure.on_submit(self._save_figure_handler)
        self._widgets.delay_box.observe(self._delay_box_handler, 'value')
        self._widgets.save_all_plots.observe(
            self._save_all_plots_handler,
            'value'
        )
        self._widgets.show_solver_time.observe(
            self._show_solver_time_handler,
            'value'
        )

        # PLEASE NOTE:
        # All widget handlers take in 'change' as an argument. This is usually
        # a dictionary containing information about the widget and the change
        # in state. However, these functions are also used outside of the use
        # case of a user-triggered-event, and in these scenarios None should
        # be passed as the argument. This is of particular significance
        # because in some of these functions plt.figure.show() gets called
        # only if the argument passed is not None.

        for array_name in self._widgets.particles.keys():
            pa_widgets = self._widgets.particles[array_name]
            pa_widgets.scalar.observe(self._scalar_handler, 'value')
            pa_widgets.vector.observe(self._vector_handler, 'value')
            pa_widgets.vector_width.observe(
                self._vector_width_handler,
                'value'
            )
            pa_widgets.vector_scale.observe(
                self._vector_scale_handler,
                'value'
            )
            pa_widgets.scalar_size.observe(self._scalar_size_handler, 'value')
            pa_widgets.legend.observe(self._legend_handler, 'value')
            pa_widgets.scalar_cmap.observe(self._scalar_cmap_handler, 'value')
            pa_widgets.is_visible.observe(self._is_visible_handler, 'value')
            pa_widgets.legend_lower_lim.observe(
                self._legend_lim_handler,
                'value',
            )
            pa_widgets.legend_upper_lim.observe(
                self._legend_lim_handler,
                'value',
            )

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

    def interactive_plot(self):

        '''
        Set plotting attributes, create widgets and display them
        along with the interactive plot.
        '''

        self._configure_plot()
        self._create_widgets()
        display(self._widgets._create_tabs())
        temp_data = self.get_frame(self._widgets.frame.value)
        self.time = str(temp_data['solver_data']['t'])
        temp_data = temp_data['arrays']

        for sct in self._scatters.values():
            if sct in self._scatter_ax.collections:
                self._scatter_ax.collections.remove(sct)

        self._scatters = {}
        for array_name in self._widgets.particles.keys():
            pa_widgets = self._widgets.particles[array_name]
            if (pa_widgets.scalar.value != 'None' and
                    pa_widgets.is_visible.value is True):
                sct = self._scatters[array_name] = self._scatter_ax.scatter(
                    temp_data[array_name].x,
                    temp_data[array_name].y,
                    s=pa_widgets.scalar_size.value,
                )
                c = getattr(
                        temp_data[array_name],
                        pa_widgets.scalar.value
                )
                colormap = getattr(
                    plt.cm,
                    pa_widgets.scalar_cmap.value
                )
                min_c, max_c, c_norm = self._cmap_helper(
                    c,
                    array_name
                )
                sct.set_facecolors(colormap(c_norm))

        self._scatter_ax.axis('equal')
        self.solver_time_textbox = None
        # So that _show_solver_time_handler does not glitch at intialization.
        self._show_solver_time_handler(None)
        self._legend_handler(None)

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
                temp_data_arr = temp_data[array_name]
                x = temp_data_arr.x
                y = temp_data_arr.y

                try:
                    v1 = getattr(
                        temp_data_arr,
                        pa_widgets.vector.value.split(",")[0]
                        )
                    v2 = getattr(
                        temp_data_arr,
                        pa_widgets.vector.value.split(",")[1]
                        )
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
                sct = self._scatters[array_name]
                sct.set_offsets(
                    np.vstack(
                        (temp_data[array_name].x, temp_data[array_name].y)
                        ).T
                    )

                c = getattr(
                        temp_data[array_name],
                        pa_widgets.scalar.value
                )
                colormap = getattr(
                    plt.cm,
                    pa_widgets.scalar_cmap.value
                )
                min_c, max_c, c_norm = self._cmap_helper(
                    c,
                    array_name
                )
                sct.set_facecolors(colormap(c_norm))

        self._legend_handler(None)
        self._vector_handler(None)
        self._show_solver_time_handler(None)
        self._adjust_axes()
        self.figure.show()

    def _scalar_handler(self, change):

        array_name = change['owner'].owner
        pa_widgets = self._widgets.particles[array_name]
        if pa_widgets.is_visible.value is True:
            temp_data = self.get_frame(
                self._widgets.frame.value
            )['arrays']
            sct = self._scatters[array_name]

            new = change['new']
            old = change['old']

            if (new == 'None' and old == 'None'):
                pass

            elif (new == 'None' and old != 'None'):
                sct.set_offsets(None)

            elif (new != 'None' and old == 'None'):
                sct.set_offsets(
                    np.vstack(
                        (temp_data[array_name].x, temp_data[array_name].y)
                    ).T
                )
                c = getattr(
                        temp_data[array_name],
                        pa_widgets.scalar.value
                )
                colormap = getattr(
                        plt.cm,
                        pa_widgets.scalar_cmap.value
                )
                min_c, max_c, c_norm = self._cmap_helper(
                    c,
                    array_name
                )
                sct.set_facecolors(colormap(c_norm))

            else:
                c = getattr(
                        temp_data[array_name],
                        pa_widgets.scalar.value
                )
                colormap = getattr(
                        plt.cm,
                        pa_widgets.scalar_cmap.value
                )
                min_c, max_c, c_norm = self._cmap_helper(
                    c,
                    array_name
                )
                sct.set_facecolors(colormap(c_norm))

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
            temp_data = self.get_frame(
                self._widgets.frame.value
            )['arrays']
            c = getattr(
                    temp_data[array_name],
                    pa_widgets.scalar.value
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
            sct.set_facecolors(colormap(c_norm))
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
                    c = getattr(
                            temp_data[array_name],
                            pa_widgets.scalar.value
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
                        boundaries = np.linspace(0, 1, 25)
                        #norm = mpl.colors.Normalize(vmin=0, vmax=1)
                    elif max_c == min_c:
                        # this occurs at initialization for some properties
                        # like pressure, and stays true throughout for
                        # others like mass of the particles
                        boundaries = np.linspace(0, max_c, 25)
                        #norm = mpl.colors.Normalize(vmin=0, vmax=actual_maxm)
                    else:
                        boundaries = np.linspace(min_c, max_c, 25)
                        #norm = mpl.colors.Normalize(vmin=minm, vmax=maxm)

                    self._cbars[array_name] = mpl.colorbar.ColorbarBase(
                        ax=self._cbar_ax[array_name],
                        cmap=colormap,
                        boundaries=boundaries,
                        ticks=boundaries,
                    )
                    self._cbars[array_name].set_label(
                        array_name + " : " +
                        pa_widgets.scalar.value
                    )
        if len(self._cbars.keys()) == 0:
            self._scatter_ax.set_position(
                [0, 0, 1, 1]
            )
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

    def _delay_box_handler(self, change):

        self._widgets.play_button.interval = change['new']*1000

    def _save_all_plots_handler(self, change):

        if change['new'] is True:
            mkdir('all_plots')
            self._widgets.frame.disabled = True
            self._widgets.play_button.disabled = True
            self._widgets.delay_box.disabled = True
            self._widgets.save_figure.disabled = True
            self._widgets.save_all_plots.disabled = True

            for array_name in self._widgets.particles.keys():
                pa_widgets = self._widgets.particles[array_name]
                pa_widgets.scalar.disabled = True
                pa_widgets.scalar_cmap.disabled = True
                pa_widgets.legend.disabled = True
                pa_widgets.vector.disabled = True
                pa_widgets.vector_width.disabled = True
                pa_widgets.vector_scale.disabled = True
                pa_widgets.scalar_size.disabled = True
                pa_widgets.is_visible.disabled = True

            file_count = len(self.paths_list) - 1

            for i in np.arange(0, file_count + 1):
                self._widgets.frame.value = i
                self._frame_handler(None)
                self.figure.savefig(
                    'all_plots/frame_%s.png' % i,
                    dpi=300
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

            self._widgets.save_all_plots.value = False

            for array_name in self._widgets.particles.keys():
                pa_widgets = self._widgets.particles[array_name]
                pa_widgets.scalar.disabled = False
                pa_widgets.scalar_cmap.disabled = False
                pa_widgets.legend.disabled = False
                pa_widgets.vector.disabled = False
                pa_widgets.vector_width.disabled = False
                pa_widgets.vector_scale.disabled = False
                pa_widgets.scalar_size.disabled = False
                pa_widgets.is_visible.disabled = False

    def _is_visible_handler(self, change):

        array_name = change['owner'].owner
        pa_widgets = self._widgets.particles[array_name]
        temp_data = self.get_frame(self._widgets.frame.value)['arrays']
        sct = self._scatters[array_name]

        if change['new'] is False:
            sct.set_offsets(None)
        elif change['new'] is True:
            sct.set_offsets(
                np.vstack(
                    (temp_data[array_name].x, temp_data[array_name].y)
                ).T
            )
            c = getattr(
                temp_data[array_name],
                pa_widgets.scalar.value
            )
            colormap = getattr(
                plt.cm,
                pa_widgets.scalar_cmap.value
            )
            min_c, max_c, c_norm = self._cmap_helper(
                c,
                array_name
            )
            sct.set_facecolors(colormap(c_norm))

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
                bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 3},
            )
        elif self._widgets.show_solver_time.value is False:
            if self.solver_time_textbox is not None:
                self.solver_time_textbox.remove()
            self.solver_time_textbox = None
        if change is not None:
            self.figure.show()

    def _legend_lim_handler(self, change):

        array_name = change['owner'].owner
        pa_widgets = self._widgets.particles[array_name]
        temp_data = self.get_frame(
            self._widgets.frame.value
        )['arrays']
        sct = self._scatters[array_name]
        c = getattr(
            temp_data[array_name],
            pa_widgets.scalar.value
        )
        colormap = getattr(
            plt.cm,
            pa_widgets.scalar_cmap.value
        )
        min_c, max_c, c_norm = self._cmap_helper(
            c,
            array_name
        )
        sct.set_facecolors(colormap(c_norm))
        
        self._legend_handler(None)
        self.figure.show()


class ParticleArrayWidgets3D(object):

    def __init__(self, particlearray):
        self.array_name = particlearray.name
        self.scalar = widgets.Dropdown(
            options=[
                'None'
            ] + particlearray.output_property_arrays,
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

    def _tab_config(self):
        
        VBox1 = widgets.VBox(
            [
                self.scalar,
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
            ]
        )
        hbox = widgets.HBox([VBox1, VBox2, VBox3])
        return hbox


class Viewer3DWidgets(object):

    def __init__(self, file, file_count):

        self.temp_data = load(file)
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
        )
        self.solver_time = widgets.HTML(
                value=self.time,
                description='Solver time:',
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
                     ]
                    ),
                    widgets.HBox(
                     [
                        self.save_all_plots,
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

    def _create_widgets(self):

        self._widgets = Viewer3DWidgets(
            file=self.paths_list[0],
            file_count=len(self.paths_list) - 1,
        )

        self._widgets.frame.observe(self._frame_handler, 'value')
        self._widgets.save_figure.on_submit(self._save_figure_handler)
        self._widgets.delay_box.observe(self._delay_box_handler, 'value')
        self._widgets.save_all_plots.observe(
            self._save_all_plots_handler,
            'value'
        )

        for array_name in self._widgets.particles.keys():
            pa_widgets = self._widgets.particles[array_name]
            pa_widgets.scalar.observe(self._scalar_handler, 'value')
            pa_widgets.velocity_vectors.observe(
                self._velocity_vectors_handler,
                'value'
            )
            pa_widgets.vector_size.observe(
                self._vector_size_handler,
                'value'
            )
            pa_widgets.scalar_size.observe(self._scalar_size_handler, 'value')
            pa_widgets.scalar_cmap.observe(self._scalar_cmap_handler, 'value')
            pa_widgets.legend.observe(self._legend_handler, 'value')
            pa_widgets.is_visible.observe(self._is_visible_handler, 'value')
            pa_widgets.legend_upper_lim.observe(
                self._legend_lim_handler,
                'value'
            )
            pa_widgets.legend_lower_lim.observe(
                self._legend_lim_handler,
                'value'
            )

    def interactive_plot(self):

        import ipyvolume.pylab as p3

        self._create_widgets()
        self.scatters = {}
        self.vectors = {}
        self._cbars = {}
        self._cbar_ax = {}        

        self.pltfigure = plt.figure(figsize=(9, 1), dpi=100)
        self._initial_ax = self.pltfigure.add_axes([0, 0, 1, 1])
        self._initial_ax.axis('off')
        # Creating a dummy axes element, that prevents the plot
        # from glitching and showing random noise when no legends are
        # being displayed.

        self.plot = p3.figure(width=800)
        data = self.get_frame(self._widgets.frame.value)['arrays']
        for array_name in self._widgets.particles.keys():
            pa_widgets = self._widgets.particles[array_name]
            if pa_widgets.scalar.value != 'None':
                colormap = getattr(mpl.cm, pa_widgets.scalar_cmap.value)
                c = getattr(data[array_name], pa_widgets.scalar.value)
                min_c, max_c, c_norm = self._cmap_helper(
                    c,
                    array_name
                )
                self.scatters[array_name] = p3.scatter(
                    data[array_name].x,
                    data[array_name].y,
                    data[array_name].z,
                    color=colormap(c_norm),
                    size=pa_widgets.scalar_size.value,
                    marker='sphere',
                )
        p3.squarelim()  # Makes sure the figure doesn't appear distorted.
        #self.plot = p3.gcf()  # Used in 'self._save_figure_handler()'.
        self._legend_handler(None)
        display(self.plot)
        display(self.pltfigure)
        display(self._widgets._create_tabs())

    def _frame_handler(self, change):

        data = self.get_frame(self._widgets.frame.value)
        self.time = str(data['solver_data']['t'])
        self._widgets.solver_time.value = self.time
        data = data['arrays']
        for array_name in self._widgets.particles.keys():
            pa_widgets = self._widgets.particles[array_name]
            if pa_widgets.is_visible.value is True:
                if pa_widgets.scalar.value != 'None':
                    colormap = getattr(mpl.cm, pa_widgets.scalar_cmap.value)
                    scatters = self.scatters[array_name]
                    c = getattr(data[array_name], pa_widgets.scalar.value)
                    min_c, max_c, c_norm = self._cmap_helper(
                        c,
                        array_name
                    )
                    scatters.x = data[array_name].x
                    scatters.y = data[array_name].y,
                    scatters.z = data[array_name].z,
                    scatters.color = colormap(c_norm)
                if pa_widgets.velocity_vectors.value is True:
                    vectors = self.vectors[array_name]
                    vectors.x = data[array_name].x
                    vectors.y = data[array_name].y
                    vectors.z = data[array_name].z
                    vectors.vx = getattr(data[array_name], 'u')
                    vectors.vy = getattr(data[array_name], 'v')
                    vectors.vz = getattr(data[array_name], 'w')
                    vectors.size = pa_widgets.vector_size.value
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
                self.scatters[array_name].visible = False
            else:
                colormap = getattr(mpl.cm, pa_widgets.scalar_cmap.value)
                data = self.get_frame(self._widgets.frame.value)['arrays']
                c = getattr(data[array_name], pa_widgets.scalar.value)
                min_c, max_c, c_norm = self._cmap_helper(
                    c,
                    array_name
                )
                if old != 'None' and new != 'None': 
                    self.scatters[array_name].color = colormap(c_norm)
                else:                
                    self.scatters[array_name] = p3.scatter(
                        data[array_name].x,
                        data[array_name].y,
                        data[array_name].z,
                        color=colormap(c_norm),
                        size=pa_widgets.scalar_size.value,
                        marker='sphere',
                    )
        self._legend_handler(None)

    def _velocity_vectors_handler(self, change):

        import ipyvolume.pylab as p3

        data = self.get_frame(self._widgets.frame.value)['arrays']
        array_name = change['owner'].owner
        pa_widgets = self._widgets.particles[array_name]
        if pa_widgets.is_visible.value is True:
            if change['new'] is False:
                self.vectors[array_name].size = 0
            elif array_name in self.vectors.keys():
                # change['new'] is True and the vectors have
                # already been plotted, just need updation.
                vectors = self.vectors[array_name]
                vectors.x = data[array_name].x
                vectors.y = data[array_name].y
                vectors.z = data[array_name].z
                vectors.u = getattr(data[array_name], 'u')
                vectors.v = getattr(data[array_name], 'v')
                vectors.w = getattr(data[array_name], 'w')
                vectors.size = pa_widgets.vector_size.value
            else:
                # change['new'] is True, and the vectors are
                # being plotted for the first time.
                self.vectors[array_name] = p3.quiver(
                    x=data[array_name].x,
                    y=data[array_name].y,
                    z=data[array_name].z,
                    u=getattr(data[array_name], 'u'),
                    v=getattr(data[array_name], 'v'),
                    w=getattr(data[array_name], 'w'),
                    size=pa_widgets.vector_size.value,
                )

    def _scalar_size_handler(self, change):

        array_name = change['owner'].owner
        if (array_name in self.scatters.keys() and
                self._widgets.particles[array_name].is_visible.value is True):
            self.scatters[array_name].size = change['new']

    def _vector_size_handler(self, change):

        array_name = change['owner'].owner
        pa_widgets = self._widgets.particles[array_name]
        if array_name in self.vectors.keys():
            if (pa_widgets.velocity_vectors.value is True and
                    pa_widgets.is_visible.value is True):
                self.vectors[array_name].size = change['new']

    def _scalar_cmap_handler(self, change):

        array_name = change['owner'].owner
        pa_widgets = self._widgets.particles[array_name]
        data = self.get_frame(self._widgets.frame.value)['arrays']

        if pa_widgets.is_visible.value is True:
            colormap = getattr(mpl.cm, change['new'])
            c = getattr(data[array_name], pa_widgets.scalar.value)
            min_c, max_c, c_norm = self._cmap_helper(
                c,
                array_name
            )
            self.scatters[array_name].color = colormap(c_norm)
            self._legend_handler(None)

    def _legend_handler(self, change):

        temp_data = self.get_frame(self._widgets.frame.value)['arrays']
        for _cbar_ax in self._cbar_ax.values():
            self.pltfigure.delaxes(_cbar_ax)
        self._cbar_ax = {}
        self._cbars = {}
        for array_name in self._widgets.particles.keys():
            pa_widgets = self._widgets.particles[array_name]
            if (pa_widgets.scalar.value != 'None' and
                    pa_widgets.legend.value is True):
                if pa_widgets.is_visible.value is True:
                    cmap = getattr(mpl.cm, pa_widgets.scalar_cmap.value)
                    c = getattr(
                        temp_data[array_name],
                        pa_widgets.scalar.value
                    )
                    self._initial_ax.set_position(
                        [
                            0,
                            0,
                            1,
                            0.75 - 0.25*len(self._cbar_ax)
                        ]
                    )
                    min_c, max_c, c_norm = self._cmap_helper(
                        c,
                        array_name
                    )

                    if max_c == 0:
                        boundaries = np.linspace(0, 1, 10)
                    elif min_c == max_c:
                        # This occurs at initialization for some properties
                        # like pressure, and stays true throughout for
                        # others like mass of the particles.
                        boundaries = np.linspace(0, max_c, 10)
                    else:
                        boundaries = np.linspace(min_c, max_c, 10)

                    self._cbar_ax[array_name] = self.pltfigure.add_axes(
                        [
                            0.05,
                            0.75 - 0.25*len(self._cbar_ax),
                            0.9,
                            0.25
                        ]
                    )
                    self._cbars[array_name] = mpl.colorbar.ColorbarBase(
                        ax=self._cbar_ax[array_name],
                        cmap=cmap,
                        boundaries=boundaries,
                        ticks=boundaries,
                        orientation='horizontal'
                    )
                    self._cbars[array_name].set_label(
                        array_name + " : " + pa_widgets.scalar.value
                    )
        if len(self._cbars) == 0:
            self._initial_ax.set_position(
                [0, 0, 1, 1]
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

    def _delay_box_handler(self, change):

        self._widgets.play_button.interval = change['new']*1000

    def _save_all_plots_handler(self, change):

        import ipyvolume.pylab as p3

        if change['new'] is True:
            mkdir('all_plots')
            self._widgets.frame.disabled = True
            self._widgets.play_button.disabled = True
            self._widgets.delay_box.disabled = True
            self._widgets.save_figure.disabled = True
            self._widgets.save_all_plots.disabled = True

            for array_name in self._widgets.particles.keys():
                pa_widgets = self._widgets.particles[array_name]
                pa_widgets.scalar.disabled = True
                pa_widgets.scalar_cmap.disabled = True
                pa_widgets.velocity_vectors.disabled = True
                pa_widgets.vector_size.disabled = True
                pa_widgets.scalar_size.disabled = True
                pa_widgets.is_visible.disabled = True

            file_count = len(self.paths_list) - 1

            for i in np.arange(0, file_count + 1):
                self._widgets.frame.value = i
                self._frame_handler(None)
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
            self._widgets.save_all_plots.value = False

            for array_name in self._widgets.particles.keys():
                pa_widgets = self._widgets.particles[array_name]
                pa_widgets.scalar.disabled = False
                pa_widgets.scalar_cmap.disabled = False
                pa_widgets.velocity_vectors.disabled = False
                pa_widgets.vector_size.disabled = False
                pa_widgets.scalar_size.disabled = False
                pa_widgets.is_visible.disabled = False

    def _is_visible_handler(self, change):

        import ipyvolume.pylab as p3

        array_name = change['owner'].owner
        pa_widgets = self._widgets.particles[array_name]

        if change['new'] is False:
            self.scatters[array_name].visible = False
            if array_name in self.vectors.keys():
                self.vectors[array_name].size = 0
        elif change['new'] is True:
            colormap = getattr(mpl.cm, pa_widgets.scalar_cmap.value)
            temp_data = self.get_frame(self._widgets.frame.value)['arrays']
            c = getattr(temp_data[array_name], pa_widgets.scalar.value)
            min_c, max_c, c_norm = self._cmap_helper(
                c,
                array_name
            )
            self.scatters[array_name] = p3.scatter(
                temp_data[array_name].x,
                temp_data[array_name].y,
                temp_data[array_name].z,
                color=colormap(c_norm),
                size=pa_widgets.scalar_size.value,
                marker='sphere',
            )
            if pa_widgets.velocity_vectors.value is True:
                if array_name in self.vectors.keys():
                    # The vectors have been plotted, just need updation.
                    vectors = self.vectors[array_name]
                    vectors.x = temp_data[array_name].x
                    vectors.y = temp_data[array_name].y
                    vectors.z = temp_data[array_name].z
                    vectors.u = getattr(temp_data[array_name], 'u')
                    vectors.v = getattr(temp_data[array_name], 'v')
                    vectors.w = getattr(temp_data[array_name], 'w')
                    vectors.size = pa_widgets.vector_size.value
                else:
                    # The vectors are being plotted for the first time.
                    self.vectors[array_name] = p3.quiver(
                        x=temp_data[array_name].x,
                        y=temp_data[array_name].y,
                        z=temp_data[array_name].z,
                        u=getattr(temp_data[array_name], 'u'),
                        v=getattr(temp_data[array_name], 'v'),
                        w=getattr(temp_data[array_name], 'w'),
                        size=pa_widgets.vector_size.value,
                    )

        self._legend_handler(None)

    def _legend_lim_handler(self, change):

        array_name = change['owner'].owner
        pa_widgets = self._widgets.particles[array_name]
        temp_data = self.get_frame(
            self._widgets.frame.value
        )['arrays']
        sct = self.scatters[array_name]
        c = getattr(
            temp_data[array_name],
            pa_widgets.scalar.value
        )
        colormap = getattr(
            plt.cm,
            pa_widgets.scalar_cmap.value
        )
        min_c, max_c, c_norm = self._cmap_helper(
            c,
            array_name
        )
        sct.color = colormap(c_norm)
        
        self._legend_handler(None)
