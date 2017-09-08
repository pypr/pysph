import json
import glob
from pysph.solver.utils import load, get_files
from IPython.display import display, clear_output, Image
import ipywidgets as widgets
import matplotlib.pyplot as plt
import matplotlib as mpl


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
        path = self.path + "*.log"
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

        path = self.path + "*.info"
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
            description="legend",
            disabled=False,
            layout=widgets.Layout(width='100px', display='flex')
        )
        self.legend.owner = self.array_name
        self.vector = widgets.Text(
            value='',
            placeholder='variable1,variable2',
            description='vector',
            disabled=False,
            layout=widgets.Layout(width='240px', display='flex')
        )
        self.vector.owner = self.array_name
        self.vector_width = widgets.FloatSlider(
            min=1,
            max=100,
            step=1,
            value=25,
            description='vector width',
            layout=widgets.Layout(width='300px'),
        )
        self.vector_width.owner = self.array_name
        self.vector_scale = widgets.FloatSlider(
            min=1,
            max=100,
            step=1,
            value=55,
            description='vector scale',
            layout=widgets.Layout(width='300px'),
        )
        self.vector_scale.owner = self.array_name
        self.scalar_size = widgets.FloatSlider(
            min=0,
            max=50,
            step=1,
            value=10,
            description='scalar size',
            layout=widgets.Layout(width='300px'),
        )
        self.scalar_size.owner = self.array_name

    def _create_vbox(self):

        from ipywidgets import VBox, HTML, Layout
        return VBox([
            HTML('<b>' + self.array_name.upper() + '</b>'),
            self.scalar,
            self.vector,
            self.vector_scale,
            self.vector_width,
            self.scalar_size,
            self.scalar_cmap,
            self.legend,

        ],
            layout=Layout(border='1px solid', margin='3px', min_width='320px')
        )


class Viewer2DWidgets(object):

    def __init__(self, file, file_count):

        self.temp_data = load(file)['arrays']
        self.frame = widgets.IntSlider(
            min=0,
            max=file_count,
            step=1,
            value=0,
            description='frame',
            layout=widgets.Layout(width='600px'),
        )
        self.save_figure = widgets.Text(
                value='',
                placeholder='example.pdf',
                description='Save figure',
                disabled=False,
                layout=widgets.Layout(width='240px', display='flex')
        )
        self.particles = {}
        for array_name in self.temp_data.keys():
            self.particles[array_name] = ParticleArrayWidgets(
                self.temp_data[array_name],
            )

    def _create_vbox(self):

        from ipywidgets import HBox, VBox, Label, Layout
        items = []
        for array_name in self.particles.keys():
            items.append(self.particles[array_name]._create_vbox())

        return VBox(
                [
                    HBox(
                        items,
                    ),
                    self.frame,
                    self.save_figure
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
            file=self.paths_list[0],
            file_count=len(self.paths_list) - 1,
        )
        widgets = self._widgets
        widgets.frame.observe(self._frame_handler, 'value')
        widgets.save_figure.on_submit(self._save_figure_handler)

        for array_name in self._widgets.particles.keys():
            pa_widgets = widgets.particles[array_name]
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

    def _configure_plot(self):
        '''
        Set attributes for plotting.
        '''

        self.figure = plt.figure()
        self._scatter_ax = self.figure.add_axes([0, 0, 1, 1])
        self._scatters = {}
        self._cbar_ax = {}
        self._cbars = {}
        self._vectors = {}

    def interactive_plot(self):
        '''
        Set plotting attributes, create widgets and display them
        along with the interactive plot.

        Use %matplotlib notebook for more interactivity.
        '''

        self._configure_plot()
        self._create_widgets()
        self._frame_handler(None)
        display(self._widgets._create_vbox())

    def _frame_handler(self, change):

        temp_data = self.get_frame(self._widgets.frame.value)

        # scalars #

        self.figure.set_label(
            "Time : " + str(temp_data['solver_data']['t'])
        )
        temp_data = temp_data['arrays']

        for sct in self._scatters.values():
            if sct in self._scatter_ax.collections:
                self._scatter_ax.collections.remove(sct)

        self._scatters = {}
        for array_name in self._widgets.particles.keys():
            pa_widgets = self._widgets.particles[array_name]
            if pa_widgets.scalar.value != 'None':
                self._scatters[array_name] = self._scatter_ax.scatter(
                    temp_data[array_name].x,
                    temp_data[array_name].y,
                    c=getattr(
                            temp_data[array_name],
                            pa_widgets.scalar.value
                    ),
                    s=pa_widgets.scalar_size.value,
                    cmap=pa_widgets.scalar_cmap.value,
                )
        self._legend_handler(None, manual=True)

        # _vectors #

        for vct in self._vectors.values():
            if vct in self._scatter_ax.collections:
                self._scatter_ax.collections.remove(vct)

        self._vectors = {}
        for array_name in self._widgets.particles.keys():
            if self._widgets.particles[array_name].vector.value != '':
                pa_widgets = self._widgets.particles[array_name]
                temp_data = temp_data[array_name]
                x = temp_data.x
                y = temp_data.y
                v1 = getattr(temp_data, pa_widgets.vector.value.split(",")[0])
                v2 = getattr(temp_data, pa_widgets.vector.value.split(",")[1])
                vmag = (v1**2 + v2**2)**0.5
                self._vectors[array_name] = self._scatter_ax.quiver(
                    x,
                    y,
                    v1,
                    v2,
                    vmag,
                    scale=pa_widgets.vector_scale.value,
                    width=(pa_widgets.vector_width.value)/10000,
                )

        # show the changes #
        clear_output(wait=True)
        display(self.figure)

    def _scalar_handler(self, change):
        array_name = change['owner'].owner
        temp_data = self.get_frame(
            self._widgets.frame.value
        )['arrays'][array_name]
        scatter = self._scatters[array_name]
        pa_widgets = self._widgets.particles[array_name]
        if array_name in self._scatters.keys():
            if scatter in self._scatter_ax.collections:
                self._scatter_ax.collections.remove(
                    scatter
                )

        if change['new'] != 'None':
            self._scatters[array_name] = self._scatter_ax.scatter(
                temp_data.x,
                temp_data.y,
                c=getattr(temp_data, change['new']),
                s=pa_widgets.scalar_size.value,
                cmap=pa_widgets.scalar_cmap.value,
            )

        self._legend_handler(None, manual=True)

        clear_output(wait=True)
        display(self.figure)

    def _vector_handler(self, change):
        '''
        Bug : Arrows go out of the figure
        '''

        array_name = change['owner'].owner
        temp_data = self.get_frame(
            self._widgets.frame.value
        )['arrays'][array_name]
        pa_widgets = self._widgets.particles[array_name]
        if array_name in self._vectors.keys():
            if self._vectors[array_name] in self._scatter_ax.collections:
                self._scatter_ax.collections.remove(
                    self._vectors[array_name]
                )

        if change['new'] != '':
            x = temp_data.x
            y = temp_data.y
            v1 = getattr(temp_data, change['new'].split(",")[0])
            v2 = getattr(temp_data, change['new'].split(",")[0])
            vmag = (v1**2 + v2**2)**0.5
            self._vectors[array_name] = self._scatter_ax.quiver(
                x,
                y,
                v1,
                v2,
                vmag,
                scale=pa_widgets.vector_scale.value,
                width=(pa_widgets.vector_width.value)/10000,
            )

        clear_output(wait=True)
        display(self.figure)

    def _vector_scale_handler(self, change):
        # the widget value must already have updated.
        array_name = change['owner'].owner
        if array_name in self._vectors.keys():
            self._vectors[array_name].scale = change['new']
        clear_output(wait=True)
        display(self.figure)

    def _scalar_size_handler(self, change):
        array_name = change['owner'].owner
        if array_name in self._scatters.keys():
            self._scatters[array_name].set_sizes([change['new']])
        clear_output(wait=True)
        display(self.figure)

    def _vector_width_handler(self, change):
        # the widget value must already have updated.
        array_name = change['owner'].owner
        if array_name in self._vectors.keys():
            self._vectors[array_name].width = change['new']/10000
        clear_output(wait=True)
        display(self.figure)

    def _scalar_cmap_handler(self, change):
        array_name = change['owner'].owner
        pa_widgets = self._widgets.particles[array_name]
        change['new'] = pa_widgets.scalar.value
        self._scalar_handler(change)

    def _legend_handler(self, change, manual=False):

        for _cbar_ax in self._cbar_ax.values():
            self.figure.delaxes(_cbar_ax)
        self._cbar_ax = {}
        self._cbars = {}
        self._scatter_ax.set_position([0, 0, 1, 1])
        for array_name in self._widgets.particles.keys():
            pa_widgets = self._widgets.particles[array_name]
            if pa_widgets.legend.value:
                if pa_widgets.scalar.value != 'None':
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
                    self._cbars[array_name] = self.figure.colorbar(
                            self._scatters[array_name],
                            cax=self._cbar_ax[array_name]
                    )
                    self._cbars[array_name].set_label(
                            array_name + " : " +
                            pa_widgets.scalar.value
                    )
        if not manual:
            clear_output(wait=True)
            display(self.figure)

    def _save_figure_handler(self, change):

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
                break
        self._widgets.save_figure.value = ""


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
            options=map(str, plt.colormaps()),
            value='viridis',
            description="Colormap",
            disabled=False,
            layout=widgets.Layout(width='240px', display='flex')
        )
        self.scalar_cmap.owner = self.array_name
        self.velocity_vectors = widgets.Checkbox(
            value=False,
            description="Vectors",
            disabled=False,
            layout=widgets.Layout(width='100px', display='flex')
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

    def _create_vbox(self):

        from ipywidgets import VBox, Layout, HTML
        return VBox([
            HTML('<b>' + self.array_name.upper() + '</b>'),
            self.scalar,
            self.velocity_vectors,
            self.vector_size,
            self.scalar_size,
            self.scalar_cmap,

        ],
            layout=Layout(border='1px solid', margin='3px', min_width='320px')
        )


class Viewer3DWidgets(object):

    def __init__(self, file, file_count):

        self.temp_data = load(file)['arrays']
        self.frame = widgets.IntSlider(
            min=0,
            max=file_count,
            step=1,
            value=0,
            description='frame',
            layout=widgets.Layout(width='600px'),
        )

        self.particles = {}
        for array_name in self.temp_data.keys():
            self.particles[array_name] = ParticleArrayWidgets3D(
                self.temp_data[array_name],
            )

    def _create_vbox(self):

        from ipywidgets import HBox, VBox, Label, Layout
        items = []
        for array_name in self.particles.keys():
            items.append(self.particles[array_name]._create_vbox())

        return VBox(
                [
                    HBox(
                        items,
                    ),
                    self.frame,
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
        widgets = self._widgets
        widgets.frame.observe(self._frame_handler, 'value')

        for array_name in self._widgets.particles.keys():
            pa_widgets = widgets.particles[array_name]
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

    def interactive_plot(self):
        self._create_widgets()
        self.scatters = {}
        display(self._widgets._create_vbox())
        self.vectors = {}
        self.legend = widgets.Output()

        import ipyvolume.pylab as p3

        p3.clear()
        data = self.get_frame(self._widgets.frame.value)['arrays']
        for array_name in self._widgets.particles.keys():
            pa_widgets = self._widgets.particles[array_name]
            colormap = getattr(mpl.cm, pa_widgets.scalar_cmap.value)
            c = colormap(
                getattr(data[array_name], pa_widgets.scalar.value)
            )
            self.scatters[array_name] = p3.scatter(
                data[array_name].x,
                data[array_name].y,
                data[array_name].z,
                color=c,
                size=pa_widgets.scalar_size.value,
            )
        self._legend_handler(None)
        display(widgets.VBox((p3.gcc(), self.legend)))
        # HBox does not allow custom layout.

    def _frame_handler(self, change):

        data = self.get_frame(self._widgets.frame.value)['arrays']
        for array_name in self._widgets.particles.keys():
            pa_widgets = self._widgets.particles[array_name]
            colormap = getattr(mpl.cm, pa_widgets.scalar_cmap.value)

            scatters = self.scatters[array_name]
            c = colormap(
                getattr(data[array_name], pa_widgets.scalar.value)
            )
            scatters.x = data[array_name].x
            scatters.y = data[array_name].y,
            scatters.z = data[array_name].z,
            scatters.color = c
            pa_widgets = self._widgets.particles[array_name]
            if hasattr(self.vectors, array_name):
                vectors = self.vectors[array_name]
                if pa_widgets.velocity_vectors.value is True:
                    vectors.x = data[array_name].x
                    vectors.y = data[array_name].y
                    vectors.z = data[array_name].z
                    vectors.vx = getattr(data[array_name], 'u')
                    vectors.vy = getattr(data[array_name], 'v')
                    vectors.vz = getattr(data[array_name], 'w')
        self._legend_handler(None)

    def _scalar_handler(self, change):
        array_name = change['owner'].owner
        pa_widgets = self._widgets.particles[array_name]
        colormap = getattr(mpl.cm, pa_widgets.scalar_cmap.value)
        data = self.get_frame(self._widgets.frame.value)['arrays']
        array_name = change['owner'].owner
        c = colormap(getattr(data[array_name], pa_widgets.scalar.value))
        self.scatters[array_name].color = c
        self._legend_handler(None)

    def _velocity_vectors_handler(self, change):
        import ipyvolume.pylab as p3
        data = self.get_frame(self._widgets.frame.value)['arrays']
        array_name = change['owner'].owner
        pa_widgets = self._widgets.particles[array_name]
        if change['new'] is False:
            self.vectors[array_name].size = 0
        else:
            if array_name in self.vectors.keys():
                self.vectors[
                    array_name
                ].size = pa_widgets.vector_size.value
            else:
                self.vectors[array_name] = p3.quiver(
                    data[array_name].x,
                    data[array_name].y,
                    data[array_name].z,
                    getattr(data[array_name], 'u'),
                    getattr(data[array_name], 'v'),
                    getattr(data[array_name], 'w'),
                    size=pa_widgets.vector_size.value,
                )

    def _scalar_size_handler(self, change):
        array_name = change['owner'].owner
        if array_name in self.scatters.keys():
            self.scatters[array_name].size = change['new']

    def _vector_size_handler(self, change):
        array_name = change['owner'].owner
        if array_name in self.vectors.keys():
            self.vectors[array_name].size = change['new']

    def _scalar_cmap_handler(self, change):
        array_name = change['owner'].owner
        pa_widgets = self._widgets.particles[array_name]
        change['new'] = pa_widgets.scalar.value
        self._scalar_handler(change)
        self._legend_handler(None)

    def _legend_handler(self, change):

        import ipyvolume.pylab as p3
        import numpy as np
        temp_data = self.get_frame(self._widgets.frame.value)
        self.pltfigure = plt.figure(figsize=(8, 8))
        self.cbars = {}
        self.cbars_ax = {}
        for array_name in self._widgets.particles.keys():
            pa_widgets = self._widgets.particles[array_name]
            cmap = getattr(mpl.cm, pa_widgets.scalar_cmap.value)
            ticks = set(list(np.sort(
                getattr(
                    temp_data['arrays'][array_name],
                    pa_widgets.scalar.value
                )
            )))
            ticks = list(ticks)
            ticks.sort()
            if len(ticks) == 1:
                ticks.append(ticks[0] + 0.00000001)
                # To avoid passing a singleton set

            self.cbars_ax[array_name] = self.pltfigure.add_axes(
                [
                    0.2*len(self.cbars_ax.keys()),
                    0,
                    0.05,
                    0.5
                ]
            )
            self.cbars[array_name] = mpl.colorbar.ColorbarBase(
                ax=self.cbars_ax[array_name],
                cmap=cmap,
                boundaries=ticks,
            )
            self.cbars[array_name].set_label(
                            array_name + " : " + pa_widgets.scalar.value
                    )
        clear_output()
        with self.legend:
            self.legend.clear_output()
            display(self.pltfigure)
