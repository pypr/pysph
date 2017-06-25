class Viewer(object):

    '''A class for the pysph jupyter notebook Viewer2D.

    Attributes
    ----------

        path              String containing the path of the output directory
        paths_list        List of paths to the output files
        cache             Cached data

    Methods
    -------

        get_buffer
        show_log
        show_info
        show_results
        show_all

    '''

    def __init__(self, path, cache=True):

        # Imports #

        from pysph.solver.utils import get_files, load

        # Configuring the path #

        self.path = path
        self.paths_list = get_files(path)

        # Caching #
        # Note : Caching is only used by get_buffer and widget handlers.
        if cache:
            self.cache = {}
        else:
            pass

    def get_buffer(self, frame):
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
        >>> sample.get_buffer(12)
        {'arrays': {'fluid': <pysph.base.particle_array.ParticleArray at 0x7f3f7d144d60>,
        'inlet': <pysph.base.particle_array.ParticleArray at 0x7f3f7d144b98>,
        'outlet': <pysph.base.particle_array.ParticleArray at 0x7f3f7d144c30>},
        'solver_data': {'count': 240, 'dt': 0.01, 't': 2.399999999999993}}


        '''
        if hasattr(self, "_cbars"):
            if frame in self.cache.keys():
                temp_data = self.cache[frame]
            else:
                from pysph.solver.utils import load
                self.cache[frame] = temp_data = load(self.paths_list[frame])
        else:
            from pysph.solver.utils import load
            temp_data = load(self.paths_list[frame])

        return temp_data

    def show_log(self):
        '''
        Prints the content of log file.
        '''
        import glob
        print("Printing log : \n\n")
        path = self.path + "*.log"
        logfile = open(glob.glob(path)[0], 'r')
        for lines in logfile:
            print(lines)
        logfile.close()

    def show_results(self):
        '''
        Show if there are any png, jpeg, jpg, or bmp images.
        '''

        import glob
        from IPython.display import Image, display

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

        Examples
        --------

        >>> sample = Viewer2D('/home/deep/pysph/trivial_inlet_outlet_output/')
        >>> sample.show_info()
        Printing info :

        cpu_time : 2.58390402794
        args : []
        completed : True
        fname : trivial_inlet_outlet
        output_dir : /home/deep/pysph_coverage/pysph_coverage/trivial_inlet_outlet_output
        Number of files : 31
          inlet :
            Number of particles : 55
            Output Property Arrays : ['x', 'y', 'z', 'u', 'v', 'w', 'rho',
            'm', 'h', 'pid', 'gid', 'tag']
          fluid :
            Number of particles : 0
            Output Property Arrays : ['x', 'y', 'z', 'u', 'v', 'w', 'rho',
             'm', 'h', 'pid', 'gid', 'tag']
          outlet :
            Number of particles : 0
            Output Property Arrays : ['x', 'y', 'z', 'u', 'v', 'w', 'rho',
            'm', 'h', 'pid', 'gid', 'tag']

        Keys in results.npz :
        ['t', 'x_max']
        '''
        # Imports #

        import json
        import glob
        from pysph.solver.utils import load

        # General Info #

        path = self.path + "*.info"
        infofile = open(glob.glob(path)[0], 'r')
        data = json.load(infofile)
        infofile.close
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

        import glob
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



    def _display_widgets(self):
        '''
        Display the widgets created using _create_widgets() method.
        '''

        from ipywidgets import HBox, VBox, Label, Layout
        from IPython.display import display
        items = []
        for particles_type in self._widgets['particles'].keys():
            items.append(
                VBox([
                    Label(particles_type),
                    self._widgets['particles'][particles_type]['scalar'],
                    self._widgets['particles'][particles_type]['legend'],
                    self._widgets['particles'][particles_type]['vector'],
                    self._widgets[
                        'particles'
                    ][particles_type]['vector_scale'],
                    self._widgets['particles'][particles_type]['vector_width'],
                    self._widgets['particles'][particles_type]['scalar_size'],
                ],
                    layout=Layout(display='flex'))
            )

        display(
            VBox(
                [
                    HBox(items, layout=Layout(display='flex')),
                    self._widgets['frame'],
                    self._widgets['save_figure']
                ]
            )
        )

class Viewer2D(Viewer):

    def _create_widgets(self):
        '''
        For handlers to work,
        make sure description of each widget starts like fluid_widname.

        '''

        from pysph.solver.utils import load
        import ipywidgets as widgets

        temp_data = load(self.paths_list[0])['arrays']
        self._widgets = {}
        file_count = len(self.paths_list) - 1
        self._widgets['frame'] = widgets.IntSlider(
            min=0,
            max=file_count,
            step=1,
            value=0,
            description='frame',
            layout=widgets.Layout(width='600px'),
        )
        self._widgets['frame'].observe(self._frame_handler, 'value')
        self._widgets['save_figure'] = widgets.Text(
                value='',
                placeholder='example.pdf',
                description='Save figure',
                disabled=False,
                layout=widgets.Layout(width='240px', display='flex')
        )
        self._widgets['save_figure'].on_submit(self._save_figure_handler)
        self._widgets['particles'] = {}
        for particles_type in temp_data.keys():
            self._widgets['particles'][particles_type] = {}
            self._widgets[
                'particles'
            ][particles_type]['scalar'] = widgets.Dropdown(
                options=[
                    'None'
                ]+temp_data[particles_type].output_property_arrays,
                value='rho',
                description=particles_type + "_scalar",
                disabled=False,
                layout=widgets.Layout(width='240px', display='flex')
            )
            self._widgets['particles'][particles_type]['scalar'].observe(
                self._scalar_handler,
                'value',
            )
            self._widgets[
                'particles'
            ][particles_type]['legend'] = widgets.ToggleButton(
                value=False,
                description=particles_type + "_legend",
                disabled=False,
                tooltip='Description',
                layout=widgets.Layout(width='240px', display='flex')
            )
            self._widgets['particles'][particles_type]['legend'].observe(
                self._legend_handler,
                'value',
            )
            self._widgets[
                'particles'
            ][particles_type]['vector'] = widgets.Text(
                value='',
                placeholder='variable1,variable2',
                description=particles_type + '_vector',
                disabled=False,
                layout=widgets.Layout(width='240px', display='flex')
            )
            self._widgets['particles'][particles_type]['vector'].observe(
                self._vector_handler,
                'value',
            )
            self._widgets[
                'particles'
            ][particles_type]['vector_width'] = widgets.FloatSlider(
                min=1,
                max=100,
                step=1,
                value=25,
                description=particles_type + '_vector_width',
                layout=widgets.Layout(width='300px'),
            )

            self._widgets[
                'particles'
            ][particles_type]['vector_width'].observe(
                self._vector_width_handler,
                'value',
            )
            self._widgets[
                'particles'
            ][particles_type]['vector_scale'] = widgets.FloatSlider(
                min=1,
                max=100,
                step=1,
                value=55,
                description=particles_type + '_vector_scale',
                layout=widgets.Layout(width='300px'),
            )
            self._widgets[
                'particles'
            ][particles_type]['vector_scale'].observe(
                self._vector_scale_handler,
                'value',
            )
            self._widgets[
                'particles'
            ][particles_type]['scalar_size'] = widgets.FloatSlider(
                min=0,
                max=50,
                step=1,
                value=10,
                description=particles_type + '_scalar_size',
                layout=widgets.Layout(width='300px'),
            )
            self._widgets[
                'particles'
            ][particles_type]['scalar_size'].observe(
                self._scalar_size_handler,
                'value',
            )

    def _configure_plot(self):
        '''
        Set attributes for plotting.
        '''

        from matplotlib import pyplot as plt

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
        self._display_widgets()

    def _frame_handler(self, change):
        from IPython.display import clear_output, display

        temp_data = self.get_buffer(self._widgets['frame'].value)

        # scalars #

        self.figure.set_label(
            "Time : " + str(temp_data['solver_data']['t'])
        )
        temp_data = temp_data['arrays']

        for sct in self._scatters.values():
            if sct in self._scatter_ax.collections:
                self._scatter_ax.collections.remove(sct)

        self._scatters = {}
        for particles_type in self._widgets['particles'].keys():
            if self._widgets[
                'particles'
            ][particles_type]['scalar'].value != 'None':
                self._scatters[particles_type] = self._scatter_ax.scatter(
                    temp_data[particles_type].x,
                    temp_data[particles_type].y,
                    c=getattr(
                            temp_data[particles_type],
                            self._widgets[
                                'particles'
                            ][particles_type]['scalar'].value
                    ),
                    s=self._widgets[
                        'particles'
                    ][particles_type]['scalar_size'].value,
                )
        self._legend_handler(None, manual=True)

        # _vectors #

        for vct in self._vectors.values():
            if vct in self._scatter_ax.collections:
                self._scatter_ax.collections.remove(vct)

        self._vectors = {}
        for particles_type in self._widgets['particles'].keys():
            if self._widgets[
                'particles'
            ][particles_type]['vector'].value != '':
                self._vectors[particles_type] = self._scatter_ax.quiver(
                    temp_data[particles_type].x,
                    temp_data[particles_type].y,
                    getattr(
                        temp_data[particles_type],
                        self._widgets[
                            'particles'
                        ][particles_type]['vector'].value.split(",")[0]
                    ),
                    getattr(
                        temp_data[particles_type],
                        self._widgets[
                            'particles'
                        ][particles_type]['vector'].value.split(",")[1]
                    ),
                    ((getattr(
                        temp_data[particles_type],
                        self._widgets[
                            'particles'
                        ][particles_type]['vector'].value.split(",")[0]
                    ))**2 + (getattr(
                        temp_data[particles_type],
                        self._widgets[
                            'particles'
                        ][particles_type]['vector'].value.split(",")[1]
                        ))**2
                    )**0.5,
                    scale=self._widgets[
                        'particles'
                    ][particles_type]['vector_scale'].value,
                    width=(self._widgets[
                        'particles'
                    ][particles_type]['vector_width'].value)/10000,
                )

        # show the changes #
        clear_output(wait=True)
        display(self.figure)

    def _scalar_handler(self, change):

        from IPython.display import clear_output

        temp_data = self.get_buffer(self._widgets['frame'].value)['arrays']

        particles_type = change['owner'].description.split('_')[0]

        if particles_type in self._scatters.keys():
            if self._scatters[particles_type] in self._scatter_ax.collections:
                self._scatter_ax.collections.remove(
                    self._scatters[particles_type]
                )

        if change['new'] != 'None':
            self._scatters[particles_type] = self._scatter_ax.scatter(
                temp_data[particles_type].x,
                temp_data[particles_type].y,
                c=getattr(temp_data[particles_type], change['new']),
                s=self._widgets[
                    'particles'
                ][particles_type]['scalar_size'].value
            )

        self._legend_handler(None, manual=True)
        clear_output(wait=True)
        display(self.figure)

    def _vector_handler(self, change):
        '''
        Bug : Arrows go out of the figure
        '''

        from IPython.display import clear_output

        temp_data = self.get_buffer(
            self._widgets['frame'].value
        )['arrays']

        particles_type = change['owner'].description.split('_')[0]

        if particles_type in self._vectors.keys():
            if self._vectors[particles_type] in self._scatter_ax.collections:
                self._scatter_ax.collections.remove(
                    self._vectors[particles_type]
                )

        if change['new'] != '':
            self._vectors[particles_type] = self._scatter_ax.quiver(
                temp_data[particles_type].x,
                temp_data[particles_type].y,
                getattr(
                    temp_data[particles_type],
                    change['new'].split(",")[0]
                ),
                getattr(
                    temp_data[particles_type],
                    change['new'].split(",")[1]
                ),
                ((getattr(
                    temp_data[particles_type],
                    change['new'].split(",")[0]
                ))**2 + (getattr(
                    temp_data[particles_type], change['new'].split(",")[1]
                ))**2
                )**0.5,
                scale=self._widgets[
                    'particles'
                ][particles_type]['vector_scale'].value,
                width=(self._widgets[
                    'particles'
                ][particles_type]['vector_width'].value)/10000,
            )

        clear_output(wait=True)
        display(self.figure)

    def _vector_scale_handler(self, change):
        from IPython.display import display, clear_output
        # the widget value must already have updated.
        particles_type = change['owner'].description.split('_')[0]
        if particles_type in self._vectors.keys():
            self._vectors[particles_type].scale = change['new']
        clear_output(wait=True)
        display(self.figure)

    def _scalar_size_handler(self, change):
        from IPython.display import display, clear_output
        particles_type = change['owner'].description.split('_')[0]
        if particles_type in self._scatters.keys():
            self._scatters[particles_type].set_sizes([change['new']])
        clear_output(wait=True)
        display(self.figure)

    def _vector_width_handler(self, change):
        from IPython.display import display, clear_output
        # the widget value must already have updated.
        particles_type = change['owner'].description.split('_')[0]
        if particles_type in self._vectors.keys():
            self._vectors[particles_type].width = change['new']/10000
        clear_output(wait=True)
        display(self.figure)

    def _legend_handler(self, change, manual=False):
        from IPython.display import display, clear_output

        for _cbar_ax in self._cbar_ax.values():
            self.figure.delaxes(_cbar_ax)
        self._cbar_ax = {}
        self._cbars = {}
        self._scatter_ax.set_position([0, 0, 1, 1])
        for particles_type in self._widgets['particles'].keys():
            if self._widgets['particles'][particles_type]['legend'].value:
                if self._widgets[
                    'particles'
                ][particles_type]['scalar'].value != 'None':
                    self._scatter_ax.set_position(
                        [0, 0, 0.84 - 0.15*len(self._cbars.keys()), 1]
                    )
                    self._cbar_ax[particles_type] = self.figure.add_axes(
                            [
                                0.85 - 0.15*len(self._cbars.keys()),
                                0.02,
                                0.02,
                                0.82
                            ]
                    )
                    self._cbars[particles_type] = self.figure.colorbar(
                            self._scatters[particles_type],
                            cax=self._cbar_ax[particles_type]
                    )
                    self._cbars[particles_type].set_label(
                            particles_type + " : " +
                            self._widgets[
                                'particles'
                            ][particles_type]['scalar'].value
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
            if self._widgets['save_figure'].value.endswith(extension):
                self.figure.savefig(self._widgets['save_figure'].value)
                print(
                    "Saved figure as {} in the present working directory"
                    .format(
                        self._widgets['save_figure'].value
                    )
                )
                break
        self._widgets['save_figure'].value = ""




class Viewer3D(Viewer):

    def _create_widgets(self):
        '''
        For handlers to work,
        make sure description of each widget starts like fluid_widname.

        '''

        from pysph.solver.utils import load
        import ipywidgets as widgets

        temp_data = load(self.paths_list[0])['arrays']
        self._widgets = {}
        file_count = len(self.paths_list) - 1
        self._widgets['frame'] = widgets.IntSlider(
            min=0,
            max=file_count,
            step=1,
            value=0,
            description='frame',
            layout=widgets.Layout(width='600px'),
        )
        self._widgets['frame'].observe(self._frame_handler, 'value')
        self._widgets['save_figure'] = widgets.Text(
                value='',
                placeholder='example.pdf',
                description='Save figure',
                disabled=False,
                layout=widgets.Layout(width='240px', display='flex')
        )
        self._widgets['save_figure'].on_submit(self._save_figure_handler)
        self._widgets['particles'] = {}
        for particles_type in temp_data.keys():
            self._widgets['particles'][particles_type] = {}
            self._widgets[
                'particles'
            ][particles_type]['scalar'] = widgets.Dropdown(
                options=temp_data[particles_type].output_property_arrays,
                value='rho',
                description=particles_type + "_scalar",
                disabled=False,
                layout=widgets.Layout(width='240px', display='flex')
            )
            self._widgets['particles'][particles_type]['scalar'].observe(
                self._scalar_handler,
                'value',
            )
            self._widgets[
                'particles'
            ][particles_type]['legend'] = widgets.ToggleButton(
                value=False,
                description=particles_type + "_legend",
                disabled=False,
                tooltip='Description',
                layout=widgets.Layout(width='240px', display='flex')
            )
            self._widgets['particles'][particles_type]['legend'].observe(
                self._legend_handler,
                'value',
            )
            self._widgets[
                'particles'
            ][particles_type]['vector'] = widgets.Text(
                value='',
                placeholder='variable1,variable2',
                description=particles_type + '_vector',
                disabled=False,
                layout=widgets.Layout(width='240px', display='flex')
            )
            self._widgets['particles'][particles_type]['vector'].observe(
                self._vector_handler,
                'value',
            )
            self._widgets[
                'particles'
            ][particles_type]['vector_width'] = widgets.FloatSlider(
                min=1,
                max=100,
                step=1,
                value=25,
                description=particles_type + '_vector_width',
                layout=widgets.Layout(width='300px'),
            )

            self._widgets[
                'particles'
            ][particles_type]['vector_width'].observe(
                self._vector_width_handler,
                'value',
            )
            self._widgets[
                'particles'
            ][particles_type]['vector_scale'] = widgets.FloatSlider(
                min=1,
                max=100,
                step=1,
                value=55,
                description=particles_type + '_vector_scale',
                layout=widgets.Layout(width='300px'),
            )
            self._widgets[
                'particles'
            ][particles_type]['vector_scale'].observe(
                self._vector_scale_handler,
                'value',
            )
            self._widgets[
                'particles'
            ][particles_type]['scalar_size'] = widgets.FloatSlider(
                min=0,
                max=10,
                step=0.1,
                value=10,
                description=particles_type + '_scalar_size',
                layout=widgets.Layout(width='300px'),
            )
            self._widgets[
                'particles'
            ][particles_type]['scalar_size'].observe(
                self._scalar_size_handler,
                'value',
            )

    def interactive_plot(self):
        self._create_widgets()
        self.scatters={}
        self._display_widgets()
        self.vectors = {}
        import ipyvolume.pylab as p3
        import numpy as np
        import matplotlib.cm
        colormap = matplotlib.cm.viridis
        p3.clear()
        data = self.get_buffer(self._widgets['frame'].value)['arrays']
        for particles_type in self._widgets['particles'].keys():
            c = colormap(getattr(data[particles_type], self._widgets['particles'][particles_type]['scalar'].value))
            self.scatters[particles_type] = p3.scatter(
                data[particles_type].x,
                data[particles_type].y,
                data[particles_type].z,
                color = c,
            )
        p3.show()

    def _frame_handler(self, change):
        import numpy as np
        import matplotlib.cm
        colormap = matplotlib.cm.viridis
        data = self.get_buffer(self._widgets['frame'].value)['arrays']
        for particles_type in self._widgets['particles'].keys():

            c = colormap(getattr(data[particles_type], self._widgets['particles'][particles_type]['scalar'].value))

            self.scatters[particles_type].x = data[particles_type].x
            self.scatters[particles_type].y = data[particles_type].y,
            self.scatters[particles_type].z = data[particles_type].z,
            self.scatters[particles_type].color = c

#            if self._widgets['particles'][particles_type]['vector'].value != '':
#                if particles_type in self.vectors.keys():
#                    if self.vectors[particles_type].size != 0:
#                        self.vectors[particles_type].x = data[particles_type].x
#                        self.vectors[particles_type].y = data[particles_type].y
#                        self.vectors[particles_type].z = data[particles_type].z
#                        self.vectors[particles_type].vx = getattr(data[particles_type], self._widgets['particles'][particles_type]['vector'].value.split(",")[0])
#                        self.vectors[particles_type].vy = getattr(data[particles_type], self._widgets['particles'][particles_type]['vector'].value.split(",")[1])
#                        self.vectors[particles_type].vz = getattr(data[particles_type], self._widgets['particles'][particles_type]['vector'].value.split(",")[2])



    def _scalar_handler(self, change):
        import numpy as np
        import matplotlib.cm
        colormap = matplotlib.cm.viridis
        data = self.get_buffer(self._widgets['frame'].value)['arrays']
        particles_type = change['owner'].description.split("_")[0]
        c = colormap(getattr(data[particles_type], self._widgets['particles'][particles_type]['scalar'].value))
        self.scatters[particles_type].color = c

    def _vector_handler(self, change):
        pass
#        import numpy as np
#        import ipyvolume.pylab as p3
#        import matplotlib.cm
#        data = self.get_buffer(self._widgets['frame'].value)['arrays']
#        particles_type = change['owner'].description.split("_")[0]
#        if change['new'] != '':
#            if particles_type in self.vectors.keys():
#                self.vectors[particles_type].vx = getattr(data[particles_type], self._widgets['particles'][particles_type]['vector'].value.split(",")[0])
#                self.vectors[particles_type].vy = getattr(data[particles_type], self._widgets['particles'][particles_type]['vector'].value.split(",")[1])
#                self.vectors[particles_type].vz = getattr(data[particles_type], self._widgets['particles'][particles_type]['vector'].value.split(",")[2])
#                self.vectors[particles_type].size = self._widgets['particles'][particles_type]['vector_width'].value
#            else:
#                self.vectors[particles_type] = p3.quiver(
#                    data[particles_type].x,
#                    data[particles_type].y,
#                    data[particles_type].z,
#                    getattr(data[particles_type], self._widgets['particles'][particles_type]['vector'].value.split(",")[0]),
#                    getattr(data[particles_type], self._widgets['particles'][particles_type]['vector'].value.split(",")[1]),
#                    getattr(data[particles_type], self._widgets['particles'][particles_type]['vector'].value.split(",")[2]),
#                    size=self._widgets['particles'][particles_type]['vector_width'].value,
#                )
#        else:
#            if particles_type in self.vectors.keys():
#                self.vectors[particles_type].size = 0

    def _vector_scale_handler(self, change):
        pass

    def _scalar_size_handler(self, change):
        particles_type = change['owner'].description.split("_")[0]
        self.scatters[particles_type].size = self._widgets['particles'][particles_type]['scalar_size'].value

    def _vector_width_handler(self, change):
        pass

    def _legend_handler(self, change, manual=False):
        pass

    def _save_figure_handler(self, change):
        pass
#        import ipyvolume.pylab as p3
#        for extension in [
#            '.png',
#            '.jpeg',
#        ]:
#            if self._widgets['save_figure'].value.endswith(extension):
#                p3.savefig(self._widgets['save_figure'].value)
#                print(
#                    "Saved figure as {} in the present working directory"
#                    .format(
#                        self._widgets['save_figure'].value
#                    )
#                )
#                break
#        self._widgets['save_figure'].value = ""
