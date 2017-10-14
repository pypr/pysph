import numpy as np
import pyopencl as cl
import pyopencl.array

from pysph.base.opencl import get_config


class GPUDomainManager(object):
    def __init__(self, xmin=-1000., xmax=1000., ymin=0.,
                 ymax=0., zmin=0., zmax=0.,
                 periodic_in_x=False, periodic_in_y=False,
                 periodic_in_z=False):
        """Constructor"""
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.zmin = zmin
        self.zmax = zmax

        # Indicates if the domain is periodic
        self.periodic_in_x = periodic_in_x
        self.periodic_in_y = periodic_in_y
        self.periodic_in_z = periodic_in_z
        self.is_periodic = periodic_in_x or periodic_in_y or periodic_in_z

        # get the translates in each coordinate direction
        self.xtranslate = xmax - xmin
        self.ytranslate = ymax - ymin
        self.ztranslate = zmax - zmin

        # empty list of particle array wrappers for now
        self.pa_wrappers = []
        self.narrays = 0

        # default value for the cell size
        self.cell_size = 1.0
        self.hmin = 1.0

        # default DomainManager in_parallel is set to False
        self.in_parallel = False

        use_double = get_config().use_double
        self.dtype = np.float64 if use_double else np.float32

        self.dtype_max = np.finfo(self.dtype).max

    def set_pa_wrappers(self, wrappers):
        self.pa_wrappers = wrappers
        self.narrays = len(wrappers)

    def set_cell_size(self, cell_size):
        self.cell_size = cell_size

    def set_in_parallel(self, in_parallel):
        self.in_parallel = in_parallel

    def set_radius_scale(self, radius_scale):
        self.radius_scale = radius_scale

    def update(self, *args, **kwargs):
        """General method that is called before NNPS can bin particles.

        This method is responsible for the computation of cell sizes
        and creation of any ghost particles for periodic or wall
        boundary conditions.

        """
        # compute the cell sizes
        self.compute_cell_size_for_binning()

        # Periodicity is handled by adjusting particles according to a
        # given cubic domain box. In parallel, it is expected that the
        # appropriate parallel NNPS is responsible for the creation of
        # ghost particles.
        if self.is_periodic and not self.in_parallel:
            self._update_from_gpu()

            # remove periodic ghost particles from a previous step
            self._remove_ghosts()

            # box-wrap current particles for periodicity
            self._box_wrap_periodic()

            # create new periodic ghosts
            self._create_ghosts_periodic()

            # Update GPU.
            self._update_gpu()

    def compute_cell_size_for_binning(self):
        """Compute the cell size for the binning.

        The cell size is chosen as the kernel radius scale times the
        maximum smoothing length in the local processor. For parallel
        runs, we would need to communicate the maximum 'h' on all
        processors to decide on the appropriate binning size.

        """
        _hmax, hmax = -1.0, -1.0
        _hmin, hmin = self.dtype_max, self.dtype_max

        for pa_wrapper in self.pa_wrappers:
            h = pa_wrapper.pa.gpu.h

            _hmax = float(cl.array.max(h).get())
            _hmin = float(cl.array.min(h).get())
            if _hmax > hmax:
                hmax = _hmax
            if _hmin < hmin:
                hmin = _hmin

        cell_size = self.radius_scale * hmax
        self.hmin = self.radius_scale * hmin

        if cell_size < 1e-6:
            cell_size = 1.0

        self.cell_size = cell_size

        # set the cell size for the DomainManager
        self.set_cell_size(cell_size)
