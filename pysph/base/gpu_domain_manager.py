from __future__ import print_function
import numpy as np

from pysph.base.nnps_base import DomainManagerBase

from compyle.config import get_config
from compyle.array import Array, get_backend
from compyle.parallel import Elementwise, Reduction, Scan
from compyle.types import annotate, dtype_to_ctype

from pytools import memoize_method


class GPUDomainManager(DomainManagerBase):
    def __init__(self, xmin=-1000., xmax=1000., ymin=0.,
                 ymax=0., zmin=0., zmax=0.,
                 periodic_in_x=False, periodic_in_y=False,
                 periodic_in_z=False, n_layers=2.0, backend=None, props=None,
                 mirror_in_x=False, mirror_in_y=False, mirror_in_z=False):
        """Constructor"""
        DomainManagerBase.__init__(self, xmin=xmin, xmax=xmax,
                                   ymin=ymin, ymax=ymax, zmin=zmin, zmax=zmax,
                                   periodic_in_x=periodic_in_x,
                                   periodic_in_y=periodic_in_y,
                                   periodic_in_z=periodic_in_z,
                                   n_layers=n_layers, props=props)

        self.use_double = get_config().use_double
        self.dtype = np.float64 if self.use_double else np.float32

        self.dtype_max = np.finfo(self.dtype).max
        self.backend = get_backend(backend)

        self.ghosts = None

    def update(self):
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
            # remove periodic ghost particles from a previous step
            self._remove_ghosts()

            # box-wrap current particles for periodicity
            self._box_wrap_periodic()

            # create new periodic ghosts
            self._create_ghosts_periodic()

    def _compute_cell_size_for_binning(self):
        """Compute the cell size for the binning.

        The cell size is chosen as the kernel radius scale times the
        maximum smoothing length in the local processor. For parallel
        runs, we would need to communicate the maximum 'h' on all
        processors to decide on the appropriate binning size.

        """
        _hmax, hmax = -1.0, -1.0
        _hmin, hmin = self.dtype_max, self.dtype_max

        for pa_wrapper in self.pa_wrappers:
            h = pa_wrapper.pa.gpu.get_device_array('h')
            pa_wrapper.pa.gpu.update_minmax_cl(['h'])

            _hmax = h.maximum
            _hmin = h.minimum

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

    @memoize_method
    def _get_box_wrap_kernel(self):
        @annotate
        def box_wrap(i, x, y, z, xmin, ymin, zmin, xmax, ymax, zmax,
                     xtranslate, ytranslate, ztranslate,
                     periodic_in_x, periodic_in_y, periodic_in_z):
            if periodic_in_x:
                if x[i] < xmin:
                    x[i] = x[i] + xtranslate
                if x[i] > xmax:
                    x[i] = x[i] - xtranslate

            if periodic_in_y:
                if y[i] < ymin:
                    y[i] = y[i] + ytranslate
                if y[i] > ymax:
                    y[i] = y[i] - ytranslate

            if periodic_in_z:
                if z[i] < zmin:
                    z[i] = z[i] + ztranslate
                if z[i] > zmax:
                    z[i] = z[i] - ztranslate

        return Elementwise(box_wrap, backend=self.backend)

    ###########################CHANGE FROM HERE###############################

    def _box_wrap_periodic(self):
        """Box-wrap particles for periodicity

        The periodic domain is a rectangular box defined by minimum
        and maximum values in each coordinate direction. These values
        are used in turn to define translation values used to box-wrap
        particles that cross a periodic boundary.

        The periodic domain is specified using the DomainManager object

        """
        # minimum and maximum values of the domain
        xmin, xmax = self.xmin, self.xmax
        ymin, ymax = self.ymin, self.ymax
        zmin, zmax = self.zmin, self.zmax

        # translations along each coordinate direction
        xtranslate = self.xtranslate
        ytranslate = self.ytranslate
        ztranslate = self.ztranslate

        # periodicity flags for NNPS
        periodic_in_x = self.periodic_in_x
        periodic_in_y = self.periodic_in_y
        periodic_in_z = self.periodic_in_z

        box_wrap_knl = self._get_box_wrap_kernel()

        # iterate over each array and mark for translation
        for pa_wrapper in self.pa_wrappers:
            x = pa_wrapper.pa.gpu.x
            y = pa_wrapper.pa.gpu.y
            z = pa_wrapper.pa.gpu.z

            box_wrap_knl(x, y, z, xmin, ymin, zmin, xmax, ymax, zmax,
                         xtranslate, ytranslate, ztranslate,
                         periodic_in_x, periodic_in_y, periodic_in_z)

    @memoize_method
    def _get_ghosts_reduction_kernel(self):
        @annotate
        def map_func(i, periodic_in_x, periodic_in_y, periodic_in_z,
                     x, y, z, xmin, ymin, zmin, xmax, ymax, zmax, cell_size):
            x_copies, y_copies, z_copies = declare('int', 3)

            x_copies = 1
            y_copies = 1
            z_copies = 1

            if periodic_in_x:
                if (x[i] - xmin) <= cell_size:
                    x_copies += 1
                if (xmax - x[i]) <= cell_size:
                    x_copies += 1

            if periodic_in_y:
                if (y[i] - ymin) <= cell_size:
                    y_copies += 1
                if (ymax - y[i]) <= cell_size:
                    y_copies += 1

            if periodic_in_z:
                if (z[i] - zmin) <= cell_size:
                    z_copies += 1
                if (zmax - z[i]) <= cell_size:
                    z_copies += 1

            return x_copies * y_copies * z_copies - 1

        return Reduction('a+b', map_func=map_func, dtype_out=np.int32,
                         backend=self.backend)

    @memoize_method
    def _get_ghosts_scan_kernel(self):
        @annotate
        def inp_fill_ghosts(i, periodic_in_x, periodic_in_y, periodic_in_z,
                            x, y, z, xmin, ymin, zmin, xmax, ymax, zmax,
                            cell_size):
            x_copies, y_copies, z_copies = declare('int', 3)

            x_copies = 1
            y_copies = 1
            z_copies = 1

            if periodic_in_x:
                if (x[i] - xmin) <= cell_size:
                    x_copies += 1
                if (xmax - x[i]) <= cell_size:
                    x_copies += 1

            if periodic_in_y:
                if (y[i] - ymin) <= cell_size:
                    y_copies += 1
                if (ymax - y[i]) <= cell_size:
                    y_copies += 1

            if periodic_in_z:
                if (z[i] - zmin) <= cell_size:
                    z_copies += 1
                if (zmax - z[i]) <= cell_size:
                    z_copies += 1

            return x_copies * y_copies * z_copies - 1

        @annotate
        def out_fill_ghosts(i, item, prev_item, periodic_in_x,
                            periodic_in_y, periodic_in_z, x, y, z,
                            xmin, ymin, zmin, xmax, ymax, zmax, cell_size,
                            masks, indices):
            xleft, yleft, zleft = declare('int', 3)
            xright, yright, zright = declare('int', 3)

            xleft = 0
            yleft = 0
            zleft = 0

            xright = 0
            yright = 0
            zright = 0

            if periodic_in_x:
                if (x[i] - xmin) <= cell_size:
                    xright = 1
                if (xmax - x[i]) <= cell_size:
                    xleft = -1

            if periodic_in_y:
                if (y[i] - ymin) <= cell_size:
                    yright = 1
                if (ymax - y[i]) <= cell_size:
                    yleft = -1

            if periodic_in_z:
                if (z[i] - zmin) <= cell_size:
                    zright = 1
                if (zmax - z[i]) <= cell_size:
                    zleft = -1

            xp, yp, zp = declare('int', 3)
            idx, mask = declare('int', 2)

            idx = prev_item

            for xp in range(-1, 2):
                if xp != 0 and ((xleft == 0 and xright == 0) or
                                (xp != xleft and xp != xright)):
                    continue
                for yp in range(-1, 2):
                    if yp != 0 and ((yleft == 0 and yright == 0) or
                                    (yp != yleft and yp != yright)):
                        continue
                    for zp in range(-1, 2):
                        if zp != 0 and ((zleft == 0 and zright == 0) or
                                        (zp != zleft and zp != zright)):
                            continue
                        if xp == 0 and yp == 0 and zp == 0:
                            continue
                        mask = (xp + 1) * 9 + (yp + 1) * 3 + (zp + 1)
                        masks[idx] = mask
                        indices[idx] = i
                        idx += 1

        return Scan(inp_fill_ghosts, out_fill_ghosts, 'a+b',
                    dtype=np.int32, backend=self.backend)

    @memoize_method
    def _get_translate_kernel(self):
        @annotate
        def translate(i, x, y, z, tag, xtranslate, ytranslate,
                      ztranslate, masks):
            xmask, ymask, zmask, mask = declare('int', 4)
            mask = masks[i]

            zmask = mask % 3
            mask /= 3
            ymask = mask % 3
            mask /= 3
            xmask = mask % 3

            x[i] = x[i] + (xmask - 1) * xtranslate
            y[i] = y[i] + (ymask - 1) * ytranslate
            z[i] = z[i] + (zmask - 1) * ztranslate

            tag[i] = 2

        return Elementwise(translate, backend=self.backend)

    def _create_ghosts_periodic(self):
        """Identify boundary particles and create images.

        We need to find all particles that are within a specified
        distance from the boundaries and place image copies on the
        other side of the boundary. Corner reflections need to be
        accounted for when using domains with multiple periodicity.

        The periodic domain is specified using the DomainManager object

        """
        copy_props = self.copy_props
        pa_wrappers = self.pa_wrappers
        narrays = self.narrays

        # cell size used to check for periodic ghosts. For summation density
        # like operations, we need to create two layers of ghost images, this
        # is configurable via the n_layers argument to the constructor.
        cell_size = self.n_layers * self.cell_size

        # periodic domain values
        xmin, xmax = self.xmin, self.xmax
        ymin, ymax = self.ymin, self.ymax
        zmin, zmax = self.zmin, self.zmax

        xtranslate = self.xtranslate
        ytranslate = self.ytranslate
        ztranslate = self.ztranslate

        # periodicity flags
        periodic_in_x = self.periodic_in_x
        periodic_in_y = self.periodic_in_y
        periodic_in_z = self.periodic_in_z

        reduce_knl = self._get_ghosts_reduction_kernel()
        scan_knl = self._get_ghosts_scan_kernel()
        translate_knl = self._get_translate_kernel()

        if not self.ghosts:
            self.ghosts = [paw.pa.empty_clone(props=copy_props[i])
                           for i, paw in enumerate(pa_wrappers)]
        else:
            for ghost_pa in self.ghosts:
                ghost_pa.resize(0)
            for i in range(narrays):
                self.ghosts[i].ensure_properties(
                    pa_wrappers[i].pa, props=copy_props[i]
                )

        for i, pa_wrapper in enumerate(self.pa_wrappers):
            ghost_pa = self.ghosts[i]

            x = pa_wrapper.pa.gpu.x
            y = pa_wrapper.pa.gpu.y
            z = pa_wrapper.pa.gpu.z

            num_extra_particles = reduce_knl(periodic_in_x, periodic_in_y,
                                             periodic_in_z, x, y, z, xmin,
                                             ymin, zmin, xmax, ymax, zmax,
                                             cell_size)

            num_extra_particles = int(num_extra_particles)

            indices = Array(np.int32, n=num_extra_particles)
            masks = Array(np.int32, n=num_extra_particles)

            scan_knl(periodic_in_x=periodic_in_x, periodic_in_y=periodic_in_y,
                     periodic_in_z=periodic_in_z, x=x, y=y, z=z, xmin=xmin,
                     ymin=ymin, zmin=zmin, xmax=xmax, ymax=ymax, zmax=zmax,
                     cell_size=cell_size, masks=masks, indices=indices)

            pa_wrapper.pa.extract_particles(
                indices, ghost_pa, align=False, props=copy_props[i]
            )

            translate_knl(ghost_pa.gpu.x, ghost_pa.gpu.y, ghost_pa.gpu.z,
                          ghost_pa.gpu.tag, xtranslate, ytranslate,
                          ztranslate, masks)

            pa_wrapper.pa.append_parray(ghost_pa, align=False)
