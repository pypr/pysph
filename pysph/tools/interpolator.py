# Standard library imports
from functools import reduce

# Library imports.
import numpy as np

# Package imports.
from pysph.base.utils import get_particle_array
from pysph.base.kernels import Gaussian
from pysph.base.nnps import LinkedListNNPS as NNPS
from pysph.sph.equation import Equation, Group
from pysph.sph.acceleration_eval import AccelerationEval
from pysph.sph.sph_compiler import SPHCompiler
from compyle.api import declare
from pysph.sph.wc.linalg import gj_solve, augmented_matrix
from pysph.sph.basic_equations import SummationDensity


class InterpolateFunction(Equation):
    def initialize(self, d_idx, d_prop, d_number_density):
        d_prop[d_idx] = 0.0
        d_number_density[d_idx] = 0.0

    def loop(self, s_idx, d_idx, s_temp_prop, d_prop, d_number_density, WIJ):
        d_number_density[d_idx] += WIJ
        d_prop[d_idx] += WIJ*s_temp_prop[s_idx]

    def post_loop(self, d_idx, d_prop, d_number_density):
        if d_number_density[d_idx] > 1e-12:
            d_prop[d_idx] /= d_number_density[d_idx]


class InterpolateSPH(Equation):
    def initialize(self, d_idx, d_prop):
        d_prop[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_rho, s_m, s_temp_prop, d_prop, WIJ):
        d_prop[d_idx] += s_m[s_idx]/s_rho[s_idx]*WIJ*s_temp_prop[s_idx]


class SplashInterpolateProperty(Equation):
    def __init__(self, dest, sources, nx=1, ny=1, nz=1):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        super().__init__(dest, sources)

    def loop(self, d_idx, s_idx, WI, s_ix, s_iy, s_iz, d_interpolated,
             d_m, d_rho, d_prop2interp):
        pxidx = declare('int')
        pxidx = s_ix[s_idx] * self.ny * self.nz + s_iy[s_idx] * self.nz + \
                s_iz[s_idx]

        d_interpolated[pxidx] += (d_m[d_idx] /
                                  d_rho[d_idx]) * WI * d_prop2interp[d_idx]


class SplashInterpolateUnity(SplashInterpolateProperty):
    def loop(self, d_idx, s_idx, WI, s_ix, s_iy, s_iz, d_unity,
             d_m, d_rho, d_prop2interp):
        pxidx = declare('int')
        pxidx = s_ix[s_idx] * self.ny * self.nz + s_iy[s_idx] * self.nz + \
                s_iz[s_idx]

        d_unity[pxidx] += (d_m[d_idx] / d_rho[d_idx]) * WI


class SPHFirstOrderApproximationPreStep(Equation):
    def __init__(self, dest, sources, dim=1):
        self.dim = dim

        super(SPHFirstOrderApproximationPreStep, self).__init__(dest, sources)

    def initialize(self, d_idx, d_moment):
        i, j = declare('int', 2)

        for i in range(4):
            for j in range(4):
                d_moment[16*d_idx + j+4*i] = 0.0

    def loop(self, d_idx, s_idx, d_h, s_h, s_x, s_y, s_z, d_x, d_y, d_z, s_rho,
             s_m, WIJ, XIJ, DWIJ, d_moment):
        Vj = s_m[s_idx] / s_rho[s_idx]
        i16 = declare('int')
        i16 = 16*d_idx

        d_moment[i16+0] += WIJ * Vj

        d_moment[i16+1] += -XIJ[0] * WIJ * Vj
        d_moment[i16+2] += -XIJ[1] * WIJ * Vj
        d_moment[i16+3] += -XIJ[2] * WIJ * Vj

        d_moment[i16+4] += DWIJ[0] * Vj
        d_moment[i16+8] += DWIJ[1] * Vj
        d_moment[i16+12] += DWIJ[2] * Vj

        d_moment[i16+5] += -XIJ[0] * DWIJ[0] * Vj
        d_moment[i16+6] += -XIJ[1] * DWIJ[0] * Vj
        d_moment[i16+7] += -XIJ[2] * DWIJ[0] * Vj

        d_moment[i16+9] += - XIJ[0] * DWIJ[1] * Vj
        d_moment[i16+10] += -XIJ[1] * DWIJ[1] * Vj
        d_moment[i16+11] += -XIJ[2] * DWIJ[1] * Vj

        d_moment[i16+13] += -XIJ[0] * DWIJ[2] * Vj
        d_moment[i16+14] += -XIJ[1] * DWIJ[2] * Vj
        d_moment[i16+15] += -XIJ[2] * DWIJ[2] * Vj


class SPHFirstOrderApproximation(Equation):
    """First order SPH approximation.

    The method used to solve the linear system in this function is not same as
    in the reference. In the function :math:`Ax=b` is solved where :math:`A :=
    moment` (Moment matrix) and :math:`b := p_sph` (Property calculated using
    basic SPH). The calculation need the "moment" to be evaluated before this
    step which is done in `SPHFirstOrderApproximationPreStep`

    References
    -----------

    .. [Liu2006] M.B. Liu, G.R. Liu, "Restoring particle consistency in
       smoothed particle hydrodynamics", Applied Numerical Mathematics
       Volume 56, Issue 1 2006, Pages 19-36, ISSN 0168-9274

    """
    def _get_helpers_(self):
        return [gj_solve, augmented_matrix]

    def __init__(self, dest, sources, dim=1):
        self.dim = dim

        super(SPHFirstOrderApproximation, self).__init__(dest, sources)

    def initialize(self, d_idx, d_prop, d_p_sph):
        i = declare('int')

        for i in range(3):
            d_prop[4*d_idx+i] = 0.0
            d_p_sph[4*d_idx+i] = 0.0

    def loop(self, d_idx, d_h, s_h, s_x, s_y, s_z, d_x, d_y, d_z, s_rho,
             s_m, WIJ, DWIJ, s_temp_prop, d_p_sph, s_idx):
        i4 = declare('int')
        Vj = s_m[s_idx] / s_rho[s_idx]
        pj = s_temp_prop[s_idx]
        i4 = 4*d_idx

        d_p_sph[i4+0] += pj * WIJ * Vj
        d_p_sph[i4+1] += pj * DWIJ[0] * Vj
        d_p_sph[i4+2] += pj * DWIJ[1] * Vj
        d_p_sph[i4+3] += pj * DWIJ[2] * Vj

    def post_loop(self, d_idx, d_moment, d_prop, d_p_sph):

        a_mat = declare('matrix(16)')
        aug_mat = declare('matrix(20)')
        b = declare('matrix(4)')
        res = declare('matrix(4)')

        i, n, i16, i4 = declare('int', 4)
        i16 = 16*d_idx
        i4 = 4*d_idx
        for i in range(16):
            a_mat[i] = d_moment[i16+i]
        for i in range(20):
            aug_mat[i] = 0.0
        for i in range(4):
            b[i] = d_p_sph[4*d_idx+i]
            res[i] = 0.0

        n = self.dim + 1
        augmented_matrix(a_mat, b, n, 1, 4, aug_mat)
        gj_solve(aug_mat, n, 1, res)
        for i in range(4):
            d_prop[i4+i] = res[i]


def get_bounding_box(particle_arrays, tight=False, stretch=0.05):
    """Find the size of the domain given a sequence of particle arrays.

    If `tight` is True, the bounds are tight, if not the domain is stretched
    along each dimension by an amount `stretch` specified as a percentage of
    the length along that dimension is added in each dimension.

    """
    xmin, xmax = 1e20, -1e20
    ymin, ymax = 1e20, -1e20
    zmin, zmax = 1e20, -1e20
    for pa in particle_arrays:
        x, y, z = pa.x, pa.y, pa.z
        xmin = min(xmin, x.min())
        xmax = max(xmax, x.max())
        ymin = min(ymin, y.min())
        ymax = max(ymax, y.max())
        zmin = min(zmin, z.min())
        zmax = max(zmax, z.max())

    bounds = np.asarray((xmin, xmax, ymin, ymax, zmin, zmax))
    if not tight:
        # Add the extra space.
        lengths = stretch*np.repeat(bounds[1::2] - bounds[::2], 2)
        lengths[::2] *= -1.0
        bounds += lengths

    return bounds


def get_nx_ny_nz(num_points, bounds):
    """Given a number of points to use and the bounds, return a triplet
    of integers for a uniform mesh with approximately that many points.
    """
    bounds = np.asarray(bounds, dtype=float)
    length = bounds[1::2] - bounds[::2]
    total_length = length.sum()
    rel_length = length/total_length
    non_zero = rel_length > 1e-3
    dim = int(non_zero.sum())
    volume = np.prod(length[non_zero])
    delta = pow(volume/num_points, 1.0/dim)
    dimensions = np.ones(3, dtype=int)
    for i in range(3):
        if rel_length[i] > 1e-4:
            dimensions[i] = int(round(length[i]/delta))

    return dimensions

def infer_nx_ny_nz_from_shape(shape):
    if len(shape) == 1:
        nx, = shape
        ny, nz = 1, 1
    elif len(shape) == 2:
        nx, ny = shape
        nz = 1
    else:
        nx, ny, nz = shape
    return nx, ny, nz

class Interpolator(object):
    """Convenient class to interpolate particle properties onto a uniform grid
    or given set of particles.  This is particularly handy for visualization.

    """

    def __init__(self, particle_arrays, num_points=125000, kernel=None,
                 x=None, y=None, z=None, domain_manager=None,
                 equations=None, method='shepard'):
        """
        The x, y, z coordinates need not be specified, and if they are not,
        the bounds of the interpolated domain is automatically computed and
        `num_points` number of points are used in this domain uniformly placed.

        Parameters
        ----------

        particle_arrays: list
            A list of particle arrays.
        num_points: int
            the number of points to interpolate on to.
        kernel: Kernel
            the kernel to use for interpolation.
        x: ndarray
            the x-coordinate of points on which to interpolate.
        y: ndarray
            the y-coordinate of points on which to interpolate.
        z: ndarray
            the z-coordinate of points on which to interpolate.
        domain_manager: DomainManager
            An optional Domain manager for periodic domains.
        equations: sequence
            A sequence of equations or groups.  Defaults to None.  This is
            used only if the default interpolation equations are inadequate.
        method : str
            String with the following allowed methods: 'shepard', 'sph',
            'order1'
        """
        self._set_particle_arrays(particle_arrays)
        bounds = get_bounding_box(self.particle_arrays)
        shape = get_nx_ny_nz(num_points, bounds)
        self.dim = 3 - list(shape).count(1)

        if kernel is None:
            self.kernel = Gaussian(dim=self.dim)
        else:
            self.kernel = kernel

        self.pa = None
        self.nnps = None
        self.equations = equations
        self.func_eval = None
        self.domain_manager = domain_manager
        self.method = method
        if method not in ['sph', 'shepard', 'order1']:
            raise RuntimeError('%s method is not implemented' % (method))
        if x is None and y is None and z is None:
            self.set_domain(bounds, shape)
        else:
            self.set_interpolation_points(x=x, y=y, z=z)

    # ## Interpolator protocol ###############################################
    def set_interpolation_points(self, x=None, y=None, z=None):
        """Set the points on which we must interpolate the arrays.

        If any of x, y, z is not passed it is assumed to be 0.0 and shaped
        like the other non-None arrays.


        Parameters
        ----------

        x: ndarray
            the x-coordinate of points on which to interpolate.
        y: ndarray
            the y-coordinate of points on which to interpolate.
        z: ndarray
            the z-coordinate of points on which to interpolate.

        """
        tmp = None
        for tmp in (x, y, z):
            if tmp is not None:
                break
        if tmp is None:
            raise RuntimeError('At least one non-None array must be given.')

        def _get_array(_t):
            return np.asarray(_t) if _t is not None else np.zeros_like(tmp)

        x, y, z = _get_array(x), _get_array(y), _get_array(z)

        self.shape = x.shape
        self.pa = self._create_particle_array(x, y, z)
        arrays = self.particle_arrays + [self.pa]

        if self.func_eval is None:
            self._compile_acceleration_eval(arrays)

        self.update_particle_arrays(self.particle_arrays)

    def set_domain(self, bounds, shape):
        """Set the domain to interpolate into.

        Parameters
        ----------

        bounds: tuple
            (xmin, xmax, ymin, ymax, zmin, zmax)
        shape: tuple
            (nx, ny, nz)
        """
        self.bounds = np.asarray(bounds)
        self.shape = np.asarray(shape)
        x, y, z = self._create_default_points(self.bounds, self.shape)
        self.set_interpolation_points(x, y, z)

    def interpolate(self, prop, comp=0):
        """Interpolate given property.

        Parameters
        ----------

        prop: str
            The name of the property to interpolate.

        comp: int
            The component of the gradient required

        Returns
        -------
        A numpy array suitably shaped with the property interpolated.
        """
        assert isinstance(comp, int), 'Error: only interger value is allowed'
        for array in self.particle_arrays:
            if prop not in array.properties:
                data = 0.0
            else:
                data = array.get(prop, only_real_particles=False)

            array.get('temp_prop', only_real_particles=False)[:] = data

        self.func_eval.compute(0.0, 0.1)  # These are junk arguments.
        if comp and (self.method in ['sph', 'shepard']):
            raise RuntimeError("Error: use 'order1' method to evaluate"
                               "gradient")
        elif self.method in ['sph', 'shepard']:
            result = self.pa.prop.copy()
        else:
            if comp > 3:
                raise RuntimeError("Error: Only 0, 1, 2, 3 allowed")
            result = self.pa.prop[comp::4].copy()

        result.shape = self.shape
        return result.squeeze()

    def update(self, update_domain=True):
        """Update the NNPS when particles have moved.

        If the update_domain is False, the domain is not updated.

        Use this when the arrays are the same but the particles have themselves
        changed. If the particle arrays themselves change use the
        `update_particle_arrays` method instead.
        """
        if update_domain:
            self.nnps.update_domain()
        self.nnps.update()

    def update_particle_arrays(self, particle_arrays):
        """Call this for a new set of particle arrays which have the
        same properties as before.

        For example, if you are reading the particle array data from files,
        each time you load a new file a new particle array is read with the
        same properties.  Call this function to reset the arrays.
        """
        self._set_particle_arrays(particle_arrays)
        arrays = self.particle_arrays + [self.pa]
        self._create_nnps(arrays)
        self.func_eval.update_particle_arrays(arrays)

    # ### Private protocol ###################################################

    def _create_nnps(self, arrays):
        # create the neighbor locator object
        self.nnps = NNPS(dim=self.kernel.dim, particles=arrays,
                         radius_scale=self.kernel.radius_scale,
                         domain=self.domain_manager,
                         cache=True)
        self.func_eval.set_nnps(self.nnps)

    def _create_default_points(self, bounds, shape):
        b = bounds
        n = shape
        x, y, z = np.mgrid[
            b[0]:b[1]:n[0]*1j,
            b[2]:b[3]:n[1]*1j,
            b[4]:b[5]:n[2]*1j,
        ]
        return x, y, z

    def _create_particle_array(self, x, y, z):
        xr = x.ravel()
        yr = y.ravel()
        zr = z.ravel()
        self.x, self.y, self.z = x.squeeze(), y.squeeze(), z.squeeze()

        hmax = self._get_max_h_in_arrays()
        h = hmax*np.ones_like(xr)
        pa = get_particle_array(
            name='interpolate',
            x=xr, y=yr, z=zr, h=h,
            number_density=np.zeros_like(xr)
        )
        if self.method in ['sph', 'shepard']:
            pa.add_property('prop')
        else:
            pa.add_property('moment', stride=16)
            pa.add_property('p_sph', stride=4)
            pa.add_property('prop', stride=4)

        return pa

    def _compile_acceleration_eval(self, arrays):
        names = [x.name for x in self.particle_arrays]
        if self.equations is None:
            if self.method == 'shepard':
                equations = [
                    InterpolateFunction(dest='interpolate',
                                        sources=names)
                ]
            elif self.method == 'sph':
                equations = [
                    InterpolateSPH(dest='interpolate',
                                        sources=names)
                ]
            else:
                equations = [
                    Group(equations=[
                        SummationDensity(dest=name, sources=names)
                        for name in names],
                          real=False),
                    Group(equations=[
                        SPHFirstOrderApproximationPreStep(dest='interpolate',
                                                          sources=names,
                                                          dim=self.dim)],
                          real=True),
                    Group(equations=[
                        SPHFirstOrderApproximation(dest='interpolate',
                                                   sources=names,
                                                   dim=self.dim)],
                          real=True)
                ]
        else:
            equations = self.equations
        self.func_eval = AccelerationEval(arrays, equations, self.kernel)
        compiler = SPHCompiler(self.func_eval, None)
        compiler.compile()

    def _get_max_h_in_arrays(self):
        hmax = -1.0
        for array in self.particle_arrays:
            hmax = max(array.h.max(), hmax)
        return hmax

    def _set_particle_arrays(self, particle_arrays):
        self.particle_arrays = particle_arrays
        for array in self.particle_arrays:
            if 'temp_prop' not in array.properties:
                array.add_property('temp_prop')


class SplashInterpolator(Interpolator):
    """
    Just another :py:class:`~pysph.tools.Interpolator` based on [Price2007]_.

    Notes
    -----

    Not everything from [Price2007]_ is implemented here. For example, the
    case of really coarse interpolation grid where the consecutive points on
    the interpolation grid are father than twice the smoothing
    length(read half of the pixel width > smoothing length, in the paper
    terminology) is not implemented.

    Some details about how the SPLASH method from [Price2007]_ is implemented
    here:

    - Make a new particle array named *interp_data* with all particles from
      all the source arrays.
    - Add a new property named *prop2interp* and copy over the property to be
      interpolated *interp_data* particle array.
    - Add a constant array named *interpolated* to the particle array named
      *interp_data*. Since this is supposed to hold interpolated data, that
      should dictate the size of this array.
    - Create a new particle array named *dummy4interp*. This contains the
      points onto which the properties need to be interpolated. So, the *x*,
      *y* and *z* values would represent the location of the points onto which
      the property of interest need to be interpolated. *ix*, *iy* and *iz*
      which are the x, y and z indices of the points on the interpolation grid
      respectively.
    - Evaluate :py:class:`~pysph.tools.SplashInterpolateProperty` equation
      with *dummy4interp* and *interp_data* as the source and destination
      particle arrays respectively.


    References
    ----------

    .. [Price2007] Price, Daniel J. "SPLASH: An interactive visualisation tool
       for Smoothed Particle Hydrodynamics simulations." Publications of the
       Astronomical Society of Australia 24, no. 3 (2007): 159-173.
       https://doi.org/10.1071/AS07022.
    """

    def __init__(self, particle_arrays, num_points=125000, kernel=None,
                 x=None, y=None, z=None, domain_manager=None,
                 equations=None, normalize=False):
        """
        The x, y, z coordinates need not be specified, and if they are not,
        the bounds of the interpolated domain is automatically computed and
        `num_points` number of points are used in this domain uniformly placed.

        Parameters
        ----------

        particle_arrays: list
            A list of particle arrays.
        num_points: int
            the number of points to interpolate on to.
        kernel: Kernel
            the kernel to use for interpolation.
        x: ndarray
            the x-coordinate of points on which to interpolate.
        y: ndarray
            the y-coordinate of points on which to interpolate.
        z: ndarray
            the z-coordinate of points on which to interpolate.
        domain_manager: DomainManager
            An optional Domain manager for periodic domains.
        equations: sequence
            A sequence of equations or groups.  Defaults to None.  This is
            used only if the default interpolation equations are inadequate.
        normalize: bool
            Weather to divide the interpolated property at each point
            by an interpolated unity at the same point.
        """
        self.normalize = normalize
        main_pa = self._create_main_pa(particle_arrays)

        self.particle_arrays = []
        self._set_particle_arrays([main_pa])
        bounds = get_bounding_box(self.particle_arrays)
        shape = get_nx_ny_nz(num_points, bounds)
        self.nx, self.ny, self.nz = infer_nx_ny_nz_from_shape(shape)
        self.dim = 3 - list(shape).count(1)

        if kernel is None:
            self.kernel = Gaussian(dim=self.dim)
        else:
            self.kernel = kernel

        self.shape = shape
        self.pa = None
        self.nnps = None
        self.equations = equations
        self.func_eval = None
        self.domain_manager = domain_manager
        if x is None and y is None and z is None:
            self.set_domain(bounds, shape)
        else:
            self.set_interpolation_points(x=x, y=y, z=z)

    def _create_main_pa(self, particle_arrays):
        main_pa = get_particle_array(name='interp_data')
        for array in particle_arrays:
            main_pa.append_parray(array)
            for key in main_pa.default_values.keys():
                main_pa.default_values[key] = 0
        return main_pa

    def interpolate(self, prop):
        """Interpolate given property.

        Parameters
        ----------

        prop: str
            The name of the property to interpolate.

        Returns
        -------
        A numpy array suitably shaped with the property interpolated.
        """
        for array in self.particle_arrays:
            if prop not in array.properties:
                data = 0.0
            else:
                data = array.get(prop, only_real_particles=False)

            array.get('prop2interp', only_real_particles=False)[:] = data

        main_pa = self.particle_arrays[0]
        main_pa.interpolated[:] = 0.0
        if self.normalize:
            main_pa.unity[:] = 0.0

        self.func_eval.compute(0.0, 0.1)  # These are junk arguments.

        if self.normalize:
            main_pa.interpolated /= main_pa.unity

        result = main_pa.interpolated.copy()

        result.shape = self.shape
        return result.squeeze()

    def set_interpolation_points(self, x=None, y=None, z=None):
        tmp = None
        for tmp in (x, y, z):
            if tmp is not None:
                break
        if tmp is None:
            raise RuntimeError('At least one non-None array must be given.')

        def _get_array(_t):
            return np.asarray(_t) if _t is not None else np.zeros_like(tmp)

        x, y, z = _get_array(x), _get_array(y), _get_array(z)

        self.shape = x.shape
        nx, ny, nz = infer_nx_ny_nz_from_shape(self.shape)
        if nx == self.nx and ny == self.ny and nz == self.nz:
            recompile_func_eval = False
        else:
            self.nx, self.ny, self.nz = nx, ny, nz
            recompile_func_eval = True  # since nx, ny or nz have changed

        self.pa = self._create_particle_array(x, y, z)

        main_pa = self.particle_arrays[0]
        if 'interpolated' in main_pa.constants:
            # Since there is no pa.remove_constant() main_pa need to be nuked
            # and created again.
            if main_pa.interpolated.size != x.size:
                main_pa = self._create_main_pa(self.particle_arrays)
                self._set_particle_arrays([main_pa])

        main_pa = self.particle_arrays[0]
        main_pa.add_constant('interpolated', np.zeros_like(x))
        if self.normalize:
            main_pa.add_constant('unity', np.zeros_like(x))

        arrays = self.particle_arrays + [self.pa]

        if self.func_eval is None or recompile_func_eval:
            self._compile_acceleration_eval(arrays)

        self.update_particle_arrays(self.particle_arrays)

    def _create_particle_array(self, x, y, z):
        self.nx, self.ny, self.nz = infer_nx_ny_nz_from_shape(self.shape)

        xr = x.ravel()
        yr = y.ravel()
        zr = z.ravel()
        self.x, self.y, self.z = x.squeeze(), y.squeeze(), z.squeeze()

        ix, iy, iz = np.mgrid[0:self.nx, 0:self.ny, 0:self.nz]

        pa = get_particle_array(name='dummy4interp', x=xr, y=yr, z=zr)
        pa.add_property('ix', 'int', data=ix)
        pa.add_property('iy', 'int', data=iy)
        pa.add_property('iz', 'int', data=iz)

        return pa

    def update_particle_arrays(self, particle_arrays):
        main_pa = self._create_main_pa(particle_arrays)
        main_pa.add_constant('interpolated', np.zeros_like(self.x))
        if self.normalize:
            main_pa.add_constant('unity', np.zeros_like(self.x))
        self._set_particle_arrays([main_pa])

        arrays = self.particle_arrays + [self.pa]
        self._create_nnps(arrays)
        self.func_eval.update_particle_arrays(arrays)

    def _set_particle_arrays(self, particle_arrays):
        self.particle_arrays = particle_arrays
        for array in self.particle_arrays:
            if 'prop2interp' not in array.properties:
                array.add_property('prop2interp')

    def _compile_acceleration_eval(self, arrays):
        if self.equations is None:
            equations = [
                SplashInterpolateProperty(dest='interp_data',
                                          sources=['dummy4interp'],
                                          nx=self.nx,
                                          ny=self.ny,
                                          nz=self.nz)]
            if self.normalize:
                equations.append(
                    SplashInterpolateUnity(
                        dest='interp_data',
                        sources=['dummy4interp'],
                        nx=self.nx,
                        ny=self.ny,
                        nz=self.nz)
                )

            equation_groups = [Group(equations, real=False)]
        else:
            equation_groups = self.equations
        self.func_eval = AccelerationEval(arrays, equation_groups, self.kernel)
        compiler = SPHCompiler(self.func_eval, None)
        compiler.compile()


def main(fname, prop, npoint):
    from pysph.solver.utils import load
    print("Loading", fname)
    data = load(fname)
    arrays = list(data['arrays'].values())
    interp = Interpolator(arrays, num_points=npoint)
    print(interp.shape)
    print("Interpolating")
    prop = interp.interpolate(prop)
    print("Visualizing")
    from mayavi import mlab
    src = mlab.pipeline.scalar_field(interp.x, interp.y, interp.z, prop)
    if interp.dim == 3:
        mlab.pipeline.scalar_cut_plane(src)
    else:
        mlab.pipeline.surface(src)
    mlab.pipeline.outline(src)
    mlab.show()


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 4:
        print("Usage: interpolator.py filename property num_points")
        sys.exit(1)
    else:
        main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
