import numpy as np
from stl import stl

# PySPH base and carray imports
from pysph.base.utils import get_particle_array
from pysph.base.kernels import QuinticSpline
from pysph.solver.solver import Solver
from pysph.sph.integrator import EulerIntegrator, IntegratorStep
from pysph.sph.integrator_step import EulerStep
from pysph.sph.equation import Group, Equation
from pysph.solver.application import Application
from pysph.tools.geometry import remove_overlap_particles

from pysph.tools.particle_packing import (
    ParticlePacking, calculate_normal_2d_surface, shift_surface_inside)


def get_packing_folders(folder, dx):
    """Get all the required folder and files names for the packing.
    This is done to avoid mixing of multiple particle spacing

     Parameters
    ----------
    folder : name of the output folder from Application subclass
    dx : required particles spacing

    Returns
    -------
    preprocess_folder : The output folder for packing Application
    layer_folder : The output folder for layer packing around the hexagonal
                   packing
    res_file : the name of the packed file containing the coordinates
    """
    import os
    parent = os.path.dirname(folder)
    basename = os.path.basename(folder)
    preprocess_folder = os.path.join(parent, 'packing_%.4f' % dx)
    layer_folder = os.path.join(parent, 'layer_%.4f' % dx)

    res_folder = os.path.join(parent, 'preprocess')
    os.makedirs(res_folder, exist_ok=True)

    res_file = os.path.join(parent, 'preprocess', basename + '_%.4f.npz' % dx)

    return preprocess_folder, layer_folder, res_file


def readdata(resfile):
    data = np.load(resfile)
    xs = data['xs']
    ys = data['ys']
    zs = data['zs']
    xf = data['xf']
    yf = data['yf']
    zf = data['zf']
    return xs, ys, zs, xf, yf, zf


class Packer(Application):
    def __init__(self, fname, output_dir, domain, add_opt_func, dx, out,
                 dim=None, x=None, y=None, z=None, L=0.0, B=0.0,
                 H=0.0, filename=None,
                 hardpoints=None, use_prediction=False,
                 filter_layers=False, reduce_dfreq=False,
                 tol=1e-2, scale=1.0, shift=False,
                 invert_normal=False, pb=None, nu=None,
                 k=None, dfreq=-1, no_solid=False):
        self.hdx = 1.2
        self.dx = dx
        self.x = x
        self.y = y
        self.z = z
        self.L = L
        self.B = B
        self.H = H
        self.filename = filename
        self.dfreq = dfreq
        self.hardpoints = {} if hardpoints is None else hardpoints
        self.use_prediction = use_prediction
        self.filter_layers = filter_layers
        self.reduce_dfreq = reduce_dfreq
        self.tol = tol
        self.dim = dim
        self.scale = scale
        self.shift = shift
        self.invert_normal = invert_normal
        self.pb = pb
        self.nu = nu
        self.k = k
        self.out = out
        self.no_solid = no_solid
        self.add_opt_func = add_opt_func

        self.bound = self._get_bound()

        super(Packer, self).__init__(fname, output_dir, domain)

    def add_user_options(self, group):
        self.add_opt_func(group)

    def _get_bound(self):
        import sys
        import os
        from pysph.tools.particle_packing import get_bounding_box
        if self.filename is not None:
            file, ext = os.path.splitext(self.filename)
            print(ext)
            if (ext == '.txt') or (ext == '.csv'):
                self.dim = 2
            elif ext == '.stl':
                self.dim = 3
            else:
                print('file extension %s not supported' % ext)
                sys.exit()

            if self.dim == 2:
                try:
                    self.x, self.y = np.loadtxt(self.filename, unpack=True)
                    self.x *= self.scale
                    self.y *= self.scale
                except IOError:
                    print('read the file')
                    print('The supported file format is \"x y\"')
                    sys.exit()

            if self.dim == 3:
                try:
                    data = stl.StlMesh(self.filename)
                    self.x = self.scale * data.x
                    self.y = self.scale * data.y
                    self.z = self.scale * data.z
                except IOError:
                    print('read the file')
                    print('The supported file format is \"x y\"')
                    sys.exit()

        if self.z is None:
            self.dim = 2
            self.z = np.zeros_like(self.x)

        return get_bounding_box(
            self.dx, self.x, self.y, self.z, self.L, self.B, self.H)

    def create_particles(self):
        s = self.scheme

        bound = self.bound
        free = s.create_free_particles(bound, name='free')
        frozen = s.create_frozen_container(bound, name='frozen')
        particles = [free, frozen]
        if (self.filename is None) and (self.x is None):
            free = s.create_free_particles(bound, name='free', rect=True)
            frozen = s.create_frozen_container(bound, name='frozen', rect=True)
            np.random.seed(10)
            x = free.x
            free.x += (np.random.random(len(x)) - 0.5) * self.dx
            free.y += (np.random.random(len(x)) - 0.5) * self.dx
            if self.dim == 3:
                free.z += (np.random.random(len(x)) - 0.5) * self.dx
            particles = [free, frozen]
        else:
            nodes = None
            if self.filename is None:
                nodes = s.create_boundary_node(
                    self.filename, [self.x, self.y], scale=self.scale,
                    shift=self.shift, invert=self.invert_normal,
                    name='nodes')
            else:
                nodes = s.create_boundary_node(
                    self.filename, scale=self.scale, shift=self.shift,
                    invert=self.invert_normal, name='nodes')

            boundary = get_particle_array(name='boundary')
            particles.extend([boundary, nodes])

        s.setup_properties(particles)
        for pa in particles:
            pa.dt_adapt[:] = 1e20
        return particles

    def create_scheme(self):
        hardpoints = self.hardpoints
        if self.no_solid:
            s = ParticlePacking(
                fluids=['free'], solids={}, frozen=['frozen'],
                dim=self.dim, hdx=self.hdx, dx=self.dx, nu=self.nu,
                pb=self.pb, k=self.k, tol=self.tol)
        else:
            s = ParticlePacking(
                fluids=['free'], solids={'boundary': 'nodes'},
                frozen=['frozen'], dim=self.dim,
                use_prediction=self.use_prediction,
                filter_layers=self.filter_layers,
                reduce_dfreq=self.reduce_dfreq,
                hdx=self.hdx, dx=self.dx, hardpoints=hardpoints,
                nu=self.nu, pb=self.pb, k=self.k, tol=self.tol,
                dfreq=self.dfreq)

        s.configure_solver(dt=1e-5)
        return s

    def post_step(self, solver):
        self.scheme.post_step(self.particles, solver)

    def post_process(self, info_fname):
        import os
        from pysph.solver.utils import load
        self.read_info(info_fname)
        if len(self.output_files) == 0:
            return
        res = self.out
        filename = self.output_files[-1]
        data = load(filename)
        free = data['arrays']['free']
        frozen = data['arrays']['frozen']
        solid = None
        solid_nodes = None
        if len(data['arrays']) > 2:
            solid = data['arrays']['boundary']
            solid_nodes = data['arrays']['nodes']
        self.scheme.post_process(
            free, solid, solid_nodes, frozen, self.dx, res)


class HexaToRectLayer(Packer):
    def create_particles(self):
        s = self.scheme

        bound = self.bound
        free = s.create_free_particles(bound, name='free', outer=True)
        frozen = s.create_frozen_container(bound, name='frozen', outer=True)
        particles = [free, frozen]
        s.setup_properties(particles)
        for pa in particles:
            pa.dt_adapt[:] = 1e20
        return particles
