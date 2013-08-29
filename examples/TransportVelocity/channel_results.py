"""Results post-processing for the Poiseuille and Couette flow examples"""

import pysph.solver.utils as utils
from pysph.tools import pprocess
from pysph.base.utils import get_particle_array as gpa
import numpy as np

class PoieuilleChannelResults(pprocess.Results):
    def __init__(self, dirname='poiseuille_output', fname='poiseuille', endswith=".npz",
                 d=0.5, Re=0.0125, nu=1.0):
        super(PoieuilleChannelResults, self).__init__(dirname, fname, endswith)

        self.d = d
        self.L = 2*d
        self.Re = Re
        self.nu = nu

        # maximum velocity based on Reynolds number and geometry
        self.Vmax = Vmax = nu*Re/(2*d)

        # body force
        self.fx = Vmax * 2*nu/(d**2)

    def get_results(self, array="fluid"):
        files = self.files
        nfiles = self.nfiles

        # compute the kinetic energy history for the array
        self.get_ke_history(array)

        # interpolate the u-velocity profile along the centerline
        y = np.linspace(0,1,101)
        x = np.ones_like(y) * 0.2
        h = np.ones_like(y) * 1.5 * 0.01

        dst = gpa('test', x=x, y=y, h=h)

        # take the last solution data
        fname = self.files[-1]
        data = utils.load(fname)
        self.pa = src = data['arrays'][array]

        interp = utils.SPHInterpolate(dim=2, dst=dst, src=src)
        self.ui = ui = interp.interpolate(src.u)
        self.y = y

        # exact parabolic profile for the u-velocity
        self.ye = ye = np.arange(-self.d, self.d+1e-3, 0.01)
        self.ue = -0.5 * self.fx/self.nu * (ye**2 - self.d*self.d)
        ye += self.d

class CouetteFlowResults(pprocess.Results):
    def __init__(self, dirname='couette_output', fname='couette', endswith=".npz",
                 d=0.5, Re=0.0125, nu=1.0):
        super(CouetteFlowResults, self).__init__(dirname, fname, endswith)

        self.d = d
        self.L = 2*d
        self.Re = Re
        self.nu = nu

        # maximum velocity based on Reynolds number and geometry
        self.Vmax = Vmax = nu*Re/(self.L)

    def get_results(self, array="fluid"):
        files = self.files
        nfiles = self.nfiles

        # compute the kinetic energy history for the array
        self.get_ke_history(array)

        # interpolate the u-velocity profile along the centerline
        y = np.linspace(0,1,101)
        x = np.ones_like(y) * 0.2
        h = np.ones_like(y) * 1.5 * 0.01

        dst = gpa('test', x=x, y=y, h=h)

        # take the last solution data
        fname = self.files[-1]
        data = utils.load(fname)
        self.pa = src = data['arrays'][array]

        interp = utils.SPHInterpolate(dim=2, dst=dst, src=src)
        self.ui = ui = interp.interpolate(src.u)
        self.y = y

        # exact parabolic profile for the u-velocity
        self.ye = y
        self.ue = self.Vmax*y/self.L

