"""Solution monitors for the driven cavity problem"""

import pysph.solver.utils as utils
from pysph.tools import pprocess
from pysph.base.utils import get_particle_array as gpa
import numpy as np

MATPLOTLIB_STREAMPLOT=True
import matplotlib.pyplot as plt
if not hasattr(plt, 'streamplot'):
    from streamplot import streamplot
    MATPLOTLIB_STREAMPLOT=False
else:
    streamplot = plt.streamplot

class LDCavityResults(pprocess.Results):
    def __init__(self, dirname='cavity_output', fname='cavity', endswith=".npz",
                 Re=100):
        super(LDCavityResults, self).__init__(dirname, fname, endswith)

        self.Re = Re
        self.nu = 1./Re

    def get_results(self, array="fluid"):
        files = self.files
        nfiles = self.nfiles

        # compute the kinetic energy history for the array
        self.get_ke_history(array)

        # interpolated velocities
        dx = 0.01
        self._x = _x = np.linspace(0,1,101)
        xx, yy = np.meshgrid(_x, _x)
        xgrid = xx.ravel(); ygrid = yy.ravel()
        hgrid = np.ones_like(xgrid) * 1.3 * dx
        self.grid = grid = gpa('grid', x=xgrid, y=ygrid, h=hgrid)


        self.xx = xx; self.yy = yy

        # take the last solution data
        fname = self.files[-1]
        data = utils.load(fname)
        self.pa = src = data['arrays'][array]

        interp = utils.SPHInterpolate(dim=2, dst=grid, src=src)
        self.ui = ui = interp.interpolate(src.u)
        self.vi = vi = interp.interpolate(src.v)

        ui.shape = 101,101
        vi.shape = 101,101

        # velocity magnitude
        self.vmag = vmag = np.sqrt( ui**2 + vi**2 )

    def streamplot(self, density=2):
        f = plt.figure()

        if not MATPLOTLIB_STREAMPLOT:
            streamplot(
                self._x, self._x, self.ui, self.vi,
                color=self.vmag,density=(density, density),
                INTEGRATOR='RK4', linewidth=5*self.vmag/self.vmag.max() )
        else:
            streamplot(
                self._x, self._x, self.ui, self.vi, density=(density, density),
                linewidth=5*self.vmag/self.vmag.max(),
                color=self.vmag)
        plt.show()

    def centerline_velocities(self):
        import matplotlib.pyplot as plt
        f = plt.figure()

        ui = self.ui[:, 50]
        vi = self.vi[50]

        s1 = plt.subplot(211)
        s1.plot(ui, self._x)
        s1.set_xlabel(r'$v_x$')
        s1.set_ylabel(r'$y$')

        s2 = plt.subplot(212)
        s2.plot(self._x, vi)
        s2.set_xlabel(r'$x$')
        s2.set_ylabel(r'$v_y$')

        plt.show()

    def vorticity(self):
        """Get the Vorticity over the domain"""
        pass

