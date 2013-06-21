"""Analysis for the taylor green vortex case"""

import pysph.solver.utils as utils
from pysph.tools import pprocess
import numpy as np

def velocity(b, t, x, y):
    pi = np.pi; sin = np.sin; cos = np.cos
    factor = U * np.exp(b*t)

    u = cos( 2 * pi * x ) * sin( 2 * pi * y)
    v = sin( 2 * pi * x ) * cos( 2 * pi * y)

    return factor * u, factor * v

class TaylorGreenResults(pprocess.Results):
    def __init__(self, dirname=None, fname=None, endswith=".npz",
                 U=1, Re=100):
        super(TaylorGreenResults, self).__init__(dirname, fname, endswith)
        
        self.U = U
        self.Re = Re
        self.decay_rate_constant = -8*np.pi**2/Re
        
    def comute_stats(self, array="fluid"):
        files = self.files
        nfiles = self.nfiles
        
        times = np.zeros( nfiles )
        decay = np.zeros( nfiles )
        linf = np.zeros( nfiles )
        for i in range(nfiles):
            data = utils.load(files[i])
            
            pa = data['arrays'][array]
            vmag = np.sqrt( pa.vmag )
            
            t = data['solver_data']['t']
            times[i] = t
            decay[i] = vmag.max()

            # compute the error norm
            theoretical_max = self.U * np.exp(self.decay_rate_constant * t)
            linf[i] = abs( (vmag.max() - theoretical_max)/theoretical_max )

        self.times = times
        self.decay = decay
        self.linf = linf

        

    

