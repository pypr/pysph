"""General post-processing utility for solution data"""

VV=True
try:
    import visvis
except ImportError:
    VV=False

import numpy as np
import pysph.solver.utils as utils

class Results(object):
    def __init__(self, dirname=None, fname=None, endswith=".npz"):
        self.dirname = dirname
        self.fname = fname
        self.endswith = endswith

        if ( (dirname is not None) and (fname is not None) ):
            self.load()

    def set_dirname(self, dirname):
        self.dirname=dirname

    def set_fname(self, fname):
        self.fname = fname

    def load(self):
        self.files = files = utils.get_files(
            self.dirname, self.fname, self.endswith)

        self.nfiles = len(files)

    def get_ke_history(self, array_name):
        nfiles = self.nfiles
        ke = np.zeros(nfiles, dtype=np.float64)
        t = np.zeros(nfiles, dtype=np.float64)

        for i in range(nfiles):
            data = utils.load(self.files[i])

            # save the time array
            t[i] = data['solver_data']['t']

            array = data['arrays'][array_name]
            
            m, u, v, w = array.get('m', 'u', 'v', 'w')
            ke[i] = 0.5 * np.sum( m * (u**2 + v**2) )
            
        return t, ke
