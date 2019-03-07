"""
Inlet boundary
"""
from pysph.sph.bc.inlet_outlet_manager import IOEvaluate
import numpy as np


class Inlet(object):
    def __init__(self, inlet_pa, dest_pa, inletinfo, kernel, dim,
                 active_stages=[1], callback=None):
        """An API to add/delete particle when moving between inlet-fluid

        Parameters
        ----------

        inlet_pa : particle_array
            particle array for inlet
        dest_pa : particle_array
            particle_array of the fluid
        inletinfo : InletInfo instance
            contains information fo inlet
        kernel : Kernel instance
            Kernel to be used for computations
        dim : int
            dimnesion of the problem
        active_stages : list
            stages of integrator at which update should be active
        callback : function
            callback after the update function
        """
        self.inlet_pa = inlet_pa
        self.dest_pa = dest_pa
        self.callback = callback
        self.dim = dim
        self.kernel = kernel
        self.inletinfo = inletinfo
        self.x = self.y = self.z = 0.0
        self.xn = self.yn = self.zn = 0.0
        self.length = 0.0
        self.active_stages = active_stages
        self.io_eval = None
        self._init = False

    def initialize(self):
        inletinfo = self.inletinfo
        self.x = inletinfo.refpoint[0]
        self.y = inletinfo.refpoint[1]
        self.z = inletinfo.refpoint[2]
        self.xn = inletinfo.normal[0]
        self.yn = inletinfo.normal[1]
        self.zn = inletinfo.normal[2]
        self.length = inletinfo.length

    def _io_eval(self):
        if self.io_eval is None:
            from pysph.sph.equation import Group
            from pysph.tools.sph_evaluator import SPHEvaluator
            i_name = self.inlet_pa.name
            f_name = self.dest_pa.name
            eqns = []
            eqns.append(Group(equations=[
                IOEvaluate(
                    i_name, [], x=self.x, y=self.y, z=self.z,
                    xn=self.xn, yn=self.yn, zn=self.zn,
                    maxdist=self.length)],
                real=False, update_nnps=False))

            eqns.append(Group(equations=[
                IOEvaluate(
                    f_name, [], x=self.x, y=self.y, z=self.z, xn=self.xn,
                    yn=self.yn, zn=self.zn)],
                real=False, update_nnps=False))

            arrays = [self.inlet_pa] + [self.dest_pa]
            io_eval = SPHEvaluator(
                arrays=arrays, equations=eqns, dim=self.dim,
                kernel=self.kernel)
            return io_eval
        else:
            return self.io_eval

    def update(self, time, dt, stage):
        if not self._init:
            self.initialize()
            self._init = True
        if stage in self.active_stages:
            dest_pa = self.dest_pa
            inlet_pa = self.inlet_pa

            self.io_eval = self._io_eval()
            self.io_eval.update()
            self.io_eval.evaluate()

            io_id = inlet_pa.ioid
            cond = (io_id == 0)
            all_idx = np.where(cond)[0]
            inlet_pa.extract_particles(all_idx, dest_pa)

            # moving the moved particles back to the array beginning.
            inlet_pa.x[all_idx] += self.length * self.xn
            inlet_pa.y[all_idx] += self.length * self.yn
            inlet_pa.z[all_idx] += self.length * self.zn

            if self.callback is not None:
                self.callback(dest_pa, inlet_pa)
