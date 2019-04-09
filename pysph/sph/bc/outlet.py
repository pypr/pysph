"""
Outlet boundary
"""

from pysph.sph.bc.inlet_outlet_manager import IOEvaluate
import numpy as np


class Outlet(object):
    def __init__(self, outlet_pa, source_pa, outletinfo, kernel, dim,
                 active_stages=[1], callback=None):
        """An API to add/delete particle when miving between fluid-outlet

        Parameters
        ----------

        outlet_pa : particle_array
            particle array for outlet
        source_pa : particle_array
            particle_array of the fluid
        outletinfo : OutletInfo instance
            contains information fo outlet
        kernel : Kernel instance
            Kernel to be used for computations
        dim : int
            dimnesion of the problem
        active_stages : list
            stages of integrator at which update should be active
        callback : function
            callback after the update function
        """
        self.outlet_pa = outlet_pa
        self.source_pa = source_pa
        self.dim = dim
        self.kernel = kernel
        self.outletinfo = outletinfo
        self.x = self.y = self.z = 0.0
        self.xn = self.yn = self.zn = 0.0
        self.length = 0.0
        self.callback = callback
        self.active_stages = active_stages
        self.io_eval = None
        self._init = False

    def initialize(self):
        outletinfo = self.outletinfo
        self.x = outletinfo.refpoint[0]
        self.y = outletinfo.refpoint[1]
        self.z = outletinfo.refpoint[2]
        self.xn = outletinfo.normal[0]
        self.yn = outletinfo.normal[1]
        self.zn = outletinfo.normal[2]
        self.length = outletinfo.length

    def _create_io_eval(self):
        if self.io_eval is None:
            from pysph.sph.equation import Group
            from pysph.tools.sph_evaluator import SPHEvaluator
            o_name = self.outlet_pa.name
            f_name = self.source_pa.name
            eqns = []

            eqns.append(Group(equations=[
                IOEvaluate(
                    o_name, [], x=self.x, y=self.y, z=self.z, xn=self.xn,
                    yn=self.yn, zn=self.zn, maxdist=self.length)],
                real=False, update_nnps=False))

            eqns.append(Group(equations=[
                IOEvaluate(
                    f_name, [], x=self.x, y=self.y, z=self.z, xn=self.xn,
                    yn=self.yn, zn=self.zn,)], real=False, update_nnps=False))

            arrays = [self.outlet_pa] + [self.source_pa]
            io_eval = SPHEvaluator(arrays=arrays, equations=eqns, dim=self.dim,
                                   kernel=self.kernel)
            return io_eval
        else:
            return self.io_eval

    def update(self, time, dt, stage):
        if not self._init:
            self.initialize()
            self._init = True
        if stage in self.active_stages:
            outlet_pa = self.outlet_pa
            source_pa = self.source_pa

            self.io_eval = self._create_io_eval()
            self.io_eval.update()
            self.io_eval.evaluate()

            # adding particles to the destination array.
            io_id = source_pa.ioid
            cond = (io_id == 1)
            all_idx = np.where(cond)[0]
            source_pa.extract_particles(all_idx, outlet_pa)
            source_pa.remove_particles(all_idx)

            io_id = outlet_pa.ioid
            cond = (io_id == 2)
            all_idx = np.where(cond)[0]
            outlet_pa.remove_particles(all_idx)

            if self.callback is not None:
                self.callback(source_pa, outlet_pa)
