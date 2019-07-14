"""
Outlet boundary
"""

from pysph.sph.bc.inlet_outlet_manager import OutletBase
import numpy as np


class Outlet(OutletBase):
    def update(self, time, dt, stage):
        if not self._init:
            self.initialize()
            self._init = True
        if stage in self.active_stages:
            props_to_copy = [
                'x0', 'y0', 'z0', 'uhat', 'vhat', 'what', 'x', 'y', 'z',
                'u', 'v', 'w', 'm', 'h', 'rho', 'p', 'uta', 'pta',
                'ioid', 'u0', 'v0', 'w0', 'p0']
            outlet_pa = self.outlet_pa
            source_pa = self.source_pa

            self.io_eval = self._create_io_eval()
            self.io_eval.update()
            self.io_eval.evaluate()

            # adding particles to the destination array.
            io_id = source_pa.ioid
            cond = (io_id == 1)
            all_idx = np.where(cond)[0]
            pa_add = source_pa.extract_particles(
                all_idx, props=props_to_copy)
            outlet_pa.add_particles(**pa_add.get_property_arrays())
            source_pa.remove_particles(all_idx)

            io_id = outlet_pa.ioid
            cond = (io_id == 2)
            all_idx = np.where(cond)[0]
            outlet_pa.remove_particles(all_idx)

            if self.callback is not None:
                self.callback(source_pa, outlet_pa)
