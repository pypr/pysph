"""
Outlet boundary
"""

from pysph.sph.bc.inlet_outlet_manager import OutletBase
import numpy as np


class Outlet(OutletBase):
    def _get_ghost_xyz(self, x, y, z):
        xij = x - self.x
        yij = y - self.y
        zij = z - self.z

        disp = xij * self.xn + yij * self.yn + zij * self.zn
        x = x - 2 * disp * self.xn
        y = y - 2 * disp * self.yn
        z = z - 2 * disp * self.zn

        return x, y, z

    def update(self, time, dt, stage):
        if not self._init:
            self.initialize()
            self._init = True
        if stage in self.active_stages:
            props_to_copy = self.props_to_copy
            outlet_pa = self.outlet_pa
            source_pa = self.source_pa
            ghost_pa = self.ghost_pa

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

            if ghost_pa:
                if len(all_idx) > 0:
                    x, y, z = self._get_ghost_xyz(
                        pa_add.x, pa_add.y, pa_add.z)
                    pa_add.x = x
                    pa_add.y = y
                    pa_add.z = z
                    pa_add.u = -1. * pa_add.u
                    ghost_pa.add_particles(**pa_add.get_property_arrays())
            source_pa.remove_particles(all_idx)

            io_id = outlet_pa.ioid
            cond = (io_id == 2)
            all_idx = np.where(cond)[0]
            outlet_pa.remove_particles(all_idx)
            if ghost_pa:
                ghost_pa.remove_particles(all_idx)

            if self.callback is not None:
                self.callback(source_pa, outlet_pa)
