"""
Inlet boundary
"""
import numpy as np
from pysph.sph.bc.inlet_outlet_manager import InletBase


class Inlet(InletBase):
    def update(self, time, dt, stage):
        dest_pa = self.dest_pa
        inlet_pa = self.inlet_pa
        ghost_pa = self.ghost_pa

        dest_pa.uref[0] = 0.5 * (inlet_pa.uref[0] + dest_pa.uref[0])

        if not self._init:
            self.initialize()
            self._init = True
        if stage in self.active_stages:

            self.io_eval = self._create_io_eval()
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

            if ghost_pa:
                ghost_pa.x[all_idx] -= self.length * self.xn
                ghost_pa.y[all_idx] -= self.length * self.yn
                ghost_pa.z[all_idx] -= self.length * self.zn

            if self.callback is not None:
                self.callback(dest_pa, inlet_pa)
