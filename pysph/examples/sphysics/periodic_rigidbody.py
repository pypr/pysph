from pysph.sph.equation import Equation
import numpy as np


class GroupParticles(Equation):

    def __init__(self, dest, sources=None, xmin=0.0, xmax=0.0, ymin=0.0,
                 ymax=0.0, zmin=0.0, zmax=0.0, periodic_in_x=False,
                 periodic_in_y=False, periodic_in_z=False):
        self.periodic_in_x = periodic_in_x
        self.periodic_in_y = periodic_in_y
        self.periodic_in_z = periodic_in_z
        self.xlen = abs(xmax - xmin)
        self.xmin = xmin
        self.xmax = xmax
        self.ylen = abs(ymax - ymin)
        self.ymin = ymin
        self.ymax = ymax
        self.zlen = abs(zmax - zmin)
        self.zmin = zmin
        self.zmax = zmax

        super(GroupParticles, self).__init__(dest, sources)

    def loop(self, d_idx, d_cm, d_body_id, d_x, d_y, d_z):
        b = declare('int')
        b = d_body_id[d_idx]
        if self.periodic_in_x:
            if (abs(d_x[d_idx] - d_cm[3 * b]) > (self.xlen / 2.0)):
                if (d_cm[3 * b] > self.xmin + self.xlen / 2.0):
                    d_x[d_idx] += self.xlen
                else:
                    d_x[d_idx] -= self.xlen
        if self.periodic_in_y:
            if (abs(d_y[d_idx] - d_cm[3 * b + 1]) > (self.ylen / 2.0)):
                if (d_cm[3 * b + 1] > self.ymin + self.ylen / 2.0):
                    d_y[d_idx] += self.ylen
                else:
                    d_y[d_idx] -= self.ylen
        if self.periodic_in_z:
            if (abs(d_z[d_idx] - d_cm[3 * b + 2]) > (self.zlen / 2.0)):
                if (d_cm[3 * b + 2] > self.zmin + self.zlen / 2.0):
                    d_z[d_idx] += self.zlen
                else:
                    d_z[d_idx] -= self.zlen
