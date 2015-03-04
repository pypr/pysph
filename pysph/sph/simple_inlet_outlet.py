"""Simple Inlet and Outlet along the x-axis.

Copyright (c) 2015, Prabhu Ramachandran
License: BSD
"""

import numpy as np


class SimpleInlet(object):
    """This inlet has particles stacked along the positive x-axis.  These
    particles should move along the x direction and as they flow out of the
    domain they are copied into the destination particle array at each
    timestep.

    Inlet particles should be given for the particles closest to the domain.
    These are copied when the inlet is created.  The particles that cross
    the inlet domain's xmax are copied over to the destination particle array
    and moved back to the front of the inlet.

    """
    def __init__(self, inlet_pa, dest_pa, dx, n = 5):
        """Constructor.

        Arguments
        -----------

         inlet_pa - Particle array for the inlet particles.

         dest_pa - Particle array for the destination into which inlet flows.

         dx - Spacing of particles in the inlet domain.

         n - Total number of copies of the initial particles.

        """
        self.inlet_pa = inlet_pa
        self.dest_pa = dest_pa
        self.dx = dx
        self.n = n
        self.xmax = max(self.inlet_pa.x)
        self._create_inlet_particles()

    def _create_inlet_particles(self):
        props =  self.inlet_pa.get_property_arrays()
        inlet_props = {}
        for prop, array in props.iteritems():
            new_array = np.array([], dtype=array.dtype)
            for i in range(1, self.n):
                if prop == 'x':
                    new_array = np.append(new_array, array - i*self.dx)
                else:
                    new_array = np.append(new_array, array)
            inlet_props[prop] = new_array
        self.inlet_pa.add_particles(**inlet_props)
        self.xmin = min(self.inlet_pa.x)

    def update(self, solver=None):
        """This is called by the solver after each timestep and is passed
        the solver instance.
        """
        pa_add = {}
        inlet_pa = self.inlet_pa
        idx = np.where(inlet_pa.x >= self.xmax)[0]

        # adding particles to the destination array.
        props = inlet_pa.get_property_arrays()
        for prop, array in props.iteritems():
            pa_add[prop] = np.array(array[idx])
        self.dest_pa.add_particles(**pa_add)

        # moving the moved particles back to the array beginning.
        inlet_pa.x[idx] +=  self.xmin - self.xmax


class SimpleOutlet(object):

    """This outlet simply moves the particles that comes into it from the
    source and removes any that leave the box.

    """
    def __init__(self, outlet_pa, source_pa, xmin=-1.0, xmax=1.0,
                 ymin=-1.0, ymax=1.0, zmin=-1.0, zmax=1.0):
        """Constructor.

        Arguments
        -----------

         outlet_pa - Particle array for the outlet particles.

         source_pa - Particle array from which the particles flow in.

         xmin, xmax, ymin, ymax, zmin, zmax - Domain of the outlet.

        """
        self.outlet_pa = outlet_pa
        self.source_pa = source_pa
        self.xmin, self.xmax = xmin, xmax
        self.ymin, self.ymax = ymin, ymax
        self.zmin, self.zmax = zmin, zmax

    def update(self, solver=None):
        """This is called by the solver after each timestep and is passed
        the solver instance.
        """
        xmin, xmax, ymin, ymax = self.xmin, self.xmax, self.ymin, self.ymax
        zmin, zmax = self.zmin, self.zmax
        outlet_pa = self.outlet_pa
        source_pa = self.source_pa
        x, y, z = source_pa.x, source_pa.y, source_pa.z
        idx = np.where((x <= xmax) & (x >= xmin) & (y <= ymax) & (y >= ymin) &
                       (z <= zmax) & (z >= zmin))[0]

        # adding particles to the destination array.
        pa_add = {}
        props = source_pa.get_property_arrays()
        for prop, array in props.iteritems():
            pa_add[prop] = np.array(array[idx])
        outlet_pa.add_particles(**pa_add)

        # removing the particles that moved into the outlet
        source_pa.remove_particles(idx)

        x, y, z = outlet_pa.x, outlet_pa.y, outlet_pa.z
        idx = np.where((x > xmax) | (x < xmin) | (y > ymax) | (y < ymin) |
                       (z > zmax) | (z < zmin))[0]
        outlet_pa.remove_particles(idx)
