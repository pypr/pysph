from pysph.examples.flow_past_cylinder_2d import WindTunnel
from pysph.base.utils import get_particle_array
from math import sin, cos, pi
import numpy as np
import tempfile
import os

# fluid mechanical/numerical parameters
rho = 1000
umax = 1.0
c0 = 10 * umax
p0 = rho * c0 * c0
use_coords = True

# creating the files and coordinates of the cylinder surface for demonstration
xc, yc = [], []
cyl_file = os.path.join(tempfile.gettempdir(), 'cylinder.txt')
print(cyl_file)
fp = open(cyl_file, 'w')
for i in range(0, 100):
    _x = cos(2 * pi * i / 100) + 5.0
    _y = sin(2 * pi * i / 100)
    xc.append(_x)
    yc.append(_y)
    fp.write('%.3f %.3f\n' % (_x, _y))
fp.close()


class FPCWithPackedCylinder(WindTunnel):
    def _get_packed_points(self):
        '''
        returns
        xs, ys, zs, xf, yf, zf
        '''
        from pysph.tools.geometry import (
            get_packed_2d_particles_from_surface_coordinates,
            get_packed_2d_particles_from_surface_file)
        folder = self.output_dir
        dx = self.dx
        if use_coords:
            return get_packed_2d_particles_from_surface_coordinates(
                self.add_user_options, folder, dx, x=np.array(xc),
                y=np.array(yc), shift=True)
        else:
            return get_packed_2d_particles_from_surface_file(
                self.add_user_options, folder, dx, filename=cyl_file,
                shift=True)

    def _create_solid(self):
        xs, ys, zs, xf, yf, zf = self._get_packed_points()
        dx = self.dx
        h0 = self.h
        volume = dx*dx
        solid = get_particle_array(
            name='solid', x=xs-dx/2, y=ys,
            m=volume*rho, rho=rho, h=h0, V=1.0/volume)
        return solid

    def _create_fluid(self):
        from pysph.tools.geometry import create_fluid_around_packing
        xs, ys, zs, xf, yf, zf = self._get_packed_points()
        dx = self.dx
        h0 = self.h
        volume = dx*dx
        L = self.Lt
        B = self.Wt * 2.0

        fluid = create_fluid_around_packing(
            dx, xf-dx/2, yf, L, B, m=volume*rho, rho=rho, h=h0, V=1.0/volume,
            u=umax, p=0.0, uhat=umax)

        return fluid

    def create_particles(self):
        dx = self.dx
        fluid = self._create_fluid()
        solid = self._create_solid()
        outlet = self._create_outlet()
        inlet = self._create_inlet()
        wall = self._create_wall()

        ghost_inlet = self.iom.create_ghost(inlet, inlet=True)
        ghost_outlet = self.iom.create_ghost(outlet, inlet=False)

        particles = [fluid, inlet, outlet, solid, wall]
        if ghost_inlet:
            particles.append(ghost_inlet)
        if ghost_outlet:
            particles.append(ghost_outlet)

        self.scheme.setup_properties(particles)
        self._set_wall_normal(wall)

        if self.io_method == 'hybrid':
            fluid.uag[:] = 1.0
            fluid.uta[:] = 1.0
            outlet.uta[:] = 1.0

        return particles


if __name__ == '__main__':
    app = FPCWithPackedCylinder()
    app.run()
    app.post_process(app.info_filename)
