from pysph.examples.flow_past_cylinder_2d import WindTunnel
from math import sin, cos, pi
import numpy as np
import tempfile
import os

USE_COORDS = True
xc, yc = [], []
cyl_file = os.path.join(tempfile.gettempdir(), 'cylinder.txt')
print(cyl_file)
fp = open(cyl_file, 'w')
for i in range(0, 100):
    _x = cos(2*pi*i/100)
    _y = sin(2*pi*i/100)
    xc.append(_x)
    yc.append(_y)
    fp.write('%.3f %.3f\n'%(_x, _y))
print(xc, yc)
fp.close()

class FPCWithPackedCylinder(WindTunnel):
    def _create_solid(self):
        from pysph.tools.geometry import get_packed_particles
        pass

    def _create_fluid(self):
        from pysph.tools.geometry import get_packed_particles

        folder = self.output_dir
        dx = self.dx

        points = None
        if USE_COORDS:
            points = get_packed_particles(
                folder, dx, x=np.array(xc), y=np.array(yc))
        else:
            points = get_packed_particles(folder, dx, filename=cyl_file)
        print(points)


if __name__ == '__main__':
    app = FPCWithPackedCylinder()
    app.run()
    app.post_process(app.info_filename)