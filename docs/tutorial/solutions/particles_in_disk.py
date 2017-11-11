import numpy as np
from pysph.base.utils import get_particle_array_wcsph
x, y = np.mgrid[-1:1:50j, -1:1:50j]
mask = x*x + y*y < 1.0
pa = get_particle_array_wcsph(name='fluid', x=x[mask], y=y[mask])
plt.scatter(pa.x, pa.y, marker='.')