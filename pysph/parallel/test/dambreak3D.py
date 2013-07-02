"""3D dam break with an obstacle simulation using PySPH.

For testing, we use the input geometry and discretization as the
SPHYSICS Case 5
(https://wiki.manchester.ac.uk/sphysics/index.php/SPHYSICS_Home_Page)

"""
import numpy as np

from pysph.base.utils import get_particle_array_wcsph as gpa
from pysph.base.kernels import CubicSpline
from pysph.sph.wc.basic import TaitEOS, ContinuityEquation, MomentumEquation,\
     XSPHCorrection

from pysph.solver.application import Application
from pysph.solver.solver import Solver

dim = 3

dt = 3e-5
tf = 1.0

hdx = 1.2
ro = 1000.0
co = 10.0 * np.sqrt(9.81*0.3)
gamma = 7.0
alpha = 0.1
beta = 0.0
B = co*co*ro/gamma

def create_particles(empty=False, **kwargs):
    ns = 13987
    dx = dy = dz = 0.0225
    #h0 = 0.9 * np.sqrt(3 * dx**2)
    h0 = hdx * dx

    if empty:
        fluid = gpa(name='fluid')
        solid = gpa(name='boundary')
    else:
        ipart = np.loadtxt('IPART.txt.gz')
        x = ipart[:, 0]; y = ipart[:, 1]; z = ipart[:, 2]
        u = ipart[:, 3]; v = ipart[:, 4]; w = ipart[:, 5]
        rho = ipart[:, 6]; p = ipart[:, 7]; m = ipart[:, 8]

        # the fluid particles
        xf = x[ns:]; yf = y[ns:]; zf = z[ns:]
        rhof = rho[ns:]; pf = p[ns:]; mf = m[ns:]

        hf = np.ones_like(xf) * h0

        fluid = gpa(name='fluid', x=xf, y=yf, z=zf,
                    rho=rhof, p=pf, m=mf, h=hf)

        # the solid particles
        xs = x[:ns]; ys = y[:ns]; zs = z[:ns]
        rhos = rho[:ns]; ps = p[:ns]; ms = m[:ns]

        hs = np.ones_like(xs) * h0

        solid = gpa(name='boundary', x=xs, y=ys, z=zs,
                    rho=rhos, p=ps, m=ms, h=hs)

    particles = [fluid, solid]

    # add requisite variables
    for pa in particles:
        pa.add_property( {'name': 'arho'} )
        pa.add_property( {'name': 'au'} )
        pa.add_property( {'name': 'av'} )
        pa.add_property( {'name': 'aw'} )

        pa.add_property( {'name': 'ax'} )
        pa.add_property( {'name': 'ay'} )
        pa.add_property( {'name': 'az'} )

        pa.add_property( {'name': 'rho0'} )

        pa.add_property( {'name': 'u0'} )
        pa.add_property( {'name': 'v0'} )
        pa.add_property( {'name': 'w0'} )

        pa.add_property( {'name': 'x0'} )
        pa.add_property( {'name': 'y0'} )
        pa.add_property( {'name': 'z0'} )

    return particles

# Create the application.
app = Application()

# Create the kernel
kernel = CubicSpline(dim=3)

# Create a solver.
solver = Solver(kernel=kernel, dim=dim)

# Setup default parameters.
solver.set_time_step(dt)
solver.set_final_time(tf)

# create the equations
equations = [

    # Equation of state
    TaitEOS(dest='fluid', sources=None, rho0=ro, c0=co, gamma=gamma),
    TaitEOS(dest='boundary', sources=None, rho0=ro, c0=co, gamma=gamma),

    # Continuity equation
    ContinuityEquation(dest='fluid', sources=['fluid', 'boundary']),
    ContinuityEquation(dest='boundary', sources=['fluid']),

    # Momentum equation
    MomentumEquation(dest='fluid', sources=['fluid', 'boundary'],
                     alpha=alpha, beta=beta, gz=-9.81),

    # Position step with XSPH
    XSPHCorrection(dest='fluid', sources=['fluid'], eps=0.5)

    ]

# Setup the application and solver.  This also generates the particles.
app.setup(solver=solver, equations=equations,
          particle_factory=create_particles)

app.run()
