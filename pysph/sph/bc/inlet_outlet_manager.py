from pysph.sph.equation import Equation


class InletInfo(object):
    def __init__(self, pa_name, normal, refpoint, freesurface=False,
                 basenormal=None, equations=None):
        """Create object with information of inlets, all the others parameters
           which are not passed here get evaluated by `InletOutletManager`once
           the inlet is created.

        Parameters
        ----------

        pa_name : str
            Name of the inlet
        normal : list
            Components of normal (float)
        refpoint : list
            Point at the fluid-inlet interface (float)
        freesurface : bool
            if True, the inlet will need as basenormal
        basenormal : list
            List of normal of the base wall (float)
        equations : list
            List of equations (optional)
        """
        self.pa_name = pa_name
        self.normal = normal
        self.refpoint = refpoint
        self.length = 0.0
        self.freesurface = freesurface
        self.dx = 0.1
        self.bound = []
        self.basenormal = [0.0, 0.0, 0.0] if basenormal is None else basenormal
        self.equations = [] if equations is None else equations


class OutletInfo(InletInfo):
    """Create object with information of outlet

    The name is kept different for distinction only.
    """
    pass


class InletOutletManager(object):
    def __init__(self, fluid_arrays, inletinfo, outletinfo, extraeqns=None):
        """Create the object to manage inlet outlet boundary conditions.
           Most of the variables are evaluated after the scheme and particles
           are created.

        Parameters
        -----------

        fluid_arrays : list
            List of fluid particles array names (str)
        inletinfo : list
            List of inlets (InletInfo)
        outletinfo : list
            List of outlet (OutletInfo)
        extraeqns : dict
            Dict of custom equations
        """
        self.fluids = fluid_arrays
        self.dim = None
        self.kernel = None
        self.inlets = [x.pa_name for x in inletinfo]
        self.outlets = [x.pa_name for x in outletinfo]
        self.inlet_pairs = {}
        self.outlet_pairs = {}
        self.inletinfo = inletinfo
        self.outletinfo = outletinfo
        self.extraeqns = {} if extraeqns is None else extraeqns
        self.active_stages = []

    def update_dx(self, dx):
        """Update the discretisation length

        Parameters
        -----------

        dx : float
            The discretisation length
        """
        all_info = self.inletinfo + self.outletinfo
        for info in all_info:
            info.dx = dx

    def _update_inlet_outlet_info(self, pa):
        """Updates refpoint, length and bound

        Parameters
        -----------

        pa : Particle_array
            Particle array of inlet/outlet
        """
        all_info = self.inletinfo + self.outletinfo
        for info in all_info:
            dx = info.dx
            if (info.pa_name == pa.name):
                x = pa.x
                y = pa.y
                z = pa.z
                xmax, xmin = max(x)+dx/2, min(x)-dx/2
                ymax, ymin = max(y)+dx/2, min(y)-dx/2
                zmax, zmin = max(z)+dx/2, min(z)-dx/2
                xdist = xmax - xmin
                ydist = ymax - ymin
                zdist = zmax - zmin
                xn, yn, zn = info.normal[0], info.normal[1], info.normal[2]
                info.length = abs(xdist*xn+ydist*yn+zdist*zn)
                info.bound = [[xmin, ymin, zmin], [xmax, ymax, zmax]]

    def add_io_properties(self, pa, scheme=None):
        """Add properties to be used in inlet/outlet equations

        Parameters
        ----------

        pa : particle_array
            Particle array of inlet/outlet
        scheme : pysph.sph.scheme
            The insance of scheme class
        """
        pass

    def get_io_names(self):
        """return all the names of inlets and outlets"""
        return self.inlets + self.outlets

    def get_stepper(self, scheme, integrator, **kw):
        """Returns the steppers for inlet/outlet

        Parameters
        ----------

        scheme : pysph.sph.scheme
            The instance of the scheme class
        intergrator : pysph.sph.integrator
            The parent class of the integrator
        **kw : extra arguments
            Extra arguments depending upon the scheme used
        """
        raise NotImplementedError()

    def setup_iom(self, dim, kernel):
        """Essential data passed

        Parameters
        ----------

        dim : int
            dimension of the problem
        kernel : pysph.base.kernel
            the kernel instance
        """
        self.dim = dim
        self.kernel = kernel

    def get_equations(self, scheme, **kw):
        """Returns the equations for inlet/outlet

        Parameters
        ----------

        scheme : pysph.sph.scheme
            The instance of the scheme class
        **kw : extra arguments
            Extra arguments depending upon the scheme used
        """
        raise NotImplementedError()

    def get_inlet_outlet(self, particle_array):
        """ Returns list of `Inlet` and `Outlet` instances which
            performs the change in inlet particles to outlet
            particles.

        Parameters
        -----------

        particle_array : list
            List of all particle_arrays
        """
        result = []
        from pysph.sph.bc.inlet import Inlet
        from pysph.sph.bc.outlet import Outlet
        for inlet in self.inletinfo:
            i_name = inlet.pa_name
            self._update_inlet_outlet_info(particle_array[i_name])
            for fluid in self.fluids:
                in1 = Inlet(
                        particle_array[i_name],
                        particle_array[fluid],
                        inlet,
                        self.kernel,
                        self.dim,
                        self.active_stages
                        )
            result.append(in1)

        for outlet in self.outletinfo:
            o_name = outlet.pa_name
            self._update_inlet_outlet_info(particle_array[o_name])
            for fluid in self.fluids:
                out1 = Outlet(
                        particle_array[o_name],
                        particle_array[fluid],
                        outlet,
                        self.kernel,
                        self.dim,
                        self.active_stages
                        )
            result.append(out1)

        return result


class IOEvaluate(Equation):
    def __init__(self, dest, sources, x, y, z, xn, yn, zn,
                 maxdist=1000.0):
        """Compute ioid for the particles
           0 : particle is in fluid
           1 : particle is inside the inlet/outlet
           2 : particle is out of inlet/outlet

        parameters:
        ----------

        dest : str
            destination particle array name
        sources : list
            List of source particle arrays
        x : float
            x coordinate of interface point
        y : float
            y coordinate of interface point
        z : float
            z coordinate of interface point
        xn : float
            x component of interface outward normal
        yn : float
            y component of interface outward normal
        zn : float
            z component of interface outward normal
        maxdist : float
            Maximum length of inlet/outlet

        """
        self.x = x
        self.y = y
        self.z = z
        self.xn = xn
        self.yn = yn
        self.zn = zn
        self.maxdist = maxdist

        super(IOEvaluate, self).__init__(dest, sources)

    def initialize(self, d_ioid, d_idx):
        d_ioid[d_idx] = 1

    def loop(self, d_idx, d_x, d_y, d_z, d_ioid, d_disp):
        delx = d_x[d_idx] - self.x
        dely = d_y[d_idx] - self.y
        delz = d_z[d_idx] - self.z

        d_disp[d_idx] = delx * self.xn + dely * self.yn + delz * self.zn

        if ((d_disp[d_idx] > -0.000001) and
           (d_disp[d_idx]-self.maxdist < 0.000001)):
            d_ioid[d_idx] = 1
        elif (d_disp[d_idx] - self.maxdist > 0.000001):
            d_ioid[d_idx] = 2
        else:
            d_ioid[d_idx] = 0
