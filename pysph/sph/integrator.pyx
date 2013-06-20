"""Implementation for the integrator"""
cimport cython
from libc.math cimport sqrt

cdef class Integrator:

    def __init__(self, object evaluator, list particles):
        """Constructor for the Integrator"""
        self.evaluator = evaluator
        self.particles = particles
        self.nnps = evaluator.nnps

        # default cfl number
        self.cfl = 0.5

    def set_parallel_manager(self, object pm):
        self.pm = pm

    def set_solver(self, object solver):
        self.solver = solver

    def set_cfl_number(self, double cfl):
        self.cfl = cfl

    cpdef integrate(self, double t, double dt, int count):
        raise RuntimeError("Integrator::integrate called!")

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef _set_initial_values(self):
        cdef int i, npart
        cdef object pa_wrapper
        
        # Quantities at the start of a time step
        cdef DoubleArray x0, y0, z0, u0, v0, w0, rho0

        # Particle properties
        cdef DoubleArray x, y, z, u, v, w, rho

        for pa in self.particles:
            pa_wrapper = getattr(self.evaluator.calc, pa.name)

            rho = pa_wrapper.rho; rho0 = pa_wrapper.rho0

            x = pa_wrapper.x  ; y = pa_wrapper.y  ; z = pa_wrapper.z
            x0 = pa_wrapper.x0; y0 = pa_wrapper.y0; z0 = pa_wrapper.z0

            u = pa_wrapper.u  ; v = pa_wrapper.v  ; w = pa_wrapper.w
            u0 = pa_wrapper.u0; v0 = pa_wrapper.v0; w0 = pa_wrapper.w0

            npart = pa.get_number_of_particles()

            for i in range(npart):
                x0.data[i] = x.data[i]
                y0.data[i] = y.data[i]
                z0.data[i] = z.data[i]

                u0.data[i] = u.data[i]
                v0.data[i] = v.data[i]
                w0.data[i] = w.data[i]
                
                rho0.data[i] = rho.data[i]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef _reset_accelerations(self):
        cdef int i, npart
        cdef object pa_wrapper
        
        # acc properties
        cdef DoubleArray ax, ay, az, au, av, aw, arho

        for pa in self.particles:
            pa_wrapper = getattr(self.evaluator.calc, pa.name)

            arho = pa_wrapper.arho

            ax = pa_wrapper.ax  ; ay = pa_wrapper.ay  ; az = pa_wrapper.az
            au = pa_wrapper.au  ; av = pa_wrapper.av  ; aw = pa_wrapper.aw

            npart = pa.get_number_of_particles()

            for i in range(npart):
                ax.data[i] = 0.0
                ay.data[i] = 0.0
                az.data[i] = 0.0

                au.data[i] = 0.0
                av.data[i] = 0.0
                aw.data[i] = 0.0
                
                arho.data[i] = 0.0

    def compute_time_step(self, double dt):
        raise RuntimeError("Integrator::compute_time_step called!")

cdef class WCSPHRK2Integrator(Integrator):
    cpdef integrate(self, double t, double dt, int count):
        """Main step routine"""
        cdef object pa_wrapper
        cdef object evaluator = self.evaluator
        cdef object pm = self.pm
        cdef NNPS nnps = self.nnps
        cdef str name

        # Particle properties
        cdef DoubleArray x, y, z, u, v, w, rho

        # Quantities at the start of a time step
        cdef DoubleArray x0, y0, z0, u0, v0, w0, rho0

        # accelerations for the variables
        cdef DoubleArray ax, ay, az, au, av, aw, arho

        # number of particles
        cdef size_t npart
        cdef size_t i

        # save the values at the beginning of a time step
        self._set_initial_values()

        #############################################################
        # Predictor
        #############################################################
        cdef double dtb2 = 0.5*dt

        for pa in self.particles:
            name = pa.name
            pa_wrapper = getattr(self.evaluator.calc, name)

            rho = pa_wrapper.rho; rho0 = pa_wrapper.rho0; arho=pa_wrapper.arho

            x = pa_wrapper.x  ; y = pa_wrapper.y  ; z = pa_wrapper.z
            x0 = pa_wrapper.x0; y0 = pa_wrapper.y0; z0 = pa_wrapper.z0
            ax = pa_wrapper.ax; ay = pa_wrapper.ay; az = pa_wrapper.az

            u = pa_wrapper.u  ; v = pa_wrapper.v  ; w = pa_wrapper.w
            u0 = pa_wrapper.u0; v0 = pa_wrapper.v0; w0 = pa_wrapper.w0
            au = pa_wrapper.au; av = pa_wrapper.av; aw = pa_wrapper.aw

            npart = pa.get_number_of_particles()

            for i in range(npart):
                # Update velocities
                u.data[i] = u0.data[i] + dtb2*au.data[i]
                v.data[i] = v0.data[i] + dtb2*av.data[i]
                w.data[i] = w0.data[i] + dtb2*aw.data[i]

                # Positions are updated using XSPH
                x.data[i] = x0.data[i] + dtb2 * ax.data[i]
                y.data[i] = y0.data[i] + dtb2 * ay.data[i]
                z.data[i] = z0.data[i] + dtb2 * az.data[i]

                # Update densities and smoothing lengths from the accelerations
                rho.data[i] = rho0.data[i] + dtb2 * arho.data[i]

        # Update NNPS since particles have moved
        if pm: pm.update()
        nnps.update()

        # compute accelerations
        self._reset_accelerations()
        evaluator.compute(t, dt)

        #############################################################
        # Corrector
        #############################################################
        for pa in self.particles:
            name = pa.name
            pa_wrapper = getattr(self.evaluator.calc, name)

            rho = pa_wrapper.rho; rho0 = pa_wrapper.rho0; arho=pa_wrapper.arho

            x = pa_wrapper.x  ; y = pa_wrapper.y  ; z = pa_wrapper.z
            x0 = pa_wrapper.x0; y0 = pa_wrapper.y0; z0 = pa_wrapper.z0
            ax = pa_wrapper.ax; ay = pa_wrapper.ay; az = pa_wrapper.az

            u = pa_wrapper.u  ; v = pa_wrapper.v  ; w = pa_wrapper.w
            u0 = pa_wrapper.u0; v0 = pa_wrapper.v0; w0 = pa_wrapper.w0
            au = pa_wrapper.au; av = pa_wrapper.av; aw = pa_wrapper.aw

            npart = pa.get_number_of_particles()
            
            for i in range(npart):

                # Update velocities
                u.data[i] = u0.data[i] + dt*au.data[i]
                v.data[i] = v0.data[i] + dt*av.data[i]
                w.data[i] = w0.data[i] + dt*aw.data[i]

                # Positions are updated using the velocities and XSPH
                x.data[i] = x0.data[i] + dt * ax.data[i]
                y.data[i] = y0.data[i] + dt * ay.data[i]
                z.data[i] = z0.data[i] + dt * az.data[i]

                # Update densities and smoothing lengths from the accelerations
                rho.data[i] = rho0.data[i] + dt * arho.data[i]

    def compute_time_step(self, double dt, double c0):
        """Compute a stable time step"""
        cdef double cfl = self.cfl
        cdef double dt_cfl = self.evaluator.dt_cfl
        cdef DoubleArray h
        cdef double hmin = 1.0

        # if the dt_cfl is not defined, return default dt
        if dt_cfl < 0:
            return dt

        # iterate over particles and find the stable time step
        for pa in self.particles:
            pa_wrapper = getattr(self.evaluator.calc, pa.name)

            h = pa_wrapper.h
            h.update_min_max()

            if h.minimum < hmin:
                hmin = h.minimum

        # return the courant limited time step
        return cfl * hmin/dt_cfl

cdef class EulerIntegrator(Integrator):
    cpdef integrate(self, double t, double dt, int count):
        """Main step routine"""
        cdef object pa_wrapper
        cdef object evaluator = self.evaluator
        cdef object pm = self.pm
        cdef NNPS nnps = self.nnps
        cdef str name

        # Particle properties
        cdef DoubleArray x, y, z, u, v, w, rho

        # Quantities at the start of a time step
        cdef DoubleArray x0, y0, z0, u0, v0, w0, rho0

        # accelerations for the variables
        cdef DoubleArray ax, ay, az, au, av, aw, arho

        # number of particles
        cdef size_t npart
        cdef size_t i

        # Update NNPS since particles have moved
        if pm: pm.update()
        nnps.update()
                
        # compute accelerations
        self._reset_accelerations()
        evaluator.compute(t, dt)

        #############################################################
        # integrate
        #############################################################
        for pa in self.particles:
            name = pa.name
            pa_wrapper = getattr(self.evaluator.calc, name)

            rho = pa_wrapper.rho; rho0 = pa_wrapper.rho0; arho=pa_wrapper.arho

            x = pa_wrapper.x  ; y = pa_wrapper.y  ; z = pa_wrapper.z
            ax = pa_wrapper.ax; ay = pa_wrapper.ay; az = pa_wrapper.az

            u = pa_wrapper.u  ; v = pa_wrapper.v  ; w = pa_wrapper.w
            au = pa_wrapper.au; av = pa_wrapper.av; aw = pa_wrapper.aw

            npart = pa.get_number_of_particles()
            
            for i in range(npart):

                # Update velocities
                u.data[i] += dt*au.data[i]
                v.data[i] += dt*av.data[i]
                w.data[i] += dt*aw.data[i]

                # Positions are updated using the velocities and XSPH
                x.data[i] += dt * ax.data[i]
                y.data[i] += dt * ay.data[i]
                z.data[i] += dt * az.data[i]

                # Update densities and smoothing lengths from the accelerations
                rho.data[i] = dt * arho.data[i]

    def compute_time_step(self, double dt):
        return dt

cdef class TransportVelocityIntegrator(Integrator):
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef _reset_accelerations(self):
        cdef int i, npart
        cdef object pa_wrapper
        
        # acc properties to reset
        cdef DoubleArray au, av, aw, auhat, avhat, V, rho
        cdef DoubleArray u, v, p, wij

        for pa in self.particles:
            name = pa.name

            pa_wrapper = getattr(self.evaluator.calc, pa.name)

            if not name.startswith('fluid'):
                u = pa_wrapper.u; v = pa_wrapper.v
                p = pa_wrapper.p; wij = pa_wrapper.wij

                npart = pa.get_number_of_particles()
                for i in range(npart):
                    u.data[i] = 0.0
                    v.data[i] = 0.0
                    p.data[i] = 0.0
                    wij.data[i] = 0.0

            else:
                au = pa_wrapper.au; av = pa_wrapper.av; aw = pa_wrapper.aw
                auhat = pa_wrapper.auhat; avhat = pa_wrapper.avhat
                V = pa_wrapper.V; rho = pa_wrapper.rho

                npart = pa.get_number_of_particles()
                for i in range(npart):
                    au.data[i] = 0.0
                    av.data[i] = 0.0
                    aw.data[i] = 0.0

                    auhat.data[i] = 0.0
                    avhat.data[i] = 0.0

                    V.data[i] = 0.0
                    rho.data[i] = 0.0

    cpdef integrate(self, double t, double dt, int count):
        """Main step routine"""
        cdef object pa_wrapper
        cdef object evaluator = self.evaluator
        cdef object pm = self.pm
        cdef NNPS nnps = self.nnps
        cdef str name

        # Particle properties
        cdef DoubleArray x, y, z, u, v, w, uhat, vhat
        cdef DoubleArray vmag

        # accelerations for the variables
        cdef DoubleArray au, av, aw, auhat, avhat

        # number of particles
        cdef size_t npart
        cdef size_t i

        # half time step
        cdef double dtb2 = 0.5*dt

        #############################################################
        # Kick
        #############################################################
        for pa in self.particles:
            name = pa.name

            if not name.startswith('fluid'):
                continue

            pa_wrapper = getattr(self.evaluator.calc, name)

            x = pa_wrapper.x  ; y = pa_wrapper.y  ; z = pa_wrapper.z

            u = pa_wrapper.u  ; v = pa_wrapper.v  ; w = pa_wrapper.w
            au = pa_wrapper.au; av = pa_wrapper.av; aw = pa_wrapper.aw

            uhat = pa_wrapper.uhat; vhat = pa_wrapper.vhat
            auhat = pa_wrapper.auhat; avhat = pa_wrapper.avhat

            vmag = pa_wrapper.vmag

            npart = pa.get_number_of_particles()

            for i in range(npart):
                # Update momentum velocities
                u.data[i] = u.data[i] + dtb2*au.data[i]
                v.data[i] = v.data[i] + dtb2*av.data[i]

                # update advection velocities
                uhat.data[i] = u.data[i] + dtb2*auhat.data[i]
                vhat.data[i] = v.data[i] + dtb2*avhat.data[i]

                #############################################################
                # Drift positions with the advection velocity
                #############################################################
                x.data[i] = x.data[i] + dt * uhat.data[i]
                y.data[i] = y.data[i] + dt * vhat.data[i]

        # Update NNPS since particles have moved
        if pm: pm.update()
        nnps.update()

        # compute accelerations
        self._reset_accelerations()
        evaluator.compute(t, dt)

        #############################################################
        # Kick
        #############################################################
        for pa in self.particles:
            name = pa.name

            if not name.startswith('fluid'):
                continue

            pa_wrapper = getattr(self.evaluator.calc, name)

            u = pa_wrapper.u  ; v = pa_wrapper.v
            au = pa_wrapper.au; av = pa_wrapper.av

            npart = pa.get_number_of_particles()
            for i in range(npart):
                # Update velocities
                u.data[i] = u.data[i] + dtb2*au.data[i]
                v.data[i] = v.data[i] + dtb2*av.data[i]
    
                vmag.data[i] = sqrt( u.data[i]*u.data[i] + v.data[i]*v.data[i] )
                                         
