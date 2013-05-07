"""Implementation for the integrator"""
cimport cython

cdef class WCSPHRK2Integrator:

    def __init__(self, object evaluator, list particles):
        """Constructor for the Integrator"""
        self.evaluator = evaluator
        self.particles = particles
        self.nnps = evaluator.nnps

    cpdef integrate(self, double dt):
        """Main step routine"""
        cdef object pa_wrapper
        cdef object evaluator = self.evaluator
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
        nnps.update()
                
        # compute accelerations
        self._reset_accelerations()
        evaluator.compute()

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

        # Re-bin because particles have moved
        nnps.update()

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
