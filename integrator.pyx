"""Implementation for the integrator"""

cdef class WCSPHRK2Integrator:

    def __init__(self, object evaluator, list particles):
        """Constructor for the Integrator"""
        self.evaluator = evaluator
        self.particles = particles

    cpdef integrate(self, double dt):
        """Main step routine"""
        cdef object pa_wrapper
        cdef object evaluator = self.evaluator

        # Particle properties
        cdef DoubleArray x, y, z, u, v, w, rho

        # Quantities at the start of a time step
        cdef DoubleArray x0, y0, z0, u0, v0, w0, rho0

        # accelerations for the variables
        cdef DoubleArray ax, ay, az, au, av, aw, arho

        # particle tags and number may change from iterations so we
        # want to ensure these are freshly created
        #cdef size_t npart = pa.get_number_of_particles()
        cdef size_t npart
        cdef size_t i

        # Set initial values
        #self._set_initial_values()

        #############################################################
        # Predictor
        #############################################################
        cdef double dtb2 = 0.5*dt

        for pa in self.particles:
            name = pa.name
            pa_wrapper = getattr(self.evaluator.calc, name)

            rho = pa_wrapper.rho; rho0 = pa_wrapper.rho0; arho=pa_wrapper.arho
            x = pa_wrapper.x; y = pa_wrapper.y; z = pa_wrapper.z
            x0 = pa_wrapper.x0; y0 = pa_wrapper.y0; z0 = pa_wrapper.z0
            ax = pa_wrapper.ax; ay = pa_wrapper.ay; az = pa_wrapper.az

            u = pa_wrapper.u; v = pa_wrapper.v; w = pa_wrapper.w
            u0 = pa_wrapper.u0; v0 = pa_wrapper.v0; w0 = pa_wrapper.w0
            au = pa_wrapper.au; av = pa_wrapper.av; aw = pa_wrapper.aw

            npart = x.length

            # saving X^n
            for i in range(npart):
                u0.data[i] = u.data[i]; v0.data[i] = v.data[i]
                w0.data[i] = w.data[i]; x0.data[i] = x.data[i]
                y0.data[i] = y.data[i]; z0.data[i] = y.data[i]
                rho0.data[i] = rho.data[i]

            for i in range(npart):
                # Update velocities
                u.data[i] = u0.data[i] + dtb2*au.data[i]
                v.data[i] = v0.data[i] + dtb2*av.data[i]
                w.data[i] = w0.data[i] + dtb2*aw.data[i]

                # Positions are updated using XSPH
                x.data[i] = x0.data[i] + dtb2 * ax.data[i]
                y.data[i] = y0.data[i] + dtb2 * ay.data[i]
                w.data[i] = z0.data[i] + dtb2 * aw.data[i]

                # Update densities and smoothing lengths from the accelerations
                rho.data[i] = rho0.data[i] + dtb2 * arho.data[i]

            # Update NNPS since particles have moved
            #nnps.update()
                
            # compute accelerations
            evaluator.compute()

            #############################################################
            # Corrector
            #############################################################
            for i in range(npart):

                # Update velocities
                u.data[i] = u0.data[i] + dt*au.data[i]
                v.data[i] = v0.data[i] + dt*av.data[i]
                w.data[i] = w0.data[i] + dt*aw.data[i]

                # Positions are updated using the velocities and XSPH
                x.data[i] = x0.data[i] + dt * ax.data[i]
                y.data[i] = y0.data[i] + dt * ay.data[i]
                z.data[i] = z0.data[i] + dt * aw.data[i]

                # Update densities and smoothing lengths from the accelerations
                rho.data[i] = rho0.data[i] + dt * arho.data[i]

            # Re-bin because particles have moved
            #nnps.update()

