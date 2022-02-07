from pysph.sph.integrator import IntegratorStep


class PECStep(IntegratorStep):
    """Predictor Corrector integrator for Gas-dynamics"""

    def initialize(self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z, d_h,
                   d_u0, d_v0, d_w0, d_u, d_v, d_w, d_e, d_e0, d_h0,
                   d_converged, d_rho, d_rho0):
        d_x0[d_idx] = d_x[d_idx]
        d_y0[d_idx] = d_y[d_idx]
        d_z0[d_idx] = d_z[d_idx]

        d_u0[d_idx] = d_u[d_idx]
        d_v0[d_idx] = d_v[d_idx]
        d_w0[d_idx] = d_w[d_idx]

        d_e0[d_idx] = d_e[d_idx]

        d_h0[d_idx] = d_h[d_idx]
        d_rho0[d_idx] = d_rho[d_idx]

        # set the converged attribute to 0 at the beginning of a Group
        d_converged[d_idx] = 0

    def stage1(self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z,
               d_u0, d_v0, d_w0, d_u, d_v, d_w, d_e0, d_e, d_au, d_av,
               d_aw, d_ae, d_rho, d_rho0, d_arho, d_h, d_h0, d_ah, dt):

        dtb2 = 0.5 * dt

        d_u[d_idx] = d_u0[d_idx] + dtb2 * d_au[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dtb2 * d_av[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dtb2 * d_aw[d_idx]

        d_x[d_idx] = d_x0[d_idx] + dtb2 * d_u[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dtb2 * d_v[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dtb2 * d_w[d_idx]

        # update thermal energy
        d_e[d_idx] = d_e0[d_idx] + dtb2 * d_ae[d_idx]

        # predict density and smoothing lengths for faster
        # convergence. NNPS need not be explicitly updated since it
        # will be called at the end of the predictor stage.
        d_h[d_idx] = d_h0[d_idx] + dtb2 * d_ah[d_idx]
        d_rho[d_idx] = d_rho0[d_idx] + dtb2 * d_arho[d_idx]

    def stage2(self, d_idx, d_x0, d_y0, d_z0, d_x, d_y, d_z,
               d_u0, d_v0, d_w0, d_u, d_v, d_w, d_e0, d_e, d_au, d_av,
               d_aw, d_ae, dt):
        d_u[d_idx] = d_u0[d_idx] + dt * d_au[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dt * d_av[d_idx]
        d_w[d_idx] = d_w0[d_idx] + dt * d_aw[d_idx]

        d_x[d_idx] = d_x0[d_idx] + dt * d_u[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dt * d_v[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dt * d_w[d_idx]

        # Update densities and smoothing lengths from the accelerations
        d_e[d_idx] = d_e0[d_idx] + dt * d_ae[d_idx]
