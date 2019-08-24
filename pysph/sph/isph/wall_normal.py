from math import sqrt
from compyle.api import declare

from pysph.sph.equation import Equation


class ComputeNormals(Equation):
    """Compute normals using a simple approach

    .. math::

       -\frac{m_j}{\rho_j} DW_{ij}

    First compute the normals, then average them and finally normalize them.

    """

    def initialize(self, d_idx, d_normal_tmp, d_normal):
        idx = declare('int')
        idx = 3*d_idx
        d_normal_tmp[idx] = 0.0
        d_normal_tmp[idx + 1] = 0.0
        d_normal_tmp[idx + 2] = 0.0
        d_normal[idx] = 0.0
        d_normal[idx + 1] = 0.0
        d_normal[idx + 2] = 0.0

    def loop(self, d_idx, d_normal_tmp, s_idx, s_m, s_rho, DWIJ):
        idx = declare('int')
        idx = 3*d_idx
        fac = -s_m[s_idx]/s_rho[s_idx]
        d_normal_tmp[idx] += fac*DWIJ[0]
        d_normal_tmp[idx + 1] += fac*DWIJ[1]
        d_normal_tmp[idx + 2] += fac*DWIJ[2]

    def post_loop(self, d_idx, d_normal_tmp, d_h):
        idx = declare('int')
        idx = 3*d_idx
        mag = sqrt(d_normal_tmp[idx]**2 + d_normal_tmp[idx + 1]**2 +
                   d_normal_tmp[idx + 2]**2)
        if mag > 0.25/d_h[d_idx]:
            d_normal_tmp[idx] /= mag
            d_normal_tmp[idx + 1] /= mag
            d_normal_tmp[idx + 2] /= mag
        else:
            d_normal_tmp[idx] = 0.0
            d_normal_tmp[idx + 1] = 0.0
            d_normal_tmp[idx + 2] = 0.0


class SmoothNormals(Equation):
    def loop(self, d_idx, d_normal, s_normal_tmp, s_idx, s_m, s_rho, WIJ):
        idx = declare('int')
        idx = 3*d_idx
        fac = s_m[s_idx]/s_rho[s_idx]*WIJ
        d_normal[idx] += fac*s_normal_tmp[3*s_idx]
        d_normal[idx + 1] += fac*s_normal_tmp[3*s_idx + 1]
        d_normal[idx + 2] += fac*s_normal_tmp[3*s_idx + 2]

    def post_loop(self, d_idx, d_normal, d_h):
        idx = declare('int')
        idx = 3*d_idx
        mag = sqrt(d_normal[idx]**2 + d_normal[idx + 1]**2 +
                   d_normal[idx + 2]**2)
        if mag > 1e-3:
            d_normal[idx] /= mag
            d_normal[idx + 1] /= mag
            d_normal[idx + 2] /= mag
        else:
            d_normal[idx] = 0.0
            d_normal[idx + 1] = 0.0
            d_normal[idx + 2] = 0.0


class SetWallVelocityNew(Equation):
    r"""Modified SetWall velocity which sets a suitable normal velocity.

    This requires that the destination array has a 3-strided "normal"
    property.
    """
    def initialize(self, d_idx, d_uf, d_vf, d_wf, d_wij):
        d_uf[d_idx] = 0.0
        d_vf[d_idx] = 0.0
        d_wf[d_idx] = 0.0
        d_wij[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_uf, d_vf, d_wf,
             s_u, s_v, s_w, d_wij, XIJ, RIJ, HIJ, SPH_KERNEL):
        wij = SPH_KERNEL.kernel(XIJ, RIJ, 0.5*HIJ)

        d_wij[d_idx] += wij
        d_uf[d_idx] += s_u[s_idx] * wij
        d_vf[d_idx] += s_v[s_idx] * wij
        d_wf[d_idx] += s_w[s_idx] * wij

    def post_loop(self, d_uf, d_vf, d_wf, d_wij, d_idx,
                  d_ug, d_vg, d_wg, d_u, d_v, d_w, d_normal):
        idx = declare('int')
        idx = 3*d_idx
        # calculation is done only for the relevant boundary particles.
        # d_wij (and d_uf) is 0 for particles sufficiently away from the
        # solid-fluid interface
        if d_wij[d_idx] > 1e-12:
            d_uf[d_idx] /= d_wij[d_idx]
            d_vf[d_idx] /= d_wij[d_idx]
            d_wf[d_idx] /= d_wij[d_idx]

        # Dummy velocities at the ghost points using Eq. (23),
        # d_u, d_v, d_w are the prescribed wall velocities.
        d_ug[d_idx] = 2*d_u[d_idx] - d_uf[d_idx]
        d_vg[d_idx] = 2*d_v[d_idx] - d_vf[d_idx]
        d_wg[d_idx] = 2*d_w[d_idx] - d_wf[d_idx]

        vn = (d_ug[d_idx]*d_normal[idx] + d_vg[d_idx]*d_normal[idx+1]
              + d_wg[d_idx]*d_normal[idx+2])
        if vn < 0:
            d_ug[d_idx] -= vn*d_normal[idx]
            d_vg[d_idx] -= vn*d_normal[idx+1]
            d_wg[d_idx] -= vn*d_normal[idx+2]
