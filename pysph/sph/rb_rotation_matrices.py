import numpy as np
import numpy
import unittest
from numpy import (cos as c, sin as s)

from pysph.base.utils import get_particle_array
from pysph.sph.equation import Equation
from compyle.types import declare
from pysph.sph.integrator import Integrator
from pysph.sph.integrator_step import IntegratorStep


def get_particle_array_rigid_body(constants=None, **props):
    extra_props = ['fx', 'fy', 'fz', 'dx0', 'dy0', 'dz0']

    # add body id if not available
    body_id = props.pop('body_id', None)
    n_body = 1 if body_id is None else numpy.max(body_id) + 1

    consts = {
        'total_mass': numpy.zeros(n_body, dtype=float),
        'xcm': numpy.zeros(3 * n_body, dtype=float),
        'xcm_0': numpy.zeros(3 * n_body, dtype=float),
        'moi_body_inv': numpy.zeros(9 * n_body, dtype=float),
        'moi_body': numpy.zeros(9 * n_body, dtype=float),
        'orientation_R': [1., 0., 0., 0., 1., 0., 0., 0., 1.] * n_body,
        'orientation_R_0': [1., 0., 0., 0., 1., 0., 0., 0., 1.] * n_body,
        # we only use this since, used at angular velocity computation
        'moi_global_inv': numpy.zeros(9 * n_body, dtype=float),
        # we never compute this.
        'moi_global': numpy.zeros(9 * n_body, dtype=float),
        # total force at the center of mass
        'force': numpy.zeros(3 * n_body, dtype=float),
        # torque about the center of mass
        'torque': numpy.zeros(3 * n_body, dtype=float),
        # velocity, acceleration of CM.
        'vcm': numpy.zeros(3 * n_body, dtype=float),
        'vcm_0': numpy.zeros(3 * n_body, dtype=float),
        # angular momentum
        'ang_mom': numpy.zeros(3 * n_body, dtype=float),
        'ang_mom_0': numpy.zeros(3 * n_body, dtype=float),
        # angular velocity in global frame
        'omega': numpy.zeros(3 * n_body, dtype=float),
        'omega_mat': numpy.zeros(9 * n_body, dtype=float),
    }

    if constants:
        consts.update(constants)

    # print(consts)
    pa = get_particle_array(constants=consts, additional_props=extra_props,
                            **props)

    if body_id is None:
        body_id = numpy.zeros(len(pa.x), dtype=int)

    pa.add_property('body_id', type='int', data=body_id)

    pa.add_constant('n_body', np.array([n_body], dtype=int))

    # set the index limits of the body
    limits = get_limits_of_rigid_bodies(pa)
    pa.add_constant('body_limits', limits)

    # no of bodies array for compyle parallel computing
    pa.add_constant('bodies', np.arange(0, pa.n_body[0], 1))

    # total mass of the body
    set_total_mass(pa)

    # center of mass of the body
    set_center_of_mass(pa)

    # compute moment of inertia
    set_moi_in_body_frame(pa)

    # setup particle positions in body frame
    set_body_frame_position_vectors(pa)

    pa.set_output_arrays(['x', 'y', 'z', 'u', 'v', 'w', 'fx', 'fy', 'fz'])
    return pa


def get_limits_of_rigid_bodies(pa):
    if 'body_id' in pa.properties:
        body_id = pa.body_id
        n_body = pa.n_body[0]
        limits = np.zeros(2 * n_body, dtype=int)
        for i in range(n_body):
            tmp_lim = np.where(body_id == i)[0]
            limits[2 * i] = tmp_lim[0]
            limits[2 * i + 1] = tmp_lim[-1] + 1
        return limits


def set_total_mass(pa):
    for i in range(pa.n_body[0]):
        # left limit of body i
        l = pa.body_limits[2 * i]
        # right limit of body i
        r = pa.body_limits[2 * i + 1]
        pa.total_mass[i] = np.sum(pa.m[l:r])
        assert pa.total_mass[i] > 0., "Total mass has to be greater than zero"
        # assert np.sum(pa.m) == pa.total_mass[0]


def set_center_of_mass(pa):
    for i in range(pa.n_body[0]):
        # left limit of body i
        l = pa.body_limits[2 * i]
        # right limit of body i
        r = pa.body_limits[2 * i + 1]

        j = 3 * i

        pa.xcm[j] = np.sum(pa.m[l:r] * pa.x[l:r]) / pa.total_mass[i]
        pa.xcm[j + 1] = np.sum(pa.m[l:r] * pa.y[l:r]) / pa.total_mass[i]
        pa.xcm[j + 2] = np.sum(pa.m[l:r] * pa.z[l:r]) / pa.total_mass[i]


def set_moi_in_body_frame(pa):
    """This method assumes the center of mass is already computed."""
    for i in range(pa.n_body[0]):
        xcm_i = pa.xcm[3 * i:3 * i + 3]
        # left limit of body i
        l = pa.body_limits[2 * i]
        # right limit of body i
        r = pa.body_limits[2 * i + 1]

        I = np.zeros(9)
        for j in range(l, r):
            # print(j)
            # Ixx
            I[0] += pa.m[j] * (
                (pa.y[j] - xcm_i[1])**2. + (pa.z[j] - xcm_i[2])**2.)

            # Iyy
            I[4] += pa.m[j] * (
                (pa.x[j] - xcm_i[0])**2. + (pa.z[j] - xcm_i[2])**2.)

            # Iyy
            I[8] += pa.m[j] * (
                (pa.x[j] - xcm_i[0])**2. + (pa.y[j] - xcm_i[1])**2.)

            # Ixy
            I[1] -= pa.m[j] * (pa.x[j] - xcm_i[0]) * (pa.y[j] - xcm_i[1])

            # Ixz
            I[2] -= pa.m[j] * (pa.x[j] - xcm_i[0]) * (pa.z[j] - xcm_i[2])

            # Iyz
            I[5] -= pa.m[j] * (pa.y[j] - xcm_i[1]) * (pa.z[j] - xcm_i[2])

        I[3] = I[1]
        I[6] = I[2]
        I[7] = I[5]
        I_inv = np.linalg.inv(I.reshape(3, 3))
        I_inv = I_inv.ravel()
        pa.moi_body[9 * i:9 * i + 9] = I[:]
        pa.moi_body_inv[9 * i:9 * i + 9] = I_inv[:]


def compute_moment_of_inertia_from_global_positions(pa):
    """This method assumes the center of mass is already computed."""
    I_all = np.array([])
    for i in range(pa.n_body[0]):
        xcm_i = pa.xcm[3 * i:3 * i + 3]
        # left limit of body i
        l = pa.body_limits[2 * i]
        # right limit of body i
        r = pa.body_limits[2 * i + 1]

        I = np.zeros(9)
        for j in range(l, r):
            # Ixx
            I[0] += pa.m[j] * (
                (pa.y[j] - xcm_i[1])**2. + (pa.z[j] - xcm_i[2])**2.)

            # Iyy
            I[4] += pa.m[j] * (
                (pa.x[j] - xcm_i[0])**2. + (pa.z[j] - xcm_i[2])**2.)

            # Iyy
            I[8] += pa.m[j] * (
                (pa.x[j] - xcm_i[0])**2. + (pa.y[j] - xcm_i[1])**2.)

            # Ixy
            I[1] -= pa.m[j] * (pa.x[j] - xcm_i[0]) * (pa.y[j] - xcm_i[1])

            # Ixz
            I[2] -= pa.m[j] * (pa.x[j] - xcm_i[0]) * (pa.z[j] - xcm_i[2])

            # Iyz
            I[5] -= pa.m[j] * (pa.y[j] - xcm_i[1]) * (pa.z[j] - xcm_i[2])

        I[3] = I[1]
        I[6] = I[2]
        I[7] = I[5]

        I_all = np.concatenate(I_all, I)
    return I_all


def set_body_frame_position_vectors(pa):
    for i in range(pa.n_body[0]):
        # left limit of body i
        l = pa.body_limits[2 * i]
        # right limit of body i
        r = pa.body_limits[2 * i + 1]

        j = 3 * i

        pa.dx0[l:r] = pa.x[l:r] - pa.xcm[j]
        pa.dy0[l:r] = pa.y[l:r] - pa.xcm[j + 1]
        pa.dz0[l:r] = pa.z[l:r] - pa.xcm[j + 2]


class RigidBodyCollision(Equation):
    def __init__(self, dest, sources, kn=1e3, mu=0.5, en=0.8):
        """Initialise the required coefficients for force calculation.


        Keyword arguments:
        kn -- Normal spring stiffness (default 1e3)
        mu -- friction coefficient (default 0.5)
        en -- coefficient of restitution (0.8)

        Given these coefficients, tangential spring stiffness, normal and
        tangential damping coefficient are calculated by default.

        """
        self.kn = kn
        self.kt = 2. / 7. * kn
        m_eff = np.pi * 0.5**2 * 1e-6 * 2120
        self.gamma_n = -(2 * np.sqrt(kn * m_eff) * np.log(en)) / (
            np.sqrt(np.pi**2 + np.log(en)**2))
        self.gamma_t = 0.5 * self.gamma_n
        self.mu = mu
        super(RigidBodyCollision, self).__init__(dest, sources)

    def loop(self, d_idx, d_fx, d_fy, d_fz, d_h, d_total_mass, d_rad_s, s_idx,
             s_rad_s, XIJ, R2IJ, RIJ, VIJ):
        overlap = 0
        if RIJ > 1e-9:
            overlap = d_rad_s[d_idx] + s_rad_s[s_idx] - RIJ

        if overlap > 0:
            # normal vector passing from particle i to j
            nij_x = -XIJ[0] / RIJ
            nij_y = -XIJ[1] / RIJ
            nij_z = -XIJ[2] / RIJ

            # overlap speed: a scalar
            vijdotnij = VIJ[0] * nij_x + VIJ[1] * nij_y + VIJ[2] * nij_z

            # normal velocity
            vijn_x = vijdotnij * nij_x
            vijn_y = vijdotnij * nij_y
            vijn_z = vijdotnij * nij_z

            # normal force with conservative and dissipation part
            fn_x = -self.kn * overlap * nij_x - self.gamma_n * vijn_x
            fn_y = -self.kn * overlap * nij_y - self.gamma_n * vijn_y
            fn_z = -self.kn * overlap * nij_z - self.gamma_n * vijn_z

            d_fx[d_idx] += fn_x
            d_fy[d_idx] += fn_y
            d_fz[d_idx] += fn_z


class SumUpExternalForces(Equation):
    def reduce(self, dst, t, dt):
        frc = declare('object')
        trq = declare('object')
        body_limits = declare('object')
        n_body = declare('int')
        fx = declare('object')
        fy = declare('object')
        fz = declare('object')
        x = declare('object')
        y = declare('object')
        z = declare('object')
        xcm = declare('object')
        i = declare('int')
        j = declare('int')
        i3 = declare('int')
        l = declare('int')
        r = declare('int')

        frc = dst.force
        trq = dst.torque
        fx = dst.fx
        fy = dst.fy
        fz = dst.fz
        x = dst.x
        y = dst.y
        z = dst.z
        xcm = dst.xcm
        n_body = dst.n_body[0]
        body_limits = dst.body_limits

        frc[:] = 0
        trq[:] = 0

        for i in range(n_body):
            # left limit of body i
            l = body_limits[2 * i]
            # right limit of body i
            r = body_limits[2 * i + 1]

            # step i
            i3 = 3 * i
            for j in range(l, r):
                frc[i3] += fx[j]
                frc[i3 + 1] += fy[j]
                frc[i3 + 2] += fz[j]

                # torque due to force on particle i
                # (r_i - com) \cross f_i
                dx = x[j] - xcm[i3]
                dy = y[j] - xcm[i3 + 1]
                dz = z[j] - xcm[i3 + 2]

                # torque due to force on particle i
                # dri \cross fi
                trq[i3] += (dy * fz[j] - dz * fy[j])
                trq[i3 + 1] += (dz * fx[j] - dx * fz[j])
                trq[i3 + 2] += (dx * fy[j] - dy * fx[j])


def normalize_orientation(orien):
    a1 = np.array([orien[0], orien[3], orien[6]])
    a2 = np.array([orien[1], orien[4], orien[7]])
    a3 = np.array([orien[2], orien[5], orien[8]])
    # norm of col0
    na1 = np.linalg.norm(a1)

    b1 = a1 / na1

    b2 = a2 - np.dot(b1, a2) * b1
    nb2 = np.linalg.norm(b2)
    b2 = b2 / nb2

    b3 = a3 - np.dot(b1, a3) * b1 - np.dot(b2, a3) * b2
    nb3 = np.linalg.norm(b3)
    b3 = b3 / nb3

    orien[0] = b1[0]
    orien[3] = b1[1]
    orien[6] = b1[2]
    orien[1] = b2[0]
    orien[4] = b2[1]
    orien[7] = b2[2]
    orien[2] = b3[0]
    orien[5] = b3[1]
    orien[8] = b3[2]


class RK2StepRigidBody(IntegratorStep):
    def py_initialize(self, dst, t, dt):
        for i in range(dst.n_body[0]):
            i3 = 3 * i
            i9 = 9 * i
            for j in range(i3, i3 + 3):
                # save the center of mass and center of mass velocity
                dst.xcm_0[j] = dst.xcm[j]
                dst.vcm_0[j] = dst.vcm[j]

                # save the current angular momentum
                dst.ang_mom_0[j] = dst.ang_mom[j]

            # save the current orientation
            # print(dst.orientation_R)
            for j in range(i9, i9 + 9):
                dst.orientation_R_0[j] = dst.orientation_R[j]

    def initialize(self):
        pass

    def py_stage1(self, dst, t, dt):
        dtb2 = dt / 2.
        for i in range(dst.n_body[0]):
            i3 = 3 * i
            i9 = 9 * i
            for j in range(i3, i3 + 3):
                # update center of mass position and velocity
                dst.xcm[j] = dst.xcm_0[j] + dtb2 * dst.vcm[j]
                dst.vcm[j] = dst.vcm_0[j] + (
                    dtb2 * dst.force[j] / dst.total_mass[i])

                # update the angular momentum
                dst.ang_mom[j] = dst.ang_mom_0[j] + dst.torque[j] * dtb2

            # angular velocity in terms of matrix
            omega_mat = np.array([[0, -dst.omega[i3 + 2], dst.omega[i3 + 1]],
                                  [dst.omega[i3 + 2], 0, -dst.omega[i3]],
                                  [-dst.omega[i3 + 1], dst.omega[i3], 0]])

            # Currently the orientation is
            orientation = dst.orientation_R[i9:i9 + 9].reshape(3, 3)

            # Rate of change of orientation is
            r_dot = np.matmul(omega_mat, orientation)
            r_dot = r_dot.ravel()

            # update the orientation to next time step
            dst.orientation_R[i9:i9 + 9] = (
                dst.orientation_R_0[i9:i9 + 9] + r_dot[:] * dtb2)
            # normalize the orientation
            normalize_orientation(dst.orientation_R[i9:i9 + 9])

            # update the moment of inertia
            R = dst.orientation_R[i9:i9 + 9].reshape(3, 3)
            R_t = dst.orientation_R[i9:i9 + 9].reshape(3, 3).transpose()
            tmp = np.matmul(R, dst.moi_body_inv[i9:i9+9].reshape(3, 3))
            dst.moi_global_inv[i9:i9 + 9] = (np.matmul(tmp, R_t)).ravel()

            # update the angular velocity
            dst.omega[i3:i3 + 3] = np.matmul(
                dst.moi_global_inv[i9:i9 + 9].reshape(3, 3),
                dst.ang_mom[i3:i3 + 3])

    def stage1(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_dx0, d_dy0, d_dz0,
               d_xcm, d_vcm, d_orientation_R, d_omega, d_body_id):
        i3, i9, b_id = declare('int', 3)
        b_id = d_body_id[d_idx]
        i9 = 9 * b_id
        i3 = 3 * b_id

        ###########################
        # Update position vectors #
        ###########################
        # rotate the position of the vector in the body frame to global frame
        dx = (d_orientation_R[i9+0] * d_dx0[d_idx] +
              d_orientation_R[i9+1] * d_dy0[d_idx] +
              d_orientation_R[i9+2] * d_dz0[d_idx])
        dy = (d_orientation_R[i9+3] * d_dx0[d_idx] +
              d_orientation_R[i9+4] * d_dy0[d_idx] +
              d_orientation_R[i9+5] * d_dz0[d_idx])
        dz = (d_orientation_R[i9+6] * d_dx0[d_idx] +
              d_orientation_R[i9+7] * d_dy0[d_idx] +
              d_orientation_R[i9+8] * d_dz0[d_idx])

        d_x[d_idx] = d_xcm[i3+0] + dx
        d_y[d_idx] = d_xcm[i3+1] + dy
        d_z[d_idx] = d_xcm[i3+2] + dz

        ###########################
        # Update velocity vectors #
        ###########################
        # here du, dv, dw are velocities due to angular velocity
        # dV = omega \cross dr
        # where dr = x - cm
        du = d_omega[i3+1] * dz - d_omega[i3+2] * dy
        dv = d_omega[i3+2] * dx - d_omega[i3+0] * dz
        dw = d_omega[i3+0] * dy - d_omega[i3+1] * dx

        d_u[d_idx] = d_vcm[i3+0] + du
        d_v[d_idx] = d_vcm[i3+1] + dv
        d_w[d_idx] = d_vcm[i3+2] + dw

    def py_stage2(self, dst, t, dt):
        for i in range(dst.n_body[0]):
            i3 = 3 * i
            i9 = 9 * i
            for j in range(i3, i3 + 3):
                # update center of mass position and velocity
                dst.xcm[j] = dst.xcm_0[j] + dt * dst.vcm[j]
                dst.vcm[j] = dst.vcm_0[j] + (
                    dt * dst.force[j] / dst.total_mass[i])

                # update the angular momentum
                dst.ang_mom[j] = dst.ang_mom_0[j] + dst.torque[j] * dt

            # angular velocity in terms of matrix
            omega_mat = np.array([[0, -dst.omega[i3 + 2], dst.omega[i3 + 1]],
                                  [dst.omega[i3 + 2], 0, -dst.omega[i3]],
                                  [-dst.omega[i3 + 1], dst.omega[i3], 0]])

            # Currently the orientation is
            orientation = dst.orientation_R[i9:i9 + 9].reshape(3, 3)

            # Rate of change of orientation is
            r_dot = np.matmul(omega_mat, orientation)
            r_dot = r_dot.ravel()

            # update the orientation to next time step
            dst.orientation_R[i9:i9 + 9] = (
                dst.orientation_R_0[i9:i9 + 9] + r_dot[:] * dt)

            # update the moment of inertia
            R = dst.orientation_R[i9:i9 + 9].reshape(3, 3)
            R_t = dst.orientation_R[i9:i9 + 9].reshape(3, 3).transpose()
            tmp = np.matmul(R, dst.moi_body_inv[i9:i9+9].reshape(3, 3))
            dst.moi_global_inv[i9:i9 + 9] = (np.matmul(tmp, R_t)).ravel()

            # update the angular velocity
            dst.omega[i3:i3 + 3] = np.matmul(
                dst.moi_global_inv[i9:i9 + 9].reshape(3, 3),
                dst.ang_mom[i3:i3 + 3])

    def stage2(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_dx0, d_dy0, d_dz0,
               d_xcm, d_vcm, d_orientation_R, d_omega, d_body_id):
        i3, i9, b_id = declare('int', 3)
        b_id = d_body_id[d_idx]
        i9 = 9 * b_id
        i3 = 3 * b_id

        ###########################
        # Update position vectors #
        ###########################
        # rotate the position of the vector in the body frame to global frame
        dx = (d_orientation_R[i9+0] * d_dx0[d_idx] +
              d_orientation_R[i9+1] * d_dy0[d_idx] +
              d_orientation_R[i9+2] * d_dz0[d_idx])
        dy = (d_orientation_R[i9+3] * d_dx0[d_idx] +
              d_orientation_R[i9+4] * d_dy0[d_idx] +
              d_orientation_R[i9+5] * d_dz0[d_idx])
        dz = (d_orientation_R[i9+6] * d_dx0[d_idx] +
              d_orientation_R[i9+7] * d_dy0[d_idx] +
              d_orientation_R[i9+8] * d_dz0[d_idx])

        d_x[d_idx] = d_xcm[i3+0] + dx
        d_y[d_idx] = d_xcm[i3+1] + dy
        d_z[d_idx] = d_xcm[i3+2] + dz

        ###########################
        # Update velocity vectors #
        ###########################
        # here du, dv, dw are velocities due to angular velocity
        # dV = omega \cross dr
        # where dr = x - cm
        du = d_omega[i3+1] * dz - d_omega[i3+2] * dy
        dv = d_omega[i3+2] * dx - d_omega[i3+0] * dz
        dw = d_omega[i3+0] * dy - d_omega[i3+1] * dx

        d_u[d_idx] = d_vcm[i3+0] + du
        d_v[d_idx] = d_vcm[i3+1] + dv
        d_w[d_idx] = d_vcm[i3+2] + dw
