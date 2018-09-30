import numpy

from pysph.base.utils import get_particle_array
from pysph.sph.integrator_step import IntegratorStep
from pysph.sph.equation import Equation
from pysph.cpy.types import declare


def get_particle_array_rigid_body(constants=None, **props):
    extra_props = [
        'rbx', 'rby', 'rbz', 'rx', 'ry', 'rz', 'fx', 'fy', 'fz', 'tang_disp_y',
        'tang_velocity_z', 'tang_velocity_y', 'tang_velocity_x', 'tang_disp_z',
        'tang_disp_x'
    ]

    body_id = props.pop('body_id', None)
    nb = 1 if body_id is None else numpy.max(body_id) + 1

    consts = {
        'total_mass': numpy.zeros(nb, dtype=float),
        'num_body': numpy.asarray(nb, dtype=int),
        'xcm': numpy.zeros(3 * nb, dtype=float),
        'xcm0': numpy.zeros(3 * nb, dtype=float),
        'mi_b_inv': numpy.zeros(9 * nb, dtype=float),
        'force': numpy.zeros(3 * nb, dtype=float),
        'torque': numpy.zeros(3 * nb, dtype=float),
        'L': numpy.zeros(3 * nb, dtype=float),
        'L0': numpy.zeros(3 * nb, dtype=float),
        # velocity CM.
        'vcm': numpy.zeros(3 * nb, dtype=float),
        'vcm0': numpy.zeros(3 * nb, dtype=float),
        # angular velocity, acceleration of body.
        'omega': numpy.zeros(3 * nb, dtype=float),
        'omega_quat': numpy.zeros(4 * nb, dtype=float),
        # orientation of the body
        'A': numpy.asarray(list(numpy.identity(3).ravel()) * nb),
        'q': numpy.asarray([1., 0, 0., 0.] * nb),
        'q0': numpy.asarray([1., 0, 0., 0.] * nb),
        'q_dot': numpy.asarray([0., 0, 0., 0.] * nb),
    }
    if constants:
        consts.update(constants)
    pa = get_particle_array(constants=consts, additional_props=extra_props,
                            **props)
    if body_id is None:
        body_id = numpy.zeros(len(pa.x), dtype=int)

    pa.add_property('body_id', type='int', data=body_id)
    pa.add_property('indices', type='int', data=numpy.arange(len(pa.x)))

    compute_properties_of_rigid_body(pa)
    pa.set_output_arrays([
        'x', 'y', 'z', 'u', 'v', 'w', 'rho', 'h', 'm', 'p', 'pid', 'au', 'av',
        'aw', 'tag', 'gid', 'fx', 'fy', 'fz'
    ])
    return pa


def compute_properties_of_rigid_body(pa):
    """Computes the precomputed values of rigid body such as total mass,
    center of mass, body moment of inertia inverse, body frame position vector
    of particles."""
    for i in range(pa.num_body[0]):
        # indices of a given body id
        cond = pa.body_id == i
        indices = pa.indices[cond]
        min_idx = min(indices)
        max_idx = max(indices) + 1

        pa.total_mass[i] = numpy.sum(pa.m[min_idx:max_idx])
        if pa.total_mass[i] == 0.:
            print("Total mass of the rigid body is\
            zero, please check mass of particles in body")

        # Compute center of mass body i
        x_cm = 0.
        y_cm = 0.
        z_cm = 0.
        for j in range(min_idx, max_idx):
            mj = pa.m[j]
            x_cm += pa.x[j] * mj
            y_cm += pa.y[j] * mj
            z_cm += pa.z[j] * mj

        pa.xcm[3 * i] = x_cm / pa.total_mass[i]
        pa.xcm[3 * i + 1] = y_cm / pa.total_mass[i]
        pa.xcm[3 * i + 2] = z_cm / pa.total_mass[i]

        # save the body frame position vectors
        for j in range(min_idx, max_idx):
            mj = pa.m[j]
            # the posjtjon vector js from center of mass to the partjcle's
            # posjtjon jn body frame
            pa.rbx[j] = pa.x[j] - pa.xcm[3 * i]
            pa.rby[j] = pa.y[j] - pa.xcm[3 * i + 1]
            pa.rbz[j] = pa.z[j] - pa.xcm[3 * i + 2]
            # initially both body and global frames are same, so
            # global position vector from center of mass will be
            pa.rx[j] = pa.x[j] - pa.xcm[3 * i]
            pa.ry[j] = pa.y[j] - pa.xcm[3 * i + 1]
            pa.rz[j] = pa.z[j] - pa.xcm[3 * i + 2]

        # moment of inertia calculation
        i_xx = 0.
        i_xy = 0.
        i_xz = 0.
        i_yy = 0.
        i_yz = 0.
        i_zz = 0.
        for j in range(min_idx, max_idx):
            mj = pa.m[j]
            rbx = pa.rbx[j]
            rby = pa.rby[j]
            rbz = pa.rbz[j]
            i_xx += mj * (rby**2. + rbz**2.)
            i_yy += mj * (rbx**2. + rbz**2.)
            i_zz += mj * (rbx**2. + rby**2.)
            i_xy += mj * (rbx * rby)
            i_xz += mj * (rbx * rbz)
            i_yz += mj * (rby * rbz)

        mi_b = numpy.array([[i_xx, -i_xy, -i_xz], [-i_xy, i_yy, -i_yz],
                            [-i_xz, -i_yz, i_zz]])

        # set moment of inertia inverse in body frame
        mi_b_inv = numpy.linalg.inv(mi_b)
        pa.mi_b_inv[9 * i:9 * i + 9] = mi_b_inv.ravel()


def set_linear_velocity(pa, vcm=numpy.array([0., 0., 0.])):
    # XXX: When ever angular velocity or linear velocity of COM changes
    # particle velocities has to be updated

    # update each particles velocity using angular and linear velocity of com
    # of body
    pa.vcm = vcm
    omega = pa.omega

    for i in range(len(pa.x)):
        # velocity due to angular velocity is
        ang_vel = numpy.cross(omega,
                              numpy.array([pa.rx[i], pa.ry[i], pa.rz[i]]))
        pa.u[i] = vcm[0] + ang_vel[0]
        pa.v[i] = vcm[1] + ang_vel[1]
        pa.w[i] = vcm[2] + ang_vel[2]


def set_angular_momentum(pa, L=numpy.array([0., 0., 0.])):
    # XXX: When ever angular velocity or angular momentum or linear velocity of
    # COM changes particle velocities has to be updated
    # set angular velocity
    pa.L = L

    # update angular velocity
    # omega = A * I^{body}^{-1} * R^{T} * L

    # tmp = R^T * L
    A = pa.A
    L = pa.L
    mi_b_inv = pa.mi_b_inv
    tmp0 = A[0] * L[0] + A[3] * L[1] + A[6] * L[2]
    tmp1 = A[1] * L[0] + A[4] * L[1] + A[7] * L[2]
    tmp2 = A[2] * L[0] + A[5] * L[1] + A[8] * L[2]

    # tmp = I^{body}^{-1} * tmp
    tmp0 = mi_b_inv[0] * tmp0 + mi_b_inv[1] * tmp1 + mi_b_inv[2] * tmp2
    tmp1 = mi_b_inv[3] * tmp0 + mi_b_inv[4] * tmp1 + mi_b_inv[5] * tmp2
    tmp2 = mi_b_inv[6] * tmp0 + mi_b_inv[7] * tmp1 + mi_b_inv[8] * tmp2

    pa.omega[0] = A[0] * tmp0 + A[1] * tmp1 + A[2] * tmp2
    pa.omega[1] = A[3] * tmp0 + A[4] * tmp1 + A[5] * tmp2
    pa.omega[2] = A[6] * tmp0 + A[7] * tmp1 + A[8] * tmp2

    pa.omega_quat[0] = 0
    pa.omega_quat[1] = pa.omega[0]
    pa.omega_quat[2] = pa.omega[1]
    pa.omega_quat[3] = pa.omega[2]

    # update each particles velocity using angular and linear velocity of com
    # of body
    vcm = pa.vcm

    omega = pa.omega
    for i in range(len(pa.x)):
        # velocity due to angular velocity is
        ang_vel = numpy.cross(omega,
                              numpy.array([pa.rx[i], pa.ry[i], pa.rz[i]]))
        pa.u[i] = vcm[0] + ang_vel[0]
        pa.v[i] = vcm[1] + ang_vel[1]
        pa.w[i] = vcm[2] + ang_vel[2]


class SumUpExternalForces(Equation):
    def reduce(self, dst, t, dt):
        rx = declare('object')
        ry = declare('object')
        rz = declare('object')
        fx = declare('object')
        fy = declare('object')
        fz = declare('object')
        torque = declare('object')
        force = declare('object')
        cond = declare('object')
        i = declare('int')
        m = declare('int')
        M = declare('int')
        nbody = declare('int')
        indices = declare('object')
        torque = dst.torque
        force = dst.force
        rx = dst.rx
        ry = dst.ry
        rz = dst.rz
        fx = dst.fx
        fy = dst.fy
        fz = dst.fz
        nbody = dst.num_body[0]

        for i in range(nbody):
            cond = dst.body_id == i
            indices = dst.indices[cond]
            m = min(indices)
            M = max(indices) + 1

            force[3 * i] = numpy.sum(fx[m:M])
            force[3 * i + 1] = numpy.sum(fy[m:M])
            force[3 * i + 2] = numpy.sum(fz[m:M])

            torque[3 * i] = numpy.sum(ry[m:M] * fz[m:M] - rz[m:M] * fy[m:M])
            torque[3 * i + 1] = numpy.sum(rz[m:M] * fx[m:M] -
                                          rx[m:M] * fz[m:M])
            torque[3 * i + 2] = numpy.sum(rx[m:M] * fy[m:M] -
                                          ry[m:M] * fx[m:M])


def quaternion_multiplication(q1=[1., 0.], q2=[1., 0.], result=[1., 0.]):
    result[0] = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3]
    result[1] = q1[0] * q2[1] + q2[0] * q1[1] + q1[2] * q2[3] - q2[2] * q1[3]
    result[2] = q1[0] * q2[2] + q2[0] * q1[2] + q2[1] * q1[3] - q1[1] * q2[3]
    result[3] = q1[0] * q2[3] + q2[0] * q1[3] + q1[1] * q2[2] - q2[1] * q1[2]


def quaternion_to_matrix(q=[0., 0.], A=[0., 0.]):
    A[0] = 1 - 2 * q[2] * q[2] - 2 * q[3] * q[3]
    A[1] = 2 * q[1] * q[2] - 2 * q[0] * q[3]
    A[2] = 2 * q[1] * q[3] + 2 * q[0] * q[2]
    A[3] = 2 * q[1] * q[2] + 2 * q[0] * q[3]
    A[4] = 1 - 2 * q[1] * q[1] - 2 * q[3] * q[3]
    A[5] = 2 * q[2] * q[3] - 2 * q[0] * q[1]
    A[6] = 2 * q[1] * q[3] - 2 * q[0] * q[2]
    A[7] = 2 * q[2] * q[3] + 2 * q[0] * q[1]
    A[8] = 1 - 2 * q[1] * q[1] - 2 * q[2] * q[2]


class RK2StepRigidBody(IntegratorStep):
    def initialize(self, d_idx, d_A, d_xcm, d_xcm0, d_vcm, d_vcm0, d_L, d_L0,
                   d_q, d_q0, d_num_body):
        _i = declare('int')
        _j = declare('int')
        if d_idx == 0:
            for _i in range(d_num_body[0]):
                for _j in range(3):
                    d_xcm0[3 * _i + _j] = d_xcm[3 * _i + _j]
                    d_vcm0[3 * _i + _j] = d_vcm[3 * _i + _j]
                    d_L0[3 * _i + _j] = d_L[3 * _i + _j]
                for _j in range(4):
                    d_q0[4 * _i + _j] = d_q[4 * _i + _j]

    def py_stage1(self, dst, t, dt):
        dtb2 = dt / 2.
        for i in range(dst.num_body[0]):
            i3 = 3 * i
            i4 = 4 * i
            i9 = 9 * i
            # update center of mass position and velocity
            dst.xcm[i3:i3 +
                    3] = dst.xcm0[i3:i3 + 3] + dtb2 * dst.vcm[i3:i3 + 3]
            dst.vcm[i3:i3 + 3] = dst.vcm0[
                i3:i3 + 3] + dtb2 * dst.force[i3:i3 + 3] / dst.total_mass[i]

            # update orientation
            quaternion_multiplication(dst.omega_quat[i4:i4 + 4],
                                      dst.q[i4:i4 + 4], dst.q_dot[i4:i4 + 4])
            dst.q[i4:14+4] = dst.q0[i4:14+4] + dtb2 * 0.5 * dst.q_dot[i4:14+4]

            # normalize the orientation
            q_magn = numpy.linalg.norm(dst.q[i4:i4 + 4])
            dst.q[i4:i4 + 4] = dst.q[i4:i4 + 4] / q_magn
            quaternion_to_matrix(dst.q[i4:i4 + 4], dst.A[i9:i9 + 9])

            # update angular momentum
            dst.L[i3:i3 + 3] = dst.L0[i3:i3 + 3] + dtb2 * dst.torque[i3:i3 + 3]

            # update angular velocity
            # omega = A * I^{body}^{-1} * R^{T} * L

            # tmp = R^T * L
            A = dst.A[i9:i9 + 9]
            L = dst.L[i3:i3 + 3]
            omega = dst.omega[i3:i3 + 3]
            mi_b_inv = dst.mi_b_inv[i9:i9 + 9]
            tmp0 = A[0] * L[0] + A[3] * L[1] + A[6] * L[2]
            tmp1 = A[1] * L[0] + A[4] * L[1] + A[7] * L[2]
            tmp2 = A[2] * L[0] + A[5] * L[1] + A[8] * L[2]

            # tmp = I^{body}^{-1} * tmp
            tmp0 = mi_b_inv[0] * tmp0 + mi_b_inv[1] * tmp1 + mi_b_inv[2] * tmp2
            tmp1 = mi_b_inv[3] * tmp0 + mi_b_inv[4] * tmp1 + mi_b_inv[5] * tmp2
            tmp2 = mi_b_inv[6] * tmp0 + mi_b_inv[7] * tmp1 + mi_b_inv[8] * tmp2

            dst.omega[i3] = A[0] * tmp0 + A[1] * tmp1 + A[2] * tmp2
            dst.omega[i3 + 1] = A[3] * tmp0 + A[4] * tmp1 + A[5] * tmp2
            dst.omega[i3 + 2] = A[6] * tmp0 + A[7] * tmp1 + A[8] * tmp2

            dst.omega_quat[i4] = 0
            dst.omega_quat[i4 + 1] = dst.omega[i3]
            dst.omega_quat[i4 + 2] = dst.omega[i3 + 1]
            dst.omega_quat[i4 + 3] = dst.omega[i3 + 2]

            # -------------------------
            # update particle positions
            cond = dst.body_id == i
            indices = dst.indices[cond]
            m = min(indices)
            M = max(indices) + 1
            dst.rx[m:M] = A[0] * dst.rbx[m:M] + A[1] * dst.rby[m:M] + A[
                2] * dst.rbz[m:M]
            dst.ry[m:M] = A[3] * dst.rbx[m:M] + A[4] * dst.rby[m:M] + A[
                5] * dst.rbz[m:M]
            dst.rz[m:M] = A[6] * dst.rbx[m:M] + A[7] * dst.rby[m:M] + A[
                8] * dst.rbz[m:M]
            # update position of particle in global frame from origin
            dst.x[m:M] = dst.xcm[i3] + dst.rx[m:M]
            dst.y[m:M] = dst.xcm[i3 + 1] + dst.ry[m:M]
            dst.z[m:M] = dst.xcm[i3 + 2] + dst.rz[m:M]

            dst.u[m:M] = dst.vcm[i3] + (
                omega[1] * dst.rz[m:M] - omega[2] * dst.ry[m:M])
            dst.v[m:M] = dst.vcm[i3 + 1] + (
                omega[2] * dst.rx[m:M] - omega[0] * dst.rz[m:M])
            dst.w[m:M] = dst.vcm[i3 + 2] + (
                omega[0] * dst.ry[m:M] - omega[1] * dst.rx[m:M])

    def stage1(self, d_idx, d_u, d_v, d_w, d_x, d_y, d_z, d_omega, d_vcm, d_rx,
               d_ry, d_rz, d_rbx, d_rby, d_rbz, d_xcm, d_A, dt=0.0):
        pass

    def py_stage2(self, dst, t, dt):
        # update orientation
        for i in range(dst.num_body[0]):
            i3 = 3 * i
            i4 = 4 * i
            i9 = 9 * i
            # update center of mass position and velocity
            dst.xcm[i3:i3 + 3] = dst.xcm0[i3:i3 + 3] + dt * dst.vcm[i3:i3 + 3]
            dst.vcm[i3:i3 + 3] = dst.vcm0[
                i3:i3 + 3] + dt * dst.force[i3:i3 + 3] / dst.total_mass[i]
            quaternion_multiplication(dst.omega_quat[i4:i4 + 4],
                                      dst.q[i4:i4 + 4], dst.q_dot[i4:i4 + 4])
            dst.q[i4:14+4] = dst.q0[i4:14+4] + dt * 0.5 * dst.q_dot[i4:14+4]

            # normalize the orientation
            q_magn = numpy.linalg.norm(dst.q[i4:i4 + 4])
            dst.q[i4:i4 + 4] = dst.q[i4:i4 + 4] / q_magn
            quaternion_to_matrix(dst.q[i4:i4 + 4], dst.A[i9:i9 + 9])

            # update angular momentum
            dst.L[i3:i3 + 3] = dst.L0[i3:i3 + 3] + dt * dst.torque[i3:i3 + 3]

            # update angular velocity
            # omega = A * I^{body}^{-1} * R^{T} * L

            # tmp = R^T * L
            A = dst.A[i9:i9 + 9]
            L = dst.L[i3:i3 + 3]
            omega = dst.omega[i3:i3 + 3]
            mi_b_inv = dst.mi_b_inv[i9:i9 + 9]
            tmp0 = A[0] * L[0] + A[3] * L[1] + A[6] * L[2]
            tmp1 = A[1] * L[0] + A[4] * L[1] + A[7] * L[2]
            tmp2 = A[2] * L[0] + A[5] * L[1] + A[8] * L[2]

            # tmp = I^{body}^{-1} * tmp
            tmp0 = mi_b_inv[0] * tmp0 + mi_b_inv[1] * tmp1 + mi_b_inv[2] * tmp2
            tmp1 = mi_b_inv[3] * tmp0 + mi_b_inv[4] * tmp1 + mi_b_inv[5] * tmp2
            tmp2 = mi_b_inv[6] * tmp0 + mi_b_inv[7] * tmp1 + mi_b_inv[8] * tmp2

            dst.omega[i3] = A[0] * tmp0 + A[1] * tmp1 + A[2] * tmp2
            dst.omega[i3 + 1] = A[3] * tmp0 + A[4] * tmp1 + A[5] * tmp2
            dst.omega[i3 + 2] = A[6] * tmp0 + A[7] * tmp1 + A[8] * tmp2

            dst.omega_quat[i4] = 0
            dst.omega_quat[i4 + 1] = dst.omega[i3]
            dst.omega_quat[i4 + 2] = dst.omega[i3 + 1]
            dst.omega_quat[i4 + 3] = dst.omega[i3 + 2]

            # -------------------------
            # update particle positions
            cond = dst.body_id == i
            indices = dst.indices[cond]
            m = min(indices)
            M = max(indices) + 1
            dst.rx[m:M] = A[0] * dst.rbx[m:M] + A[1] * dst.rby[m:M] + A[
                2] * dst.rbz[m:M]
            dst.ry[m:M] = A[3] * dst.rbx[m:M] + A[4] * dst.rby[m:M] + A[
                5] * dst.rbz[m:M]
            dst.rz[m:M] = A[6] * dst.rbx[m:M] + A[7] * dst.rby[m:M] + A[
                8] * dst.rbz[m:M]
            # update position of particle in global frame from origin
            dst.x[m:M] = dst.xcm[i3] + dst.rx[m:M]
            dst.y[m:M] = dst.xcm[i3 + 1] + dst.ry[m:M]
            dst.z[m:M] = dst.xcm[i3 + 2] + dst.rz[m:M]

            dst.u[m:M] = dst.vcm[i3] + (
                omega[1] * dst.rz[m:M] - omega[2] * dst.ry[m:M])
            dst.v[m:M] = dst.vcm[i3 + 1] + (
                omega[2] * dst.rx[m:M] - omega[0] * dst.rz[m:M])
            dst.w[m:M] = dst.vcm[i3 + 2] + (
                omega[0] * dst.ry[m:M] - omega[1] * dst.rx[m:M])

    def stage2(self, d_idx, d_u, d_v, d_w, d_x, d_y, d_z, d_omega, d_vcm, d_rx,
               d_ry, d_rz, d_rbx, d_rby, d_rbz, d_xcm, d_A, dt=0.0):
        pass


class EulerStepRigidBody(IntegratorStep):
    def py_stage1(self, dst, t, dt):
        for i in range(dst.num_body[0]):
            i3 = 3 * i
            i4 = 4 * i
            i9 = 9 * i
            # update center of mass position and velocity
            dst.xcm[i3:i3 + 3] = dst.xcm0[i3:i3 + 3] + dt * dst.vcm[i3:i3 + 3]
            dst.vcm[i3:i3 + 3] = dst.vcm0[
                i3:i3 + 3] + dt * dst.force[i3:i3 + 3] / dst.total_mass[i]

            # update orientation
            quaternion_multiplication(dst.omega_quat[i4:i4 + 4],
                                      dst.q[i4:i4 + 4], dst.q_dot[i4:i4 + 4])
            dst.q[i4] = dst.q[i4] + dt * dst.q_dot[i4] * 0.5
            dst.q[i4 + 1] = dst.q[i4 + 1] + dt * dst.q_dot[i4 + 1] * 0.5
            dst.q[i4 + 2] = dst.q[i4 + 2] + dt * dst.q_dot[i4 + 2] * 0.5
            dst.q[i4 + 3] = dst.q[i4 + 3] + dt * dst.q_dot[i4 + 3] * 0.5

            # normalize the orientation
            q_magn = numpy.linalg.norm(dst.q[i4:i4 + 4])
            dst.q[i4:i4 + 4] = dst.q[i4:i4 + 4] / q_magn
            quaternion_to_matrix(dst.q[i4:i4 + 4], dst.A[i9:i9 + 9])

            # update angular momentum
            dst.L[i3:i3 + 3] = dst.L[i3:i3 + 3] + dt * dst.torque[i3:i3 + 3]

            # update angular velocity
            # omega = A * I^{body}^{-1} * R^{T} * L

            # tmp = R^T * L
            A = dst.A[i9:i9 + 9]
            L = dst.L[i3:i3 + 3]
            omega = dst.omega[i3:i3 + 3]
            mi_b_inv = dst.mi_b_inv[i9:i9 + 9]
            tmp0 = A[0] * L[0] + A[3] * L[1] + A[6] * L[2]
            tmp1 = A[1] * L[0] + A[4] * L[1] + A[7] * L[2]
            tmp2 = A[2] * L[0] + A[5] * L[1] + A[8] * L[2]

            # tmp = I^{body}^{-1} * tmp
            tmp0 = mi_b_inv[0] * tmp0 + mi_b_inv[1] * tmp1 + mi_b_inv[2] * tmp2
            tmp1 = mi_b_inv[3] * tmp0 + mi_b_inv[4] * tmp1 + mi_b_inv[5] * tmp2
            tmp2 = mi_b_inv[6] * tmp0 + mi_b_inv[7] * tmp1 + mi_b_inv[8] * tmp2

            dst.omega[i3] = A[0] * tmp0 + A[1] * tmp1 + A[2] * tmp2
            dst.omega[i3 + 1] = A[3] * tmp0 + A[4] * tmp1 + A[5] * tmp2
            dst.omega[i3 + 2] = A[6] * tmp0 + A[7] * tmp1 + A[8] * tmp2

            dst.omega_quat[i4] = 0
            dst.omega_quat[i4 + 1] = dst.omega[i3]
            dst.omega_quat[i4 + 2] = dst.omega[i3 + 1]
            dst.omega_quat[i4 + 3] = dst.omega[i3 + 2]

            # -------------------------
            # update particle positions
            cond = dst.body_id == i
            indices = dst.indices[cond]
            m = min(indices)
            M = max(indices) + 1
            dst.rx[m:M] = A[0] * dst.rbx[m:M] + A[1] * dst.rby[m:M] + A[
                2] * dst.rbz[m:M]
            dst.ry[m:M] = A[3] * dst.rbx[m:M] + A[4] * dst.rby[m:M] + A[
                5] * dst.rbz[m:M]
            dst.rz[m:M] = A[6] * dst.rbx[m:M] + A[7] * dst.rby[m:M] + A[
                8] * dst.rbz[m:M]
            # update position of particle in global frame from origin
            dst.x[m:M] = dst.xcm[i3] + dst.rx[m:M]
            dst.y[m:M] = dst.xcm[i3 + 1] + dst.ry[m:M]
            dst.z[m:M] = dst.xcm[i3 + 2] + dst.rz[m:M]

            dst.u[m:M] = dst.vcm[i3] + (
                omega[1] * dst.rz[m:M] - omega[2] * dst.ry[m:M])
            dst.v[m:M] = dst.vcm[i3 + 1] + (
                omega[2] * dst.rx[m:M] - omega[0] * dst.rz[m:M])
            dst.w[m:M] = dst.vcm[i3 + 2] + (
                omega[0] * dst.ry[m:M] - omega[1] * dst.rx[m:M])

    def stage1(self, d_idx, d_u, d_v, d_w, d_x, d_y, d_z, d_omega, d_vcm, d_rx,
               d_ry, d_rz, d_rbx, d_rby, d_rbz, d_xcm, d_A, dt=0.0):
        pass
