# -*- coding: utf-8 -*-
"""Rigid body related equations.
"""
from pysph.base.reduce_array import serial_reduce_array, parallel_reduce_array
from pysph.sph.equation import Equation
from pysph.sph.integrator_step import IntegratorStep


def skew(vec):
    import sympy as S
    x, y, z = vec[0], vec[1], vec[2]
    return S.Matrix([[0, -z, y],[z, 0, -x], [-y, x, 0]])

def get_alpha_dot():
    """Use sympy to perform most of the math and use the resulting formulae
    to calculate:

            inv(I) (\tau - w x (I w))
    """
    import sympy as S
    ixx, iyy, izz, ixy, ixz, iyz = S.symbols("ixx, iyy, izz, ixy, ixz, iyz")
    tx, ty, tz = S.symbols("tx, ty, tz")
    wx, wy, wz = S.symbols('wx, wy, wz')
    tau = S.Matrix([tx, ty, tz])
    I = S.Matrix([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
    w = S.Matrix([wx, wy, wz])
    Iinv = I.inv()
    Iinv.simplify()
    # inv(I) (\tau - w x (Iw))
    res = Iinv*(tau - w.cross(I*w))
    res.simplify()
    # Now do some awesome sympy magic.
    syms, result = S.cse(res, symbols=S.numbered_symbols('tmp'))
    for lhs, rhs in syms:
        print("%s = %s"%(lhs, rhs))
    for i in range(3):
        print("omega_dot[%d] ="%i, result[0][i])


def get_torque():
    """Use sympy to perform some simple math.
        R x F
        C_m x F
        w x r
    """
    import sympy as S
    x, y, z, fx, fy, fz = S.symbols("x, y, z, fx, fy, fz")
    R = S.Matrix([x, y, z])
    F = S.Matrix([fx, fy, fz])
    print("Torque:", R.cross(F))
    cx, cy, cz = S.symbols('cx, cy, cz')
    d = S.Matrix([cx, cy, cz])
    print("c_m x f = ", d.cross(F))
    wx, wy, wz = S.symbols('wx, wy, wz')
    rx, ry, rz = S.symbols('rx, ry, rz')
    w = S.Matrix([wx, wy, wz])
    r = S.Matrix([rx, ry, rz])
    print("w x r = %s"%w.cross(r))

# This is defined to silence editor warnings for the use of declare.
def declare(*args): pass

class RigidBodyMoments(Equation):
    def initialize(self, d_idx, d_m, d_x, d_y, d_z, d_fx, d_fy, d_fz,
                   d_mi, d_num_body, d_body_id):
        nbody = declare('int')
        nbody = d_num_body[0]
        _i = declare('int')
        _j = declare('int')
        if d_idx == 0:
            for _i in range(nbody):
                for _j in range(16):
                    d_mi[_i*16 + _j] = 0.0

        base = declare('int')
        base = d_body_id[d_idx]*16

        # Find the total_mass, center of mass and second moments.
        m = d_m[d_idx]
        x = d_x[d_idx]
        y = d_y[d_idx]
        z = d_z[d_idx]
        d_mi[base] += m
        d_mi[base + 1] += m*x
        d_mi[base + 2] += m*y
        d_mi[base + 3] += m*z

        # Only do the lower triangle of values moments of inertia.
        d_mi[base + 4] += m*(y*y + z*z)
        d_mi[base + 5] += m*(x*x + z*z)
        d_mi[base + 6] += m*(x*x + y*y)

        d_mi[base + 7] -= m*x*y
        d_mi[base + 8] -= m*x*z
        d_mi[base + 9] -= m*y*z

        # the total force and torque
        fx = d_fx[d_idx]
        fy = d_fy[d_idx]
        fz = d_fz[d_idx]
        d_mi[base + 10] += fx
        d_mi[base + 11] += fy
        d_mi[base + 12] += fz

        # Calculate the torque and reduce it.
        d_mi[base + 13] += (y*fz - z*fy)
        d_mi[base + 14] += (z*fx - x*fz)
        d_mi[base + 15] += (x*fy - y*fx)

    def reduce(self, dst):
        # Reduce the temporary mi values in parallel across processors.
        dst.mi.set_data(parallel_reduce_array(dst.mi))

        # Set the reduced values.
        nbody = declare('int')
        i = declare('int')
        base_mi = declare('int')
        base = declare('int')
        nbody = dst.num_body.data[0]

        for i in range(nbody):
            base_mi = i*16
            base = i*3
            m = dst.mi.data[base_mi + 0]
            dst.total_mass.data[i] = m
            cx = dst.mi.data[base_mi + 1]/m
            cy = dst.mi.data[base_mi + 2]/m
            cz = dst.mi.data[base_mi + 3]/m
            dst.cm.data[base + 0] = cx
            dst.cm.data[base + 1] = cy
            dst.cm.data[base + 2] = cz

            # The actual moment of inertia about center of mass from parallel
            # axes theorem.
            ixx = dst.mi.data[base_mi + 4] - (cy*cy + cz*cz)*m
            iyy = dst.mi.data[base_mi + 5] - (cx*cx + cz*cz)*m
            izz = dst.mi.data[base_mi + 6] - (cx*cx + cy*cy)*m
            ixy = dst.mi.data[base_mi + 7] + cx*cy*m
            ixz = dst.mi.data[base_mi + 8] + cx*cz*m
            iyz = dst.mi.data[base_mi + 9] + cy*cz*m

            dst.mi.data[base_mi + 0] = ixx
            dst.mi.data[base_mi + 1] = ixy
            dst.mi.data[base_mi + 2] = ixz
            dst.mi.data[base_mi + 3] = ixy
            dst.mi.data[base_mi + 4] = iyy
            dst.mi.data[base_mi + 5] = iyz
            dst.mi.data[base_mi + 6] = ixz
            dst.mi.data[base_mi + 7] = iyz
            dst.mi.data[base_mi + 8] = izz

            fx = dst.mi.data[base_mi + 10]
            fy = dst.mi.data[base_mi + 11]
            fz = dst.mi.data[base_mi + 12]
            dst.force.data[base + 0] = fx
            dst.force.data[base + 1] = fy
            dst.force.data[base + 2] = fz

            # Acceleration of CM.
            dst.ac.data[base + 0] = fx/m
            dst.ac.data[base + 1] = fy/m
            dst.ac.data[base + 2] = fz/m

            # Find torque about the Center of Mass and not origin.
            tx = dst.mi.data[base_mi + 13]
            ty = dst.mi.data[base_mi + 14]
            tz = dst.mi.data[base_mi + 15]
            tx -= cy*fz - cz*fy
            ty -= -cx*fz + cz*fx
            tz -= cx*fy - cy*fx
            dst.torque.data[base + 0] = tx
            dst.torque.data[base + 1] = ty
            dst.torque.data[base + 2] = tz

            wx = dst.omega.data[base + 0]
            wy = dst.omega.data[base + 1]
            wz = dst.omega.data[base + 2]
            # Find omega_dot from: omega_dot = inv(I) (\tau - w x (Iw))
            # This was done using the sympy code above.
            tmp0 = iyz**2
            tmp1 = ixy**2
            tmp2 = ixz**2
            tmp3 = ixx*iyy
            tmp4 = ixy*ixz
            tmp5 = 1./(ixx*tmp0 + iyy*tmp2 - 2*iyz*tmp4 + izz*tmp1 - izz*tmp3)
            tmp6 = ixy*izz - ixz*iyz
            tmp7 = ixz*wx + iyz*wy + izz*wz
            tmp8 = ixx*wx + ixy*wy + ixz*wz
            tmp9 = tmp7*wx - tmp8*wz + ty
            tmp10 = ixy*iyz - ixz*iyy
            tmp11 = ixy*wx + iyy*wy + iyz*wz
            tmp12 = -tmp11*wx + tmp8*wy + tz
            tmp13 = tmp11*wz - tmp7*wy + tx
            tmp14 = ixx*iyz - tmp4
            dst.omega_dot.data[base + 0] = tmp5*(-tmp10*tmp12 - tmp13*(iyy*izz - tmp0) + tmp6*tmp9)
            dst.omega_dot.data[base + 1] = tmp5*(tmp12*tmp14 + tmp13*tmp6 - tmp9*(ixx*izz - tmp2))
            dst.omega_dot.data[base + 2] = tmp5*(-tmp10*tmp13 - tmp12*(-tmp1 + tmp3) + tmp14*tmp9)


class RigidBodyMotion(Equation):
    def initialize(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w,
                   d_cm, d_vc, d_ac, d_omega, d_body_id):
        base = declare('int')
        base = d_body_id[d_idx]*3
        wx = d_omega[base + 0]; wy = d_omega[base + 1]; wz = d_omega[base + 2]
        rx = d_x[d_idx] - d_cm[base + 0]
        ry = d_y[d_idx] - d_cm[base + 1]
        rz = d_z[d_idx] - d_cm[base + 2]

        d_u[d_idx] = d_vc[base + 0] + wy*rz - wz*ry
        d_v[d_idx] = d_vc[base + 1] + wz*rx - wx*rz
        d_w[d_idx] = d_vc[base + 2] + wx*ry - wy*rx

class BodyForce(Equation):
    def __init__(self, dest, sources, gx=0.0, gy=0.0, gz=0.0):
        self.gx = gx
        self.gy = gy
        self.gz = gz
        super(BodyForce, self).__init__(dest, sources)

    def initialize(self, d_idx, d_m, d_fx, d_fy, d_fz):
        d_fx[d_idx] = d_m[d_idx]*self.gx
        d_fy[d_idx] = d_m[d_idx]*self.gy
        d_fz[d_idx] = d_m[d_idx]*self.gz


class NumberDensity(Equation):
    def initialize(self, d_idx, d_V):
        d_V[d_idx] = 0.0

    def loop(self, d_idx, d_V, WIJ):
        d_V[d_idx] += WIJ

class SummationDensityRigidBody(Equation):
    def __init__(self, dest, sources, rho0):
        self.rho0 = rho0
        super(SummationDensityRigidBody, self).__init__(dest, sources)

    def initialize(self, d_idx, d_rho):
        d_rho[d_idx] = 0.0

    def loop(self, d_idx, d_rho, s_idx, s_V, WIJ):
        d_rho[d_idx] += self.rho0/s_V[s_idx]*WIJ


class ViscosityRigidBody(Equation):

    """The viscous acceleration on the fluid/solid due to a boundary.
    Implemented from Akinci et al. http://dx.doi.org/10.1145/2185520.2185558

    Use this with the fluid as a destination and body as source.
    """

    def __init__(self, dest, sources, rho0, nu):
        self.nu = nu
        self.rho0 = rho0
        super(ViscosityRigidBody, self).__init__(dest, sources)

    def loop(self, d_idx, d_m, d_au, d_av, d_aw, d_rho,
             s_idx, s_V, s_fx, s_fy, s_fz,
             EPS, VIJ, XIJ, R2IJ, DWIJ):
        phi_b = self.rho0/(s_V[s_idx]*d_rho[d_idx])
        vijdotxij = min(VIJ[0]*XIJ[0] + VIJ[1]*XIJ[1] + VIJ[2]*XIJ[2], 0.0)

        fac = self.nu*phi_b*vijdotxij/(R2IJ + EPS)
        ax = fac*DWIJ[0]
        ay = fac*DWIJ[1]
        az = fac*DWIJ[2]
        d_au[d_idx] += ax
        d_av[d_idx] += ay
        d_aw[d_idx] += az
        s_fx[s_idx] += -d_m[d_idx]*ax
        s_fy[s_idx] += -d_m[d_idx]*ay
        s_fz[s_idx] += -d_m[d_idx]*az

class PressureRigidBody(Equation):

    """The pressure acceleration on the fluid/solid due to a boundary.
    Implemented from Akinci et al. http://dx.doi.org/10.1145/2185520.2185558

    Use this with the fluid as a destination and body as source.
    """

    def __init__(self, dest, sources, rho0):
        self.rho0 = rho0
        super(PressureRigidBody, self).__init__(dest, sources)

    def loop(self, d_idx, d_m, d_rho, d_au, d_av, d_aw,  d_p,
             s_idx, s_V, s_fx, s_fy, s_fz, DWIJ):
        rho1 = 1.0/d_rho[d_idx]
        fac = -d_p[d_idx]*rho1*rho1*self.rho0/s_V[s_idx]
        ax = fac*DWIJ[0]
        ay = fac*DWIJ[1]
        az = fac*DWIJ[2]
        d_au[d_idx] += ax
        d_av[d_idx] += ay
        d_aw[d_idx] += az
        s_fx[s_idx] += -d_m[d_idx]*ax
        s_fy[s_idx] += -d_m[d_idx]*ay
        s_fz[s_idx] += -d_m[d_idx]*az

class RigidBodyCollision(Equation):
    """This is inspired from
    http://http.developer.nvidia.com/GPUGems3/gpugems3_ch29.html
    and

    BK Mishra's article on DEM
    http://dx.doi.org/10.1016/S0301-7516(03)00032-2

    A review of computer simulation of tumbling mills by the discrete element
    method: Part I - contact mechanics
    """
    def __init__(self, dest, sources, k=1.0, d=1.0, eta=1.0, kt=1.0):
        """Note that d is a factor multiplied with the "h" of the particle.
        """
        self.k = k
        self.d = d
        self.eta = eta
        self.kt = kt
        super(RigidBodyCollision, self).__init__(dest, sources)

    def loop(self, d_idx, d_fx, d_fy, d_fz, d_h, d_total_mass, XIJ, RIJ, R2IJ, VIJ):
        vijdotrij = VIJ[0]*XIJ[0] + VIJ[1]*XIJ[1] + VIJ[2]*XIJ[2]
        if RIJ > 1e-9:
            vijdotrij_r2ij = vijdotrij/R2IJ
            nij_x = XIJ[0]/RIJ
            nij_y = XIJ[1]/RIJ
            nij_z = XIJ[2]/RIJ
        else:
            vijdotrij_r2ij = 0.0
            nij_x = 0.0
            nij_y = 0.0
            nij_z = 0.0
        vijt_x = VIJ[0] - vijdotrij_r2ij*XIJ[0]
        vijt_y = VIJ[1] - vijdotrij_r2ij*XIJ[1]
        vijt_z = VIJ[2] - vijdotrij_r2ij*XIJ[2]

        d = self.d*d_h[d_idx]
        fac = self.k*d_total_mass[0]/d*max(d - RIJ, 0.0)

        d_fx[d_idx] += fac*nij_x - self.eta*VIJ[0] - self.kt*vijt_x
        d_fy[d_idx] += fac*nij_y - self.eta*VIJ[1] - self.kt*vijt_y
        d_fz[d_idx] += fac*nij_z - self.eta*VIJ[2] - self.kt*vijt_z


class EulerStepRigidBody(IntegratorStep):
    """Fast but inaccurate integrator. Use this for testing"""
    def initialize(self):
        pass
    def stage1(self, d_idx, d_u, d_v, d_w, d_x, d_y, d_z,
               d_omega, d_omega_dot, d_vc, d_ac, d_num_body,
               dt=0.0):
        _i = declare('int')
        _j = declare('int')
        base = declare('int')
        if d_idx == 0:
            for _i in range(d_num_body[0]):
                base = 3*_i
                for _j in range(3):
                    d_vc[base + _j] += d_ac[base + _j]*dt
                    d_omega[base + _j] += d_omega_dot[base + _j]*dt

        d_x[d_idx] += dt*d_u[d_idx]
        d_y[d_idx] += dt*d_v[d_idx]
        d_z[d_idx] += dt*d_w[d_idx]

class RK2StepRigidBody(IntegratorStep):
    def initialize(self, d_idx, d_x, d_y, d_z, d_x0, d_y0, d_z0,
                   d_omega, d_omega0, d_vc, d_vc0, d_num_body):
        _i = declare('int')
        _j = declare('int')
        base = declare('int')
        if d_idx == 0:
            for _i in range(d_num_body[0]):
                base = 3*_i
                for _j in range(3):
                    d_vc0[base + _j] = d_vc[base + _j]
                    d_omega0[base + _j] = d_omega[base + _j]

        d_x0[d_idx] = d_x[d_idx]
        d_y0[d_idx] = d_y[d_idx]
        d_z0[d_idx] = d_z[d_idx]

    def stage1(self, d_idx, d_u, d_v, d_w, d_x, d_y, d_z,d_x0, d_y0, d_z0,
               d_omega, d_omega_dot, d_vc, d_ac, d_omega0, d_vc0, d_num_body,
               dt=0.0):
        dtb2 = 0.5*dt
        _i = declare('int')
        j = declare('int')
        base = declare('int')
        if d_idx == 0:
            for _i in range(d_num_body[0]):
                base = 3*_i
                for j in range(3):
                    d_vc[base + j] = d_vc0[base + j] + d_ac[base + j]*dtb2
                    d_omega[base + j] = d_omega0[base + j] + d_omega_dot[base + j]*dtb2

        d_x[d_idx] = d_x0[d_idx] + dtb2*d_u[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dtb2*d_v[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dtb2*d_w[d_idx]

    def stage2(self, d_idx, d_u, d_v, d_w, d_x, d_y, d_z,d_x0, d_y0, d_z0,
               d_omega, d_omega_dot, d_vc, d_ac, d_omega0, d_vc0, d_num_body,
               dt=0.0):
        _i = declare('int')
        j = declare('int')
        base = declare('int')
        if d_idx == 0:
            for _i in range(d_num_body[0]):
                base = 3*_i
                for j in range(3):
                    d_vc[base + j] = d_vc0[base + j] + d_ac[base + j]*dt
                    d_omega[base + j] = d_omega0[base + j] + d_omega_dot[base + j]*dt

        d_x[d_idx] = d_x0[d_idx] + dt*d_u[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dt*d_v[d_idx]
        d_z[d_idx] = d_z0[d_idx] + dt*d_w[d_idx]
