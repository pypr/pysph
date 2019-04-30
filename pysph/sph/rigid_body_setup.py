import numpy as np


def set_total_mass(pa):
    # left limit of body i
    pa.total_mass[0] = np.sum(pa.m)
    assert pa.total_mass[0] > 0., "Total mass has to be greater than zero"


def set_center_of_mass(pa):
    pa.cm[0] = np.sum(pa.m * pa.x) / pa.total_mass[0]
    pa.cm[1] = np.sum(pa.m * pa.y) / pa.total_mass[0]
    pa.cm[2] = np.sum(pa.m * pa.z) / pa.total_mass[0]


def set_mi_in_body_frame(pa):
    """Compute the moment of inertia at the beginning of the simulation.
    And set moment of inertia inverse for further computations.
    This method assumes the center of mass is already computed."""
    cm_i = pa.cm

    I = np.zeros(9)
    for j in range(len(pa.x)):
        # Ixx
        I[0] += pa.m[j] * ((pa.y[j] - cm_i[1])**2. + (pa.z[j] - cm_i[2])**2.)

        # Iyy
        I[4] += pa.m[j] * ((pa.x[j] - cm_i[0])**2. + (pa.z[j] - cm_i[2])**2.)

        # Izz
        I[8] += pa.m[j] * ((pa.x[j] - cm_i[0])**2. + (pa.y[j] - cm_i[1])**2.)

        # Ixy
        I[1] -= pa.m[j] * (pa.x[j] - cm_i[0]) * (pa.y[j] - cm_i[1])

        # Ixz
        I[2] -= pa.m[j] * (pa.x[j] - cm_i[0]) * (pa.z[j] - cm_i[2])

        # Iyz
        I[5] -= pa.m[j] * (pa.y[j] - cm_i[1]) * (pa.z[j] - cm_i[2])

    I[3] = I[1]
    I[6] = I[2]
    I[7] = I[5]
    I_inv = np.linalg.inv(I.reshape(3, 3))
    I_inv = I_inv.ravel()
    pa.mib[0:9] = I_inv[:]


def get_mi_in_global_frame(pa):
    """Given particle array at an instant compute the global moment of inertia
    from the global positions."""
    cm_i = pa.cm

    I = np.zeros(9)
    for j in range(len(pa.x)):
        # Ixx
        I[0] += pa.m[j] * ((pa.y[j] - cm_i[1])**2. + (pa.z[j] - cm_i[2])**2.)

        # Iyy
        I[4] += pa.m[j] * ((pa.x[j] - cm_i[0])**2. + (pa.z[j] - cm_i[2])**2.)

        # Izz
        I[8] += pa.m[j] * ((pa.x[j] - cm_i[0])**2. + (pa.y[j] - cm_i[1])**2.)

        # Ixy
        I[1] -= pa.m[j] * (pa.x[j] - cm_i[0]) * (pa.y[j] - cm_i[1])

        # Ixz
        I[2] -= pa.m[j] * (pa.x[j] - cm_i[0]) * (pa.z[j] - cm_i[2])

        # Iyz
        I[5] -= pa.m[j] * (pa.y[j] - cm_i[1]) * (pa.z[j] - cm_i[2])

    I[3] = I[1]
    I[6] = I[2]
    I[7] = I[5]
    return I.reshape(3, 3)


def set_body_frame_position_vectors(pa):
    """Save the position vectors w.r.t body frame"""
    pa.dx0[:] = pa.x[:] - pa.cm[0]
    pa.dy0[:] = pa.y[:] - pa.cm[1]
    pa.dz0[:] = pa.z[:] - pa.cm[2]


def set_angular_momentum(pa):
    Ig = get_mi_in_global_frame(pa)

    # L = I omega
    pa.L = np.matmul(Ig, pa.omega)


def setup_rotation_matrix_rigid_body(pa):
    """Setup total mass, center of mass, moment of inertia and
    angular momentum of a rigid body defined using rotation matrices."""
    set_total_mass(pa)
    set_center_of_mass(pa)
    set_mi_in_body_frame(pa)
    set_body_frame_position_vectors(pa)
    set_angular_momentum(pa)


def setup_quaternion_rigid_body(pa):
    """Setup total mass, center of mass, moment of inertia and
    angular momentum of a rigid body defined using quaternion."""
    set_total_mass(pa)
    set_center_of_mass(pa)
    set_mi_in_body_frame(pa)
    set_body_frame_position_vectors(pa)
    set_angular_momentum(pa)
