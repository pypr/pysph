import numpy as np
import pytest

try:
    import setuptools  # noqa: F401 - keeps distutils importable on Python 3.14.
except Exception:
    pass

pytest.importorskip('warp')

from cyarray.carray import UIntArray

from pysph.base.kernels import CubicSpline
from pysph.base.nnps import LinkedListNNPS
from pysph.base.utils import get_particle_array
from pysph.base.warp_nnps import UniformGridWarpNNPS
from pysph.base.warp_sph import (
    compute_artificial_viscosity, compute_continuity, compute_isothermal_eos,
    compute_pressure_gradient, compute_summation_density, compute_tait_eos,
    euler_step, leapfrog_drift, leapfrog_kick, wc_sph_euler_step,
    wc_sph_leapfrog_step, wrap_periodic
)


def _neighbors(nnps, src_index, dst_index, d_idx):
    nbrs = UIntArray()
    nnps.get_nearest_particles(src_index, dst_index, d_idx, nbrs)
    return nbrs.get_npy_array()[:nbrs.length]


def _cpu_summation_density(particles, src_index, dst_index, dim,
                           radius_scale=2.0):
    nnps = LinkedListNNPS(
        dim=dim, particles=particles, radius_scale=radius_scale
    )
    kernel = CubicSpline(dim=dim)
    src = particles[src_index]
    dst = particles[dst_index]
    result = np.zeros(dst.get_number_of_particles())

    for d_idx in range(dst.get_number_of_particles()):
        total = 0.0
        for s_idx in _neighbors(nnps, src_index, dst_index, d_idx):
            xij = [
                dst.x[d_idx] - src.x[s_idx],
                0.0,
                0.0,
            ]
            if dim > 1:
                xij[1] = dst.y[d_idx] - src.y[s_idx]
            if dim > 2:
                xij[2] = dst.z[d_idx] - src.z[s_idx]
            rij = np.sqrt(xij[0]**2 + xij[1]**2 + xij[2]**2)
            hij = 0.5 * (dst.h[d_idx] + src.h[s_idx])
            total += src.m[s_idx] * kernel.kernel(xij=xij, rij=rij, h=hij)
        result[d_idx] = total
    return result


def _cpu_continuity(particles, src_index, dst_index, dim, radius_scale=2.0):
    nnps = LinkedListNNPS(
        dim=dim, particles=particles, radius_scale=radius_scale
    )
    kernel = CubicSpline(dim=dim)
    src = particles[src_index]
    dst = particles[dst_index]
    result = np.zeros(dst.get_number_of_particles())

    for d_idx in range(dst.get_number_of_particles()):
        total = 0.0
        for s_idx in _neighbors(nnps, src_index, dst_index, d_idx):
            xij = [
                dst.x[d_idx] - src.x[s_idx],
                0.0,
                0.0,
            ]
            vij = [
                dst.u[d_idx] - src.u[s_idx],
                0.0,
                0.0,
            ]
            if dim > 1:
                xij[1] = dst.y[d_idx] - src.y[s_idx]
                vij[1] = dst.v[d_idx] - src.v[s_idx]
            if dim > 2:
                xij[2] = dst.z[d_idx] - src.z[s_idx]
                vij[2] = dst.w[d_idx] - src.w[s_idx]
            rij = np.sqrt(xij[0]**2 + xij[1]**2 + xij[2]**2)
            hij = 0.5 * (dst.h[d_idx] + src.h[s_idx])
            dwij = [0.0, 0.0, 0.0]
            kernel.gradient(xij=xij, rij=rij, h=hij, grad=dwij)
            total += src.m[s_idx] * (
                vij[0]*dwij[0] + vij[1]*dwij[1] + vij[2]*dwij[2]
            )
        result[d_idx] = total
    return result


def _cpu_pressure_gradient(particles, src_index, dst_index, dim,
                           radius_scale=2.0):
    nnps = LinkedListNNPS(
        dim=dim, particles=particles, radius_scale=radius_scale
    )
    kernel = CubicSpline(dim=dim)
    src = particles[src_index]
    dst = particles[dst_index]
    result = np.zeros((dst.get_number_of_particles(), 3))

    for d_idx in range(dst.get_number_of_particles()):
        acc = np.zeros(3)
        rhoi21 = 1.0/(dst.rho[d_idx]*dst.rho[d_idx])
        tmpi = dst.p[d_idx]*rhoi21
        for s_idx in _neighbors(nnps, src_index, dst_index, d_idx):
            xij = [
                dst.x[d_idx] - src.x[s_idx],
                0.0,
                0.0,
            ]
            if dim > 1:
                xij[1] = dst.y[d_idx] - src.y[s_idx]
            if dim > 2:
                xij[2] = dst.z[d_idx] - src.z[s_idx]
            rij = np.sqrt(xij[0]**2 + xij[1]**2 + xij[2]**2)
            hij = 0.5 * (dst.h[d_idx] + src.h[s_idx])
            dwij = [0.0, 0.0, 0.0]
            kernel.gradient(xij=xij, rij=rij, h=hij, grad=dwij)
            rhoj21 = 1.0/(src.rho[s_idx]*src.rho[s_idx])
            tmp = tmpi + src.p[s_idx]*rhoj21
            acc += -src.m[s_idx] * tmp * np.asarray(dwij)
        result[d_idx, :] = acc
    return result


def _cpu_tait_eos(rho, rho0, c0, gamma=7.0, p0=0.0):
    ratio = rho / rho0
    p = p0 + (rho0*c0*c0/gamma) * (ratio**gamma - 1.0)
    cs = c0 * ratio**(0.5 * (gamma - 1.0))
    return p, cs


def _cpu_artificial_viscosity(particles, src_index, dst_index, dim,
                              alpha, beta, c0, radius_scale=2.0):
    nnps = LinkedListNNPS(
        dim=dim, particles=particles, radius_scale=radius_scale
    )
    kernel = CubicSpline(dim=dim)
    src = particles[src_index]
    dst = particles[dst_index]
    result = np.zeros((dst.get_number_of_particles(), 3))
    src_has_cs = 'cs' in src.properties
    dst_has_cs = 'cs' in dst.properties

    for d_idx in range(dst.get_number_of_particles()):
        acc = np.zeros(3)
        for s_idx in _neighbors(nnps, src_index, dst_index, d_idx):
            xij = [
                dst.x[d_idx] - src.x[s_idx],
                0.0,
                0.0,
            ]
            vij = [
                dst.u[d_idx] - src.u[s_idx],
                0.0,
                0.0,
            ]
            if dim > 1:
                xij[1] = dst.y[d_idx] - src.y[s_idx]
                vij[1] = dst.v[d_idx] - src.v[s_idx]
            if dim > 2:
                xij[2] = dst.z[d_idx] - src.z[s_idx]
                vij[2] = dst.w[d_idx] - src.w[s_idx]
            vdotx = vij[0]*xij[0] + vij[1]*xij[1] + vij[2]*xij[2]
            if vdotx < 0.0:
                rij2 = xij[0]**2 + xij[1]**2 + xij[2]**2
                rij = np.sqrt(rij2)
                hij = 0.5 * (dst.h[d_idx] + src.h[s_idx])
                mu = hij * vdotx / (rij2 + 0.01*hij*hij)
                rhoij1 = 2.0 / (dst.rho[d_idx] + src.rho[s_idx])
                csi = dst.cs[d_idx] if dst_has_cs else c0
                csj = src.cs[s_idx] if src_has_cs else c0
                cij = 0.5 * (csi + csj)
                piij = (-alpha*cij*mu + beta*mu*mu) * rhoij1
                dwij = [0.0, 0.0, 0.0]
                kernel.gradient(xij=xij, rij=rij, h=hij, grad=dwij)
                acc += -src.m[s_idx] * piij * np.asarray(dwij)
        result[d_idx, :] = acc
    return result


def test_warp_isothermal_eos_matches_cpu_and_pulls_pressure():
    pa = get_particle_array(
        name='fluid',
        rho=[900.0, 1000.0, 1100.0],
        p=[0.0, 0.0, 0.0],
        backend='warp',
    )
    expected = 5.0 + 20.0*20.0*(pa.rho - 1000.0)

    actual = compute_isothermal_eos(pa, rho0=1000.0, c0=20.0,
                                    p0=5.0).get()
    pa.gpu.pull('p')

    assert np.allclose(actual, expected)
    assert np.allclose(pa.p, expected)


def test_warp_tait_eos_matches_cpu_and_pulls_pressure_and_sound_speed():
    rho = np.asarray([0.9, 1.0, 1.1, 1.25])
    pa = get_particle_array(
        name='fluid',
        rho=rho,
        p=np.zeros_like(rho),
        cs=np.zeros_like(rho),
        backend='warp',
    )
    expected_p, expected_cs = _cpu_tait_eos(
        rho, rho0=1.0, c0=20.0, gamma=7.0, p0=0.5
    )

    p, cs = compute_tait_eos(
        pa, rho0=1.0, c0=20.0, gamma=7.0, p0=0.5
    )
    pa.gpu.pull('p', 'cs')

    assert np.allclose(p.get(), expected_p)
    assert np.allclose(cs.get(), expected_cs)
    assert np.allclose(pa.p, expected_p)
    assert np.allclose(pa.cs, expected_cs)


def test_warp_summation_density_matches_cpu_in_2d():
    pa = get_particle_array(
        name='fluid',
        x=[0.0, 0.2, 0.4, 1.5],
        y=[0.0, 0.0, 0.1, 1.5],
        z=[0.0, 0.0, 0.0, 0.0],
        h=[0.25, 0.25, 0.35, 0.25],
        m=[1.0, 2.0, 1.5, 1.0],
        backend='warp',
    )
    particles = [pa]
    expected = _cpu_summation_density(particles, 0, 0, dim=2)
    nnps = UniformGridWarpNNPS(dim=2, particles=particles, radius_scale=2.0)

    actual = compute_summation_density(nnps, 0, 0).get()

    assert np.allclose(actual, expected)


def test_warp_summation_density_matches_cpu_cross_array_in_3d_and_pulls_rho():
    fluid = get_particle_array(
        name='fluid',
        x=[0.0, 0.2, 0.4, 1.5],
        y=[0.0, 0.0, 0.1, 1.5],
        z=[0.0, 0.1, 0.0, 1.5],
        h=[0.25, 0.25, 0.35, 0.25],
        m=[1.0, 2.0, 1.5, 1.0],
        backend='warp',
    )
    solid = get_particle_array(
        name='solid',
        x=[0.1, 0.8],
        y=[0.0, 0.0],
        z=[0.0, 0.0],
        h=[0.25, 0.25],
        m=[1.0, 1.0],
        backend='warp',
    )
    particles = [fluid, solid]
    expected = _cpu_summation_density(particles, 0, 1, dim=3)
    nnps = UniformGridWarpNNPS(dim=3, particles=particles, radius_scale=2.0)

    compute_summation_density(nnps, 0, 1)
    solid.gpu.pull('rho')

    assert np.allclose(solid.rho, expected)


def test_warp_continuity_matches_cpu_in_2d():
    pa = get_particle_array(
        name='fluid',
        x=[0.0, 0.2, 0.4, 1.5],
        y=[0.0, 0.0, 0.1, 1.5],
        z=[0.0, 0.0, 0.0, 0.0],
        h=[0.25, 0.25, 0.35, 0.25],
        m=[1.0, 2.0, 1.5, 1.0],
        u=[1.0, 0.5, -0.5, 0.0],
        v=[0.0, 0.2, 0.4, -0.1],
        w=[0.0, 0.0, 0.0, 0.0],
        arho=[0.0, 0.0, 0.0, 0.0],
        backend='warp',
    )
    particles = [pa]
    expected = _cpu_continuity(particles, 0, 0, dim=2)
    nnps = UniformGridWarpNNPS(dim=2, particles=particles, radius_scale=2.0)

    actual = compute_continuity(nnps, 0, 0).get()

    assert np.allclose(actual, expected)


def test_warp_continuity_matches_cpu_cross_array_in_3d_and_pulls_arho():
    fluid = get_particle_array(
        name='fluid',
        x=[0.0, 0.2, 0.4, 1.5],
        y=[0.0, 0.0, 0.1, 1.5],
        z=[0.0, 0.1, 0.0, 1.5],
        h=[0.25, 0.25, 0.35, 0.25],
        m=[1.0, 2.0, 1.5, 1.0],
        u=[1.0, 0.5, -0.5, 0.0],
        v=[0.0, 0.2, 0.4, -0.1],
        w=[0.1, 0.3, -0.2, 0.0],
        backend='warp',
    )
    solid = get_particle_array(
        name='solid',
        x=[0.1, 0.8],
        y=[0.0, 0.0],
        z=[0.0, 0.0],
        h=[0.25, 0.25],
        m=[1.0, 1.0],
        u=[-0.1, 0.2],
        v=[0.3, -0.2],
        w=[0.0, 0.1],
        arho=[0.0, 0.0],
        backend='warp',
    )
    particles = [fluid, solid]
    expected = _cpu_continuity(particles, 0, 1, dim=3)
    nnps = UniformGridWarpNNPS(dim=3, particles=particles, radius_scale=2.0)

    compute_continuity(nnps, 0, 1)
    solid.gpu.pull('arho')

    assert np.allclose(solid.arho, expected)


def test_warp_pressure_gradient_matches_cpu_in_2d():
    pa = get_particle_array(
        name='fluid',
        x=[0.0, 0.2, 0.4, 1.5],
        y=[0.0, 0.0, 0.1, 1.5],
        z=[0.0, 0.0, 0.0, 0.0],
        h=[0.25, 0.25, 0.35, 0.25],
        m=[1.0, 2.0, 1.5, 1.0],
        rho=[1.0, 1.1, 0.9, 1.2],
        p=[2.0, 3.0, 1.5, 0.5],
        au=[0.0, 0.0, 0.0, 0.0],
        av=[0.0, 0.0, 0.0, 0.0],
        aw=[0.0, 0.0, 0.0, 0.0],
        backend='warp',
    )
    particles = [pa]
    expected = _cpu_pressure_gradient(particles, 0, 0, dim=2)
    nnps = UniformGridWarpNNPS(dim=2, particles=particles, radius_scale=2.0)

    au, av, aw = compute_pressure_gradient(nnps, 0, 0)

    assert np.allclose(au.get(), expected[:, 0])
    assert np.allclose(av.get(), expected[:, 1])
    assert np.allclose(aw.get(), expected[:, 2])


def test_warp_pressure_gradient_matches_cpu_cross_array_in_3d_and_pulls_accel():
    fluid = get_particle_array(
        name='fluid',
        x=[0.0, 0.2, 0.4, 1.5],
        y=[0.0, 0.0, 0.1, 1.5],
        z=[0.0, 0.1, 0.0, 1.5],
        h=[0.25, 0.25, 0.35, 0.25],
        m=[1.0, 2.0, 1.5, 1.0],
        rho=[1.0, 1.1, 0.9, 1.2],
        p=[2.0, 3.0, 1.5, 0.5],
        backend='warp',
    )
    solid = get_particle_array(
        name='solid',
        x=[0.1, 0.8],
        y=[0.0, 0.0],
        z=[0.0, 0.0],
        h=[0.25, 0.25],
        m=[1.0, 1.0],
        rho=[1.05, 0.95],
        p=[2.5, 1.0],
        au=[0.0, 0.0],
        av=[0.0, 0.0],
        aw=[0.0, 0.0],
        backend='warp',
    )
    particles = [fluid, solid]
    expected = _cpu_pressure_gradient(particles, 0, 1, dim=3)
    nnps = UniformGridWarpNNPS(dim=3, particles=particles, radius_scale=2.0)

    compute_pressure_gradient(nnps, 0, 1)
    solid.gpu.pull('au', 'av', 'aw')

    assert np.allclose(solid.au, expected[:, 0])
    assert np.allclose(solid.av, expected[:, 1])
    assert np.allclose(solid.aw, expected[:, 2])


def test_warp_artificial_viscosity_matches_cpu_and_adds_to_acceleration():
    pa = get_particle_array(
        name='fluid',
        x=[0.0, 0.2, 0.45, 1.2],
        y=[0.0, 0.03, -0.02, 0.1],
        z=[0.0, 0.0, 0.0, 0.0],
        h=[0.35, 0.35, 0.4, 0.35],
        m=[1.0, 1.5, 1.2, 0.8],
        rho=[1.0, 1.1, 0.9, 1.2],
        cs=[4.0, 5.0, 6.0, 7.0],
        u=[1.0, -1.0, -0.2, 0.0],
        v=[0.0, 0.05, -0.1, 0.0],
        w=[0.0, 0.0, 0.0, 0.0],
        au=[0.5, -0.25, 0.1, 0.0],
        av=[0.0, 0.2, -0.1, 0.0],
        aw=[0.0, 0.0, 0.0, 0.0],
        backend='warp',
    )
    particles = [pa]
    alpha = 0.1
    beta = 0.2
    c0 = 5.0
    initial = np.column_stack([pa.au.copy(), pa.av.copy(), pa.aw.copy()])
    expected = initial + _cpu_artificial_viscosity(
        particles, 0, 0, dim=2, alpha=alpha, beta=beta, c0=c0
    )
    nnps = UniformGridWarpNNPS(dim=2, particles=particles, radius_scale=2.0)

    au, av, aw = compute_artificial_viscosity(
        nnps, 0, 0, alpha=alpha, beta=beta, c0=c0
    )

    assert np.allclose(au.get(), expected[:, 0])
    assert np.allclose(av.get(), expected[:, 1])
    assert np.allclose(aw.get(), expected[:, 2])


def test_warp_wc_sph_euler_step_with_tait_eos_uses_sound_speed_in_avisc():
    x = np.asarray([0.0, 0.2, 0.45, 1.2])
    y = np.asarray([0.0, 0.1, -0.05, 0.2])
    z = np.zeros_like(x)
    h = np.asarray([0.35, 0.35, 0.4, 0.35])
    m = np.asarray([1.0, 1.5, 1.2, 0.8])
    u = np.asarray([0.5, -0.4, 0.2, 0.0])
    v = np.asarray([0.0, 0.15, -0.1, 0.05])
    w = np.zeros_like(x)
    dt = 1.0e-3
    rho0 = 1.0
    c0 = 5.0
    gamma = 7.0
    p0 = 0.1
    alpha = 0.1
    beta = 0.2
    pa = get_particle_array(
        name='fluid',
        x=x.copy(),
        y=y.copy(),
        z=z.copy(),
        h=h.copy(),
        m=m.copy(),
        rho=np.zeros_like(x),
        p=np.zeros_like(x),
        cs=np.zeros_like(x),
        u=u.copy(),
        v=v.copy(),
        w=w.copy(),
        au=np.zeros_like(x),
        av=np.zeros_like(x),
        aw=np.zeros_like(x),
        backend='warp',
    )
    particles = [pa]
    expected_rho = _cpu_summation_density(particles, 0, 0, dim=2)
    expected_p, expected_cs = _cpu_tait_eos(
        expected_rho, rho0=rho0, c0=c0, gamma=gamma, p0=p0
    )
    expected_pa = get_particle_array(
        name='expected',
        x=x.copy(),
        y=y.copy(),
        z=z.copy(),
        h=h.copy(),
        m=m.copy(),
        rho=expected_rho,
        p=expected_p,
        cs=expected_cs,
        u=u.copy(),
        v=v.copy(),
        w=w.copy(),
        backend='warp',
    )
    expected_acc = (
        _cpu_pressure_gradient([expected_pa], 0, 0, dim=2) +
        _cpu_artificial_viscosity(
            [expected_pa], 0, 0, dim=2, alpha=alpha, beta=beta, c0=c0
        )
    )
    expected_u = u + dt*expected_acc[:, 0]
    expected_v = v + dt*expected_acc[:, 1]
    expected_w = w + dt*expected_acc[:, 2]

    nnps = UniformGridWarpNNPS(dim=2, particles=particles, radius_scale=2.0)
    wc_sph_euler_step(
        nnps, dt=dt, rho0=rho0, c0=c0, p0=p0, alpha=alpha, beta=beta,
        eos='tait', gamma=gamma
    )
    pa.gpu.pull('rho', 'p', 'cs', 'au', 'av', 'aw', 'u', 'v', 'w')

    assert np.all(np.isfinite(pa.cs))
    assert np.allclose(pa.rho, expected_rho)
    assert np.allclose(pa.p, expected_p)
    assert np.allclose(pa.cs, expected_cs)
    assert np.allclose(pa.au, expected_acc[:, 0], rtol=1e-5, atol=1e-5)
    assert np.allclose(pa.av, expected_acc[:, 1], rtol=1e-5, atol=1e-5)
    assert np.allclose(pa.aw, expected_acc[:, 2], rtol=1e-5, atol=1e-5)
    assert np.allclose(pa.u, expected_u, rtol=1e-5, atol=1e-5)
    assert np.allclose(pa.v, expected_v, rtol=1e-5, atol=1e-5)
    assert np.allclose(pa.w, expected_w, rtol=1e-5, atol=1e-5)


def test_warp_euler_step_updates_velocity_and_position_on_device():
    pa = get_particle_array(
        name='fluid',
        x=[0.0, 1.0, 2.0],
        y=[0.5, 1.5, 2.5],
        z=[1.0, 2.0, 3.0],
        u=[1.0, -1.0, 0.5],
        v=[0.0, 2.0, -0.5],
        w=[0.25, -0.25, 1.0],
        au=[0.1, 0.2, -0.3],
        av=[-0.2, 0.4, 0.1],
        aw=[0.5, -0.5, 0.25],
        backend='warp',
    )
    dt = 0.25
    old_x = pa.x.copy()
    old_y = pa.y.copy()
    old_z = pa.z.copy()
    old_u = pa.u.copy()
    old_v = pa.v.copy()
    old_w = pa.w.copy()
    au = pa.au.copy()
    av = pa.av.copy()
    aw = pa.aw.copy()

    euler_step(pa, dt=dt, dim=2)
    pa.gpu.pull('x', 'y', 'z', 'u', 'v', 'w')

    expected_u = old_u + dt*au
    expected_v = old_v + dt*av
    expected_w = old_w + dt*aw
    assert np.allclose(pa.u, expected_u)
    assert np.allclose(pa.v, expected_v)
    assert np.allclose(pa.w, expected_w)
    assert np.allclose(pa.x, old_x + dt*expected_u)
    assert np.allclose(pa.y, old_y + dt*expected_v)
    assert np.allclose(pa.z, old_z)


def test_warp_leapfrog_kick_drift_and_wrap_update_device_state():
    pa = get_particle_array(
        name='fluid',
        x=[0.95, -0.2, 1.8],
        y=[0.9, 1.2, -0.1],
        z=[0.0, 0.0, 0.0],
        u=[0.4, 0.5, -0.25],
        v=[0.3, -0.4, 0.1],
        w=[0.0, 0.0, 0.0],
        au=[0.2, -0.1, 0.4],
        av=[-0.3, 0.2, 0.0],
        aw=[0.0, 0.0, 0.0],
        backend='warp',
    )
    dt = 0.5
    expected_u = pa.u + 0.5*dt*pa.au
    expected_v = pa.v + 0.5*dt*pa.av
    expected_x = pa.x + dt*expected_u
    expected_y = pa.y + dt*expected_v
    expected_x = expected_x - np.floor(expected_x)
    expected_y = expected_y - np.floor(expected_y)

    leapfrog_kick(pa, dt=0.5*dt, dim=2)
    leapfrog_drift(pa, dt=dt, dim=2, push=False)
    wrap_periodic(
        pa,
        {'xmin': 0.0, 'xmax': 1.0, 'ymin': 0.0, 'ymax': 1.0},
        dim=2,
    )
    pa.gpu.pull('x', 'y', 'z', 'u', 'v', 'w')

    assert np.allclose(pa.u, expected_u)
    assert np.allclose(pa.v, expected_v)
    assert np.allclose(pa.x, expected_x)
    assert np.allclose(pa.y, expected_y)
    assert np.allclose(pa.z, np.zeros(3))


def test_warp_wc_sph_euler_step_matches_cpu_expected_state():
    x = np.asarray([0.0, 0.2, 0.45, 1.2])
    y = np.asarray([0.0, 0.1, -0.05, 0.2])
    z = np.zeros_like(x)
    h = np.asarray([0.35, 0.35, 0.4, 0.35])
    m = np.asarray([1.0, 1.5, 1.2, 0.8])
    u = np.asarray([0.1, -0.05, 0.2, 0.0])
    v = np.asarray([0.0, 0.15, -0.1, 0.05])
    w = np.zeros_like(x)
    dt = 1.0e-3
    rho0 = 1.0
    c0 = 5.0
    p0 = 0.1
    pa = get_particle_array(
        name='fluid',
        x=x.copy(),
        y=y.copy(),
        z=z.copy(),
        h=h.copy(),
        m=m.copy(),
        rho=np.zeros_like(x),
        p=np.zeros_like(x),
        u=u.copy(),
        v=v.copy(),
        w=w.copy(),
        au=np.zeros_like(x),
        av=np.zeros_like(x),
        aw=np.zeros_like(x),
        backend='warp',
    )
    particles = [pa]
    expected_rho = _cpu_summation_density(particles, 0, 0, dim=2)
    expected_p = p0 + c0*c0*(expected_rho - rho0)
    expected_pa = get_particle_array(
        name='expected',
        x=x.copy(),
        y=y.copy(),
        z=z.copy(),
        h=h.copy(),
        m=m.copy(),
        rho=expected_rho,
        p=expected_p,
        backend='warp',
    )
    expected_acc = _cpu_pressure_gradient([expected_pa], 0, 0, dim=2)
    expected_u = u + dt*expected_acc[:, 0]
    expected_v = v + dt*expected_acc[:, 1]
    expected_w = w + dt*expected_acc[:, 2]
    expected_x = x + dt*expected_u
    expected_y = y + dt*expected_v

    nnps = UniformGridWarpNNPS(dim=2, particles=particles, radius_scale=2.0)
    wc_sph_euler_step(nnps, dt=dt, rho0=rho0, c0=c0, p0=p0)
    pa.gpu.pull('rho', 'p', 'au', 'av', 'aw', 'x', 'y', 'z', 'u', 'v', 'w')

    assert np.all(np.isfinite(pa.rho))
    assert np.all(np.isfinite(pa.p))
    assert np.all(np.isfinite(pa.au))
    assert np.allclose(pa.rho, expected_rho)
    assert np.allclose(pa.p, expected_p)
    assert np.allclose(pa.au, expected_acc[:, 0], rtol=1e-5, atol=1e-5)
    assert np.allclose(pa.av, expected_acc[:, 1], rtol=1e-5, atol=1e-5)
    assert np.allclose(pa.aw, expected_acc[:, 2], rtol=1e-5, atol=1e-5)
    assert np.allclose(pa.u, expected_u, rtol=1e-5, atol=1e-5)
    assert np.allclose(pa.v, expected_v, rtol=1e-5, atol=1e-5)
    assert np.allclose(pa.w, expected_w, rtol=1e-5, atol=1e-5)
    assert np.allclose(pa.x, expected_x, rtol=1e-5, atol=1e-5)
    assert np.allclose(pa.y, expected_y, rtol=1e-5, atol=1e-5)
    assert np.allclose(pa.z, z)


def test_warp_wc_sph_leapfrog_step_matches_cpu_expected_state():
    x = np.asarray([0.0, 0.2, 0.45, 1.2])
    y = np.asarray([0.0, 0.1, -0.05, 0.2])
    z = np.zeros_like(x)
    h = np.asarray([0.35, 0.35, 0.4, 0.35])
    m = np.asarray([1.0, 1.5, 1.2, 0.8])
    u = np.asarray([0.1, -0.05, 0.2, 0.0])
    v = np.asarray([0.0, 0.15, -0.1, 0.05])
    w = np.zeros_like(x)
    dt = 1.0e-3
    rho0 = 1.0
    c0 = 5.0
    p0 = 0.1
    pa = get_particle_array(
        name='fluid',
        x=x.copy(),
        y=y.copy(),
        z=z.copy(),
        h=h.copy(),
        m=m.copy(),
        rho=np.zeros_like(x),
        p=np.zeros_like(x),
        u=u.copy(),
        v=v.copy(),
        w=w.copy(),
        au=np.zeros_like(x),
        av=np.zeros_like(x),
        aw=np.zeros_like(x),
        backend='warp',
    )
    particles = [pa]
    rho_n = _cpu_summation_density(particles, 0, 0, dim=2)
    p_n = p0 + c0*c0*(rho_n - rho0)
    force_pa_n = get_particle_array(
        name='expected_n',
        x=x.copy(),
        y=y.copy(),
        z=z.copy(),
        h=h.copy(),
        m=m.copy(),
        rho=rho_n,
        p=p_n,
        backend='warp',
    )
    acc_n = _cpu_pressure_gradient([force_pa_n], 0, 0, dim=2)
    u_half = u + 0.5*dt*acc_n[:, 0]
    v_half = v + 0.5*dt*acc_n[:, 1]
    w_half = w + 0.5*dt*acc_n[:, 2]
    x_np1 = x + dt*u_half
    y_np1 = y + dt*v_half
    z_np1 = z + dt*w_half

    force_pa_np1 = get_particle_array(
        name='expected_np1',
        x=x_np1.copy(),
        y=y_np1.copy(),
        z=z_np1.copy(),
        h=h.copy(),
        m=m.copy(),
        rho=np.zeros_like(x),
        p=np.zeros_like(x),
        backend='warp',
    )
    rho_np1 = _cpu_summation_density([force_pa_np1], 0, 0, dim=2)
    p_np1 = p0 + c0*c0*(rho_np1 - rho0)
    force_pa_np1.rho[:] = rho_np1
    force_pa_np1.p[:] = p_np1
    acc_np1 = _cpu_pressure_gradient([force_pa_np1], 0, 0, dim=2)
    expected_u = u_half + 0.5*dt*acc_np1[:, 0]
    expected_v = v_half + 0.5*dt*acc_np1[:, 1]
    expected_w = w_half + 0.5*dt*acc_np1[:, 2]

    nnps = UniformGridWarpNNPS(dim=2, particles=particles, radius_scale=2.0)
    wc_sph_leapfrog_step(nnps, dt=dt, rho0=rho0, c0=c0, p0=p0)
    pa.gpu.pull('rho', 'p', 'au', 'av', 'aw', 'x', 'y', 'z', 'u', 'v', 'w')

    assert np.all(np.isfinite(pa.rho))
    assert np.all(np.isfinite(pa.p))
    assert np.all(np.isfinite(pa.au))
    assert np.allclose(pa.rho, rho_np1)
    assert np.allclose(pa.p, p_np1)
    assert np.allclose(pa.au, acc_np1[:, 0], rtol=1e-5, atol=1e-5)
    assert np.allclose(pa.av, acc_np1[:, 1], rtol=1e-5, atol=1e-5)
    assert np.allclose(pa.aw, acc_np1[:, 2], rtol=1e-5, atol=1e-5)
    assert np.allclose(pa.u, expected_u, rtol=1e-5, atol=1e-5)
    assert np.allclose(pa.v, expected_v, rtol=1e-5, atol=1e-5)
    assert np.allclose(pa.w, expected_w, rtol=1e-5, atol=1e-5)
    assert np.allclose(pa.x, x_np1, rtol=1e-5, atol=1e-5)
    assert np.allclose(pa.y, y_np1, rtol=1e-5, atol=1e-5)
    assert np.allclose(pa.z, z_np1, rtol=1e-5, atol=1e-5)
