"""Analytical material checks and compiled predictor/corrector regression."""
import inspect

import numpy as np
import pytest

from pysph.base.kernels import CubicSpline
from pysph.base.nnps import LinkedListNNPS
from pysph.sph.acceleration_eval import AccelerationEval
from pysph.sph.equation import Equation, Group
from pysph.sph.integrator import EPECIntegrator
from pysph.sph.solid_mech.basic import HookesDeviatoricStressRate
from pysph.sph.solid_mech.hjc import (
    HJCStep, get_particle_array_hjc, hjc_parameters, hjc_return)
from pysph.sph.sph_compiler import SPHCompiler


def parameters(**kw):
    # Simple units make the independent expected values transparent.
    values = dict(rho0=2., G=100., A=.3, B=1., C=.02, N=1., fc=10.,
                  T=1., edot0=1., efmin=.01, sfmax=5., pc=2., muc=.01,
                  pl=20., mul=.1, D1=.04, D2=1., K1=200., K2=0., K3=0.)
    values.update(kw)
    return hjc_parameters(**values)


def returned(p, q=0., mu=0., rate=1., history=(0., 0., 0., 0.)):
    result = np.zeros(8)
    hjc_return(q, mu, rate, *history, p, result)
    return result


@pytest.mark.parametrize('kw', [
    dict(G=0), dict(efmin=0), dict(T=-1), dict(C=-1),
    dict(K2=-1), dict(A=np.nan), dict(pl=1), dict(muc=.3)])
def test_invalid_parameters(kw):
    with pytest.raises(ValueError):
        parameters(**kw)


def test_elastic_pressure_and_unconfined_plastic_shear():
    p = parameters(C=0)
    r = returned(p, q=2., mu=.001)
    assert r[5] == pytest.approx(.2)
    assert r[0] == 1 and r[1] == 0
    r = returned(p, q=6.)
    assert r[7] == pytest.approx(3.)
    assert r[2] == pytest.approx(.01)
    # EFMIN is a floor, so zero-pressure plastic shear damages the material.
    assert r[1] == pytest.approx(1.)
    r = returned(p, q=3., history=(1., .01, 0., 0.))
    assert r[7] == 0 and r[2] == pytest.approx(.02)


def test_rate_cap_and_tensile_cutoff():
    p = parameters()
    r = returned(p, q=6., rate=np.e)
    assert r[7] == pytest.approx(3. * 1.02)
    assert returned(p, q=6., rate=.01)[7] == pytest.approx(3.)
    assert returned(p, q=1e4, mu=1.)[7] == pytest.approx(50.)
    r = returned(p, q=6., mu=-.001, history=(.2, 0., 0., 0.))
    assert r[7] == pytest.approx(3. * (.8 - .2))
    assert r[5] >= -p[7] * (1 - r[1])
    assert returned(p, mu=-.1)[5] == pytest.approx(-1.)


def test_crushing_unloading_and_dense_eos_continuity():
    p = parameters()
    peak = .11
    r = returned(p, mu=peak)
    assert r[3] == pytest.approx(.05)
    assert r[5] == pytest.approx(11.)
    assert r[1] == pytest.approx(min(1., .05 / (.04 * 1.2)))
    # The damage is bounded even under purely volumetric plastic loading.
    assert r[1] <= 1
    history = tuple(r[1:5])
    unloaded = returned(p, mu=.05, history=history)
    assert unloaded[5] == pytest.approx(0.)
    np.testing.assert_allclose(unloaded[1:5], r[1:5])
    for join in (p[12], p[20]):
        left = returned(p, mu=join - 1e-10)[5]
        right = returned(p, mu=join + 1e-10)[5]
        assert abs(left - right) < 1e-6
    locked = returned(p, mu=p[20])
    unloaded = returned(p, mu=p[14], history=tuple(locked[1:5]))
    assert unloaded[5] == pytest.approx(0.)


def test_simple_shear_softening_converges_to_closed_form():
    # q=A*fc*(1-D), D=ep/ef, q=3G*(e-ep). Once plastic, the
    # consistency solution is ep=(3G*e-A*fc)/(3G-A*fc/ef).
    p = parameters(C=0, efmin=.1)
    target = .025
    exact_ep = (300. * target - 3.) / (300. - 30.)
    errors = []
    for steps in (100, 200, 400):
        r = np.zeros(8)
        for _ in range(steps):
            r = returned(p, q=r[7] + 300. * target / steps,
                         history=tuple(r[1:5]))
        errors.append(abs(r[2] - exact_ep))
        assert 0 <= r[1] <= 1
    assert errors[2] < .55 * errors[1] < .31 * errors[0]


class PrescribedGradient(Equation):
    def initialize(self, d_idx, d_v01, d_v10, d_v02, d_v20,
                   d_v12, d_v21, d_loading):
        d_v01[d_idx] = d_loading[0]
        d_v10[d_idx] = d_loading[1]
        d_v02[d_idx] = d_loading[2]
        d_v20[d_idx] = d_loading[3]
        d_v12[d_idx] = d_loading[4]
        d_v21[d_idx] = d_loading[5]


def compiled_integrator(pa):
    integrator = EPECIntegrator(solid=HJCStep())
    equations = [Group([PrescribedGradient('solid', None)]),
                 Group([HookesDeviatoricStressRate('solid', None)])]
    kernel = CubicSpline(dim=3)
    evaluator = AccelerationEval([pa], equations, kernel, backend='cython')
    SPHCompiler(evaluator, integrator).compile()
    nnps = LinkedListNNPS(dim=3, particles=[pa])
    evaluator.set_nnps(nnps)
    integrator.set_nnps(nnps)
    return integrator


def particles(**kw):
    pa = get_particle_array_hjc(parameters(**kw), name='solid',
                                x=[0.], rho=[2.], m=[1.], h=[1.])
    pa.add_constant('loading', np.zeros(6))
    return pa


def call_stage(step, method, pa, dt):
    args = dict(d_idx=0, dt=dt)
    names = inspect.signature(getattr(step, method)).parameters
    for name in names:
        if name.startswith('d_') and name != 'd_idx':
            args[name] = pa.get(name[2:])
    return getattr(step, method)(**{k: args[k] for k in names})


def test_corrector_restarts_history_and_repeated_stage_is_idempotent():
    pa = particles(C=0, efmin=.1)
    step = HJCStep()
    pa.as01[:] = 100.
    pa.v01[:] = pa.v10[:] = .5
    call_stage(step, 'initialize', pa, .1)
    call_stage(step, 'stage1', pa, .1)
    assert pa.damage[0] > 0
    call_stage(step, 'stage2', pa, .1)
    # Full trial q=sqrt(3)*10; no predictor plasticity may be added again.
    ep = (np.sqrt(3) * 10. - 3.) / 300.
    assert pa.plastic_strain[0] == pytest.approx(ep)
    assert pa.damage[0] == pytest.approx(ep / .1)
    call_stage(step, 'stage2', pa, .1)
    assert pa.plastic_strain[0] == pytest.approx(ep)


def test_compiled_three_dimensional_shear_and_rotation():
    pa = particles(C=0, efmin=.1)
    integrator = compiled_integrator(pa)
    # Symmetric 3D shear: all three tensor shear components contribute.
    pa.loading[:] = [.5, .5, .25, .25, .125, .125]
    for i in range(20):
        integrator.step(i * .001, .001)
    assert pa.damage[0] > 0
    # Independently integrate the scalar proportional-loading return. This
    # catches accidental addition of predictor damage to the corrector.
    q, damage, ep = 0., 0., 0.
    for _ in range(20):
        trial = q + 300. * np.sqrt(7. / 16.) * .001
        q = min(trial, 3. * (1. - damage))
        ep += (trial - q) / 300.
        damage = min(1., ep / .1)
    assert pa.damage[0] == pytest.approx(damage)
    assert pa.plastic_strain[0] == pytest.approx(ep)
    assert pa.s01[0] == pytest.approx(2 * pa.s02[0])
    assert pa.s02[0] == pytest.approx(2 * pa.s12[0])
    assert pa.s00[0] + pa.s11[0] + pa.s22[0] == pytest.approx(0.)
    # New, sub-yield state with pure spin. Stress should rotate without damage.
    pa = particles(A=10., sfmax=100.)
    integrator = compiled_integrator(pa)
    pa.s00[:] = 1.
    pa.s11[:] = -1.
    pa.loading[:] = [-1., 1., 0., 0., 0., 0.]
    dt = .001
    for i in range(1000):
        integrator.step(i * dt, dt)
    assert pa.s00[0] == pytest.approx(np.cos(2.), abs=2e-6)
    assert pa.s01[0] == pytest.approx(np.sin(2.), abs=2e-6)
    assert pa.damage[0] == 0
    assert pa.plastic_strain[0] == 0


def test_compiled_plane_strain_retains_out_of_plane_stress():
    pa = particles(C=0)
    integrator = compiled_integrator(pa)
    pa.v00[:] = -.2
    pa.arho[:] = .4
    integrator.step(0., .01)
    assert pa.rho[0] == pytest.approx(2.004)
    assert pa.p[0] == pytest.approx(.4)
    assert pa.q[0] == pytest.approx(.4)
    assert pa.s22[0] == pytest.approx(2. / 15.)
    assert pa.damage[0] == 0


def test_taylor_plane_contact_matches_elastic_particle_rebound():
    from pysph.examples.solid_mech.hjc_taylor_bar import RigidPlaneContact

    # An isolated particle reaches the plane at t=.01, spends pi/100 in
    # harmonic contact, then leaves with its initial speed and no friction.
    errors = []
    for dt in (1e-4, 5e-5):
        pa = particles()
        pa.z[:] = .06
        pa.w[:] = -1.
        pa.u[:] = 2.
        pa.add_property('wall_acceleration')
        kernel = CubicSpline(dim=3)
        integrator = EPECIntegrator(solid=HJCStep())
        equations = [Group([HookesDeviatoricStressRate('solid', None)]),
                     Group([ClearAcceleration('solid', None)]),
                     Group([RigidPlaneContact('solid', None, .1, 1e4)])]
        evaluator = AccelerationEval([pa], equations, kernel, backend='cython')
        SPHCompiler(evaluator, integrator).compile()
        nnps = LinkedListNNPS(dim=3, particles=[pa])
        evaluator.set_nnps(nnps)
        integrator.set_nnps(nnps)
        impulse, penetration, error = 0., 0., 0.
        for i in range(round(.06/dt)):
            integrator.step(i*dt, dt)
            assert pa.wall_acceleration[0] >= 0
            impulse += dt * pa.wall_acceleration[0]
            penetration = max(penetration, .05-pa.z[0])
            time = (i+1)*dt
            # Measure order inside smooth contact; the discrete release
            # event can make individual final-state errors nonmonotone.
            if .01 < time < .035:
                theta = 100*(time-.01)
                error = max(error, abs(pa.w[0]+np.cos(theta))
                            + 100*abs(pa.z[0]-(.05-.01*np.sin(theta))))
        exact_z = .05 + .06 - (.01 + np.pi/100)
        errors.append(error)
        assert pa.z[0] == pytest.approx(exact_z, abs=2e-6)
        assert pa.w[0] == pytest.approx(1., abs=3e-5)
        assert pa.u[0] == 2.
        assert penetration == pytest.approx(.01, abs=2e-6)
        assert impulse == pytest.approx(pa.w[0]+1., abs=1e-12)
        assert pa.damage[0] == 0
    assert errors[1] < .4 * errors[0]


class ClearAcceleration(Equation):
    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.
        d_av[d_idx] = 0.
        d_aw[d_idx] = 0.
