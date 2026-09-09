import numpy as np
import pytest

from compyle.config import use_config
from pysph.base.kernels import QuinticSpline
from pysph.base.nnps import LinkedListNNPS
from pysph.base.utils import get_particle_array
from pysph.sph.acceleration_eval import make_acceleration_evals
from pysph.sph.sph_compiler import SPHCompiler
from pysph.sph.wc.crksph import CRKSPHScheme


def evaluate_crksph(backend):
    x, y = np.mgrid[-0.6:0.6:7j, -0.6:0.6:7j]
    x, y = x.ravel(), y.ravel()
    pa = get_particle_array(
        name='fluid', x=x, y=y, h=0.3, m=0.04,
        rho=1.0 + 0.1*x, e=2.0 + 0.2*y,
        u=-0.3*x + 0.1*y, v=0.2*x - 0.4*y
    )
    scheme = CRKSPHScheme(
        fluids=['fluid'], dim=2, rho0=1.0, c0=1.0, nu=0.0,
        h0=0.3, p0=1.0, gamma=1.4
    )
    scheme.setup_properties([pa])
    pa.u0[:] = pa.u
    pa.v0[:] = pa.v
    kernel = QuinticSpline(dim=2)
    evaluators = make_acceleration_evals(
        [pa], scheme.get_equations(), kernel, backend=backend
    )
    SPHCompiler(evaluators, integrator=None).compile()
    if backend == 'opencl':
        from pysph.base.gpu_nnps import ZOrderGPUNNPS
        nnps = ZOrderGPUNNPS(
            dim=2, particles=[pa], radius_scale=kernel.radius_scale,
            cache=True, backend=backend
        )
    else:
        nnps = LinkedListNNPS(
            dim=2, particles=[pa], radius_scale=kernel.radius_scale,
            cache=True
        )
    for evaluator in evaluators:
        evaluator.set_nnps(nnps)
        evaluator.compute(0.0, 0.001)
    props = ('rho', 'p', 'cs', 'ai', 'bi', 'gradai', 'gradbi',
             'gradv', 'au', 'av', 'ae')
    if backend == 'opencl':
        pa.gpu.pull(*props)
    return {name: pa.get(name).copy() for name in props}


@pytest.mark.parametrize('use_double', [False, True])
def test_crksph_opencl_matches_cython(use_double):
    pytest.importorskip('pyopencl')
    pytest.importorskip('pysph.base.gpu_nnps')
    from compyle.opencl import get_context
    devices = get_context().devices
    if use_double and not all(d.double_fp_config for d in devices):
        pytest.skip('OpenCL device does not support double precision')

    # Exercise both stages, including the correction solve, viscosity limiter,
    # energy equation and speed of sound used by the Kelvin-Helmholtz example.
    with use_config(use_opencl=False, use_double=use_double):
        expected = evaluate_crksph('cython')
        actual = evaluate_crksph('opencl')
    for name in expected:
        assert np.isfinite(actual[name]).all(), name
        rtol, atol = (1e-9, 1e-10) if use_double else (2e-4, 2e-5)
        np.testing.assert_allclose(
            actual[name], expected[name], rtol=rtol, atol=atol,
            err_msg=name
        )
