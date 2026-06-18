#!/usr/bin/env python3
"""Fresh million-particle, 100-fixed-step CPU-vs-Warp headline comparison.

Warp: the grid-direct WCSPH continuity path (ADR-0004) via the elliptical-drop
runner. CPU: the real PySPH Cython Application
(``pysph/examples/elliptical_drop_no_scheme.py``: Gaussian / TaitEOS /
ContinuityEquation / MomentumEquation / XSPHCorrection / WCSPHStep), forced to
fixed timestep with ``--no-adaptive-timestep`` so it runs exactly the same 100
steps Warp does.

Both sides use identical physics (gaussian kernel, tait EOS, continuity density,
radius_scale=3, c0=1400, alpha=0.1, xsph_eps=0.5, dt=3.7328e-07). No numbers are
reused -- both are measured in this run on this machine.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp

from pysph.base.warp_nnps import UniformGridWarpNNPS
from pysph.base.warp_sph import wc_sph_leapfrog_step
from pysph.solver.utils import load
from warp_elliptical_drop_runner import WarpEllipticalDropRunner

DT = 3.732778967800475e-07
RHO0, C0, ALPHA, BETA, GAMMA, XSPH = 1.0, 1400.0, 0.1, 0.0, 7.0, 0.5


def run_warp(nx, steps):
    runner = WarpEllipticalDropRunner(
        nx=nx, steps=steps, dt=DT, rho0=RHO0, c0=C0, p0=0.0, alpha=ALPHA,
        beta=BETA, eos='tait', gamma=GAMMA, kernel='gaussian', xsph_eps=XSPH,
        density_mode='continuity',
    )
    t0 = time.perf_counter()
    pa = runner.create_particles()
    nnps = UniformGridWarpNNPS(
        dim=2, particles=[pa], radius_scale=runner.radius_scale
    )
    wp.synchronize_device(nnps.device)
    setup_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    for _ in range(steps):
        wc_sph_leapfrog_step(
            nnps, dt=DT, rho0=RHO0, c0=C0, p0=0.0, alpha=ALPHA, beta=BETA,
            eos='tait', gamma=GAMMA, kernel='gaussian', xsph_eps=XSPH,
            adaptive_dt=False, density_mode='continuity', return_dt=True,
        )
    wp.synchronize_device(nnps.device)
    steps_s = time.perf_counter() - t1

    pa.gpu.pull('x', 'y', 'rho', 'u', 'v')
    ke = 0.5 * float(np.sum(pa.m * (pa.u*pa.u + pa.v*pa.v)))
    finite = bool(np.all(np.isfinite(pa.x)) and np.all(np.isfinite(pa.rho)))
    return {
        'particles': int(pa.get_number_of_particles()),
        'setup_s': setup_s,
        'steps_s': steps_s,
        'total_s': setup_s + steps_s,
        'per_step_s': steps_s / steps,
        'kinetic_energy': ke,
        'all_finite': finite,
    }


def run_cpu(nx, steps, out_dir):
    repo_root = Path(__file__).resolve().parents[5]
    app_dir = Path(out_dir) / 'cpu-app-output'
    app_dir.mkdir(parents=True, exist_ok=True)
    tf = steps * DT
    command = [
        sys.executable, 'pysph/examples/elliptical_drop_no_scheme.py',
        '--nx', str(nx), '--tf', repr(tf), '--timestep', repr(DT),
        '--no-adaptive-timestep', '--n-damp', '0', '--pfreq', str(steps),
        '--fname', 'cpu', '--directory', str(app_dir),
        '--logfile', '', '--quiet',
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(command, cwd=repo_root, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError('CPU app failed:\n' + proc.stdout[-3000:])
    files = sorted(app_dir.glob('cpu_*.npz')) + sorted(app_dir.glob('cpu_*.hdf5'))
    last = load(str(files[-1]))
    count = int(last['solver_data']['count'])
    pa = last['arrays']['fluid']
    ke = 0.5 * float(np.sum(pa.m * (pa.u*pa.u + pa.v*pa.v)))
    finite = bool(np.all(np.isfinite(pa.x)) and np.all(np.isfinite(pa.rho)))
    return {
        'particles': int(pa.get_number_of_particles()),
        'total_s': wall,
        'steps_recorded': count,
        'per_step_s': wall / count if count else None,
        'kinetic_energy': ke,
        'all_finite': finite,
        'command': command,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nx', type=int, default=565)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--output', default=None)
    parser.add_argument('--out-dir', default='/tmp/headline_million')
    args = parser.parse_args()

    warp = run_warp(args.nx, args.steps)
    cpu = run_cpu(args.nx, args.steps, args.out_dir)

    result = {
        'case': {
            'nx': args.nx, 'steps': args.steps, 'dt': DT,
            'particles': warp['particles'], 'mode': 'fixed timestep',
            'kernel': 'gaussian', 'eos': 'tait', 'density_mode': 'continuity',
            'cpu_backend': 'pysph Application (elliptical_drop_no_scheme), single-threaded, --no-adaptive-timestep',
            'warp_path': 'grid-direct (ADR-0004)',
        },
        'warp': warp,
        'cpu': cpu,
        'speedup_total_wall': cpu['total_s'] / warp['total_s'],
        'speedup_per_step': (cpu['per_step_s'] / warp['per_step_s'])
        if cpu['per_step_s'] else None,
        'kinetic_energy_rel_delta': abs(warp['kinetic_energy'] - cpu['kinetic_energy'])
        / abs(cpu['kinetic_energy']) if cpu['kinetic_energy'] else None,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2, sort_keys=True)


if __name__ == '__main__':
    main()
