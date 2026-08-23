#!/usr/bin/env python3
"""Generate the production results report for the Warp WCSPH backend.

Orchestrates the two headline comparisons and assembles a single Markdown report
(parity + speedup tables, method, hardware). Intended to be run on a capable GPU
machine (see ``BUILD.md`` for environment + GPU compatibility):

    PYTHONPATH=<this dir> python generate_results_report.py \
        --out-dir results-report --gpu-label "NVIDIA H100 80GB"

What it runs (both fresh, same machine -- no reused numbers):

1. Resolved elliptical drop, nx=100, continuity density, PySPH timestep policy:
   the apples-to-apples PySPH CPU Application vs Warp comparison (identical step
   count), via ``resolved_elliptical_drop_comparison.py``. Gives parity at the
   output checkpoints + wall-time speedup.
2. Million-particle (nx=565) 100 fixed steps: CPU PySPH Application vs grid-direct
   Warp, via ``headline_million_100step.py``. Gives throughput speedup.

Optionally (``--resolved-tf``) a longer resolved run to a larger final time for a
fuller trajectory comparison -- this is the multi-hour CPU run best done on the
capable machine.

Use ``--quick`` first to validate the pipeline end-to-end at tiny resolution
before committing to the full (CPU-heavy) runs.

The CPU baseline is the real single-threaded PySPH Cython Application
(``pysph/examples/elliptical_drop_no_scheme.py``); Warp runs the grid-direct
WCSPH path. Both honor the same physics (Gaussian kernel, Tait EOS, continuity
density, radius_scale=3). The Warp device path is fp32 (compyle use_double=False).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[4]


def _run(cmd, env_pythonpath=None):
    import os
    env = dict(os.environ)
    if env_pythonpath:
        env['PYTHONPATH'] = env_pythonpath + os.pathsep + env.get('PYTHONPATH', '')
    print('+ ' + ' '.join(str(c) for c in cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, text=True)
    if proc.returncode != 0:
        raise SystemExit('command failed (%d): %s' % (proc.returncode, cmd))


def run_resolved(out_dir, nx, tf):
    prefix = 'report-resolved-nx%d' % nx
    cmd = [
        sys.executable, str(HERE / 'resolved_elliptical_drop_comparison.py'),
        '--nx', str(nx),
        '--warp-timestep-policy', 'pysph', '--warp-density-mode', 'continuity',
        '--output-dir', str(out_dir), '--prefix', prefix,
    ]
    if tf is not None:
        cmd += ['--output-times', tf]
    _run(cmd, env_pythonpath=str(HERE))
    matches = sorted(out_dir.glob(prefix + '*summary*.json'))
    return json.load(open(matches[-1])) if matches else None


def run_million(out_dir, nx, steps):
    out_json = out_dir / ('report-million-nx%d-%dstep.json' % (nx, steps))
    cmd = [
        sys.executable, str(HERE / 'headline_million_100step.py'),
        '--nx', str(nx), '--steps', str(steps),
        '--out-dir', str(out_dir / 'million-work'),
        '--output', str(out_json),
    ]
    _run(cmd, env_pythonpath=str(HERE))
    return json.load(open(out_json)) if out_json.exists() else None


def _fmt(v, nd=3):
    try:
        return ('%.{}g'.format(nd)) % float(v)
    except (TypeError, ValueError):
        return str(v)


def build_report(gpu_label, resolved, million):
    L = []
    L.append('# Warp WCSPH backend -- production results report')
    L.append('')
    L.append('- GPU: **%s**' % gpu_label)
    L.append('- CPU baseline: single-threaded PySPH Cython Application '
             '(`pysph/examples/elliptical_drop_no_scheme.py`)')
    L.append('- Warp path: grid-direct WCSPH (fp32; compyle `use_double=False`)')
    L.append('- Physics: Gaussian kernel, Tait EOS, continuity density, '
             'radius_scale=3')
    L.append('')

    if resolved is not None:
        cpu = resolved.get('cpu', {})
        warp = resolved.get('warp', {})
        perf = resolved.get('performance', {})
        L.append('## 1. Resolved elliptical drop (nx=%d) -- apples-to-apples'
                 % resolved.get('case', {}).get('nx', '?'))
        L.append('')
        L.append('Real PySPH CPU Application vs Warp, identical adaptive '
                 'timestep policy and step count.')
        L.append('')
        L.append('| | steps | wall (s) | s/step |')
        L.append('|---|---:|---:|---:|')
        L.append('| CPU PySPH Application | %s | %s | %s |' % (
            cpu.get('steps'), _fmt(cpu.get('wall_time_s')),
            _fmt(cpu.get('average_step_time_s'), 4)))
        L.append('| Warp (grid-direct) | %s | %s | %s |' % (
            warp.get('steps'), _fmt(warp.get('wall_time_s')),
            _fmt(warp.get('average_step_time_s'), 4)))
        L.append('')
        L.append('**Speedup (wall): %sx**' % _fmt(perf.get('speedup_wall_time')))
        L.append('')
        comps = resolved.get('comparisons', {})
        if comps:
            L.append('Final-state deltas (Warp vs CPU) at checkpoints:')
            L.append('')
            L.append('| checkpoint | KE delta | axis_major | axis_minor | '
                     'rho_min | rho_max |')
            L.append('|---|---:|---:|---:|---:|---:|')
            for label, c in sorted(comps.items()):
                d = c.get('deltas', {})
                L.append('| %s | %s | %s | %s | %s | %s |' % (
                    label, _fmt(d.get('kinetic_energy'), 3),
                    _fmt(d.get('axis_major_estimate'), 3),
                    _fmt(d.get('axis_minor_estimate'), 3),
                    _fmt(d.get('rho_min'), 3), _fmt(d.get('rho_max'), 3)))
            L.append('')

    if million is not None:
        warp = million.get('warp', {})
        cpu = million.get('cpu', {})
        L.append('## 2. Million particles (nx=%d, %d fixed steps) -- throughput'
                 % (million.get('case', {}).get('nx', '?'),
                    million.get('case', {}).get('steps', '?')))
        L.append('')
        L.append('| | particles | total wall (s) | s/step |')
        L.append('|---|---:|---:|---:|')
        L.append('| CPU PySPH Application | %s | %s | %s |' % (
            cpu.get('particles'), _fmt(cpu.get('total_s')),
            _fmt(cpu.get('per_step_s'), 4)))
        L.append('| Warp (grid-direct) | %s | %s | %s |' % (
            warp.get('particles'), _fmt(warp.get('total_s')),
            _fmt(warp.get('per_step_s'), 4)))
        L.append('')
        L.append('**Speedup: %sx wall / %sx per-step.** KE relative delta %s; '
                 'all finite: %s.' % (
                     _fmt(million.get('speedup_total_wall')),
                     _fmt(million.get('speedup_per_step')),
                     _fmt(million.get('kinetic_energy_rel_delta'), 2),
                     warp.get('all_finite')))
        L.append('')

    L.append('## Method notes')
    L.append('')
    L.append('- No reused numbers: both CPU and Warp are measured fresh on this '
             'machine in this run.')
    L.append('- CPU is single-threaded PySPH Cython; quote that framing when '
             'reporting the speedup.')
    L.append('- The million-particle run uses fixed timesteps (`--no-adaptive-'
             'timestep --n-damp 0`) so both sides do identical work.')
    L.append('- Warp first-run cold compile of the fused grid kernel is a '
             'one-time per-machine cost (deterministic-name disk cache).')
    L.append('')
    return '\n'.join(L) + '\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', default='results-report')
    parser.add_argument('--gpu-label', default='(unspecified GPU)')
    parser.add_argument('--resolved-nx', type=int, default=100)
    parser.add_argument('--resolved-tf', default=None,
                        help="output-times for a longer resolved run, e.g. "
                             "'0.0008,0.0038' (default) or a larger final time")
    parser.add_argument('--million-nx', type=int, default=565)
    parser.add_argument('--million-steps', type=int, default=100)
    parser.add_argument('--skip-resolved', action='store_true')
    parser.add_argument('--skip-million', action='store_true')
    parser.add_argument('--quick', action='store_true',
                        help='tiny smoke run to validate the pipeline')
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    resolved_nx = 20 if args.quick else args.resolved_nx
    million_nx = 30 if args.quick else args.million_nx
    million_steps = 5 if args.quick else args.million_steps

    resolved = None
    if not args.skip_resolved:
        resolved = run_resolved(out_dir, resolved_nx, args.resolved_tf)
    million = None
    if not args.skip_million:
        million = run_million(out_dir, million_nx, million_steps)

    report = build_report(args.gpu_label, resolved, million)
    report_path = out_dir / 'RESULTS_REPORT.md'
    report_path.write_text(report)
    print('\n=== wrote %s ===\n' % report_path)
    print(report)


if __name__ == '__main__':
    main()
