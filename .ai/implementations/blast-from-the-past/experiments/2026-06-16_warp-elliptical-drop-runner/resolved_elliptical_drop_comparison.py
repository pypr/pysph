#!/usr/bin/env python3
"""Resolved elliptical-drop CPU/Warp performance and result comparison."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

from pysph.examples.elliptical_drop import exact_solution
from pysph.solver.utils import load
from pysph.base.warp_nnps import UniformGridWarpNNPS
from pysph.base.warp_sph import wc_sph_leapfrog_step

from warp_elliptical_drop_runner import WarpEllipticalDropRunner


def _initial_dt(nx, hdx, c0):
    dx = 1.0 / nx
    return 0.25 * hdx * dx / (141.0 + c0)


def _checkpoint_label(t):
    return f't{t:.7f}'.replace('.', 'p')


def _metrics_from_arrays(arrays, solver_time, dt_history=None):
    x = arrays['x']
    y = arrays['y']
    m = arrays['m']
    rho = arrays['rho']
    p = arrays['p']
    u = arrays['u']
    v = arrays['v']
    speed2 = u*u + v*v
    a, _A, _po, _xe, _ye = exact_solution(solver_time)
    major = 1.0 / a
    axis_x = max(abs(float(np.min(x))), abs(float(np.max(x))))
    axis_y = max(abs(float(np.min(y))), abs(float(np.max(y))))
    result = {
        'all_finite': bool(all(np.all(np.isfinite(arrays[name]))
                               for name in ('x', 'y', 'rho', 'p', 'u', 'v'))),
        'particles': int(x.size),
        'time': float(solver_time),
        'x_min': float(np.min(x)),
        'x_max': float(np.max(x)),
        'y_min': float(np.min(y)),
        'y_max': float(np.max(y)),
        'axis_x_abs': axis_x,
        'axis_y_abs': axis_y,
        'axis_major_estimate': max(axis_x, axis_y),
        'axis_minor_estimate': min(axis_x, axis_y),
        'exact_major_axis': float(major),
        'exact_minor_axis': float(a),
        'rho_min': float(np.min(rho)),
        'rho_max': float(np.max(rho)),
        'p_min': float(np.min(p)),
        'p_max': float(np.max(p)),
        'kinetic_energy': float(0.5 * np.sum(m * speed2)),
    }
    if dt_history is not None and len(dt_history) > 0:
        dts = np.asarray(dt_history)
        result.update({
            'steps': int(dts.size),
            'dt_min': float(np.min(dts)),
            'dt_max': float(np.max(dts)),
            'dt_mean': float(np.mean(dts)),
        })
    return result


def _save_warp_checkpoint(path, pa, solver_time, dt_history):
    pull_props = [
        'x', 'y', 'z', 'h', 'm', 'rho', 'p', 'cs', 'u', 'v', 'w',
        'au', 'av', 'aw'
    ]
    for optional in ('ax', 'ay', 'az', 'arho', 'dt_cfl', 'dt_force'):
        if optional in pa.properties:
            pull_props.append(optional)
    pa.gpu.pull(*pull_props)
    arrays = {name: getattr(pa, name).copy() for name in pull_props}
    metrics = _metrics_from_arrays(arrays, solver_time, dt_history)
    np.savez(
        path,
        **arrays,
        dt_history=np.asarray(dt_history),
        solver_time=np.asarray([solver_time]),
        metrics=json.dumps(metrics, sort_keys=True),
    )
    return metrics


def _run_warp(args, out_dir, output_times):
    runner = WarpEllipticalDropRunner(
        nx=args.nx, steps=1, dt=args.dt, rho0=args.rho0, c0=args.c0,
        p0=args.p0, hdx=args.hdx, alpha=args.alpha, beta=args.beta,
        eos='tait', gamma=args.gamma, kernel='gaussian', xsph_eps=args.xsph_eps,
        adaptive_dt=True, cfl=args.cfl, dt_min=args.dt_min,
        dt_max=args.dt, density_mode=args.warp_density_mode,
    )
    pa = runner.create_particles()
    nnps = UniformGridWarpNNPS(
        dim=2, particles=[pa], radius_scale=runner.radius_scale
    )
    dt_history = []
    checkpoints = {}
    target_index = 0
    t = 0.0
    step = 0
    start = time.perf_counter()
    while target_index < len(output_times):
        target = output_times[target_index]
        remaining = target - t
        if remaining <= args.time_epsilon:
            label = _checkpoint_label(target)
            path = out_dir / f'{args.prefix}-warp-{label}.npz'
            metrics = _save_warp_checkpoint(path, pa, t, dt_history)
            checkpoints[label] = {'path': str(path), 'metrics': metrics}
            target_index += 1
            continue
        dt_cap = min(args.dt, remaining)
        _result, dt_used = wc_sph_leapfrog_step(
            nnps, dt=args.dt, rho0=args.rho0, c0=args.c0, p0=args.p0,
            alpha=args.alpha, beta=args.beta, eos='tait', gamma=args.gamma,
            kernel='gaussian', xsph_eps=args.xsph_eps, adaptive_dt=True,
            cfl=args.cfl, dt_min=args.dt_min, dt_max=dt_cap, return_dt=True,
            density_mode=args.warp_density_mode
        )
        dt_history.append(dt_used)
        t += dt_used
        step += 1
        if step >= args.max_steps:
            raise RuntimeError(
                f"Warp exceeded --max-steps={args.max_steps} at t={t}"
            )
    wall = time.perf_counter() - start
    return {
        'backend': 'warp',
        'wall_time_s': wall,
        'steps': step,
        'average_step_time_s': wall / step if step else 0.0,
        'final_time': t,
        'dt_min': float(np.min(dt_history)),
        'dt_max': float(np.max(dt_history)),
        'dt_mean': float(np.mean(dt_history)),
        'checkpoints': checkpoints,
    }


def _run_pysph_application(args, out_dir, output_times):
    app_dir = out_dir / f'{args.prefix}-pysph-app-output'
    app_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        'pysph/examples/elliptical_drop_no_scheme.py',
        '--nx', str(args.nx),
        '--tf', str(max(output_times)),
        '--timestep', str(args.dt),
        '--adaptive-timestep',
        '--cfl', str(args.cfl),
        '--n-damp', str(args.n_damp),
        '--fname', f'{args.prefix}-pysph',
        '--directory', str(app_dir),
        '--logfile', '',
        '--quiet',
    ]
    start = time.perf_counter()
    proc = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[5],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    wall = time.perf_counter() - start
    if proc.returncode != 0:
        raise RuntimeError(
            'PySPH Application baseline failed with exit code '
            f'{proc.returncode}\n{proc.stdout}'
        )
    files = sorted(app_dir.glob(f'{args.prefix}-pysph_*.npz'))
    files.extend(sorted(app_dir.glob(f'{args.prefix}-pysph_*.hdf5')))
    checkpoints = {}
    dt_values = []
    for file_path in files:
        data = load(str(file_path))
        solver_data = data['solver_data']
        solver_time = float(solver_data['t'])
        if solver_time > 0.0:
            dt_values.append(float(solver_data['dt']))
        for target in output_times:
            if abs(solver_time - target) <= args.output_tolerance:
                label = _checkpoint_label(target)
                pa = data['arrays']['fluid']
                arrays = {
                    name: getattr(pa, name)
                    for name in ('x', 'y', 'h', 'm', 'rho', 'p', 'u', 'v')
                }
                metrics = _metrics_from_arrays(arrays, solver_time)
                checkpoints[label] = {
                    'path': str(file_path),
                    'metrics': metrics,
                }
    missing = [
        _checkpoint_label(t) for t in output_times
        if _checkpoint_label(t) not in checkpoints
    ]
    if missing:
        raise RuntimeError(
            f"PySPH Application missing checkpoints {missing}; files={files}"
        )
    steps = int(max((load(str(p))['solver_data']['count'] for p in files),
                    default=0))
    summary = {
        'backend': 'pysph-application',
        'command': command,
        'wall_time_s': wall,
        'steps': steps,
        'average_step_time_s': wall / steps if steps else 0.0,
        'final_time': max(output_times),
        'stdout_tail': proc.stdout[-4000:],
        'output_dir': str(app_dir),
        'checkpoints': checkpoints,
    }
    if dt_values:
        summary.update({
            'dt_min': float(np.min(dt_values)),
            'dt_max': float(np.max(dt_values)),
            'dt_mean': float(np.mean(dt_values)),
        })
    return summary


def _checkpoint_arrays(path):
    if '-pysph_' in Path(path).name:
        data = load(str(path))
        pa = data['arrays']['fluid']
        return {
            name: np.asarray(getattr(pa, name), dtype=np.float64).copy()
            for name in ('x', 'y', 'rho', 'u', 'v')
        }
    data = np.load(path)
    return {
        name: np.asarray(data[name], dtype=np.float64).copy()
        for name in ('x', 'y', 'rho', 'u', 'v')
    }


def _plot_indices(n, max_points=20000):
    if n <= max_points:
        return np.arange(n)
    return np.linspace(0, n - 1, max_points, dtype=np.int64)


def _plot_checkpoint(cpu_path, warp_path, image_path, solver_time):
    cpu = _checkpoint_arrays(cpu_path)
    warp = _checkpoint_arrays(warp_path)
    cpu_speed = np.sqrt(cpu['u']*cpu['u'] + cpu['v']*cpu['v'])
    warp_speed = np.sqrt(warp['u']*warp['u'] + warp['v']*warp['v'])
    vmax = max(float(cpu_speed.max()), float(warp_speed.max()))
    xmin = min(float(cpu['x'].min()), float(warp['x'].min()))
    xmax = max(float(cpu['x'].max()), float(warp['x'].max()))
    ymin = min(float(cpu['y'].min()), float(warp['y'].min()))
    ymax = max(float(cpu['y'].max()), float(warp['y'].max()))
    pad = 0.05 * max(xmax - xmin, ymax - ymin)
    _a, _A, _po, xe, ye = exact_solution(solver_time)
    cpu_idx = _plot_indices(cpu['x'].size)
    warp_idx = _plot_indices(warp['x'].size)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), constrained_layout=True)
    for ax, data, speed, title, idx in (
        (axes[0], cpu, cpu_speed, 'PySPH CPU Application', cpu_idx),
        (axes[1], warp, warp_speed, 'Warp GPU', warp_idx),
    ):
        sc = ax.scatter(
            data['x'][idx], data['y'][idx], c=speed[idx], s=1.5, vmin=0.0,
            vmax=vmax, cmap='viridis', linewidths=0.0, rasterized=True,
        )
        ax.plot(xe, ye, color='black', linewidth=0.75)
        ax.set_title(title)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(xmin - pad, xmax + pad)
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
    fig.colorbar(sc, ax=axes, label='speed')
    fig.savefig(image_path, dpi=180)
    plt.close(fig)


def _metric_deltas(cpu, warp):
    deltas = {}
    for key in (
        'axis_major_estimate', 'axis_minor_estimate', 'rho_min', 'rho_max',
        'kinetic_energy', 'x_min', 'x_max', 'y_min', 'y_max'
    ):
        deltas[key] = float(warp[key] - cpu[key])
    return deltas


def _hardware_summary():
    summary = {
        'platform': platform.platform(),
        'processor': platform.processor(),
        'python': sys.version.split()[0],
    }
    try:
        import warp as wp
    except Exception as exc:  # pragma: no cover
        summary['warp_error'] = repr(exc)
    else:
        summary['warp_version'] = getattr(wp, '__version__', 'unknown')
        try:
            summary['warp_device'] = str(wp.get_device())
        except Exception as exc:  # pragma: no cover
            summary['warp_device_error'] = repr(exc)
    return summary


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nx', type=int, default=100)
    parser.add_argument('--rho0', type=float, default=1.0)
    parser.add_argument('--c0', type=float, default=1400.0)
    parser.add_argument('--p0', type=float, default=0.0)
    parser.add_argument('--hdx', type=float, default=1.3)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=7.0)
    parser.add_argument('--xsph-eps', type=float, default=0.5)
    parser.add_argument('--cfl', type=float, default=0.3)
    parser.add_argument('--n-damp', type=int, default=50)
    parser.add_argument('--dt', type=float, default=None)
    parser.add_argument('--dt-min', type=float, default=1.0e-10)
    parser.add_argument('--output-times', default='0.0008,0.0038')
    parser.add_argument('--output-tolerance', type=float, default=1.0e-10)
    parser.add_argument('--time-epsilon', type=float, default=1.0e-14)
    parser.add_argument('--max-steps', type=int, default=10000000)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--prefix', default='resolved-nx100-continuity')
    parser.add_argument('--warp-density-mode',
                        choices=('continuity', 'summation'),
                        default='continuity')
    parser.add_argument('--skip-pysph-application', action='store_true')
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.dt is None:
        args.dt = _initial_dt(args.nx, args.hdx, args.c0)
    output_times = sorted(
        float(item.strip()) for item in args.output_times.split(',')
        if item.strip()
    )
    out_dir = (
        Path(args.output_dir) if args.output_dir is not None
        else Path(__file__).parent / 'resolved'
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        'case': {
            'nx': args.nx,
            'rho0': args.rho0,
            'c0': args.c0,
            'p0': args.p0,
            'hdx': args.hdx,
            'alpha': args.alpha,
            'beta': args.beta,
            'gamma': args.gamma,
            'xsph_eps': args.xsph_eps,
            'cfl': args.cfl,
            'n_damp': args.n_damp,
            'dt': args.dt,
            'warp_density_mode': args.warp_density_mode,
            'output_times': output_times,
        },
        'hardware': _hardware_summary(),
    }
    if not args.skip_pysph_application:
        summary['cpu'] = _run_pysph_application(args, out_dir, output_times)
    summary['warp'] = _run_warp(args, out_dir, output_times)

    comparisons = {}
    if 'cpu' in summary:
        for target in output_times:
            label = _checkpoint_label(target)
            cpu_checkpoint = summary['cpu']['checkpoints'][label]
            warp_checkpoint = summary['warp']['checkpoints'][label]
            image_path = out_dir / f'{args.prefix}-{label}.png'
            _plot_checkpoint(
                cpu_checkpoint['path'], warp_checkpoint['path'], image_path,
                target,
            )
            comparisons[label] = {
                'time': target,
                'image': str(image_path),
                'deltas': _metric_deltas(
                    cpu_checkpoint['metrics'], warp_checkpoint['metrics']
                ),
            }
        summary['comparisons'] = comparisons
        cpu_time = summary['cpu']['wall_time_s']
        warp_time = summary['warp']['wall_time_s']
        summary['performance'] = {
            'speedup_wall_time': cpu_time / warp_time if warp_time else None,
            'cpu_wall_time_s': cpu_time,
            'warp_wall_time_s': warp_time,
            'cpu_average_step_time_s': summary['cpu']['average_step_time_s'],
            'warp_average_step_time_s': summary['warp']['average_step_time_s'],
        }

    summary_path = out_dir / f'{args.prefix}-summary.json'
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    if 'cpu' in summary:
        for checkpoint in summary['cpu']['checkpoints'].values():
            if not checkpoint['metrics']['all_finite']:
                raise SystemExit("CPU checkpoint contains non-finite values")
    for checkpoint in summary['warp']['checkpoints'].values():
        if not checkpoint['metrics']['all_finite']:
            raise SystemExit("Warp checkpoint contains non-finite values")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
