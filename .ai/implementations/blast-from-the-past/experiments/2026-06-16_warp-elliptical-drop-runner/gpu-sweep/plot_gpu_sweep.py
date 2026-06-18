#!/usr/bin/env python3
"""Plot the cross-GPU sweep. Reads gpu-sweep/sweep-*.json and writes PNGs here.

Pure data plotting -- no GPU required; just matplotlib. Run:

    python .ai/.../2026-06-16_warp-elliptical-drop-runner/gpu-sweep/plot_gpu_sweep.py

Figures:
- throughput_vs_particles.png : throughput vs particle count (the headline curve)
- perstep_vs_particles.png    : per-step wall time vs particle count (log-log)
- speedup_vs_cpu_1M.png       : speedup vs a single CPU core at 1M particles
"""

import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

# Single-threaded PySPH Cython Application at 1M particles (continuity PEC step):
# ~3.5 s/step (measured 3.445 s on the 4060 host, 3.574 s on the Blackwell host).
CPU_REF_S_PER_STEP_AT_1M = 3.5
ONE_MILLION = 1002885

# RTX 4060: single 1M point only (fixed-step headline run, not a full sweep).
ANCHOR_4060 = {
    'name': 'RTX 4060 Laptop (sm_89)',
    'points': [(ONE_MILLION, 0.0598, 1.677e7)],
}


def _label(hw):
    name = hw['name'].replace('NVIDIA ', '').replace(' Server Edition', '')
    return '%s (%s)' % (name, hw['arch'])


def load_gpus():
    gpus = []
    for f in sorted(glob.glob(os.path.join(HERE, 'sweep-*.json'))):
        d = json.load(open(f))
        pts = [(r['particles'], r['per_step_s_median'],
                r['throughput_particle_steps_per_s'])
               for r in d['results'] if 'error' not in r]
        gpus.append({'name': _label(d['hardware']), 'points': sorted(pts)})
    gpus.append(ANCHOR_4060)
    # order by throughput at the largest common point (nicer legend ordering)
    gpus.sort(key=lambda g: g['points'][-1][2])
    return gpus


def plot_throughput(gpus):
    fig, ax = plt.subplots(figsize=(8, 5))
    for g in gpus:
        xs = [p[0] for p in g['points']]
        ys = [p[2] / 1e6 for p in g['points']]  # million particle-steps/s
        style = 'o' if len(xs) == 1 else 'o-'
        ax.plot(xs, ys, style, label=g['name'], markersize=5)
    ax.set_xscale('log')
    ax.set_xlabel('particles')
    ax.set_ylabel('throughput (million particle-steps / s)')
    ax.set_title('Warp grid-direct WCSPH -- throughput vs particle count (fp32)')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'throughput_vs_particles.png'), dpi=130)
    plt.close(fig)


def plot_perstep(gpus):
    fig, ax = plt.subplots(figsize=(8, 5))
    for g in gpus:
        xs = [p[0] for p in g['points']]
        ys = [p[1] for p in g['points']]
        style = 'o' if len(xs) == 1 else 'o-'
        ax.plot(xs, ys, style, label=g['name'], markersize=5)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('particles')
    ax.set_ylabel('per-step wall time (s)')
    ax.set_title('Warp grid-direct WCSPH -- per-step time vs particle count')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'perstep_vs_particles.png'), dpi=130)
    plt.close(fig)


def plot_speedup(gpus):
    rows = []
    for g in gpus:
        at1m = [p for p in g['points'] if p[0] == ONE_MILLION]
        if at1m:
            rows.append((g['name'], CPU_REF_S_PER_STEP_AT_1M / at1m[0][1]))
    rows.sort(key=lambda r: r[1])  # ascending -> largest bar at top
    names = [r[0] for r in rows]
    speedups = [r[1] for r in rows]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    bars = ax.barh(names, speedups, color='tab:green', alpha=0.85)
    for b, s in zip(bars, speedups):
        ax.text(b.get_width(), b.get_y() + b.get_height() / 2,
                ' %.0fx' % s, va='center', fontsize=9)
    ax.set_xlim(0, max(speedups) * 1.12)
    ax.set_xlabel('speedup vs single CPU core (single-thread PySPH ~3.5 s/step at 1M)')
    ax.set_title('GPU speedup vs 1 CPU core @ 1M particles\n(Warp grid-direct WCSPH, fp32)')
    ax.grid(True, axis='x', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'speedup_vs_cpu_1M.png'), dpi=130)
    plt.close(fig)


def main():
    gpus = load_gpus()
    plot_throughput(gpus)
    plot_perstep(gpus)
    plot_speedup(gpus)
    print('wrote throughput_vs_particles.png, perstep_vs_particles.png, '
          'speedup_vs_cpu_1M.png to %s' % HERE)


if __name__ == '__main__':
    main()
