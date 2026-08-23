#!/usr/bin/env python3
"""Plot the cross-GPU sweep. Reads gpu-sweep/sweep-*.json and writes PNGs here.

Pure data plotting -- no GPU required; just matplotlib. Run:

    python .ai/.../2026-06-16_warp-elliptical-drop-runner/gpu-sweep/plot_gpu_sweep.py

Figures (performance lens):
- throughput_vs_particles.png : throughput vs particle count (the headline curve)
- perstep_vs_particles.png    : per-step wall time vs particle count (log-log)
- speedup_vs_cpu_1M.png       : speedup vs a single CPU core at 1M particles

Figures ($ cost-of-compute lens; Brev hourly pricing in BREV_COST_PER_HR):
- cost_per_billion_1M.png            : $ per billion particle-steps at 1M (bar)
- cost_per_billion_vs_particles.png  : $ per billion particle-steps vs count

Figures (estimate-only lenses; modeled from datasheet TDP + a byte model, no
measured power/profiler data -- there will be no profiled re-run):
- energy_per_gpstep_1M.png : kJ per billion particle-steps at 1M (TDP estimate)
- mbu_at_1M.png            : analytic memory-bandwidth utilization at 1M
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

# NVIDIA Brev on-demand hourly rates (USD/hr), as of 2026-06-18. The RTX 4060 is
# a laptop GPU (no cloud rate) and is excluded from cost plots.
BREV_COST_PER_HR = {
    'B300': 9.49,
    'RTX PRO 6000': 2.63,
    'L40S': 1.06,
    'RTX 5090': 0.78,
}


def cost_per_hr(name):
    for key, val in BREV_COST_PER_HR.items():
        if key in name:
            return val
    return None


# --- Estimate-only lenses (no measured power/profiler data; modeled from
# --- datasheet board power and a byte-traffic model). These are ESTIMATES:
# --- there will be no profiled re-run, so they are reported with their caveats.

# Datasheet board power / TDP (W). The continuity PEC step is NOT FLOP-bound, so
# the cards will not actually pull full TDP -- these overstate the energy of the
# high-TDP idle-headroom parts and are an upper-bound bracket, not a measurement.
BOARD_TDP_W = {
    'B300': 1400.0,        # SXM6 module nameplate (Blackwell Ultra)
    'RTX PRO 6000': 600.0,  # workstation/server max
    'RTX 5090': 575.0,      # FE TDP
    'L40S': 350.0,
    'RTX 4060': 115.0,      # laptop part, max-perf config
}

# Datasheet peak DRAM bandwidth (TB/s).
PEAK_BW_TBS = {
    'B300': 8.0,            # HBM3E
    'RTX PRO 6000': 1.79,   # GDDR7
    'RTX 5090': 1.79,       # GDDR7
    'L40S': 0.864,          # GDDR6
    'RTX 4060': 0.27,       # laptop GDDR6
}

# Effective DRAM traffic per particle-step (bytes), analytic model:
# own-state read+write (~60 B) + avg_neighbors(=44.9, measured) x ~24 B per
# neighbor read. ~1.12 KB. This counts logical reads and IGNORES L2 reuse, so
# achieved-BW from it is an UPPER bound on true DRAM traffic -> MBU below is an
# UPPER bound on true utilization (a generous multi-pass count would be ~2-3x
# larger, still well under peak on every card).
B_EFF_BYTES = 1120.0


def _lookup(table, name):
    for key, val in table.items():
        if key in name:
            return val
    return None


def energy_kj_per_gpstep(throughput, tdp_w):
    """Estimated kJ to compute 1e9 particle-steps at board TDP."""
    return tdp_w / throughput * 1e6


def perf_per_watt(throughput, tdp_w):
    """particle-steps/s per watt of board TDP."""
    return throughput / tdp_w


def mbu_frac(throughput, peak_bw_tbs):
    """Analytic memory-bandwidth utilization (fraction of peak DRAM BW)."""
    return throughput * B_EFF_BYTES / (peak_bw_tbs * 1e12)

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


def cost_per_billion(throughput, cost_hr):
    """USD to compute 1e9 particle-steps at this throughput (particle-steps/s)."""
    return cost_hr / (throughput * 3600.0) * 1e9


def plot_cost_bar(gpus):
    rows = []
    for g in gpus:
        cost = cost_per_hr(g['name'])
        at1m = [p for p in g['points'] if p[0] == ONE_MILLION]
        if cost and at1m:
            rows.append((g['name'], cost_per_billion(at1m[0][2], cost)))
    rows.sort(key=lambda r: r[1], reverse=True)  # cheapest at top
    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    bars = ax.barh(names, vals, color='tab:red', alpha=0.8)
    for b, v in zip(bars, vals):
        ax.text(b.get_width(), b.get_y() + b.get_height() / 2,
                ' $%.4f' % v, va='center', fontsize=9)
    ax.set_xlim(0, max(vals) * 1.18)
    ax.set_xlabel('$ per billion particle-steps  (lower = cheaper compute)')
    ax.set_title('Cost of compute @ 1M particles (NVIDIA Brev hourly pricing)\n'
                 '(Warp grid-direct WCSPH, fp32; USD per 1e9 particle-steps)')
    ax.grid(True, axis='x', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'cost_per_billion_1M.png'), dpi=130)
    plt.close(fig)


def plot_cost_curve(gpus):
    fig, ax = plt.subplots(figsize=(8, 5))
    for g in gpus:
        cost = cost_per_hr(g['name'])
        if not cost:
            continue
        xs = [p[0] for p in g['points']]
        ys = [cost_per_billion(p[2], cost) for p in g['points']]
        style = 'o' if len(xs) == 1 else 'o-'
        ax.plot(xs, ys, style, label=g['name'], markersize=5)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('particles')
    ax.set_ylabel('$ per billion particle-steps')
    ax.set_title('Cost of compute vs particle count (NVIDIA Brev hourly pricing)\n'
                 '(Warp grid-direct WCSPH, fp32; cost rises past each GPU knee)')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'cost_per_billion_vs_particles.png'), dpi=130)
    plt.close(fig)


def plot_energy_bar(gpus):
    rows = []
    for g in gpus:
        tdp = _lookup(BOARD_TDP_W, g['name'])
        at1m = [p for p in g['points'] if p[0] == ONE_MILLION]
        if tdp and at1m:
            rows.append((g['name'], energy_kj_per_gpstep(at1m[0][2], tdp)))
    rows.sort(key=lambda r: r[1], reverse=True)  # lowest energy at top
    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    bars = ax.barh(names, vals, color='tab:orange', alpha=0.85)
    for b, v in zip(bars, vals):
        ax.text(b.get_width(), b.get_y() + b.get_height() / 2,
                ' %.1f kJ' % v, va='center', fontsize=9)
    ax.set_xlim(0, max(vals) * 1.15)
    ax.set_xlabel('kJ per billion particle-steps  (lower = less energy per work)')
    ax.set_title('Energy-to-solution @ 1M particles  (ESTIMATE from board TDP)\n'
                 '(Warp grid-direct WCSPH, fp32; not a measured power draw)')
    ax.grid(True, axis='x', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'energy_per_gpstep_1M.png'), dpi=130)
    plt.close(fig)


def plot_mbu_bar(gpus):
    rows = []
    for g in gpus:
        peak = _lookup(PEAK_BW_TBS, g['name'])
        at1m = [p for p in g['points'] if p[0] == ONE_MILLION]
        if peak and at1m:
            rows.append((g['name'], mbu_frac(at1m[0][2], peak) * 100.0))
    rows.sort(key=lambda r: r[1])  # most-utilized at top
    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    bars = ax.barh(names, vals, color='tab:purple', alpha=0.8)
    for b, v in zip(bars, vals):
        ax.text(b.get_width(), b.get_y() + b.get_height() / 2,
                ' %.1f%% of peak' % v, va='center', fontsize=9)
    ax.set_xlim(0, max(vals) * 1.25)
    ax.set_xlabel('analytic memory-bandwidth utilization @ 1M  (% of datasheet peak)')
    ax.set_title('Roofline: bandwidth utilization @ 1M  (ANALYTIC upper-bound estimate)\n'
                 '(throughput x ~1.12 KB/p-step / peak BW; no profiler counter)')
    ax.grid(True, axis='x', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'mbu_at_1M.png'), dpi=130)
    plt.close(fig)


def main():
    gpus = load_gpus()
    plot_throughput(gpus)
    plot_perstep(gpus)
    plot_speedup(gpus)
    plot_cost_bar(gpus)
    plot_cost_curve(gpus)
    plot_energy_bar(gpus)
    plot_mbu_bar(gpus)
    print('wrote throughput_vs_particles.png, perstep_vs_particles.png, '
          'speedup_vs_cpu_1M.png, cost_per_billion_1M.png, '
          'cost_per_billion_vs_particles.png, energy_per_gpstep_1M.png, '
          'mbu_at_1M.png to %s' % HERE)


if __name__ == '__main__':
    main()
