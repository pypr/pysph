#!/usr/bin/env python3
"""Cross-GPU lenses + figures for the Warp 3D dam-break sweep.

Reads gpu-sweep/sweep-dambreak-*.json (one per GPU) and writes PNGs + prints a
lens table. Pure data plotting -- no GPU required. Run:

    python .../2026-06-18_warp-dam-break-3d-runner/gpu-sweep/plot_gpu_sweep_dam_break.py

Lenses (mirrors the elliptical-drop sweep):
- throughput vs particles, per-step vs particles
- speedup vs a single CPU core at ~1M
- $ cost-of-compute (Brev hourly), at 1M (bar) + vs particles (curve)
- energy-to-solution (kJ/billion, TDP estimate) and perf-per-watt
- analytic roofline / memory-bandwidth utilization (rough byte model)

All GPU numbers are the FIXED-dt fused dam-break step (the sweep's metric) --
the same methodology as the elliptical sweep. The production runner uses adaptive
dt (one extra CFL kernel + a scalar readback per step), ~1.5x slower per step at
1M, so end-to-end production speedups are correspondingly lower.
"""

import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

# The sweep's ~1M point (dx=0.011 over the Lobovsky no-obstacle geometry).
AT_1M = 999975

# Single-thread PySPH Cython dam-break Application at 999,975 particles:
# 4.18 s/step (dx=0.011-matched measurement; run-to-run 4.2-5.0 s/step on the
# 4060 host). The CPU runs adaptive dt; its per-step compute is the reference.
CPU_REF_S_PER_STEP_AT_1M = 4.18

# NVIDIA Brev on-demand hourly rates (USD/hr), 2026-06 (same table as the
# elliptical sweep). Cards without a measured sweep are simply skipped.
BREV_COST_PER_HR = {'B300': 9.49, 'RTX PRO 6000': 2.63, 'L40S': 1.06,
                    'RTX 5090': 0.78}
BOARD_TDP_W = {'B300': 1400.0, 'RTX PRO 6000': 600.0, 'RTX 5090': 575.0,
               'L40S': 350.0, 'RTX 4060': 115.0}
PEAK_BW_TBS = {'B300': 8.0, 'RTX PRO 6000': 1.79, 'RTX 5090': 1.79,
               'L40S': 0.864, 'RTX 4060': 0.27}

# Rough effective DRAM traffic per particle-step (bytes) for the fused dam-break
# step: ~73 3D neighbours x ~44 B (geom + m/rho/p/cs + vel) in the fused
# pressure+AV+continuity kernel, + an XSPH pass (~60 nbr x ~36 B), x2 EPEC accel
# evaluations, + own-state/PEC/EOS (~0.3 KB). ~11 KB. This is an ORDER-OF-
# MAGNITUDE estimate (no profiler; ignores L2 reuse; +/- ~2x), ~10x the
# elliptical's 1.12 KB single-continuity kernel. MBU below is correspondingly
# rough.
B_EFF_BYTES = 11000.0

# RTX 4060 Laptop anchor (local 2-point run, fixed-dt fused step).
ANCHOR_4060 = {
    'name': 'RTX 4060 Laptop (sm_89)',
    'points': [(32254, 0.014128, 2283000.0), (AT_1M, 0.220072, 4544000.0)],
}


def _lookup(table, name):
    for key, val in table.items():
        if key in name:
            return val
    return None


def _label(hw):
    name = hw['name'].replace('NVIDIA ', '').replace(' Server Edition', '')
    return '%s (%s)' % (name, hw['arch'])


def load_gpus():
    gpus = []
    for f in sorted(glob.glob(os.path.join(HERE, 'sweep-dambreak-*.json'))):
        d = json.load(open(f))
        pts = [(r['particles'], r['per_step_s_median'],
                r['throughput_particle_steps_per_s'])
               for r in d['results'] if 'error' not in r]
        gpus.append({'name': _label(d['hardware']), 'points': sorted(pts)})
    gpus.append(ANCHOR_4060)
    gpus.sort(key=lambda g: g['points'][-1][2])
    return gpus


def at_1m(g):
    pts = [p for p in g['points'] if p[0] == AT_1M]
    return pts[0] if pts else None


def cost_per_billion(throughput, cost_hr):
    return cost_hr / (throughput * 3600.0) * 1e9


def main():
    gpus = load_gpus()

    # ---- lens table at ~1M ----
    print('\n=== Lenses at ~1M particles (%d), fixed-dt fused dam-break step ==='
          % AT_1M)
    hdr = ('GPU', 'thru(Mp-st/s)', 'per-step(ms)', 'vs CPU', '$/B p-st',
           'kJ/B p-st', 'perf/W(p-st/s/W)', 'MBU%~')
    print('| %-26s | %12s | %11s | %7s | %9s | %9s | %16s | %6s |' % hdr)
    print('|%s|' % ('|'.join(['-' * w for w in
                              (28, 14, 13, 9, 11, 11, 18, 8)])))
    rows = {}
    for g in sorted(gpus, key=lambda g: (at_1m(g) or (0, 0, 0))[2],
                    reverse=True):
        p = at_1m(g)
        if not p:
            continue
        thru, per = p[2], p[1]
        cost = _lookup(BREV_COST_PER_HR, g['name'])
        tdp = _lookup(BOARD_TDP_W, g['name'])
        peak = _lookup(PEAK_BW_TBS, g['name'])
        speedup = CPU_REF_S_PER_STEP_AT_1M / per
        cpb = cost_per_billion(thru, cost) if cost else None
        kjpb = tdp / thru * 1e6 if tdp else None
        ppw = thru / tdp if tdp else None
        mbu = thru * B_EFF_BYTES / (peak * 1e12) * 100 if peak else None
        rows[g['name']] = dict(thru=thru, per=per, speedup=speedup, cpb=cpb,
                               kjpb=kjpb, ppw=ppw, mbu=mbu)
        print('| %-26s | %12.1f | %11.2f | %6.0fx | %9s | %9s | %16s | %6s |' % (
            g['name'], thru / 1e6, per * 1e3, speedup,
            ('$%.4f' % cpb) if cpb else 'n/a',
            ('%.1f' % kjpb) if kjpb else 'n/a',
            ('%.0f' % ppw) if ppw else 'n/a',
            ('%.0f' % mbu) if mbu else 'n/a'))
    print('\nCPU ref: single-thread PySPH Cython %.2f s/step at %d particles. '
          'GPU = fixed-dt fused step.' % (CPU_REF_S_PER_STEP_AT_1M, AT_1M))

    # ---- figures ----
    def _curve(ykey, ylabel, title, fname, ylog=False):
        fig, ax = plt.subplots(figsize=(8, 5))
        for g in gpus:
            xs = [p[0] for p in g['points']]
            ys = [ykey(p) for p in g['points']]
            style = 'o' if len(xs) == 1 else 'o-'
            ax.plot(xs, ys, style, label=g['name'], markersize=5)
        ax.set_xscale('log')
        if ylog:
            ax.set_yscale('log')
        ax.set_xlabel('particles')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, fname), dpi=130)
        plt.close(fig)

    _curve(lambda p: p[2] / 1e6, 'throughput (million particle-steps / s)',
           'Warp 3D dam-break (fused, fp32) -- throughput vs particle count',
           'db_throughput_vs_particles.png')
    _curve(lambda p: p[1], 'per-step wall time (s)',
           'Warp 3D dam-break (fused, fp32) -- per-step time vs particle count',
           'db_perstep_vs_particles.png', ylog=True)

    def _bar(metric, fmt, xlabel, title, fname, color, reverse):
        data = [(name, r[metric]) for name, r in rows.items()
                if r.get(metric) is not None]
        data.sort(key=lambda r: r[1], reverse=reverse)
        names = [d[0] for d in data]
        vals = [d[1] for d in data]
        fig, ax = plt.subplots(figsize=(8.5, 4.4))
        bars = ax.barh(names, vals, color=color, alpha=0.85)
        for b, v in zip(bars, vals):
            ax.text(b.get_width(), b.get_y() + b.get_height() / 2,
                    ' ' + fmt % v, va='center', fontsize=9)
        ax.set_xlim(0, max(vals) * 1.18)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.grid(True, axis='x', alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, fname), dpi=130)
        plt.close(fig)

    _bar('speedup', '%.0fx',
         'speedup vs single CPU core (PySPH ~%.1f s/step at 1M)'
         % CPU_REF_S_PER_STEP_AT_1M,
         'GPU speedup vs 1 CPU core @ 1M (dam-break, fixed-dt fused, fp32)',
         'db_speedup_vs_cpu_1M.png', 'tab:green', reverse=False)
    _bar('cpb', '$%.4f', '$ per billion particle-steps (lower = cheaper)',
         'Cost of compute @ 1M (dam-break; NVIDIA Brev hourly pricing)',
         'db_cost_per_billion_1M.png', 'tab:red', reverse=True)
    _bar('kjpb', '%.1f kJ', 'kJ per billion particle-steps (TDP estimate)',
         'Energy-to-solution @ 1M (dam-break; ESTIMATE from board TDP)',
         'db_energy_per_gpstep_1M.png', 'tab:orange', reverse=True)
    _bar('mbu', '%.0f%%',
         'analytic memory-bandwidth utilization @ 1M (% of peak, ROUGH)',
         'Roofline @ 1M (dam-break; ANALYTIC ~11 KB/p-step, +/-2x)',
         'db_mbu_at_1M.png', 'tab:purple', reverse=False)

    print('\nwrote db_throughput_vs_particles.png, db_perstep_vs_particles.png, '
          'db_speedup_vs_cpu_1M.png, db_cost_per_billion_1M.png, '
          'db_energy_per_gpstep_1M.png, db_mbu_at_1M.png to %s' % HERE)


if __name__ == '__main__':
    main()
