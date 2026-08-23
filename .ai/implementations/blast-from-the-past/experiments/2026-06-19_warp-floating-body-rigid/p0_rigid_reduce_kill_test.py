#!/usr/bin/env python3
"""P0 kill-test for the Warp floating-body benchmark (ADR-0006 de-risk).

The one genuinely new GPU primitive for a moving rigid body is a SUM-reduction
over the body's particles producing the 16-slot ``mi`` vector that PySPH's
``RigidBodyMoments.reduce`` builds (rigid_body.py:90-122): total mass, m*x/y/z
(for COM), 6 second-moments about the origin, total force, and torque about the
origin. The host then finalizes COM / parallel-axis inertia tensor / torque
about COM / omega_dot (rigid_body.py:128-207) -- that stays in numpy.

This harness proves, on a synthetic body, BEFORE touching warp_sph.py:

  1. A Warp ``atomic_add`` reduction reproduces the exact 16 ``mi`` sums and the
     downstream COM/inertia/force/torque/omega_dot of ``RigidBodyMoments``.
  2. The non-determinism of fp32 ``atomic_add`` (order-dependent, non-associative
     -- the risk the adversarial critique raised, unlike the order-independent
     ``atomic_max`` dt-reduce) by launching 20x and measuring the spread.
  3. The mitigation: accumulating in f64 even from f32 particle data.
  4. The device rigid-transform (``RigidBodyMotion``: v = vc + omega x r).

Sets the parity tolerance for P1. No NNPS, no PySPH Application (the CPU rigid
reference is blocked by compyle/ast.Str on py3.14); the reference here is a
faithful numpy reimplementation of RigidBodyMoments.
"""
import numpy as np
import warp as wp

wp.init()
DEV = "cuda:0"
RNG = np.random.default_rng(7)


# --------------------------------------------------------------------------
# Synthetic rigid body: a sphere of particles, COM deliberately off-origin so
# the torque-about-origin -> torque-about-COM shift and parallel-axis theorem
# are actually exercised. Forces mimic gravity + a depth-dependent buoyancy-ish
# pressure push so SF and torque are both non-trivial.
# --------------------------------------------------------------------------
def make_body(R=0.025, center=(0.05, 0.18, 0.03), solid_rho=500.0):
    # ASYMMETRIC body: an ellipsoid (distinct semi-axes) so the inertia tensor
    # has real off-diagonal terms, with a graded density and a one-sided force
    # field so the NET TORQUE is non-trivial (omega_dot is then a real number,
    # not f32 noise about zero).
    ax, ay, az = R, 1.8 * R, 0.6 * R
    g = np.linspace(-2.0 * R, 2.0 * R, 26)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    p = ((X / ax) ** 2 + (Y / ay) ** 2 + (Z / az) ** 2) <= 1.0
    x, y, z = X[p], Y[p], Z[p]
    dx = g[1] - g[0]
    # graded density (heavier on +x, +z) -> off-axis COM & products of inertia
    rho = solid_rho * (1.0 + 0.6 * (x / ax) + 0.4 * (z / az))
    m = rho * dx**3
    x = x + center[0]; y = y + center[1]; z = z + center[2]
    n = x.size
    g_acc = 9.81
    # one-sided lateral force (acts mostly on +x half) -> real torque about COM
    fx = m * g_acc * (0.8 * (x - center[0]) / ax + 0.3)
    fy = -m * g_acc + m * g_acc * 1.6 * np.clip(0.20 - y, 0.0, None) / 0.20
    fz = m * g_acc * (0.5 * (x - center[0]) / ax)
    return (np.ascontiguousarray(x), np.ascontiguousarray(y),
            np.ascontiguousarray(z), np.ascontiguousarray(m),
            np.ascontiguousarray(fx), np.ascontiguousarray(fy),
            np.ascontiguousarray(fz))


# --------------------------------------------------------------------------
# Reference: faithful numpy reimplementation of RigidBodyMoments.
# --------------------------------------------------------------------------
def reference_mi(x, y, z, m, fx, fy, fz):
    mi = np.zeros(16, dtype=np.float64)
    mi[0] = np.sum(m)
    mi[1] = np.sum(m * x); mi[2] = np.sum(m * y); mi[3] = np.sum(m * z)
    mi[4] = np.sum(m * (y * y + z * z))
    mi[5] = np.sum(m * (x * x + z * z))
    mi[6] = np.sum(m * (x * x + y * y))
    mi[7] = -np.sum(m * x * y)
    mi[8] = -np.sum(m * x * z)
    mi[9] = -np.sum(m * y * z)
    mi[10] = np.sum(fx); mi[11] = np.sum(fy); mi[12] = np.sum(fz)
    mi[13] = np.sum(y * fz - z * fy)
    mi[14] = np.sum(z * fx - x * fz)
    mi[15] = np.sum(x * fy - y * fx)
    return mi


def finalize(mi, omega):
    """RigidBodyMoments finalize (rigid_body.py:128-207). Host-side, numpy."""
    m = mi[0]
    cx, cy, cz = mi[1] / m, mi[2] / m, mi[3] / m
    cm = np.array([cx, cy, cz])
    ixx = mi[4] - (cy * cy + cz * cz) * m
    iyy = mi[5] - (cx * cx + cz * cz) * m
    izz = mi[6] - (cx * cx + cy * cy) * m
    ixy = mi[7] + cx * cy * m
    ixz = mi[8] + cx * cz * m
    iyz = mi[9] + cy * cz * m
    I = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
    fx, fy, fz = mi[10], mi[11], mi[12]
    force = np.array([fx, fy, fz])
    ac = force / m
    tx = mi[13] - (cy * fz - cz * fy)
    ty = mi[14] - (-cx * fz + cz * fx)
    tz = mi[15] - (cx * fy - cy * fx)
    tau = np.array([tx, ty, tz])
    w = np.asarray(omega, dtype=np.float64)
    omega_dot = np.linalg.solve(I, tau - np.cross(w, I @ w))
    return dict(total_mass=m, cm=cm, I=I, force=force, ac=ac, torque=tau,
                omega_dot=omega_dot)


# --------------------------------------------------------------------------
# Warp atomic_add reduction kernels (single body -> 16-slot accumulator).
# --------------------------------------------------------------------------
@wp.kernel
def reduce_f32(x: wp.array(dtype=wp.float32), y: wp.array(dtype=wp.float32),
               z: wp.array(dtype=wp.float32), m: wp.array(dtype=wp.float32),
               fx: wp.array(dtype=wp.float32), fy: wp.array(dtype=wp.float32),
               fz: wp.array(dtype=wp.float32), mi: wp.array(dtype=wp.float32)):
    i = wp.tid()
    xi = x[i]; yi = y[i]; zi = z[i]; mm = m[i]
    fxi = fx[i]; fyi = fy[i]; fzi = fz[i]
    wp.atomic_add(mi, 0, mm)
    wp.atomic_add(mi, 1, mm * xi); wp.atomic_add(mi, 2, mm * yi)
    wp.atomic_add(mi, 3, mm * zi)
    wp.atomic_add(mi, 4, mm * (yi * yi + zi * zi))
    wp.atomic_add(mi, 5, mm * (xi * xi + zi * zi))
    wp.atomic_add(mi, 6, mm * (xi * xi + yi * yi))
    wp.atomic_add(mi, 7, -mm * xi * yi); wp.atomic_add(mi, 8, -mm * xi * zi)
    wp.atomic_add(mi, 9, -mm * yi * zi)
    wp.atomic_add(mi, 10, fxi); wp.atomic_add(mi, 11, fyi)
    wp.atomic_add(mi, 12, fzi)
    wp.atomic_add(mi, 13, yi * fzi - zi * fyi)
    wp.atomic_add(mi, 14, zi * fxi - xi * fzi)
    wp.atomic_add(mi, 15, xi * fyi - yi * fxi)


@wp.kernel
def reduce_f64acc(x: wp.array(dtype=wp.float32), y: wp.array(dtype=wp.float32),
                  z: wp.array(dtype=wp.float32), m: wp.array(dtype=wp.float32),
                  fx: wp.array(dtype=wp.float32), fy: wp.array(dtype=wp.float32),
                  fz: wp.array(dtype=wp.float32),
                  mi: wp.array(dtype=wp.float64)):
    # f32 particle data, but accumulate in f64 (the recommended mitigation).
    i = wp.tid()
    xi = wp.float64(x[i]); yi = wp.float64(y[i]); zi = wp.float64(z[i])
    mm = wp.float64(m[i])
    fxi = wp.float64(fx[i]); fyi = wp.float64(fy[i]); fzi = wp.float64(fz[i])
    wp.atomic_add(mi, 0, mm)
    wp.atomic_add(mi, 1, mm * xi); wp.atomic_add(mi, 2, mm * yi)
    wp.atomic_add(mi, 3, mm * zi)
    wp.atomic_add(mi, 4, mm * (yi * yi + zi * zi))
    wp.atomic_add(mi, 5, mm * (xi * xi + zi * zi))
    wp.atomic_add(mi, 6, mm * (xi * xi + yi * yi))
    wp.atomic_add(mi, 7, -mm * xi * yi); wp.atomic_add(mi, 8, -mm * xi * zi)
    wp.atomic_add(mi, 9, -mm * yi * zi)
    wp.atomic_add(mi, 10, fxi); wp.atomic_add(mi, 11, fyi)
    wp.atomic_add(mi, 12, fzi)
    wp.atomic_add(mi, 13, yi * fzi - zi * fyi)
    wp.atomic_add(mi, 14, zi * fxi - xi * fzi)
    wp.atomic_add(mi, 15, xi * fyi - yi * fxi)


@wp.kernel
def reduce_f64(x: wp.array(dtype=wp.float64), y: wp.array(dtype=wp.float64),
               z: wp.array(dtype=wp.float64), m: wp.array(dtype=wp.float64),
               fx: wp.array(dtype=wp.float64), fy: wp.array(dtype=wp.float64),
               fz: wp.array(dtype=wp.float64), mi: wp.array(dtype=wp.float64)):
    i = wp.tid()
    xi = x[i]; yi = y[i]; zi = z[i]; mm = m[i]
    fxi = fx[i]; fyi = fy[i]; fzi = fz[i]
    wp.atomic_add(mi, 0, mm)
    wp.atomic_add(mi, 1, mm * xi); wp.atomic_add(mi, 2, mm * yi)
    wp.atomic_add(mi, 3, mm * zi)
    wp.atomic_add(mi, 4, mm * (yi * yi + zi * zi))
    wp.atomic_add(mi, 5, mm * (xi * xi + zi * zi))
    wp.atomic_add(mi, 6, mm * (xi * xi + yi * yi))
    wp.atomic_add(mi, 7, -mm * xi * yi); wp.atomic_add(mi, 8, -mm * xi * zi)
    wp.atomic_add(mi, 9, -mm * yi * zi)
    wp.atomic_add(mi, 10, fxi); wp.atomic_add(mi, 11, fyi)
    wp.atomic_add(mi, 12, fzi)
    wp.atomic_add(mi, 13, yi * fzi - zi * fyi)
    wp.atomic_add(mi, 14, zi * fxi - xi * fzi)
    wp.atomic_add(mi, 15, xi * fyi - yi * fxi)


@wp.kernel
def transform(x: wp.array(dtype=wp.float64), y: wp.array(dtype=wp.float64),
              z: wp.array(dtype=wp.float64),
              cm: wp.vec3d, vc: wp.vec3d, omega: wp.vec3d,
              u: wp.array(dtype=wp.float64), v: wp.array(dtype=wp.float64),
              w: wp.array(dtype=wp.float64)):
    # RigidBodyMotion.initialize: v_particle = vc + omega x r  (r = pos - cm).
    i = wp.tid()
    rx = x[i] - cm[0]; ry = y[i] - cm[1]; rz = z[i] - cm[2]
    u[i] = vc[0] + omega[1] * rz - omega[2] * ry
    v[i] = vc[1] + omega[2] * rx - omega[0] * rz
    w[i] = vc[2] + omega[0] * ry - omega[1] * rx


def run_reduce(kernel, arrays, dtype):
    n = arrays[0].shape[0]
    mi = wp.zeros(16, dtype=dtype, device=DEV)
    wp.launch(kernel, dim=n, inputs=list(arrays) + [mi], device=DEV)
    wp.synchronize_device(DEV)
    return mi.numpy().astype(np.float64)


def main():
    print(f"warp {wp.config.version}  device={wp.get_device(DEV)}")
    x, y, z, m, fx, fy, fz = make_body()
    n = x.size
    print(f"synthetic body: {n} particles, solid_rho=500\n")

    # Two references: from full-f64 inputs, and from the SAME f32-rounded inputs
    # the GPU f32/f64acc kernels actually see. Comparing each GPU result to the
    # reference built from *its own* inputs isolates "is the reduction exact?"
    # from "do f32 inputs carry less precision?" (the latter is inherent, fine).
    x32, y32, z32, m32, fx32, fy32, fz32 = (
        a.astype(np.float32).astype(np.float64) for a in (x, y, z, m, fx, fy, fz))
    ref64 = reference_mi(x, y, z, m, fx, fy, fz)
    ref_f32in = reference_mi(x32, y32, z32, m32, fx32, fy32, fz32)

    # device arrays
    def arr(a, dt):
        return wp.array(a.astype(dt), dtype=(wp.float32 if dt == np.float32
                                             else wp.float64), device=DEV)
    a32 = [arr(a, np.float32) for a in (x, y, z, m, fx, fy, fz)]
    a64 = [arr(a, np.float64) for a in (x, y, z, m, fx, fy, fz)]

    mi_f32 = run_reduce(reduce_f32, a32, wp.float32)
    mi_f64acc = run_reduce(reduce_f64acc, a32, wp.float64)
    mi_f64 = run_reduce(reduce_f64, a64, wp.float64)

    def relerr(a, b):
        b = np.where(np.abs(b) < 1e-30, 1.0, b)
        return np.max(np.abs((a - b) / b))

    e_f32 = relerr(mi_f32, ref_f32in)
    e_f64acc = relerr(mi_f64acc, ref_f32in)
    e_f64 = relerr(mi_f64, ref64)
    print("=== 16-slot mi reduction: GPU vs numpy RigidBodyMoments "
          "(max rel err, each vs the ref from its OWN inputs) ===")
    print(f"  f32 accum  (f32 data)        : {e_f32:.3e}  "
          f"<- f32 atomic_add round-off")
    print(f"  f64 accum  (f32 data, MITIG) : {e_f64acc:.3e}  "
          f"<- reduction exact given f32 inputs")
    print(f"  f64 accum  (f64 data)        : {e_f64:.3e}")
    print(f"  (f64acc vs full-f64 ref: {relerr(mi_f64acc, ref64):.2e} -- the "
          f"residual is the f32 INPUT precision, inherent to the f32 path)")

    # ---- non-determinism: 20 relaunches, spread of each strategy ----
    def spread(kernel, arrays, dtype, k=20):
        runs = np.stack([run_reduce(kernel, arrays, dtype) for _ in range(k)])
        return np.max(runs, axis=0) - np.min(runs, axis=0), runs[0]
    sp32, _ = spread(reduce_f32, a32, wp.float32)
    sp64a, _ = spread(reduce_f64acc, a32, wp.float64)
    sp64, _ = spread(reduce_f64, a64, wp.float64)
    # express spread as relative to the reference magnitude
    denom = np.where(np.abs(ref64) < 1e-30, 1.0, np.abs(ref64))
    print("\n=== run-to-run NON-DETERMINISM over 20 launches (max rel spread) ===")
    print(f"  f32 accum (f32 data)      : {np.max(sp32/denom):.3e}  "
          f"{'<- non-associative atomic_add' if np.max(sp32)>0 else '(bit-exact)'}")
    print(f"  f64 accum (f32 data, MITIG): {np.max(sp64a/denom):.3e}")
    print(f"  f64 accum (f64 data)      : {np.max(sp64/denom):.3e}")

    # ---- full host finalize: f64-data path vs numpy reference (clean f64
    #      correctness check), then the f32-path physical values for sanity. ----
    omega = np.array([0.3, -0.5, 0.2])  # nonzero -> exercises w x (I w)
    R = finalize(ref64, omega)
    G = finalize(mi_f64, omega)
    Gf32 = finalize(mi_f64acc, omega)
    print("\n=== host finalize (f64 path) vs numpy RigidBodyMoments ===")
    for key in ("total_mass", "cm", "force", "ac", "torque", "omega_dot"):
        a, b = np.atleast_1d(G[key]), np.atleast_1d(R[key])
        print(f"  {key:11s} max abs err {np.max(np.abs(a-b)):.3e}   "
              f"value={np.array2string(b, precision=5)}")
    print(f"  inertia tensor max abs err {np.max(np.abs(G['I']-R['I'])):.3e}")
    print(f"  [f32 path] torque={np.array2string(Gf32['torque'], precision=5)}  "
          f"omega_dot={np.array2string(Gf32['omega_dot'], precision=5)}")

    # ---- device rigid-transform (RigidBodyMotion) vs numpy ----
    vc = np.array([0.4, -0.1, 0.05])
    cm = R["cm"]
    u = wp.zeros(n, dtype=wp.float64, device=DEV)
    v = wp.zeros(n, dtype=wp.float64, device=DEV)
    w = wp.zeros(n, dtype=wp.float64, device=DEV)
    wp.launch(transform, dim=n,
              inputs=[a64[0], a64[1], a64[2],
                      wp.vec3d(*cm), wp.vec3d(*vc), wp.vec3d(*omega), u, v, w],
              device=DEV)
    wp.synchronize_device(DEV)
    rx, ry, rz = x - cm[0], y - cm[1], z - cm[2]
    u_ref = vc[0] + omega[1] * rz - omega[2] * ry
    v_ref = vc[1] + omega[2] * rx - omega[0] * rz
    w_ref = vc[2] + omega[0] * ry - omega[1] * rx
    terr = max(np.max(np.abs(u.numpy() - u_ref)),
               np.max(np.abs(v.numpy() - v_ref)),
               np.max(np.abs(w.numpy() - w_ref)))
    print(f"\n=== device rigid-transform (v = vc + omega x r) vs numpy ===")
    print(f"  max abs err: {terr:.3e}")

    # ---- verdict (thresholds reflect what each path CAN achieve) ----
    spread_f32 = np.max(sp32 / denom)
    spread_f64acc = np.max(sp64a / denom)
    od_rel = (np.max(np.abs(G['omega_dot'] - R['omega_dot'])) /
              max(np.max(np.abs(R['omega_dot'])), 1e-300))
    ok_acc = e_f64 < 1e-11 and e_f64acc < 1e-10   # reduction exact given inputs
    ok_det = spread_f64acc < 1e-11                # f64 accum ~ deterministic
    ok_fin = od_rel < 1e-9 and np.max(np.abs(G['I'] - R['I'])) < 1e-10
    ok_xf = terr < 1e-12
    f32_risk = spread_f32 > 1e-7                   # f32 accum is NOT determini.
    print("\n=== VERDICT ===")
    print(f"  reduction exact (f64 accum, given f32 in) : {ok_acc}  "
          f"(err {e_f64acc:.1e})")
    print(f"  f64 accum ~deterministic                  : {ok_det}  "
          f"(spread {spread_f64acc:.1e})")
    print(f"  fp32 accum IS non-deterministic (risk)    : {f32_risk}  "
          f"(spread {spread_f32:.1e}) -> mitigated by f64 accum")
    print(f"  host finalize matches (incl real torque)  : {ok_fin}  "
          f"(omega_dot rel {od_rel:.1e})")
    print(f"  device rigid-transform matches            : {ok_xf}")
    passed = ok_acc and ok_det and ok_fin and ok_xf
    print(f"\n  P0 {'PASS' if passed else 'FAIL'} -- atomic_add rigid-body "
          f"reduction {'is VIABLE; design decision: accumulate in f64'
                       if passed else 'needs rework'}.")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
