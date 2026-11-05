"""Software-in-the-loop test: the firmware controller (grasp_ctrl.c, compiled
for the host) closes the loop around the same plant/sensor model used in
grasp_sim.py; results are compared with the Python reference controller.
Run from code/sil:  python3 sil_test.py
"""
import ctypes, os, sys, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import grasp_sim as gs

lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "libgrasp.so"))
lib.sil_new.restype = ctypes.c_void_p
lib.sil_new.argtypes = [ctypes.c_uint] * 3
for f in ("sil_start", "sil_step"):
    getattr(lib, f).argtypes = [ctypes.c_void_p, ctypes.c_uint]
lib.sil_step.restype = ctypes.c_uint
lib.sil_free.argtypes = [ctypes.c_void_p]


def run_c(target, k, pc, reverse=False, T=4.0, dt=1e-3, T_frame=0.005, noise=6.0, rng=None):
    """Plant in 'closing coordinate' q (us of closure); reversed joints map pw = open + q."""
    rng = rng or np.random.default_rng()
    p_open, p_min = 1320.0, 820.0
    if reverse:   # mirror the travel: open at 820, closed at 1320
        po, pcl = 820, 1320
    else:
        po, pcl = 1320, 820
    c = lib.sil_new(po, pcl, int(target))
    base = gs.reading(0, 0, rng)
    lib.sil_start(c, int(round(base)))
    p = p_open; cmd = p_open; slew = 353 * 11.1
    q = []; last = -1.0; latest = base; y_log = []
    for i in range(int(T / dt)):
        t = i * dt
        p += np.clip(cmd - p, -slew * dt, slew * dt)
        y = gs.reading(k * max(0.0, pc - p) / 20.0, 0, rng)
        if t - last >= T_frame:
            last = t; q.append((t + T_frame, y + rng.normal(0, noise)))
            while q and q[0][0] <= t:
                latest = q.pop(0)[1]
            pw = lib.sil_step(c, int(np.clip(round(latest), 0, 4095)))
            cmd = (p_open + p_min) - pw if reverse else float(pw)   # map back to plant coordinate
        y_log.append(y)
    lib.sil_free(c)
    return np.arange(len(y_log)) * dt, np.array(y_log)


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    out = {}
    for name, (k0, k1) in {"compliant": (0.04, 0.08), "stiff": (0.15, 0.30)}.items():
        ec, ep, tc, tp, rev = [], [], [], [], []
        for n in range(60):
            target = rng.choice([300, 400, 500, 600, 700, 800]); k = rng.uniform(k0, k1); pc = rng.uniform(1150, 1250)
            while gs.reading(k * (pc - 820) / 20, 0, rng) * 0.9 < target:
                k = rng.uniform(k0, k1); pc = rng.uniform(1150, 1250)
            reverse = bool(n % 2)
            t, y = run_c(target, k, pc, reverse=reverse, rng=np.random.default_rng(n))
            m = gs.metrics(t, y, target); ec.append(abs(m["err"]) * 100); tc.append(m["t_settle"])
            t, y = gs.simulate(gs.CONFIGS[3], target, k, pc, T=4.0, rng=np.random.default_rng(n))
            m = gs.metrics(t, y, target); ep.append(abs(m["err"]) * 100); tp.append(m["t_settle"])
        out[name] = dict(C_median_err=float(np.median(ec)), C_p95_err=float(np.percentile(ec, 95)), C_within5=float(np.mean(np.array(ec) <= 5) * 100),
                         C_median_settle=float(np.nanmedian(tc)), Py_median_err=float(np.median(ep)), Py_p95_err=float(np.percentile(ep, 95)),
                         Py_median_settle=float(np.nanmedian(tp)), reversed_trials=30)
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "..", "data", "sil_results.json"), "w"), indent=1)
