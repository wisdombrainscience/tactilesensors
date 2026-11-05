"""Sensor characterisation from synchronised load-cell + tactile logs.

Produces the metrics a sensors journal expects for a taxel / pad:
  * calibration fit  F = f(G)  (power law on conductance, with R^2 and RMSE)
  * sensitivity      dG/dF  (and ADC counts / N) over the range
  * hysteresis       max |loading - unloading| / full-scale  (%)
  * repeatability    std of the reading at fixed force over N cycles (% FS)
  * drift / creep    relative change of the reading under constant load (%/min)
  * response time    10-90 % rise time after a step load (s)
  * spatial crosstalk  response of neighbours / response of the loaded taxel

Inputs (CSV): load cell log with columns t, force_N; tactile log from
tactile_logger.py (.npz with t, frames).  Clocks are aligned by
cross-correlating the force with the loaded taxel's reading.

  python characterize.py --force cell.csv --tactile run1.npz --row 30 --col 4 --protocol cyclic
"""
from __future__ import annotations
import argparse, json
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

VREF, VDIODE, RPD, FS = 3.3, 0.20, 1000.0, 4095


def counts_to_G(c):
    v = np.clip(np.asarray(c, float) * VREF / FS, 1e-3, VREF - VDIODE - 1e-3)
    return (v / (RPD * (VREF - VDIODE - v))) * 1e3     # mS


def align(t_f, f, t_s, s):
    """Shift tactile time so that it best correlates with the force."""
    grid = np.arange(max(t_f[0], t_s[0]), min(t_f[-1], t_s[-1]), 0.005)
    fi, si = np.interp(grid, t_f, f), np.interp(grid, t_s, s)
    fi, si = fi - fi.mean(), si - si.mean()
    lag = np.argmax(np.correlate(si, fi, "full")) - (len(grid) - 1)
    return t_s - lag * 0.005


def power_law(G, a, b):
    return a * np.power(np.clip(G, 1e-6, None), b)


def calibrate(F, G):
    m = F > 0.02 * F.max()
    p, _ = curve_fit(power_law, G[m], F[m], p0=(F.max() / G.max(), 1.0), maxfev=20000)
    pred = power_law(G[m], *p)
    ss = np.sum((F[m] - pred) ** 2); st = np.sum((F[m] - F[m].mean()) ** 2)
    return dict(a=p[0], b=p[1], R2=1 - ss / st, RMSE_N=float(np.sqrt(ss / m.sum())))


def hysteresis(F, y, nbins=20):
    dF = np.gradient(F)
    load, unload = dF > 0, dF < 0
    edges = np.linspace(F.min(), F.max(), nbins + 1)
    c = 0.5 * (edges[1:] + edges[:-1])
    yl = [np.mean(y[load & (F >= a) & (F < b)]) if np.any(load & (F >= a) & (F < b)) else np.nan for a, b in zip(edges[:-1], edges[1:])]
    yu = [np.mean(y[unload & (F >= a) & (F < b)]) if np.any(unload & (F >= a) & (F < b)) else np.nan for a, b in zip(edges[:-1], edges[1:])]
    d = np.abs(np.array(yl) - np.array(yu))
    return float(np.nanmax(d) / (np.nanmax(y) - np.nanmin(y)) * 100), c, np.array(yl), np.array(yu)


def repeatability(F, y, F_level, tol=0.03):
    sel = np.abs(F - F_level) < tol * F.max()
    return float(np.std(y[sel]) / (y.max() - y.min()) * 100)


def drift(t, y, t0, t1):
    m = (t >= t0) & (t <= t1)
    k = np.polyfit(t[m] / 60, y[m], 1)[0]
    return float(k / np.mean(y[m]) * 100)          # % per minute


def rise_time(t, y):
    y0, y1 = np.median(y[:20]), np.median(y[-20:])
    lo, hi = y0 + 0.1 * (y1 - y0), y0 + 0.9 * (y1 - y0)
    return float(t[np.argmax(y >= hi)] - t[np.argmax(y >= lo)])


def crosstalk(frames, row, col):
    m = frames.reshape(len(frames), 44, 10).astype(float)
    base = m[:20].mean(0)
    peak = m[np.argmax(m[:, row, col])] - base
    nb = [(row + dr, col + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr or dc) and 0 <= row + dr < 44 and 0 <= col + dc < 10]
    return float(np.max([peak[r, c] for r, c in nb]) / peak[row, col] * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", required=True); ap.add_argument("--tactile", required=True)
    ap.add_argument("--row", type=int, required=True); ap.add_argument("--col", type=int, required=True)
    ap.add_argument("--out", default="characterisation.json")
    a = ap.parse_args()
    fc = pd.read_csv(a.force)
    z = np.load(a.tactile)
    ts, frames = z["t"], z["frames"]
    s = frames[:, a.row * 10 + a.col].astype(float)
    ts = align(fc.t.values, fc.force_N.values, ts, s)
    F = np.interp(ts, fc.t.values, fc.force_N.values)
    G = counts_to_G(s)
    res = dict(calibration=calibrate(F, G))
    res["hysteresis_pct_FS"], *_ = hysteresis(F, G)
    res["repeatability_pct_FS_at_50pct"] = repeatability(F, G, 0.5 * F.max())
    res["crosstalk_pct"] = crosstalk(frames, a.row, a.col)
    res["frame_rate_Hz"] = float(len(ts) / (ts[-1] - ts[0]))
    json.dump(res, open(a.out, "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
