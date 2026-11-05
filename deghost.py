"""Array-level intelligence: model-based deghosting of the as-built skin.

The as-built readout (series-diode rows, floating unselected rows, divider
columns) mixes taxels through sneak paths.  Because the network topology is
known exactly, the 440 readings of one frame (44 row scans x 10 columns)
determine the 440 taxel conductances.  We recover them with a damped
multiplicative fixed-point iteration on the forward model:

    G^(0)      = G_iso(S_meas)                    (each reading inverted alone)
    G^(n+1)    = G^(n) * [G_iso(S_meas) / G_iso(F(G^(n)))]^gamma

where F is the exact nodal scan (tactile_readout.scan) and G_iso is the
single-taxel inverse transfer.  It runs on the host (or an edge processor),
needs no hardware change, and is evaluated here against ground truth.
"""
from __future__ import annotations
import time
import numpy as np
import tactile_readout as tr

_R = np.logspace(1.8, 6.0, 800)
_C = np.array([tr.solve_row(np.array([[r]]), 0, "diode_floating")[0] for r in _R]) / tr.VCC * 4095


def G_iso(counts):
    """single-taxel inverse: ADC counts -> conductance (S)."""
    c = np.clip(counts, _C.min() + 1e-6, _C.max() - 1e-6)
    return 1.0 / np.interp(c, _C[::-1], _R[::-1])


def scan_counts(G, noise=0.0, rng=None):
    S = tr.scan(1.0 / G, "diode_floating")
    if noise:
        rng = rng or np.random.default_rng(0)
        S = np.round(S + rng.normal(0, noise, S.shape))
    return np.clip(S, 0, 4095)


def deghost(S_meas, iters=12, gamma=0.8, G_min=1 / 2e5):
    target = G_iso(S_meas)
    G = target.copy()
    hist = []
    for _ in range(iters):
        S_sim = tr.scan(1.0 / G, "diode_floating")
        r = target / G_iso(S_sim)
        G = np.maximum(G * r ** gamma, G_min)
        hist.append(float(np.sqrt(np.mean((S_sim - S_meas) ** 2))))
    return G, hist


def to_counts_iso(G):
    """express conductances as the reading each taxel would give alone."""
    return np.interp(1.0 / G, _R, _C)
