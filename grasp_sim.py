"""
Closed-loop grasp-force simulation for one opposing finger pair of the hand.

It reproduces the *original* firmware logic (main.c, StartHandPeakControl):
  close the proximal servos one PWM step every `delay` ms until the
  opposing-pair condition  maxA + maxB + 100 > 2T  and  maxA, maxB > 0.6T
  holds, then stop and apply a single +/-1-step correction,
and compares it with the *proposed* controller implemented in
firmware_patch/grasp_ctrl.c (two-phase approach + PI regulation in the
linearised conductance domain, with deadband and anti-windup).

Plant:  servo pulse width p [us] (smaller = more closed), contact begins at
p_c; normalised contact load u = k_obj * max(0, p_c - p) / 20 (per 20 us of
closure), Velostat taxel R(u) and the as-built divider/diode readout give the
ADC reading.  The MG946R slews at <= 353 deg/s (~0.17 s / 60 deg at 6 V);
servo pulse 11.1 us/deg.  Sensor frames arrive every T_frame with one frame
of pipeline latency (scan -> queue -> UART), plus Gaussian read noise.

This is a simulation for design comparison, not a substitute for hardware
experiments.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
import tactile_readout as tr

TAXEL = tr.Taxel()
_Rs = np.logspace(2.5, 5.5, 600)
_cnt = np.array([tr.solve_row(np.array([[r]]), 0, "diode_floating")[0] for r in _Rs]) / tr.VCC * 4095


def reading(u, noise, rng):
    R = TAXEL.R(u)
    return np.interp(R, _Rs, _cnt) + rng.normal(0, noise)


def counts_to_G(c):
    """inverse transfer: ADC counts -> taxel conductance (mS); near-linear in load."""
    R = np.interp(c, _cnt[::-1], _Rs[::-1])
    return 1e3 / R


@dataclass
class Config:
    name: str
    controller: str          # "original" | "proposed"
    T_frame: float           # s
    q_us: float              # PWM resolution (us)
    step_counts: int = 1     # original: CCR step per move
    delay_ms: float = 100.0  # original: ms between steps (GUI default, Fig. 8)
    noise: float = 6.0       # ADC counts rms


def simulate(cfg: Config, target, k_obj, p_c, T=6.0, dt=1e-3, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    p_open = 1320.0                     # us (CCR 66 x 20 us)
    p_min = 820.0                       # us (CCR 41 x 20 us)
    slew = 353 * 11.1                   # us/s
    p = p_open; p_cmd = p_open; p_ref = p_open   # p_ref: un-quantised controller state
    n = int(T / dt)
    t_log, y_log = np.zeros(n), np.zeros(n)
    frame_q = []                        # (t_ready, value)
    last_frame = -1.0
    latest = 0.0
    state = "close"; integ = 0.0; next_move = 0.0
    G_target = counts_to_G(target)
    base = reading(0, 0, rng)
    for k in range(n):
        t = k * dt
        # plant
        dp = np.clip(p_cmd - p, -slew * dt, slew * dt); p += dp
        u = k_obj * max(0.0, p_c - p) / 20.0
        y_true = reading(u, 0, rng)
        # sensor frames with one frame of latency
        if t - last_frame >= cfg.T_frame:
            last_frame = t
            frame_q.append((t + cfg.T_frame, y_true + rng.normal(0, cfg.noise)))
        new_frame = False
        while frame_q and frame_q[0][0] <= t:
            latest = frame_q.pop(0)[1]; new_frame = True
        # controller
        if cfg.controller == "original":
            step = cfg.step_counts * 20.0
            thr = 2 * target
            if state == "close":
                if t >= next_move:
                    p_cmd = max(p_min, p_cmd - step); next_move = t + cfg.delay_ms / 1e3
                # pair condition (symmetric pair -> both fingers read `latest`)
                if new_frame and (2 * latest + 100 > thr) and latest > 0.3 * thr:
                    state = "corrected"
                    p_cmd = p_cmd + step if latest > target else p_cmd - step
                    p_cmd = round(p_cmd / step) * step
        else:  # proposed two-phase controller (see firmware_patch/grasp_ctrl.c)
            if state == "close":
                # fast approach (same mean speed as the original: 1 mm-equivalent)
                if t >= next_move:
                    p_ref = max(p_min, p_ref - 20.0); next_move = t + 0.020
                if new_frame and latest > base + 60:          # contact detected
                    state = "regulate"; e_prev = 0.0
            elif new_frame:
                e = G_target - counts_to_G(latest)            # conductance error, mS
                if abs(e) < 0.02 * G_target:                   # +/-2 % deadband
                    e = 0.0
                Kp, Ki = 20.0, 40.0                            # us/mS, us/mS per frame
                dpu = Kp * (e - e_prev) + Ki * e               # incremental PI
                e_prev = e
                dpu = np.clip(dpu, -40.0, 40.0)                # rate limit per frame
                p_ref = np.clip(p_ref - dpu, p_min, p_open)
            p_cmd = np.round(p_ref / cfg.q_us) * cfg.q_us   # only the PWM output is quantised
        p_cmd = np.round(p_cmd / cfg.q_us) * cfg.q_us
        t_log[k], y_log[k] = t, y_true
    return t_log, y_log


def metrics(t, y, target, tol=0.05):
    final = np.mean(y[t > t[-1] - 1.0])
    err = (final - target) / target
    over = (y.max() - target) / target
    inside = np.abs(y - target) <= tol * target
    # settling time: last time it was outside the band
    out_idx = np.where(~inside)[0]
    ts = t[out_idx[-1]] if len(out_idx) and out_idx[-1] < len(t) - 1 else np.nan
    if len(out_idx) and out_idx[-1] == len(t) - 1:
        ts = np.nan
    t0 = t[np.argmax(y > y[0] + 60)] if np.any(y > y[0] + 60) else np.nan
    return dict(final=final, err=err, overshoot=over, t_settle=ts - t0 if np.isfinite(ts) else np.nan)


CONFIGS = [
    Config("Original (as built): 13 Hz frames, 20 us PWM", "original", 0.0767, 20.0),
    Config("Original logic, 200 Hz frames", "original", 0.005, 20.0),
    Config("Proposed PI, 13 Hz frames, 20 us PWM", "proposed", 0.0767, 20.0),
    Config("Proposed PI, 200 Hz frames, 1 us PWM", "proposed", 0.005, 1.0),
]
