"""Additional analyses for the special-issue version of the manuscript.

  E8  array-level intelligence: model-based deghosting vs. raw readout vs.
      zero-potential hardware, as a function of read noise and load.
  E9  vision-touch fusion ablation: vision-only (open-loop), touch-only
      (single grip level), fused (vision prior + tactile regulation), with
      detector errors injected at the measured rates.
  E10 measurement-uncertainty budget of one taxel in the conductance domain.

  python run_special_issue.py      (after run_experiments.py; ~2 min)
"""
from __future__ import annotations
import json, os
import numpy as np
import matplotlib.pyplot as plt
import tactile_readout as tr
import grasp_sim as gs
import deghost as dg
from run_experiments import C, LS, INK, INK2, GRID, COL1, COL2, FIG, DATA, save

RES = json.load(open(os.path.join(DATA, "results.json")))
TX = tr.Taxel()
_FS = dg.to_counts_iso(np.array([1 / TX.R(1.0)]))[0] - dg.to_counts_iso(np.array([1 / TX.R(0.0)]))[0]


# ------------------------------------------------------------------ E8
def e8_deghost():
    rng = np.random.default_rng(11)
    noises = [0, 1, 2, 3, 4, 6]
    levels = [0.4, 1.0]
    out = {}
    lut_R = np.logspace(2.5, 5.5, 300)
    lut_zp = np.array([tr.solve_row(np.array([[r]]), 0, "zero_potential")[0] for r in lut_R]) / tr.VCC * 4095
    fs_zp = np.interp(np.log(TX.R(1.0)), np.log(lut_R), lut_zp) - np.interp(np.log(TX.R(0)), np.log(lut_R), lut_zp)
    for lv in levels:
        u = tr.grasp_load_map(lv); Gt = 1 / TX.R(u); iso = dg.to_counts_iso(Gt); unl = u < 0.02
        S_zp_clean = tr.scan(TX.R(u), "zero_potential"); iso_zp = np.interp(np.log(TX.R(u)), np.log(lut_R), lut_zp)
        rows = []
        for sg in noises:
            reps = []
            for rep in range(3):
                S = dg.scan_counts(Gt, noise=sg, rng=rng)
                e_raw = (dg.to_counts_iso(dg.G_iso(S)) - iso) / _FS * 100
                G, _ = dg.deghost(S, iters=20)
                e_dg = (dg.to_counts_iso(G) - iso) / _FS * 100
                Szp = np.round(S_zp_clean + rng.normal(0, sg, S_zp_clean.shape)) if sg else S_zp_clean
                e_zp = (Szp - iso_zp) / fs_zp * 100
                reps.append([np.sqrt(np.mean(e_raw ** 2)), np.sqrt(np.mean(e_dg ** 2)), np.sqrt(np.mean(e_zp ** 2)),
                             np.abs(e_raw[unl]).max(), np.abs(e_dg[unl]).max(), np.abs(e_zp[unl]).max()])
            rows.append(np.mean(reps, 0))
        out[lv] = np.array(rows)
        if lv == 1.0:
            S = dg.scan_counts(Gt, noise=2, rng=np.random.default_rng(3))
            G, hist = dg.deghost(S, iters=20)
            maps = dict(truth=iso, raw=dg.to_counts_iso(dg.G_iso(S)), dg=dg.to_counts_iso(G))
            conv = hist
    # timing (python, host)
    import time
    t = time.perf_counter(); dg.deghost(dg.scan_counts(1 / TX.R(tr.grasp_load_map(1.0))), iters=20); t_dg = time.perf_counter() - t
    # figure: maps + noise sensitivity
    fig = plt.figure(figsize=(COL2, 2.35))
    gsx = fig.add_gridspec(1, 6, width_ratios=[1, 1, 1, 0.1, 0.55, 2.3], wspace=0.3)
    vmax = maps["truth"].max()
    for k, (key, ttl) in enumerate([("truth", "(a) truth"), ("raw", "(b) raw"), ("dg", "(c) deghosted")]):
        ax = fig.add_subplot(gsx[0, k])
        im = ax.imshow(maps[key], cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
        ax.set_title(ttl); ax.grid(False); ax.set_xticks([0, 9]); ax.set_yticks([0, 23, 43] if k == 0 else [])
        ax.axhline(23.5, color=INK2, lw=0.6)
        if k == 0: ax.set_ylabel("row (fingers | palm)")
        ax.set_xlabel("column")
    cax = fig.add_subplot(gsx[0, 3]); cb = fig.colorbar(im, cax=cax); cb.set_label("counts (isolated)", fontsize=6.5)
    ax = fig.add_subplot(gsx[0, 5])
    r = out[1.0]
    ax.plot(noises, r[:, 0], color=C[0], ls=LS[0], marker="o", ms=3, label="as built, raw")
    ax.plot(noises, r[:, 1], color=C[1], ls=LS[1], marker="s", ms=3, label="as built + deghosting")
    ax.plot(noises, r[:, 2], color=C[2], ls=LS[2], marker="^", ms=3, label="zero-potential hardware")
    ax.set_xlabel("read noise (counts RMS)"); ax.set_ylabel("RMS image error (% FS)")
    ax.set_title("(d) error vs. noise, full grasp"); ax.legend(loc="center right", fontsize=6.3); ax.set_ylim(0, 10)
    save(fig, "fig_deghost")
    RES["deghost"] = dict(noises=noises, load_1_0=np.round(out[1.0], 2).tolist(), load_0_4=np.round(out[0.4], 2).tolist(),
                          cols=["rms_raw", "rms_deghost", "rms_zp", "maxunl_raw", "maxunl_deghost", "maxunl_zp"],
                          convergence_rms_counts=np.round(conv, 2).tolist(), runtime_s_python=t_dg)


# ------------------------------------------------------------------ E9
# Class-specific grip targets = the class->level table implemented in the
# operator interface (DetectObjectForm.cs: nuocsuoi->3, nuocmuoi->4,
# khoaitay->5) times the firmware threshold step of 100 counts.
# Acceptance window +/-10 %.
CLASSES = {"canister": dict(T=500, k=(0.05, 0.08)), "mouthwash": dict(T=400, k=(0.06, 0.10)),
           "water": dict(T=300, k=(0.04, 0.07))}
DET = dict(p_correct=80 / 84, p_confuse=2 / 84, p_miss=2 / 84)   # measured: deployed best1.pt, conf 0.7, 84 val images
FIXED_T = 400                                           # single level for touch-only (centre level 4)


def static_reading(k, pc, p):
    return gs.reading(k * max(0.0, pc - p) / 20.0, 0, np.random.default_rng(0))


def e9_fusion(N=300):
    rng = np.random.default_rng(7)
    cfg = gs.CONFIGS[3]           # proposed controller, 200 Hz, 1 us PWM
    names = list(CLASSES)
    det_log = []
    res = {s: dict(err=[], ok=[], per={c: [] for c in names}) for s in ["vision_only", "touch_only", "fused"]}
    for _ in range(N):
        c = rng.choice(names); T = CLASSES[c]["T"]
        k = rng.uniform(*CLASSES[c]["k"]); pc = rng.uniform(1150, 1250)
        while static_reading(k, pc, 820.0) < 1.1 * T:          # object must be graspable at its target
            k = rng.uniform(*CLASSES[c]["k"]); pc = rng.uniform(1150, 1250)
        # detector outcome
        r = rng.random()
        if r < DET["p_correct"]:
            c_hat = c; det_log.append("ok")
        elif r < DET["p_correct"] + DET["p_confuse"]:
            c_hat = rng.choice([x for x in names if x != c]); det_log.append("confused")
        else:
            c_hat = None; det_log.append("missed")
        T_hat = CLASSES[c_hat]["T"] if c_hat else FIXED_T
        # vision-only: open-loop closure computed for the class-nominal object
        if c_hat:
            kn = np.mean(CLASSES[c_hat]["k"]); pcn = 1200.0
            ps = np.linspace(1320, 820, 2001)
            yy = np.array([static_reading(kn, pcn, p) for p in ps])
            p_ol = ps[np.argmin(np.abs(yy - T_hat))]
        else:
            p_ol = 1320.0     # no detection: do not grasp
        y_v = static_reading(k, pc, p_ol)
        # touch-only: one level for every object, tactile regulation
        t, y = gs.simulate(cfg, FIXED_T, k, pc, T=3.0, rng=rng); y_t = y[-500:].mean()
        # fused: detected class sets the target, tactile regulates
        t, y = gs.simulate(cfg, T_hat, k, pc, T=3.0, rng=rng); y_f = y[-500:].mean()
        for s, yv in [("vision_only", y_v), ("touch_only", y_t), ("fused", y_f)]:
            e = (yv - T) / T * 100
            res[s]["err"].append(e); res[s]["ok"].append(abs(e) <= 10); res[s]["per"][c].append(abs(e) <= 10)
    summ = {}
    for s, d in res.items():
        e = np.abs(d["err"])
        summ[s] = dict(success_pct=float(np.mean(d["ok"]) * 100), median_abs_err_pct=float(np.median(e)),
                       p90_abs_err_pct=float(np.percentile(e, 90)),
                       per_class_success=({c: float(np.mean(v) * 100) for c, v in d["per"].items()}))
    RES["fusion_det_events"] = {k: det_log.count(k) for k in ("ok", "confused", "missed")}
    RES["fusion"] = dict(N=N, classes={c: dict(T=v["T"], k=list(v["k"])) for c, v in CLASSES.items()}, detector=DET, summary=summ)
    # figure
    fig, ax = plt.subplots(1, 2, figsize=(COL1, 2.0))
    lab = {"vision_only": "vision\nonly", "touch_only": "touch\nonly", "fused": "vision +\ntouch"}
    xs = np.arange(3)
    for i, s in enumerate(lab):
        ax[0].bar(i, summ[s]["success_pct"], color=C[i], width=0.6)
        ax[0].text(i, summ[s]["success_pct"] + 2, f"{summ[s]['success_pct']:.0f}", ha="center", fontsize=6.5, color=INK)
        e = np.abs(res[s]["err"])
        ax[1].boxplot([e], positions=[i], widths=0.5, showfliers=False, whis=(5, 95), patch_artist=True,
                      boxprops=dict(facecolor=C[i], alpha=0.85, lw=0.6), medianprops=dict(color=INK, lw=1), whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
    for a in ax: a.set_xticks(xs); a.set_xticklabels(list(lab.values()), fontsize=6.5)
    ax[0].set_ylim(0, 110); ax[0].set_ylabel("grasps within ±10% (%)"); ax[0].set_title("(a) success")
    ax[1].set_ylabel("|force error| (% of target)"); ax[1].set_title("(b) error (5–95th pct.)")
    ax[1].set_ylim(0, 32)
    fig.tight_layout(pad=0.3)
    save(fig, "fig_fusion")


# ------------------------------------------------------------------ E10
def e10_uncertainty():
    """GUM-style budget for G = V / (Rpd (Vs - Vf - V)) at three operating points."""
    Vs, Vf, Rpd = 3.3, 0.20, 1000.0
    def G(counts, Vs=Vs, Vf=Vf, Rpd=Rpd, Vref=3.3):
        V = counts * Vref / 4095
        return V / (Rpd * (Vs - Vf - V))
    src = {  # name: (parameter, standard uncertainty, note)
        "ADC quantisation": ("c", 1 / np.sqrt(12), "1 LSB uniform"),
        "read noise (measured SD)": ("c", 3.0, "2-4 counts steady state"),
        "diode drop, temperature": ("Vf", 0.020 / np.sqrt(3), "-2 mV/K, +/-10 K"),
        "diode drop, part spread": ("Vf", 0.020 / np.sqrt(3), "+/-20 mV"),
        "pull-down tolerance": ("Rpd", 10 / np.sqrt(3), "1 %"),
    }
    pts = {"light (300)": 300, "medium (700)": 700, "firm (1200)": 1200}
    table = {}
    for pn, c in pts.items():
        g0 = G(c); row = {}
        for name, (par, u, _) in src.items():
            if par == "c":
                dg_ = (G(c + u) - G(c - u)) / 2
            elif par == "Vf":
                dg_ = (G(c, Vf=Vf + u) - G(c, Vf=Vf - u)) / 2
            else:
                dg_ = (G(c, Rpd=Rpd + u) - G(c, Rpd=Rpd - u)) / 2
            row[name] = abs(dg_) / g0 * 100
        row["combined (RSS)"] = float(np.sqrt(sum(v ** 2 for v in row.values())))
        row["expanded, k = 2"] = 2 * row["combined (RSS)"]
        table[pn] = row
    RES["uncertainty"] = dict(sources={k: v[2] for k, v in src.items()}, rel_pct=table,
                              note="supply and ADC reference share the 3.3 V rail, so Vs/Vref errors cancel ratiometrically")


if __name__ == "__main__":
    e10_uncertainty(); e8_deghost(); e9_fusion()
    json.dump(RES, open(os.path.join(DATA, "results.json"), "w"), indent=1, default=float)
    print(json.dumps({k: RES[k] for k in ("deghost", "fusion", "uncertainty")}, indent=1, default=float))
