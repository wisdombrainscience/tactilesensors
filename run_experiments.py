"""Reproduce every simulated / re-plotted result and figure in the manuscript.

  python run_experiments.py          # writes ../figs/*.pdf and ../data/results.json
"""
from __future__ import annotations
import json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tactile_readout as tr
import grasp_sim as gs

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figs"); DATA = os.path.join(HERE, "..", "data")
os.makedirs(FIG, exist_ok=True)

# categorical slots 1-4 of the validated reference palette + ink tokens
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
LS = ["-", "--", "-.", ":"]
plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 1.4, "savefig.bbox": "tight", "savefig.dpi": 300,
    "legend.frameon": False,
})
COL1, COL2 = 3.45, 7.16          # IEEE column / page widths (in)
R = {}


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf")); fig.savefig(os.path.join(FIG, name + ".png"))
    plt.close(fig)


# ------------------------------------------------------------ E1 readout curve
def e1_transfer():
    Rs = np.logspace(2, 5.3, 300)
    rpds = [1e3, 3.3e3, 8.2e3, 22e3]
    fig, ax = plt.subplots(1, 2, figsize=(COL2, 2.3))
    span = {}
    # operating range inferred from the measured traces
    R_lo, R_hi = [float(x) for x in tr.infer_resistance(np.array([1250, 110]))]
    R["inferred_R_full_grasp_ohm"], R["inferred_R_rest_ohm"] = R_lo, R_hi
    for k, rp in enumerate(rpds):
        v = np.array([tr.solve_row(np.array([[r]]), 0, "diode_floating", R_pd=rp)[0] for r in Rs])
        cnt = v / tr.VCC * 4095
        ax[0].semilogx(Rs, cnt, color=C[k], ls=LS[k], label=f"$R_{{pd}}$ = {rp/1e3:g} k$\\Omega$")
        lo, hi = np.interp([R_lo, R_hi], Rs, cnt)
        span[rp] = float(lo - hi)
    ax[0].axvspan(R_lo, R_hi, color=GRID, alpha=0.6, lw=0)
    ax[0].text(np.sqrt(R_lo * R_hi), 3700, "measured\noperating range", ha="center", va="top", color=INK2, fontsize=6.5)
    ax[0].set_xlabel(r"Taxel resistance $R_t$ ($\Omega$)"); ax[0].set_ylabel("ADC reading (counts)")
    ax[0].set_ylim(0, 4095); ax[0].legend(loc="lower left")
    ax[0].set_title("(a) Single-taxel transfer (as built, with diode)")
    # span + worst ghost vs R_pd
    rp_sweep = np.array([0.47e3, 1e3, 2.2e3, 3.3e3, 4.7e3, 8.2e3, 12e3, 22e3])
    spans, ghosts = [], []
    for rp in rp_sweep:
        a, b = [tr.solve_row(np.array([[r]]), 0, "diode_floating", R_pd=rp)[0] / tr.VCC * 4095 for r in (R_lo, R_hi)]
        spans.append(a - b)
        ghosts.append(ghost_test("diode_floating", R_pd=rp)["ghost_excess"])
    ax[1].plot(rp_sweep / 1e3, spans, color=C[0], marker="o", ms=3.5, label="usable span (counts)")
    ax[1].plot(rp_sweep / 1e3, ghosts, color=C[1], ls="--", marker="s", ms=3.5, label="ghost excess (counts)")
    ax[1].set_xscale("log"); ax[1].set_xlabel(r"Column pull-down $R_{pd}$ (k$\Omega$)")
    ax[1].set_ylabel("ADC counts"); ax[1].legend(loc="upper left")
    ax[1].set_title("(b) Dynamic range vs. ghosting trade-off")
    save(fig, "fig_readout_transfer")
    R["span_counts_vs_Rpd"] = {f"{k:g}": round(v) for k, v in zip(rp_sweep, spans)}
    R["ghost_counts_vs_Rpd"] = {f"{k:g}": round(v) for k, v in zip(rp_sweep, ghosts)}


# ------------------------------------------------------------ E2 ghosting
GHOST_PRESS = [(30, 2), (30, 6), (36, 2)]
GHOST_PROBE = (36, 6)


def ghost_test(scheme, R_pd=1e3, u=1.0):
    tx = tr.Taxel()
    Rm = np.full((44, 10), tx.R(0.0))
    for (i, j) in GHOST_PRESS:
        Rm[i, j] = tx.R(u)
    kw = dict(R_pd=R_pd) if scheme != "zero_potential" else {}
    S = tr.scan(Rm, scheme, **kw)
    rest = tr.scan(np.full((44, 10), tx.R(0.0)), scheme, **kw)
    single = tr.scan(np.where(np.arange(440).reshape(44, 10) == 30 * 10 + 2, tx.R(u), tx.R(0.0)), scheme, **kw)
    return dict(S=S, ghost_excess=float(S[GHOST_PROBE] - rest[GHOST_PROBE]),
                pressed_err=float(np.mean([S[p] for p in GHOST_PRESS]) - single[30, 2]),
                rest=float(rest[GHOST_PROBE]), single=float(single[30, 2]))


def e2_ghost_maps():
    names = {"diode_floating": "(a) As built: series diodes,\nfloating rows",
             "grounded_rows": "(b) No diodes, grounded\nunselected rows",
             "zero_potential": "(c) Proposed: grounded rows +\nTIA virtual-ground columns"}
    fig, ax = plt.subplots(1, 3, figsize=(COL2, 2.5))
    R["ghost"] = {}
    for k, (sch, ttl) in enumerate(names.items()):
        g = ghost_test(sch)
        S = g["S"][24:44, :]   # palm
        # normalise each scheme to its own single-press reading
        im = ax[k].imshow(S / g["single"] * 100, cmap="Blues", vmin=0, vmax=110, aspect="equal")
        for (i, j) in GHOST_PRESS:
            ax[k].add_patch(plt.Rectangle((j - .5, i - 24 - .5), 1, 1, fill=False, ec=INK, lw=1.0))
        i, j = GHOST_PROBE
        ax[k].add_patch(plt.Rectangle((j - .5, i - 24 - .5), 1, 1, fill=False, ec=C[1], lw=1.4, ls="--"))
        ax[k].text(j + 0.7, i - 24 + 2.4, f"{g['ghost_excess']:+.0f}", color=INK, fontsize=7)
        ax[k].set_title(ttl); ax[k].set_xticks([0, 9]); ax[k].set_yticks([0, 19]); ax[k].grid(False)
        ax[k].set_xlabel("column")
        R["ghost"][sch] = dict(ghost_excess_counts=g["ghost_excess"], pressed_err_counts=g["pressed_err"],
                               single_press_counts=g["single"], rest_counts=g["rest"])
    ax[0].set_ylabel("palm row")
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02); cb.set_label("reading / single-press reading (%)")
    save(fig, "fig_ghosting")


# ------------------------------------------------------------ E3 full grasp image error
def e3_grasp_error():
    levels = np.linspace(0.1, 1.0, 7)
    out = {s: dict(rms=[], ghost=[]) for s in ["diode_floating", "grounded_rows", "zero_potential"]}
    tx = tr.Taxel()
    for lv in levels:
        u = tr.grasp_load_map(lv)
        Rm = tx.R(u)
        for s in out:
            S = tr.scan(Rm, s)
            ref = tr.scan(np.full((44, 10), tx.R(0)), s)
            # isolated reference: each taxel measured alone (vectorised via 1x1 solve LUT)
            lut_R = np.logspace(2.5, 5.5, 300)
            lut = np.array([tr.solve_row(np.array([[r]]), 0, s)[0] for r in lut_R]) / tr.VCC * 4095
            iso = np.interp(np.log(Rm), np.log(lut_R), lut)
            fs = np.interp(np.log(tx.R(1.0)), np.log(lut_R), lut) - np.interp(np.log(tx.R(0)), np.log(lut_R), lut)
            err = (S - iso) / fs * 100
            out[s]["rms"].append(float(np.sqrt(np.mean(err ** 2))))
            unl = u < 0.02
            out[s]["ghost"].append(float(np.max(np.abs(err[unl]))))
    fig, ax = plt.subplots(1, 2, figsize=(COL1, 2.3))
    lab = {"diode_floating": "as built (diodes, floating rows)", "grounded_rows": "grounded rows, no diodes",
           "zero_potential": "proposed zero-potential (TIA)"}
    for k, s in enumerate(out):
        ax[0].plot(levels, out[s]["rms"], color=C[k], ls=LS[k], marker="o", ms=3, label=lab[s])
        ax[1].plot(levels, out[s]["ghost"], color=C[k], ls=LS[k], marker="o", ms=3, label=lab[s])
    ax[0].set_ylabel("error (% FS)")
    for a in ax: a.set_xlabel("grasp load (norm.)")
    ax[0].set_title("(a) RMS, whole image"); ax[1].set_title("(b) max on unloaded taxels")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=1, bbox_to_anchor=(0.55, 1.0), fontsize=6.5)
    fig.tight_layout(pad=0.3)
    save(fig, "fig_grasp_image_error")
    R["grasp_image_error"] = {s: {"levels": levels.round(2).tolist(), **{k: np.round(v, 2).tolist() for k, v in d.items()}} for s, d in out.items()}


# ------------------------------------------------------------ E4 throughput
def e4_throughput():
    t_conv = (239.5 + 12.5) / 12e6                         # s per conversion
    t_row = 5 * t_conv                                     # dual simultaneous mode: 5 per ADC
    t_adc = 44 * t_row
    legacy_B, packed_B = (440 + 2) * 2, 665
    rows = []
    for name, B, baud in [("legacy frame, UART 115.2 kbaud", legacy_B, 115200),
                          ("packed frame, UART 115.2 kbaud", packed_B, 115200),
                          ("packed frame, UART 921.6 kbaud", packed_B, 921600),
                          ("packed frame, USB-CDC FS (~0.8 MB/s)", packed_B, 8e6)]:
        t_tx = B * 10 / baud
        rows.append(dict(link=name, bytes=B, t_tx_ms=t_tx * 1e3, t_adc_ms=t_adc * 1e3,
                         fps=1 / max(t_tx, t_adc)))   # DMA pipelining: scan and TX overlap
    R["throughput"] = dict(t_conv_us=t_conv * 1e6, t_row_us=t_row * 1e6, t_adc_ms=t_adc * 1e3,
                           t_adc_single_adc_ms=440 * t_conv * 1e3, rows=rows)


# ------------------------------------------------------------ E5 real grasp traces
def e5_traces():
    files = [("bottle_full", "(a) Bottle, full closure"), ("bottle_level7", "(b) Bottle, level 7"),
             ("snack_full", "(c) Canister, full closure"), ("snack_level6", "(d) Canister, level 6")]
    fig, axs = plt.subplots(2, 2, figsize=(COL1, 3.3), sharey=True)
    axs = axs.ravel()
    lab = {"peak_proximal": "proximal pads", "peak_distal": "distal pads", "peak_palm": "palm"}
    R["traces"] = {}
    for ax, (f, ttl) in zip(axs, files):
        d = pd.read_csv(os.path.join(DATA, f + ".csv"), index_col=0)
        for k, c in enumerate(lab):
            ax.plot(d.index, d[c], color=C[k], ls=LS[k], lw=1.1, label=lab[c])
        T = {"bottle_level7": 700, "snack_level6": 600}.get(f)
        if T:
            ax.axhline(T, color=INK2, lw=0.8, ls=":"); ax.text(d.index.max(), T + 30, "target", ha="right", fontsize=6, color=INK2)
        ax.set_title(ttl, fontsize=7); ax.set_ylim(0, 1450)
        if ax in axs[2:]: ax.set_xlabel("time (s)")
        tail = d[d.index > d.index.max() - 10]
        R["traces"][f] = {c: dict(mean=round(float(tail[c].mean()), 1), sd=round(float(tail[c].std()), 1)) for c in lab}
        if T:
            R["traces"][f]["rel_err_proximal_pct"] = round((tail["peak_proximal"].mean() - T) / T * 100, 1)
    axs[0].set_ylabel("peak reading (counts)"); axs[2].set_ylabel("peak reading (counts)")
    axs[0].legend(loc="lower right", fontsize=6)
    fig.tight_layout(pad=0.4)
    save(fig, "fig_grasp_traces")


# ------------------------------------------------------------ E6 grasp-control Monte Carlo
def e6_grasp_mc(N=60):
    rng = np.random.default_rng(2026)
    classes = {"compliant (k = 0.04-0.08)": (0.04, 0.08), "stiff (k = 0.15-0.30)": (0.15, 0.30)}
    res = {}
    for cfg in gs.CONFIGS:
        res[cfg.name] = {}
        for cname, (k0, k1) in classes.items():
            ms = []
            for _ in range(N):
                target = rng.choice([300, 400, 500, 600, 700, 800])
                k = rng.uniform(k0, k1); pc = rng.uniform(1150, 1250)
                y_max = gs.reading(k * (pc - 820.0) / 20.0, 0, rng)
                while target > 0.9 * y_max:          # keep only reachable targets
                    target = rng.choice([300, 400, 500, 600, 700, 800])
                    k = rng.uniform(k0, k1); pc = rng.uniform(1150, 1250)
                    y_max = gs.reading(k * (pc - 820.0) / 20.0, 0, rng)
                t, y = gs.simulate(cfg, target, k, pc, T=8.0, rng=rng)
                ms.append(gs.metrics(t, y, target))
            e = np.abs([m["err"] for m in ms]) * 100
            ov = np.array([max(0, m["overshoot"]) for m in ms]) * 100
            ts = np.array([m["t_settle"] for m in ms])
            res[cfg.name][cname] = dict(median_abs_err_pct=float(np.median(e)), p95_abs_err_pct=float(np.percentile(e, 95)),
                                        within_5pct=float(np.mean(e <= 5) * 100), median_overshoot_pct=float(np.median(ov)),
                                        median_settle_s=float(np.nanmedian(ts)) if np.any(np.isfinite(ts)) else None)
    R["grasp_mc"] = res
    # example responses
    fig, ax = plt.subplots(1, 2, figsize=(COL1, 2.0), sharey=True)
    for a, (kk, ttl) in zip(ax, [(0.06, "(a) Compliant object"), (0.25, "(b) Stiff object")]):
        for i, cfg in enumerate(gs.CONFIGS):
            t, y = gs.simulate(cfg, 700, kk, 1200.0, T=6.0, rng=np.random.default_rng(5))
            a.plot(t, y, color=C[i], ls=LS[i], lw=1.1, label=cfg.name)
        a.axhline(700, color=INK2, lw=0.8, ls=":"); a.set_title(ttl); a.set_xlabel("time (s)")
    ax[0].set_ylabel("peak reading (counts)"); ax[0].set_ylim(0, 1500)
    fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=1, bbox_to_anchor=(0.55, 1.0), fontsize=6.5)
    fig.tight_layout(pad=0.3)
    save(fig, "fig_grasp_sim")


# ------------------------------------------------------------ E7 detector training
def e7_yolo():
    d = pd.read_csv(os.path.join(DATA, "yolo_results.csv")); d.columns = [c.strip() for c in d.columns]
    fig, ax = plt.subplots(1, 2, figsize=(COL1, 2.0))
    for k, (c, l) in enumerate([("train/box_loss", "train box"), ("val/box_loss", "val box"), ("train/obj_loss", "train obj"), ("val/obj_loss", "val obj")]):
        ax[0].plot(d.epoch + 1, d[c], color=C[k], ls=LS[k], label=l)
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend(fontsize=6); ax[0].set_title("(a) Losses")
    for k, (c, l) in enumerate([("metrics/precision", "precision"), ("metrics/recall", "recall"), ("metrics/mAP_0.5", "mAP@0.5"), ("metrics/mAP_0.5:0.95", "mAP@0.5:0.95")]):
        ax[1].plot(d.epoch + 1, d[c], color=C[k], ls=LS[k], label=l)
    ax[1].set_xlabel("epoch"); ax[1].set_ylim(0, 1.02); ax[1].legend(loc="lower right", fontsize=6); ax[1].set_title("(b) Validation")
    fig.tight_layout(pad=0.3)
    save(fig, "fig_yolo_training")
    b = d.loc[d["metrics/mAP_0.5:0.95"].idxmax()]
    R["yolo_log"] = dict(epochs=int(len(d)), best_epoch=int(b.epoch) + 1, P=float(b["metrics/precision"]), Rc=float(b["metrics/recall"]),
                         mAP50=float(b["metrics/mAP_0.5"]), mAP5095=float(b["metrics/mAP_0.5:0.95"]))


if __name__ == "__main__":
    e1_transfer(); e2_ghost_maps(); e3_grasp_error(); e4_throughput(); e5_traces(); e6_grasp_mc(); e7_yolo()
    json.dump(R, open(os.path.join(DATA, "results.json"), "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in R.items() if k not in ("grasp_image_error",)}, indent=1, default=float))
