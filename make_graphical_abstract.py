"""Graphical abstract (jsenga.png) for the IEEE Sensors Journal template."""
import json, os
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import tactile_readout as tr, deghost as dg
HERE = os.path.dirname(os.path.abspath(__file__)); M = os.path.join(HERE, "..", "figs", "src")
R = json.load(open(os.path.join(HERE, "..", "data", "results.json")))["fusion"]["summary"]
det = Image.open(os.path.join(M, "detection_mouthwash.png")).convert("RGB")
hand = Image.open(os.path.join(M, "grasp_bottle.jpg")).convert("RGB")
w, h = det.size; det = det.crop((int(.06 * w), int(.06 * h), int(.97 * w), int(.49 * h)))
w, h = hand.size; hand = hand.crop((0, int(.15 * h), w, int(.85 * h)))
TX = tr.Taxel(); u = tr.grasp_load_map(1.0)
S = dg.scan_counts(1 / TX.R(u), noise=2, rng=np.random.default_rng(3)); G, _ = dg.deghost(S, iters=20)
raw, clean = dg.to_counts_iso(dg.G_iso(S)), dg.to_counts_iso(G)
fig = plt.figure(figsize=(6.0, 2.86), dpi=250)
def ax_at(x, y, w, h):
    a = fig.add_axes([x, y, w, h]); a.axis("off"); return a
a = ax_at(0.01, 0.36, 0.27, 0.5); a.imshow(det); a.set_title("Vision: object class", fontsize=8)
a = ax_at(0.355, 0.08, 0.29, 0.78); a.imshow(hand); a.set_title("Tendon-driven hand\n+ 440-taxel e-skin", fontsize=8)
a = ax_at(0.70, 0.36, 0.13, 0.5); a.imshow(raw, cmap="Blues", vmin=0, vmax=clean.max(), aspect="auto"); a.set_title("raw\n(ghosting)", fontsize=7)
a = ax_at(0.855, 0.36, 0.13, 0.5); a.imshow(clean, cmap="Blues", vmin=0, vmax=clean.max(), aspect="auto"); a.set_title("array-level\ndeghosting", fontsize=7)
fig.text(0.145, 0.25, "grip-level prior\n(context)", ha="center", fontsize=7.5, color="#2a78d6")
fig.text(0.84, 0.25, "tactile map  ->  force\nregulation (PI, slip reflex)", ha="center", fontsize=7.5, color="#1baf7a")
for x0, dx, col in [(0.285, 0.06, "#2a78d6"), (0.69, -0.035, "#1baf7a")]:
    fig.patches.append(matplotlib.patches.FancyArrow(x0, 0.6, dx, 0, width=0.012, head_width=0.04, head_length=0.02, color=col, transform=fig.transFigure))
lo = min(R["vision_only"]["success_pct"], R["touch_only"]["success_pct"]); hi = max(R["vision_only"]["success_pct"], R["touch_only"]["success_pct"])
fig.text(0.5, 0.015, f"Visuo-tactile fusion: {R['fused']['success_pct']:.0f}% of simulated grasps within ±10% of target vs. {lo:.0f}–{hi:.0f}% for either sense alone",
         ha="center", fontsize=7, color="#0b0b0b")
fig.savefig(os.path.join(HERE, "..", "figs", "jsenga.png"), dpi=250, facecolor="white")
