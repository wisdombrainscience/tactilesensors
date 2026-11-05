"""Digitize the 'Real-time Peak Load' traces from the C# GUI screen captures
(thesis Figs. 5.14, 5.16, 5.19, 5.21) into CSV, so they can be re-plotted
without the third-party chart watermark.

Calibration: horizontal gridlines are 100 ADC counts apart and the lowest
gridline is 0; x-axis labels are 5 s apart (~86.7 px).  Accuracy is limited to
about +/-10 counts and +/-0.5 s.  Replace with raw serial logs when available.
"""
import sys, numpy as np, pandas as pd
from PIL import Image

COLORS = {"peak_proximal": (255, 0, 0), "peak_distal": (255, 165, 0), "peak_palm": (46, 139, 87)}
PX_PER_S = 86.7 / 5.0

def digitize(path):
    rgb = np.array(Image.open(path).convert("RGB")).astype(int)
    H, W, _ = rgb.shape
    g = (abs(rgb[..., 0]-rgb[..., 1]) < 6) & (abs(rgb[..., 1]-rgb[..., 2]) < 6) & (rgb[..., 0] > 190) & (rgb[..., 0] < 238)
    rows = np.where(g.sum(1) > W*0.4)[0]; rows = rows[rows > 40]
    # cluster gridline rows
    cl = [[rows[0]]]
    for r in rows[1:]:
        (cl[-1].append(r) if r-cl[-1][-1] <= 2 else cl.append([r]))
    ys = np.array([np.mean(c) for c in cl])
    base = ys.max(); step = np.median(np.diff(ys))
    to_adc = lambda y: (base - y) / step * 100.0
    out = {}
    for name, c in COLORS.items():
        m = (np.abs(rgb - np.array(c)).sum(2) < 60)
        m[:115, :115] = False  # legend
        m[int(base)+2:, :] = False
        m[:int(base - 28.5*step), :] = False  # watermark / title above 2850 counts
        xs, vals = [], []
        for x in range(60, W):
            yy = np.where(m[:, x])[0]
            if len(yy):
                xs.append(x); vals.append(to_adc(np.median(yy)))
        out[name] = pd.Series(vals, index=np.array(xs))
    df = pd.DataFrame(out).sort_index()
    x0 = df.index.min()
    df.index = (df.index - x0) / PX_PER_S
    df.index.name = "t_s"
    return df.interpolate(limit=5)

if __name__ == "__main__":
    for src, dst in zip(sys.argv[1::2], sys.argv[2::2]):
        d = digitize(src); d.to_csv(dst, float_format="%.2f"); print(dst, d.shape)
