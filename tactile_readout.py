"""
Circuit-level model of the 44 x 10 passive piezoresistive (Velostat) matrix
used in the hand, and of two alternative readout front-ends.

Topology (as built, thesis Fig. 3.5 / firmware Scan_74238.c):
  * 44 row electrodes driven by six cascaded 74HC238 decoders (3.3 V).
    Exactly one output is HIGH; every row has a series 1N5819 Schottky diode,
    so non-selected rows are FLOATING (their diode is reverse biased).
  * 10 column electrodes, each with a pull-down resistor R_pd to ground and
    sampled by the STM32 12-bit ADC (dual simultaneous mode, 5+5 channels).
  * Taxel (i, j) is a load-dependent resistance R_ij between row i and column j.

Schemes modelled:
  "diode_floating"   : as built.
  "grounded_rows"    : no diodes; non-selected decoder outputs hold rows at 0 V
                       through the driver output resistance R_on.
  "zero_potential"   : proposed. Rows as in grounded_rows; every column is held
                       at virtual ground by a transimpedance amplifier (TIA),
                       V_out = I_col * R_f (D'Alessio 1999; Wu et al. 2016).

All solutions are exact nodal analyses (Newton iteration for the diodes).
"""
from __future__ import annotations
import numpy as np

VCC = 3.3
VT = 0.025852
ADC_FS = 4095
# 1N5819 small-signal Shockley fit (Vf ~ 0.2 V at 1 mA, ~0.34 V at 100 mA)
IS_D, N_D = 3.0e-6, 1.30
R_ON_HC = 60.0          # 74HC238 output resistance at 3.3 V (typ. 50-100 ohm)


# ---------------------------------------------------------------- taxel model
class Taxel:
    """Phenomenological Velostat taxel (3 mm x 3 mm, 0.2 mm film).

    R(u) = R_sat + (R_rest - R_sat) / (1 + (u / u_c)^beta),  u in [0, 1] is a
    normalised load.  R_rest and R_sat are *inferred from our recorded data*
    (baseline ~110 ADC counts, full-power grasp ~1250 counts; see
    infer_resistance()).  The load axis is normalised because calibrated
    force data are not yet available.
    """
    def __init__(self, R_rest=33e3, R_sat=1.2e3, u_c=0.25, beta=1.6):
        self.R_rest, self.R_sat, self.u_c, self.beta = R_rest, R_sat, u_c, beta

    def R(self, u):
        u = np.clip(np.asarray(u, float), 0, None)
        return self.R_sat + (self.R_rest - self.R_sat) / (1 + (u / self.u_c) ** self.beta)


def diode_I(v):
    """current and conductance of the series Schottky for forward voltage v."""
    x = np.clip(v / (N_D * VT), -60, 40)
    e = np.exp(x)
    return IS_D * (e - 1), IS_D * e / (N_D * VT)


# ------------------------------------------------------------- network solver
def solve_row(Rm: np.ndarray, sel: int, scheme: str, R_pd=1e3, R_f=1e3):
    """Return the 10 column readings (volts) when row `sel` is selected."""
    nr, nc = Rm.shape
    G = 1.0 / Rm
    if scheme == "zero_potential":
        # columns at 0 V (ideal TIA); rows: selected at VCC through R_on,
        # others at 0 V through R_on.  Row node voltages from KCL:
        g_on = 1.0 / R_ON_HC
        vsrc = np.zeros(nr); vsrc[sel] = VCC
        vrow = vsrc * g_on / (g_on + G.sum(1))
        icol = (G * vrow[:, None]).sum(0)
        return np.clip(icol * R_f, 0, VCC)

    # unknowns: row voltages (nr) + column voltages (nc)
    n = nr + nc
    v = np.zeros(n); v[sel] = VCC - 0.2
    for _ in range(60):
        Y = np.zeros((n, n)); i_inj = np.zeros(n)
        # taxels
        Y[:nr, :nr] += np.diag(G.sum(1))
        Y[nr:, nr:] += np.diag(G.sum(0))
        Y[:nr, nr:] -= G
        Y[nr:, :nr] -= G.T
        # column pull-downs
        Y[nr:, nr:] += np.eye(nc) / R_pd
        if scheme == "diode_floating":
            va = np.zeros(nr); va[sel] = VCC
            vd = va - v[:nr]
            Id, gd = diode_I(vd)
            # linearised companion model: I = Id + gd*(dv)
            Y[np.arange(nr), np.arange(nr)] += gd
            # Norton companion of I = f(va - v_row) linearised at vd
            i_inj[:nr] = Id - gd * vd + gd * va
        elif scheme == "grounded_rows":
            g_on = 1.0 / R_ON_HC
            Y[np.arange(nr), np.arange(nr)] += g_on
            i_inj[sel] += VCC * g_on
        else:
            raise ValueError(scheme)
        v_new = np.linalg.solve(Y, i_inj)
        if np.max(np.abs(v_new - v)) < 1e-9:
            v = v_new; break
        v = v + np.clip(v_new - v, -0.3, 0.3)
    return v[nr:]


def scan(Rm, scheme, **kw):
    """Full 44-row scan -> matrix of readings in ADC counts."""
    out = np.array([solve_row(Rm, s, scheme, **kw) for s in range(Rm.shape[0])])
    return np.round(out / VCC * ADC_FS)


def ideal(Rm, scheme, **kw):
    """Reading each taxel would give if it were the only conductive element."""
    nr, nc = Rm.shape
    out = np.zeros_like(Rm)
    big = np.full((1, 1), 0.0)
    for i in range(nr):
        for j in range(nc):
            single = np.full((1, 1), Rm[i, j])
            out[i, j] = solve_row(single, 0, scheme, **kw)[0]
    return np.round(out / VCC * ADC_FS)


def infer_resistance(adc_counts, R_pd=1e3):
    """Invert the as-built single-taxel transfer curve (diode included)."""
    Rs = np.logspace(1.5, 6, 4000)
    curve = np.array([solve_row(np.array([[r]]), 0, "diode_floating", R_pd=R_pd)[0] for r in Rs])
    counts = curve / VCC * ADC_FS
    return np.interp(adc_counts, counts[::-1], Rs[::-1])


# ---------------------------------------------------------------- layouts
# Row map of the 44 x 10 matrix (firmware MyArray split, main.c):
#   rows 0-7 : distal pads of fingers 1-3 (cols 0-4 finger1? see PAD_SLICES)
#   rows 8-23: two more 8 x 10 blocks; rows 24-43: palm 20 x 10
PAD_SLICES = {
    # name: (row slice, col slice)  -- 6 finger pads of 8x5 and a 20x10 palm
    "distal_1": (slice(0, 8), slice(0, 5)),  "proximal_1": (slice(0, 8), slice(5, 10)),
    "distal_2": (slice(8, 16), slice(0, 5)), "proximal_2": (slice(8, 16), slice(5, 10)),
    "distal_3": (slice(16, 24), slice(0, 5)), "proximal_3": (slice(16, 24), slice(5, 10)),
    "palm": (slice(24, 44), slice(0, 10)),
}


def grasp_load_map(level=1.0, rng=None):
    """Synthetic but realistic grasp contact image: an elliptical blob on each
    finger pad and two blobs on the palm (cf. the measured maps, Fig. 9)."""
    rng = np.random.default_rng(0) if rng is None else rng
    u = np.zeros((44, 10))
    yy, xx = np.mgrid[0:44, 0:10]
    def blob(r0, c0, sr, sc, a):
        u[:] = np.maximum(u, a * np.exp(-(((yy - r0) / sr) ** 2 + ((xx - c0) / sc) ** 2)))
    for k, (rs, cs) in PAD_SLICES.items():
        if k == "palm":
            blob(30, 3, 2.5, 1.4, 0.9 * level); blob(36, 7, 2.0, 1.2, 0.7 * level)
        else:
            r0 = (rs.start + rs.stop) / 2 + rng.uniform(-1, 1)
            c0 = (cs.start + cs.stop) / 2 + rng.uniform(-0.8, 0.8)
            blob(r0, c0, 1.4, 0.9, level * rng.uniform(0.7, 1.0))
    # keep blobs inside their own pad
    mask = np.zeros_like(u, bool)
    for rs, cs in PAD_SLICES.values():
        mask[rs, cs] = True
    return np.where(mask, u, 0)
