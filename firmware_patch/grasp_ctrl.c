/**
 * grasp_ctrl.c -- see grasp_ctrl.h.  Mirrors code/grasp_sim.py ("proposed").
 *
 * The readout is a divider: V = Vd * Rpd / (Rpd + Rt), Vd = 3.3 V - Vf(diode).
 * Inverting it gives the taxel conductance G = 1/Rt, which is close to linear
 * in contact load for Velostat, so regulating G instead of raw ADC counts
 * gives an approximately load-independent loop gain.
 */
#include "grasp_ctrl.h"
#include <math.h>

#define GC_VREF      3.3f
#define GC_VDIODE    0.20f      /* 1N5819 at ~1 mA; refine by calibration */
#define GC_RPD_OHM   1000.0f
#define GC_ADC_FS    4095.0f

float gc_counts_to_mS(uint16_t counts)
{
    float v = (float)counts * GC_VREF / GC_ADC_FS;
    float vd = GC_VREF - GC_VDIODE;
    if (v < 1e-3f) v = 1e-3f;
    if (v > vd - 1e-3f) v = vd - 1e-3f;
    float rt = GC_RPD_OHM * (vd - v) / v;          /* ohm */
    return 1000.0f / rt;                            /* mS  */
}

/* level 0..19 -> the original firmware thresholds (100 counts per level),
 * expressed as conductance so the PI loop works in a linearised domain. */
float gc_level_to_target_mS(uint8_t level)
{
    if (level > 19) level = 19;
    return gc_counts_to_mS((uint16_t)(100u * level));
}

void grasp_ctrl_init(grasp_ctrl_t *c, uint16_t pw_open, uint16_t pw_closed, float target_G)
{
    c->target_G = target_G;
    c->kp = 20.0f; c->ki = 40.0f; c->deadband = 0.02f; c->max_step_us = 40.0f;
    c->contact_delta = 60u; c->approach_us = 4u;   /* 4 us per 5 ms frame = 800 us/s */
    c->pw_open_us = pw_open; c->pw_closed_us = pw_closed;
    c->slip_drop = 0.15f; c->slip_boost = 0.10f;
    c->state = GC_IDLE; c->pw_ref_us = (float)pw_open;
    c->e_prev = 0.0f; c->G_prev = 0.0f; c->baseline = 0u;
}

void grasp_ctrl_start(grasp_ctrl_t *c, uint16_t baseline_counts)
{
    c->baseline = baseline_counts;
    c->state = GC_APPROACH;
    c->e_prev = 0.0f;
}

void grasp_ctrl_release(grasp_ctrl_t *c) { c->state = GC_RELEASE; }

static float clampf(float x, float lo, float hi) { return x < lo ? lo : (x > hi ? hi : x); }

uint16_t grasp_ctrl_update(grasp_ctrl_t *c, uint16_t peak_prox, uint16_t peak_dist)
{
    /* the finger is regulated on the larger of its two pads */
    uint16_t peak = peak_prox > peak_dist ? peak_prox : peak_dist;
    float G = gc_counts_to_mS(peak);
    /* closing direction: -1 if a smaller pulse closes the joint, +1 if reversed */
    const float s = (c->pw_closed_us < c->pw_open_us) ? -1.0f : 1.0f;
    const float lo = (float)(s < 0 ? c->pw_closed_us : c->pw_open_us);
    const float hi = (float)(s < 0 ? c->pw_open_us : c->pw_closed_us);

    switch (c->state) {
    case GC_APPROACH:
        c->pw_ref_us += s * (float)c->approach_us;       /* close */
        if (peak > c->baseline + c->contact_delta) c->state = GC_REGULATE;
        break;
    case GC_REGULATE:
    case GC_HOLD: {
        /* slip reflex: sudden loss of contact conductance while holding */
        if (c->state == GC_HOLD && c->G_prev > 0.0f && G < (1.0f - c->slip_drop) * c->G_prev)
            c->target_G *= (1.0f + c->slip_boost);
        float e = c->target_G - G;
        if (fabsf(e) < c->deadband * c->target_G) { e = 0.0f; c->state = GC_HOLD; }
        float du = c->kp * (e - c->e_prev) + c->ki * e;  /* incremental PI */
        c->e_prev = e;
        du = clampf(du, -c->max_step_us, c->max_step_us);
        c->pw_ref_us += s * du;                            /* positive du closes */
        break; }
    case GC_RELEASE:
        c->pw_ref_us -= s * 4.0f * c->max_step_us;     /* open */
        if ((s < 0 && c->pw_ref_us >= hi) || (s > 0 && c->pw_ref_us <= lo)) c->state = GC_IDLE;
        break;
    default: break;
    }
    c->pw_ref_us = clampf(c->pw_ref_us, lo, hi);
    c->G_prev = G;
    return (uint16_t)(c->pw_ref_us + 0.5f);               /* quantise only here */
}
