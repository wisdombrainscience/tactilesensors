/**
 * grasp_ctrl.h  --  tactile grasp-force regulation for the three-finger hand
 *
 * Drop-in replacement for the threshold logic in StartHandPeakControl()
 * (main.c).  One controller instance per finger acts on that finger's
 * proximal servo.  Requires:
 *   - TIM2/3/4 prescaler changed from 1440-1 to 72-1 (1 us PWM resolution;
 *     period 20000-1 for 50 Hz).  Servo limits must then be scaled by 20
 *     (e.g. downLimit 41 -> 820, upLimit 66 -> 1320).
 *   - grasp_ctrl_update() called once per new tactile frame with the peak
 *     ADC count of the finger's proximal and distal pads.
 * NOT YET VALIDATED ON HARDWARE.
 */
#ifndef GRASP_CTRL_H
#define GRASP_CTRL_H
#include <stdint.h>

typedef enum { GC_IDLE = 0, GC_APPROACH, GC_REGULATE, GC_HOLD, GC_RELEASE } gc_state_t;

typedef struct {
    /* configuration */
    float    target_G;        /* target taxel conductance, mS (from level LUT)   */
    float    kp, ki;          /* incremental PI gains: us/mS, us/mS per frame    */
    float    deadband;        /* relative, e.g. 0.02                             */
    float    max_step_us;     /* rate limit per frame                            */
    uint16_t contact_delta;   /* ADC counts above baseline that mean "contact"   */
    uint16_t approach_us;     /* approach step per frame                         */
    uint16_t pw_open_us, pw_closed_us; /* travel limits; either order (reversed joints) */
    float    slip_drop;       /* relative drop of G within one frame -> slip     */
    float    slip_boost;      /* target increase on slip, relative               */
    /* state */
    gc_state_t state;
    float    pw_ref_us;       /* un-quantised reference                          */
    float    e_prev;
    float    G_prev;
    uint16_t baseline;
} grasp_ctrl_t;

void     grasp_ctrl_init(grasp_ctrl_t *c, uint16_t pw_open, uint16_t pw_closed, float target_G);
void     grasp_ctrl_start(grasp_ctrl_t *c, uint16_t baseline_counts);
void     grasp_ctrl_release(grasp_ctrl_t *c);
/* returns the new servo pulse width in microseconds (write to CCRx) */
uint16_t grasp_ctrl_update(grasp_ctrl_t *c, uint16_t peak_prox, uint16_t peak_dist);
float    gc_counts_to_mS(uint16_t counts);
float    gc_level_to_target_mS(uint8_t level);

#endif
