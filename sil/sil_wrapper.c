/* Thin C API around grasp_ctrl.c for software-in-the-loop tests (ctypes). */
#include <stdlib.h>
#include "grasp_ctrl.h"
void *sil_new(unsigned pw_open, unsigned pw_closed, unsigned target_counts) {
    grasp_ctrl_t *c = malloc(sizeof *c);
    grasp_ctrl_init(c, (uint16_t)pw_open, (uint16_t)pw_closed, gc_counts_to_mS((uint16_t)target_counts));
    return c;
}
void sil_start(void *c, unsigned baseline) { grasp_ctrl_start((grasp_ctrl_t *)c, (uint16_t)baseline); }
unsigned sil_step(void *c, unsigned peak) { return grasp_ctrl_update((grasp_ctrl_t *)c, (uint16_t)peak, 0); }
int sil_state(void *c) { return (int)((grasp_ctrl_t *)c)->state; }
void sil_free(void *c) { free(c); }
