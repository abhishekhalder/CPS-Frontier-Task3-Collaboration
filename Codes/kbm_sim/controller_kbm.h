 /*
 * Functions and definitions for the Kinematic Bicycle Model controller
 *
 * Author: Georgiy Antonovich Bondar
 * Date  : 03-02-2023
 */
#ifndef CTRLKBM_H
#define CTRLKBM_H

#include "sim_config.h"

/*
 * Initializes KBM controller
 */
void controller_kbm_init();

/*
 * De-initializes KBM controller
 */
void controller_kbm_deinit();

/*
 * Runs the appropriate controller to set outputs to environment
 *
 * @param env pointer to environment struct
 * @param ctrl pointer to control output struct
 */
void controller_kbm_update(Env_KBM * env, Cnt_Out * ctrl);

#endif

