 /*
 * Functions and definitions for the Kinematic Bicycle Model
 *
 * Author: Georgiy Antonovich Bondar
 * Date  : 03-02-2023
 *
 */
#ifndef ENVKBM_H
#define ENVKBM_H

#include "sim_config.h"

/*
 * Initialize the KBM enironment
 */
void env_kbm_init(Env_KBM * env);

/*
 * Update environment based on control inputs
 *
 * @param env pointer to environment struct
 * @param ctrl pointer to control output struct
 */
void env_kbm_update(Env_KBM * env, Cnt_Out * ctrl);

#endif 

