/*
 * Functions and definitions for the linear kinematic bicycle model
 *
 * Author: Georgiy Antonovich Bondar
 * Date  : 02-20-2023
 *
 */
#include "utils.h"
#include "sim_config.h"
#include "environment_kbm.h"

/*
 * Initialize the KBM enironment
 */
void env_kbm_init(Env_KBM * env){
  // Set environment parameters
  env->x = ENV_X_INIT;
  env->y = ENV_Y_INIT;
  env->yaw = ENV_YAW_INIT;
  env->v = ENV_V_INIT;
}

/*
 * Update environment based on control inputs
 *
 * @param cnt pointer to control output struct
 */
void env_kbm_update(Env_KBM * env, Cnt_Out * ctrl){

	env->x += env->v * cos(env->yaw) * DT;
	env->y += env->v * sin(env->yaw) * DT;
	env->yaw += env->v/ENV_LEN_WB * tan(ctrl->delta) * DT;
	normalize_angle(&(env->yaw));
  env->v += ctrl->throttle * DT;
	
}

