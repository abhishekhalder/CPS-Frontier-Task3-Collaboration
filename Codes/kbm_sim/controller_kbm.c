 /*
 * Functions and definitions for the Kinematic Bicycle Model controller
 *
 * Author: Georgiy Antonovich Bondar
 * Date  : 03-02-2023
 *
 */
#include "utils.h"
#include "sim_config.h"
#include "controllers_kbm/ctrl_pid_plus_stanley.h"

Controllers controller_type = CONTROLLER_TYPE;

/*
 * Initializes KBM controller
 */
void controller_kbm_init(Env_KBM * env, Cnt_Out * ctrl){
  switch (controller_type){
    case CONTROLLER_PID_PLUS_STANLEY:
    default:
      ctrl_pid_plus_stanley_init(env, ctrl);
  }
}

/*
 * De-initializes KBM controller
 */
void controller_kbm_deinit(){
  switch (controller_type){
    case CONTROLLER_PID_PLUS_STANLEY:
    default:
      ctrl_pid_plus_stanley_deinit();
  }
}

/*
 * Runs the appropriate controller to set outputs to environment
 *
 * @param env pointer to environment struct
 * @param ctrl pointer to control output struct
 */
void controller_kbm_update(Env_KBM * env, Cnt_Out * ctrl){
  switch (controller_type){
    case CONTROLLER_PID_PLUS_STANLEY:
    default:
      ctrl_pid_plus_stanley_update(env, ctrl);
  }
  // Clip outputs
  ctrl->throttle = clip_abs(ctrl->throttle, CTRL_MAX_THROTTLE);
  ctrl->delta = clip_abs(ctrl->delta, CTRL_MAX_DELTA);
}

