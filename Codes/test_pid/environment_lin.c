/*
 * Functions and definitions for the linear kinematic bicycle model
 *
 * Author: Georgiy Antonovich Bondar
 * Date  : 02-20-2023
 *
 */
//#define _USE_MATH_DEFINES
#include "utils.h"
#include "environment_lin.h"

/*
 * Update environment state based on control inputs
 *
 * @param throttle linear acceleration
 * @param delta desired steering angle
 * @return 0 on success, else nonzero
 */
int lbm_update(LinearBicycleModel *lbm, double throttle, double delta){

  //delta = (delta > MAX_DELTA) ? MAX_DELTA : ((delta < -MAX_DELTA) ? -MAX_DELTA : delta);

	lbm->x += lbm->v * cos(lbm->yaw) * DT;
	lbm->y += lbm->v * sin(lbm->yaw) * DT;
	lbm->yaw += lbm->v/LEN_WB * tan(delta) * DT;
	normalize_angle(&(lbm->yaw));
  lbm->v += throttle * DT;
	
	return 0;	
}



