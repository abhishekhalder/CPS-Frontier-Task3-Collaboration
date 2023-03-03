/*
 * Simulation of Kinematic Bicycle Model control.
 *
 * Author: Georgiy Antonovich Bondar
 * Date  : 03-02-2023
 *
 */
#include <stdio.h>
#include <stdlib.h>

#include "sim_config.h"
#include "environment_kbm.h"
#include "controller_kbm.h"


int main(){

  Env_KBM env;
  Cnt_Out ctrl;
  
  // Open file for logging
  FILE *fp = fopen(OUTFILE, "w");

  env_kbm_init( &env );
  controller_kbm_init();

  int i=0;
  while( 1 ){

    // Run controller
    controller_kbm_update( &env, &ctrl );
    
    // Update environment
    env_kbm_update( &env, &ctrl );
    
		// Log output
		printf("[%d] x=%.2f y=%.2f yaw=%.2f v=%.2f\n", i, env.x, env.y, env.yaw, env.v);
		fprintf(fp, "%f %f %f %f %f %f %f\n", DT*i, env.x, env.y, env.yaw, env.v, ctrl.throttle, ctrl.delta);

    if( RUN_TIME != -1 && i > NUM_TIMESTEPS )
      break;

    i++;
  }
  
  controller_kbm_deinit();

  fclose(fp);

  return 0;

}

