/*
 * Simulation of Kinematic Bicycle Model control.
 *
 * Author: Georgiy Antonovich Bondar
 * Date  : 03-02-2023
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "sim_config.h"
#include "environment_kbm.h"
#include "controller_kbm.h"
#include "utils.h"

/* External variables */
char TRAJ_FILE[50] = TRAJ_FILE_DEFAULT;
int TRAJ_LEN = TRAJ_LEN_DEFAULT;

int load_sim_params( void );

int main(){

  Env_KBM env;
  Cnt_Out ctrl;
  clock_t start, end;
  int looptime_ms;

  // Load any configurable parameters
  if( load_sim_params() ){
    printf("(-) Failed loading simulation parameters from %s. Terminating...\n", SIM_CONFIGURABLE);
    return 1;
  }
  
  // Open file for logging
  FILE *fp = fopen(OUTFILE, "w");

  env_kbm_init( &env );
  controller_kbm_init();

  int i=0;
  while( 1 ){

    start = clock();

    // Run controller
    controller_kbm_update( &env, &ctrl );
    
    // Update environment
    env_kbm_update( &env, &ctrl );
    
		// Log output
		if(TERMLOG_EN) printf("[%d] x=%.2f y=%.2f yaw=%.2f v=%.2f\n", i, env.x, env.y, env.yaw, env.v);

		if(FILELOG_EN) fprintf(fp, "%f %f %f %f %f %f %f\n", DT*i, env.x, env.y, env.yaw, env.v, ctrl.throttle, ctrl.delta);

    if( RUN_TIME != -1 && i > NUM_TIMESTEPS )
      break;

    end = clock();
    looptime_ms = (int)(((double)(end-start)) / CLOCKS_PER_SEC * 1000);
    // printf("Control loop took %dms\n", looptime_ms);

    // Delay
    if(TS_DELAY_MS > 0){ // Naive delay
      // printf("Naive delaying for %dms\n", TS_DELAY_MS);
      delay_ms(TS_DELAY_MS);
    }
    else if(TS_DELAY_MS < 0 && looptime_ms < -1*TS_DELAY_MS){ // Account for controller execution time
      printf("Delaying for %dms\n", -1*TS_DELAY_MS - looptime_ms);
      delay_ms(-1*TS_DELAY_MS - looptime_ms);
    }

    i++;
  }
  
  controller_kbm_deinit();

  fclose(fp);

  printf("(*) Simulation successfully run for %d timesteps. Terminating...\n", (int)NUM_TIMESTEPS);

  return 0;

}

/*
 * Load configurable parameters from config file
 *
 * @return 0 on success, else 1
 */
int load_sim_params( void ){
  FILE *fp = fopen(SIM_CONFIGURABLE, "r");
  char * line = NULL;
  size_t len = 0;
  ssize_t read;

  if( !fp ) return 1;

  // Read the trajectory/waypoints file (currently only configurable)
  read = getline(&line, &len, fp);
  if(read != -1){
    memcpy(TRAJ_FILE, line, read-1);
  }
  read = getline(&line, &len, fp);
  if(read != -1){
    TRAJ_LEN = (int) strtol(line, NULL, 10);
  }

  fclose(fp);
  if(line) free(line);

  return 0;
}

