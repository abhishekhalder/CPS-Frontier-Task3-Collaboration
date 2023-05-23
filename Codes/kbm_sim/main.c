/*
 * Simulation of Kinematic Bicycle Model control.
 *
 * Author: Georgiy Antonovich Bondar
 * Date  : 03-02-2023
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "sim_config.h"
#include "environment_kbm.h"
#include "controller_kbm.h"
#include "utils.h"

// remove below
#include <x86intrin.h>
#define BUFFER_SIZE (40 * 1024 * 1024) // 40MB buffer
void
flush_cache()
{
    char *buffer;

    // Allocate the buffer
    buffer = (char *)malloc(BUFFER_SIZE);

    if (buffer == NULL) {
        printf("Error: Failed to allocate buffer.\n");
        return -1;
    }

    // Initialize the buffer
    memset(buffer, 0, BUFFER_SIZE);

    // Flush the buffer from all cache levels
    for (int i = 0; i < BUFFER_SIZE; i += 64) {
        _mm_clflush(&buffer[i]);
    }

    // Free the buffer
    free(buffer);
}
//----------------------


int main(){

  Env_KBM env;
  Cnt_Out ctrl;
  clock_t start_app, end_app;
  clock_t start, end;
  double looptime_s;
  long double total_time_s = 0;
  double apptime_ms;

  start_app = clock();

  // Open file for logging
  FILE *fp = fopen(OUTFILE, "w");

  env_kbm_init( &env );
  controller_kbm_init();

  printf("WARNING WARNING, THIS HAS CACHE FLUSHING IN, REMOVE ME! THIS WAS JUST FOR TESTING!\n");

  int i=1;
  while( 1 ){

    // TODO REMOVE ME, THIS IS JUST FOR TESTING!!!! flush caches
    //flush_cache();

    start = clock();

    // Run controller
    controller_kbm_update( &env, &ctrl );

    // Update environment
    env_kbm_update( &env, &ctrl );

		// Log output
		if(TERMLOG_EN) printf("[%d] x=%.2f y=%.2f yaw=%.2f v=%.2f\n", i, env.x, env.y, env.yaw, env.v);

		if(FILELOG_EN) fprintf(fp, "%f %f %f %f %f %f %f\n", DT*i, env.x, env.y, env.yaw, env.v, ctrl.throttle, ctrl.delta);

    if( RUN_TIME != -1 && (i > NUM_TIMESTEPS) ) {
      break;
    }

    end = clock();
    //looptime_ms = (int)(((double)(end-start)) / CLOCKS_PER_SEC * 1000);
    looptime_s = ((double)(end-start)) / CLOCKS_PER_SEC;
    total_time_s += looptime_s;
    // printf("[T_FLAG] %f %Lf\n", looptime_s, total_time_s);

    // Delay
    if(TS_DELAY_MS > 0){ // Naive delay
      // printf("Naive delaying for %dms\n", TS_DELAY_MS);
     // delay_ms(TS_DELAY_MS);
    }
    else if(TS_DELAY_MS < 0 && looptime_s < -1*TS_DELAY_MS){ // Account for controller execution time
      //printf("Delaying for %dms\n", -1*TS_DELAY_MS - looptime_s);
      //delay_ms(-1*TS_DELAY_MS - looptime_s);
    }

    i++;
  }

  controller_kbm_deinit();

  fclose(fp);

  end_app = clock();
  apptime_ms = ((double)(end_app-start_app)) / CLOCKS_PER_SEC * 1000;
  printf("Overall time: %fms\n", apptime_ms);

  printf("(*) Simulation successfully run for %d timesteps. Terminating...\n", (int)NUM_TIMESTEPS);

  return 0;

}

