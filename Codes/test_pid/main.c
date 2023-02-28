/*
 * Function for running PID controller on the kinematic bicycle model.
 *
 * Author: Georgiy Antonovich Bondar
 * Date  : 02-15-2023
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include "environment_lin.h"
#include "stanley.h"
#include "utils.h"
#include "pid.h"

#define RUN_TIME      120
#define NUM_TIMESTEPS (RUN_TIME/DT)
#define OUTFILE       "run_data.txt"
#define WAYPT_FILE    "racetrack_waypoints.txt"

// PID gains
#define KP  (double)1
#define KD  (double)0.001
#define KI  (double)0.3

// Stanley gains
#define STANLEY_KE (double)0.3
#define STANLEY_KV (double)20.0
#define LOOKAHEAD_PTS 10

#define TRAJ_FILE "racetrack_waypoints.txt"
#define TRAJ_LEN 1724

int read_trajectory(double * wpts, double * spds ){
  FILE *fp = fopen(TRAJ_FILE, "r");

  char *line = NULL;
  size_t len = 0;
  int read = 0;

  while( getline(&line, &len, fp) != -1 ){
    if(sscanf(line, "%lf, %lf, %lf", wpts+2*read, wpts+2*read+1, spds+read) != 3){
      printf("Error parsing line %d!\n", read);
    }
    read++;
  }

  fclose(fp);
  if(line) free(line);

  return read;
}

int get_nearest_ind(LinearBicycleModel * env, double *wpts){
  double curr_dist;
  double min_dist = -1.0;
  int min_ind = 0;

  for(int i=0;i<TRAJ_LEN;i++){
    curr_dist = distance(env->x, env->y, *(wpts+2*i), *(wpts+2*i+1));
    if(min_dist < 0 || curr_dist < min_dist){
      min_dist = curr_dist;
      min_ind = i;
    }
  }

  return min_ind;
}

double get_desired_speed(double *spds, int min_ind){
  return *(spds+min_ind);
}

int main(){

  int nearest_ind;

  double des_spd;
  double throttle = 0.0;
  double delta = 0.0;

  double * waypoints = malloc(TRAJ_LEN*2*sizeof(double));
  double * speeds = malloc(TRAJ_LEN*1*sizeof(double));

  // Read trajectory
  if(read_trajectory(waypoints, speeds) != TRAJ_LEN){
    printf("Unable to read trajectory from file \"%s\". Exiting...\n", TRAJ_FILE);
    return 1;
  }
  printf("Read trajectory of length %d from file \"%s\".\n", TRAJ_LEN, TRAJ_FILE);

  /*for(int i=0;i<10;i++)
    printf("%lf,%lf\n",*(waypoints+2*i), *(waypoints+2*i+1));

  for(int i=0;i<10;i++)
    printf("%lf\n",*(speeds+i));*/

  // Set path to follow
  //double path[3][2] = { {0.0,0.0}, {0.0,10.0}, {5.0,5.0} };
  
  // Open file for logging
  FILE *fp = fopen(OUTFILE, "w");

  // Set environment parameters
  LinearBicycleModel env = { .x=-180, .y=82, .yaw=-3.0/4.0*M_PI /*M_PI/4*/, .v=1.5 };

  // Initialize PID controller (throttle control)
  PID pid = { .Kp=KP, .Kd=KD, .Ki=KI, .dt=DT, .e_size=0 };

  // Initialize PP controller (lateral control)
  //PurePursuit pp = { .ld = 6, .L=LEN_WB, .thresh=0.5, .path=&path[0][0], .last_target=0, .path_len=3 };
  Stanley lat = { .Ke=STANLEY_KE, .Kv=STANLEY_KV, .waypoints=waypoints, .wpts_len=LOOKAHEAD_PTS };

  /*for(double k=-10.0; k<10.0; k+=0.5){
    double kk = k;
    normalize_angle(&kk);
    printf("%.2f -> %.2f\n", k, kk);
  }
  return 0;*/

	for(int i=0; i<NUM_TIMESTEPS; i++){
    // Find nearest index
    nearest_ind = get_nearest_ind(&env, waypoints);

    // Get desired speed
    des_spd = get_desired_speed(speeds, nearest_ind);

		// Run PID controller for throttle and clip output
    throttle = pid_update(&pid, des_spd, env.v);
    if(throttle > MAX_THROTTLE) throttle = MAX_THROTTLE;
    else if(throttle < -MAX_THROTTLE) throttle = -MAX_THROTTLE;

    // Run stanley controller for yaw and clip output
    if(nearest_ind+LOOKAHEAD_PTS >= TRAJ_LEN)
      lat.waypoints = waypoints;
    else
      lat.waypoints=waypoints+2*nearest_ind;
    delta = stanley_update(&lat, env.x, env.y, env.yaw, env.v, nearest_ind);
    //printf("%.2f\n",delta);
    //delta = purepursuit_update(&pp, env.x, env.y, env.yaw);
    if(delta > MAX_DELTA) delta = MAX_DELTA;
    else if(delta < -MAX_DELTA) delta = -MAX_DELTA;

		// Apply control inputs, update state
		lbm_update(&env, throttle, delta);
		// Set output
		printf("[%d] x=%.2f y=%.2f yaw=%.2f v=%.2f\n", i, env.x, env.y, env.yaw, env.v);
		fprintf(fp, "%f %f %f %f %f %f %f\n", DT*i, env.x, env.y, env.yaw, env.v, throttle, delta);
		//fprintf(fp, "%f %f %f %f %f\n", DT*i, env.x, env.y, env.yaw, env.v);
	}

  fclose(fp);
  free(waypoints);
  free(speeds);

  return 0;
}
