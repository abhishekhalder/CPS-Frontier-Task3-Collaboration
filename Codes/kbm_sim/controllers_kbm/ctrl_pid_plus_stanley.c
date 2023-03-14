/*
 * PID (acceleration) + Stanley (lateral) path following controller
 *
 * Author: Georgiy Antonovich Bondar
 * Date  : 03-02-2023
 */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "../utils.h"
#include "ctrl_pid_plus_stanley.h"
#include "controllers_basic/stanley.h"
#include "controllers_basic/pid.h"

static PID pid;
static Stanley lat;

static double * waypoints;
static double * speeds;

static double get_desired_speed(double *spds, int min_ind);

int ctrl_pid_plus_stanley_init(Env_KBM * env, Cnt_Out * ctrl){
  waypoints = malloc(TRAJ_LEN*2*sizeof(double));
  speeds = malloc(TRAJ_LEN*1*sizeof(double));

  char * traj_file = malloc(50);
  strcpy(traj_file, "../");
  strcat(traj_file, (const char *)TRAJ_FILE);
  
  // Read trajectory
  if(read_trajectory(TRAJ_FILE, waypoints, speeds) != TRAJ_LEN){
    printf("Unable to read trajectory from file \"%s\". Exiting...\n", TRAJ_FILE);
    return 1;
  }
  printf("Read trajectory of length %d from file \"%s\".\n", TRAJ_LEN, TRAJ_FILE);
  
  // Initialize PID controller (throttle control)
  pid.Kp = KP;
  pid.Kd = KD;
  pid.Ki = KI;
  pid.dt = DT;
  pid.e_size = 0;

  // Initialize Stanley controller (lateral control)
  lat.Ke = STANLEY_KE;
  lat.Kv = STANLEY_KV;
  lat.waypoints = waypoints;
  lat.wpts_len = STANLEY_LOOKAHEAD_PTS;

  free(traj_file);
  return 0;
}

int ctrl_pid_plus_stanley_update(Env_KBM * env, Cnt_Out * ctrl){
  double des_spd;
  int nearest_ind;

  // Find nearest index
  nearest_ind = get_nearest_ind(env, waypoints);

  // Get desired speed
  des_spd = get_desired_speed(speeds, nearest_ind);

	// Run PID controller for throttle and clip output
  ctrl->throttle = pid_update(&pid, des_spd, env->v);

  // Run stanley controller for yaw and clip output
  if(nearest_ind+lat.wpts_len >= TRAJ_LEN)
    lat.waypoints = waypoints;
  else
    lat.waypoints=waypoints+2*nearest_ind;

  ctrl->delta = stanley_update(&lat, env->x, env->y, env->yaw, env->v, nearest_ind);
  
  return 0;
}

int ctrl_pid_plus_stanley_deinit(){
  if(waypoints) free(waypoints);
  if(speeds) free(speeds);

  return 0;
}

static double get_desired_speed(double *spds, int min_ind){
  return *(spds+min_ind);
}

