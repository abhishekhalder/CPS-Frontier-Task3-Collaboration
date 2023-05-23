/*
 * Configuration for simulation and for implemented controllers 
 *
 * Author: Georgiy Antonovich Bondar
 * Date  : 03-02-2023
 *
 */
#ifndef SIMCONF_H
#define SIMCONF_H

#define _USE_MATH_DEFINES
#include <stdbool.h>
#include <math.h>

/* Simulation Settings ( ~ ) */
#define RUN_TIME          0.3 // 120
#define DT		            (double)0.1
#define NUM_TIMESTEPS     (RUN_TIME/DT)
#define OUTFILE           "run_data.txt"
#define SIM_CONFIGURABLE  "sim_config.cfg"
#define TERMLOG_EN        false
#define FILELOG_EN        true
#define TS_DELAY_MS       -100

/* Controller Selection (controller_kbm.c) */
typedef enum __Controllers {
  CONTROLLER_PID_PLUS_STANLEY,
  CONTROLLER_MPC
} Controllers;
extern Controllers controller_type;
#define CONTROLLER_TYPE (Controllers)CONTROLLER_MPC
// #define CONTROLLER_TYPE (Controllers)CONTROLLER_PID_PLUS_STANLEY

/* Environment Initialization (environment_kbm.c) */
#define ENV_X_INIT    (double)1 // -180.0
#define ENV_Y_INIT    (double)0 // 82
#define ENV_YAW_INIT  (double)0 // -5.0/8.0*M_PI
#define ENV_V_INIT    (double)10.0
#define ENV_LEN_WB    (double)3.0	// wheelbase 

/* Controller (controller_kbm.c) */
#define CTRL_MAX_THROTTLE (double)1.0
#define CTRL_MAX_DELTA 	  (M_PI/4)

/* Trajectory file */
// #define TRAJ_FILE_DEFAULT "sim_paths/racetrack_waypoints.txt"
// #define TRAJ_LEN_DEFAULT 1724
#define TRAJ_FILE_DEFAULT "sim_paths/path_wypts_0.txt"
#define TRAJ_LEN_DEFAULT  200
extern char TRAJ_FILE[50];
extern int TRAJ_LEN;


/* Environment Struct */
typedef struct __KinematicBicycleModel{
	double x;   // x position (m)
	double y;   // y position (m)
	double yaw; // steering angle (rads)
	double v;   // forward speed (m/s)
} KinematicBicycleModel;

typedef KinematicBicycleModel Env_KBM;

/* Controller Output Struct */
typedef struct __Cnt_Out{
  double throttle; // acceleration (m/s^2)
  double delta;    // steering angle change (rads)
} Cnt_Out;

#endif

