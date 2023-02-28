 /*
 * Functions and definitions for the linear kinematic bicycle model
 *
 * Author: Georgiy Antonovich Bondar
 * Date  : 02-20-2023
 *
 */
#define _USE_MATH_DEFINES
#include <math.h>

#define MAX_DELTA 	(M_PI/4)	// radians
#define MAX_THROTTLE  1
#define LEN_WB		(double)3.0	// wheelbase 
//#define Lr		(L/2.0)		// 
//#define Lf		(L-Lr)		// 
#define DT		(double)0.1

typedef struct __LinearBicycleModel{
	double x;
	double y;
	double yaw;
	double v;
} LinearBicycleModel;

/*
 * Update environment state based on control inputs
 *
 * @param throttle linear acceleration
 * @param delta desired steering angle
 * @return 0 on success, else nonzero
 */
int lbm_update(LinearBicycleModel *lbm, double throttle, double delta);

