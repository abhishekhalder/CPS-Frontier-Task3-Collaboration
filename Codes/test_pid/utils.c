/*
 * Helper functions and definitions for KBM control
 *
 * Author: Georgiy Antonovich Bondar
 * Date  : 02-28-2023
 *
 */
#include <math.h>
#include "utils.h"
  
/*
 * Normalize given angle to interval [-pi,pi]
 */
int normalize_angle(double * angle){

  *angle = fmod(*angle+M_PI, 2*M_PI);
  if (*angle<0) *angle += 2*M_PI;
  *angle -= M_PI;

  return 0;
}

/*
 * Compute distance between two points
 */
double distance(double x1, double y1, double x2, double y2){
  return sqrt( pow(x2-x1,2) + pow(y2-y1,2) );
}

