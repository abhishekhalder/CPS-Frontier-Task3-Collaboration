/*
 * Functions and definitions for stanley lateral controller
 *
 * Author: Georgiy Antonovich Bondar
 * Date  : 02-21-2023
 *
 */
#include <stdio.h>
#include <math.h>
#include "stanley.h"
#include "../../utils.h"

double stanley_update(Stanley *stanley, double x, double y, double yaw, double speed, int n_i){
  double yaw_path, yaw_diff, yaw_diff_crosstrack, crosstrack_error;
  double yaw_cross_track, yaw_path2ct;
  double * wpts = stanley->waypoints;
  int len = stanley->wpts_len;
  double delta;

  n_i = 0;

  // Calculate heading error
  yaw_path = atan2(*(wpts+2*(len-1)+1)-*(wpts+1), *(wpts+2*(len-1))-*wpts);
  yaw_diff = yaw_path - yaw;
  if(yaw_diff > M_PI)
    yaw_diff -= 2*M_PI;
  if(yaw_diff < -M_PI)
    yaw_diff += 2*M_PI;

  // Calculate crosstrack error
  crosstrack_error = distance(x, y, *(wpts+2*n_i), *(wpts+2*n_i+1));
  yaw_cross_track = atan2(y-*(wpts+1), x-*wpts);
  yaw_path2ct = yaw_path - yaw_cross_track;
  if(yaw_path2ct > M_PI)
    yaw_path2ct -= 2*M_PI;
  if(yaw_path2ct < -M_PI)
    yaw_path2ct += 2*M_PI;
  if(yaw_path2ct > 0)
    crosstrack_error = fabs(crosstrack_error);
  else
    crosstrack_error = -1*fabs(crosstrack_error);

  yaw_diff_crosstrack = atan(stanley->Ke * crosstrack_error / (stanley->Kv + speed));
  //printf("%.2f | %.2f | %.2f\n",crosstrack_error, yaw_diff, yaw_diff_crosstrack);

  // Control
  delta = yaw_diff + yaw_diff_crosstrack;
  if(delta > M_PI)
    delta -= 2*M_PI;
  if(delta < -M_PI)
    delta += 2*M_PI;

  delta = (1.22 < delta) ? 1.22 : delta;
  delta = (-1.22 > delta) ? -1.22 : delta;

  return delta;
}
