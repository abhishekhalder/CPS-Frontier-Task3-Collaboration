/*
 * Functions and definitions for stanley lateral controller
 *
 * Author: Georgiy Antonovich Bondar
 * Date  : 02-21-2023
 *
 */
typedef struct Stanley{
  double Ke;
  double Kv;
  double * waypoints;
  double wpts_len;
} Stanley;

double stanley_update(Stanley *stanley, double x, double y, double yaw, double speed, int n_i);

