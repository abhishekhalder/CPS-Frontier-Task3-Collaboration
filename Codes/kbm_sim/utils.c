/*
 * Helper functions and definitions for KBM control
 *
 * Author: Georgiy Antonovich Bondar
 * Date  : 02-28-2023
 *
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "utils.h"

/*
 * Blocking delay with millisecond precision
 *
 * @param ms the number of milliseconds to delay for
 * @return 0 upon success
 */
int delay_ms(int ms){
  long pause;
  clock_t now,then;

  pause = ms*(CLOCKS_PER_SEC/1000);
  now = then = clock();
  while( (now-then) < pause )
    now = clock();

  return 0;
}

/*
 * Clip x to [-abs(lim), abs(lim)]
 *
 * @param x the variable to clip
 * @param lim the quantity to clip to
 */
double clip_abs(double x, double lim){
  if(x > fabs(lim))
    return fabs(lim);
  else if(x < -1.0*fabs(lim))
    return -1.0*fabs(lim);
  else
    return x;
}
  
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

/*
 * Gets index of nearest point on trajectory
 */
int get_nearest_ind(Env_KBM * env, double *wpts){
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
/*
 * Read a trajectory, a file with the format
 * (double xpos) (double ypos) (double speed)
 *
 * @param traj_file the file to read from
 * @param wpts the 2d array into which to write the (x,y) coordinates
 * @param spds the array into which to write the speeds
 */
int read_trajectory(char * traj_file, double * wpts, double * spds ){
  FILE *fp = fopen(traj_file, "r");

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

