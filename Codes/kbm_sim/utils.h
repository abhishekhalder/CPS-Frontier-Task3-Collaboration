/*
 * Helper functions and definitions for KBM control
 *
 * Author: Georgiy Antonovich Bondar
 * Date  : 02-28-2023
 *
 */

/*
 * Clip x to [-abs(lim), abs(lim)]
 *
 * @param x the variable to clip
 * @param lim the quantity to clip to
 */
double clip_abs(double x, double lim);

/*
 * Normalize given angle to interval [-pi,pi]
 */
int normalize_angle(double * angle);

/*
 * Compute distance between two points
 */
double distance(double x1, double y1, double x2, double y2);

/*
 * Read a trajectory, a file with the format
 * (double xpos) (double ypos) (double speed)
 *
 * @param traj_file the file to read from
 * @param wpts the 2d array into which to write the (x,y) coordinates
 * @param spds the array into which to write the speeds
 * @return the number of lines read
 */
int read_trajectory(char * traj_file, double * wpts, double * spds );

