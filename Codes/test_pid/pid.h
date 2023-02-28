/*
 * Functions and definitions for PID controller
 *
 * Author: Georgiy Antonovich Bondar
 * Date  : 02-15-2023
 *
 */
#define PID_BUFF_LEN  20

typedef struct __PID{
  double Kp, Kd, Ki;
  double dt;
  int e_size;
  double e_buff[PID_BUFF_LEN];
} PID;

/*
 * Run PID controller
 *
 * @param state_des the desired state
 * @param state the current state
 * @return the PID controller output
 */
double pid_update(PID *pid, double state_des, double state);

