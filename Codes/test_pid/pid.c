/*
 * Functions and definitions for PID controller
 *
 * Author: Georgiy Antonovich Bondar
 * Date  : 02-21-2023
 *
 */
#include <string.h>
#include "pid.h"

/*
 * Run PID controller
 *
 * @param state_des the desired state
 * @param state the current state
 * @return the PID controller output
 */
double pid_update(PID *pid, double state_des, double state){
  double out;

  double e = state_des - state;
  double de = 0.0;
  double ie = 0.0;

  // Append state error to buffer
  if(pid->e_size < PID_BUFF_LEN){
    pid->e_buff[pid->e_size++] = e;
  }
  else{
    memmove(&(pid->e_buff), &(pid->e_buff[1]), PID_BUFF_LEN-1);
    pid->e_buff[PID_BUFF_LEN-1] = e;
  }

  // Compute integral/derivative
  if(pid->e_size >= 2){
    de = (pid->e_buff[pid->e_size-1]-pid->e_buff[pid->e_size-2]) / pid->dt;
    for(int i=0;i<pid->e_size; i++) ie += pid->e_buff[i];
    ie *= pid->dt;
  }

  out = (pid->Kp * e) + (pid->Kd * de) + (pid->Ki * ie);
  out = (pid->Kp * e) + (pid->Kd * de / pid->dt) + (pid->Ki * ie * pid->dt);
 
  return out;
}

