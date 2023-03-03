/*
 * PID (acceleration) + Stanley (lateral) path following controller
 *
 * Author: Georgiy Antonovich Bondar
 * Date  : 03-02-2023
 */
#include "../sim_config.h"

/* PID Parameters */
#define KP  (double)1
#define KD  (double)0.001
#define KI  (double)0.3

/* Stanley Gains */
#define STANLEY_KE (double)0.3
#define STANLEY_KV (double)20.0
#define STANLEY_LOOKAHEAD_PTS 10

int ctrl_pid_plus_stanley_init(Env_KBM * env, Cnt_Out * ctrl);

int ctrl_pid_plus_stanley_update(Env_KBM * env, Cnt_Out * ctrl);

int ctrl_pid_plus_stanley_deinit();
