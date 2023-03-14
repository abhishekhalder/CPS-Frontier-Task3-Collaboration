/*
 * MPC path following controller
 *
 * Author: Georgiy Antonovich Bondar
 * Date  : 03-12-2023
 */
#include "../sim_config.h"

/* MPC Parameters */
#define MPC_N (size_t)4
#define MPC_LOOKAHEAD_PTS 10

/* MPC Reference Values */
#define MPC_REF_CTE         0.0
#define MPC_REF_EPSI        0.0
#define MPC_REF_V           25.0

/* MPC Costs */
#define MPC_COST_CTE        3000
#define MPC_COST_EPSI       500
#define MPC_COST_V          200 // 1
#define MPC_COST_DELTA      1
#define MPC_COST_DIFF_DELTA 200
#define MPC_COST_A          120 // 1
#define MPC_COST_DIFF_A     1

int ctrl_mpc_init(Env_KBM * env, Cnt_Out * ctrl);

int ctrl_mpc_update(Env_KBM * env, Cnt_Out * ctrl);

int ctrl_mpc_deinit();
