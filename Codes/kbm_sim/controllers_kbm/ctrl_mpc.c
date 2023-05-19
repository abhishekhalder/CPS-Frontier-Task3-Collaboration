/*
 * MPC path following controller
 *
 * Author: Georgiy Antonovich Bondar
 * Date  : 03-12-2023
 */
#include "IpStdCInterface.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "../environment_kbm.h"
#include "../utils.h"
#include "ctrl_mpc.h"

// Set the timestep length and duration
const size_t N = MPC_N;
const double dt = DT;

// Set cost factors
const int cost_cte_factor = MPC_COST_CTE;
const int cost_epsi_factor = MPC_COST_EPSI;
const int cost_v_factor = MPC_COST_V;
const int cost_current_delta_factor = MPC_COST_DELTA;
const int cost_diff_delta_factor = MPC_COST_DIFF_DELTA;
const int cost_current_a_factor = MPC_COST_A;
const int cost_diff_a_factor = MPC_COST_DIFF_A;

// Reference cross-track error and orientation error
double ref_cte = MPC_REF_CTE;
double ref_epsi = MPC_REF_EPSI;
double ref_v = MPC_REF_V;

//const size_t x_start = 0;
//const size_t y_start = x_start + N;
//const size_t psi_start = y_start + N;
//const size_t v_start = psi_start + N;
//const size_t cte_start = v_start + N;
//const size_t epsi_start = cte_start + N;
//const size_t delta_start = epsi_start + N;
//const size_t a_start = delta_start + N - 1;

#define x_start     0
#define y_start     (x_start + N)
#define psi_start   (y_start + N)
#define v_start     (psi_start + N)
#define cte_start   (v_start + N)
#define epsi_start  (cte_start + N)
#define delta_start (epsi_start + N)
#define a_start     (delta_start + N - 1)

static double * waypoints;
static double * speeds;

//
// Needful functions
//
static void get_cte_and_epsi(double x, double y, double yaw, double v,double * cte, double * epsi);

static
bool eval_f(
   ipindex     n,
   ipnumber*   x,
   bool        new_x,
   ipnumber*   obj_value,
   UserDataPtr user_data
)
{
  // printf("(*) In eval_f\n");

  *obj_value = 0.0;

  // Cost increases with distance from reference state
  // Multiply squared devation by 'cost factors' to adjust contribution of each deviation to cost
  for (size_t i = 0; i < N; i++) {
    *obj_value += cost_cte_factor*pow(x[cte_start + i] - ref_cte, 2);
    *obj_value += cost_epsi_factor*pow(x[epsi_start + i] - ref_epsi, 2);
    *obj_value += cost_v_factor*pow(x[v_start + i] - ref_v, 2);
  }

  // Cost increases with use of actuators
  for (size_t i = 0; i < N - 1; i++) {
    *obj_value += cost_current_delta_factor*pow(x[delta_start + i], 2);
    *obj_value += cost_current_a_factor*pow(x[a_start + i], 2);
  }

  // Cost increases with value gap between sequential actuators
  for (size_t i=0; i < N-2; i++) {
    *obj_value += cost_diff_delta_factor*pow(x[delta_start + i + 1] - x[delta_start + i], 2);
    *obj_value += cost_diff_a_factor*pow(x[a_start + i + 1] - x[a_start + i], 2);
  }

  return true;
}

static
bool eval_grad_f(
  ipindex     n,
  ipnumber*   x,
  bool        new_x,
  ipnumber*   grad_f,
  UserDataPtr user_data
)
{
  // printf("(*) In eval_grad_f\n");

  // x,y, and psi don't affect the objective function (directly)
  for (size_t i = 0; i < N; i++) {
    grad_f[x_start+i] = 0.0;
    grad_f[y_start+i] = 0.0;
    grad_f[psi_start+i] = 0.0;
  }

  // v, psi, and cte are present in first summation
  for (size_t i = 0; i < N; i++) {
    grad_f[cte_start+i] = 2*cost_cte_factor*(x[cte_start + i] - ref_cte);
    grad_f[epsi_start+i] = 2*cost_epsi_factor*(x[epsi_start + i] - ref_epsi);
    grad_f[v_start+i] = 2*cost_v_factor*(x[v_start + i] - ref_v);
  }

  // delta and a from the 2nd term in the 2nd summation
  for (size_t i = 0; i < N - 1; i++) {
    grad_f[delta_start+i] = 2*cost_current_delta_factor*x[delta_start + i];
    grad_f[a_start+i] = 2*cost_current_a_factor*x[a_start + i];
  }

  grad_f[delta_start] += 2*cost_diff_delta_factor*(x[delta_start] - x[delta_start + 1]);
  grad_f[a_start] += 2*cost_diff_a_factor*(x[a_start] - x[a_start+1]);

  grad_f[delta_start+N-2] += 2*cost_diff_delta_factor*(x[delta_start + N - 2] - x[delta_start + N - 3]);
  grad_f[a_start+N-2] += 2*cost_diff_a_factor*(x[a_start + N - 2] - x[a_start + N - 3]);

  // Cost increases with value gap between sequential actuators
  for (size_t i=1; i < N-2; i++) {
    grad_f[delta_start+i] += 2*cost_diff_delta_factor*(2*x[delta_start + i] - x[delta_start + i - 1] - x[delta_start + i + 1]);
    grad_f[a_start+i] += 2*cost_diff_a_factor*(2*x[a_start + i] - x[a_start + i - 1] - x[a_start + i + 1]);
  }

  return true;
}

static
bool eval_g(
   ipindex     n,
   ipnumber*   x,
   bool        new_x,
   ipindex     m,
   ipnumber*   g,
   UserDataPtr user_data
)
{
  // printf("(*) In eval_g\n");

  Env_KBM env;

  Cnt_Out ctrl;

  // Set initial state 
  g[x_start] = x[x_start];
  g[y_start] = x[y_start];
  g[psi_start] = x[psi_start];
  g[v_start] = x[v_start];
  g[cte_start] = x[cte_start];
  g[epsi_start] = x[epsi_start];
  
  // Update based on model
  for (size_t i = 0; i < N-1; i++) {
    // The state at time t+1 .
    const double x1 = x[x_start + i + 1];
    const double y1 = x[y_start + i + 1];
    const double psi1 = x[psi_start + i + 1];
    const double v1 = x[v_start + i + 1];
    const double cte1 = x[cte_start + i + 1];
    const double epsi1 = x[epsi_start + i + 1];

    // The state at time t.
    const double x0 = x[x_start + i];
    const double y0 = x[y_start + i];
    const double psi0 = x[psi_start + i];
    const double v0 = x[v_start + i];
    const double cte0 = x[cte_start + i];
    const double epsi0 = x[epsi_start + i];

    const double delta0 = x[delta_start + i];
    const double a0 = x[a_start + i];

    env.x = x0;
    env.y = y0;
    env.yaw = psi0;
    env.v = v0;
    ctrl.delta = delta0;
    ctrl.throttle = a0;
    env_kbm_update(&env, &ctrl);
    
    double cte_upd, epsi_upd;
    get_cte_and_epsi(env.x, env.y, env.yaw, env.v,&cte_upd, &epsi_upd);
    // printf("(Errors) CTE=%.2f | EPSI=%.2f\n", cte_upd, epsi_upd);

    // Fill remaining constraints with differences between actual and predicted states
    g[x_start+i+1] = x1 - env.x;
    g[y_start+i+1] = y1 - env.y;
    g[psi_start+i+1] = psi1 - env.yaw;
    g[v_start+i+1] = v1 - env.v;
    g[cte_start+i+1] = cte1 - cte_upd;
    g[epsi_start+i+1] = epsi1 - epsi_upd;
  }

  return true;
}

static
bool eval_jac_g(
   ipindex     n,
   ipnumber*   x,
   bool        new_x,
   ipindex     m,
   ipindex     nele_jac,
   ipindex*    iRow,
   ipindex*    jCol,
   ipnumber*   values,
   UserDataPtr user_data
)
{
  // printf("(*) In eval_jac_g\n");

  if( values == NULL ){
    // Return sparsity structure of the Jacobian
    int cnt = 0;

    for (size_t i = 0; i < 6*N; i++) {
      for (size_t j = 0; j < 6*N+2*(N-1); j++) {
        iRow[cnt] = i;
        jCol[cnt] = j;
        cnt++;
      }
    }
    return true; 

    // for (size_t i = 0; i < N; i++) {
    //   for (size_t j = 0; j < 6; j++) {
    //     iRow[cnt] = 6*i+j;
    //     jCol[cnt] = x_start+i;
    //     cnt++;

    //     iRow[cnt] = 6*i+j;
    //     jCol[cnt] = y_start+i;
    //     cnt++;

    //     iRow[cnt] = 6*i+j;
    //     jCol[cnt] = psi_start+i;
    //     cnt++;

    //     iRow[cnt] = 6*i+j;
    //     jCol[cnt] = v_start+i;
    //     cnt++;

    //     iRow[cnt] = 6*i+j;
    //     jCol[cnt] = cte_start+i;
    //     cnt++;

    //     iRow[cnt] = 6*i+j;
    //     jCol[cnt] = epsi_start+i;
    //     cnt++;

    //     if(i != 0){
    //       iRow[cnt] = 6*i+j;
    //       jCol[cnt] = delta_start+i-1;
    //       cnt++;

    //       iRow[cnt] = 6*i+j;
    //       jCol[cnt] = a_start+i-1;
    //       cnt++;
    //     }
    //   }
    // }

    // return true; 
  }

  return false;
}

static
bool eval_h(
  ipindex     n,
  ipnumber*   x,
  bool        new_x,
  ipnumber    obj_factor,
  ipindex     m,
  ipnumber*   lambda,
  bool        new_lambda,
  ipindex     nele_hess,
  ipindex*    iRow,
  ipindex*    jCol,
  ipnumber*   values,
  UserDataPtr user_data
)
{
  // printf("(*) In eval_h\n");
  return false;
}

// Globals
enum ApplicationReturnStatus status;
IpoptProblem nlp = NULL;
ipindex nele_jac;                    /* number of nonzeros in the Jacobian of the constraints */
ipindex nele_hess;                   /* number of nonzeros in the Hessian of the Lagrangian (lower or upper triangular part only) */
ipindex index_style;                 /* indexing style for matrices */
ipnumber obj;
double vars[6 * (MPC_N) + 2 * ((MPC_N) - 1)];
double vars_lowerbound[6 * (MPC_N) + 2 * ((MPC_N) - 1)];
double vars_upperbound[6 * (MPC_N) + 2 * ((MPC_N) - 1)];
size_t n_vars, n_constraints;

int ctrl_mpc_init(Env_KBM * env, Cnt_Out * ctrl){

  // Read waypoints and speeds from trajectory file
  waypoints = malloc(TRAJ_LEN*2*sizeof(double));
  speeds = malloc(TRAJ_LEN*1*sizeof(double));

  char * traj_file = malloc(50);
  strcpy(traj_file, "../");
  strcat(traj_file, (const char *)TRAJ_FILE);
  
  // Read trajectory
  if(read_trajectory(TRAJ_FILE, waypoints, speeds) != TRAJ_LEN){
    printf("Unable to read trajectory from file \"%s\". Exiting...\n", TRAJ_FILE);
    return 1;
  }
  printf("Read trajectory of length %d from file \"%s\".\n", TRAJ_LEN, TRAJ_FILE);

  // ---------------------------------- //

  // State: [x,y,psi,v,cte,epsi]
  // Actuators: [delta,a]
  n_vars = 6 * N + 2 * (N - 1);
  
  // Set the number of constraints
  n_constraints = 6 * N;
 
  // Setting the lower and upper limits for variables
  for (size_t i = 0; i < delta_start; i++) {
    vars_upperbound[i] = 1.0e19;
    vars_lowerbound[i] = -1.0e19;
  }

  // Steering angle (deltas)
  for (size_t i = delta_start; i < a_start; i++)
  {
    vars_upperbound[i] = CTRL_MAX_DELTA;
    vars_lowerbound[i] = -1 * CTRL_MAX_DELTA;
  }

  // Acceleration
  for (size_t i = a_start; i < n_vars; i++)
  {
    vars_upperbound[i] = CTRL_MAX_THROTTLE;
    vars_lowerbound[i] = -1 * CTRL_MAX_THROTTLE;
  }

  /* set the number of nonzeros in the Jacobian and Hessian */
  // nele_jac = 8 * 6*(N-1) + 6*6*(1);
  nele_jac = n_constraints * n_vars;
  nele_hess = 0; // Not using this

  /* set the indexing style to C-style (start counting of rows and column indices at 0) */
  index_style = 0;

  return 0;
}

int ctrl_mpc_update(Env_KBM * env, Cnt_Out * ctrl){
  // Initial value of the independent variables.
  // SHOULD BE 0 besides initial state.
  for (size_t i = 0; i < n_vars; i++) {
    vars[i] = 0.0;
  }
  
  // Set starting state
  const double x = env->x;
  const double y = env->y;
  const double psi = env->yaw;
  const double v = env->v;

  // Calulate initial yaw and cross-track error
  get_cte_and_epsi(x,y,psi,v,&vars[cte_start],&vars[epsi_start]);
  vars[x_start] = x;
  vars[y_start] = y;
  vars[psi_start] = psi;
  vars[v_start] = v;
  // Create initial prediction with no inputs
  Env_KBM t_env = { .x=x, .y=y, .yaw=psi, .v=v };
  Cnt_Out t_ctrl = { .throttle=0.0, .delta=0.0 };
  for (size_t i = 1; i < N; i++) {
    env_kbm_update(&t_env, &t_ctrl);
    vars[x_start+i] = t_env.x;
    vars[y_start+i] = t_env.y;
    vars[psi_start+i] = t_env.yaw;
    vars[v_start+i] = t_env.v;
    get_cte_and_epsi(t_env.x, t_env.y, t_env.yaw, t_env.v,&vars[cte_start+i], &vars[epsi_start+i]);
  }


  // Setting the lower and upper limits for variables
 
  double constraints_lowerbound[n_constraints];
  double constraints_upperbound[n_constraints];
  for (size_t i = 0; i < n_constraints; i++) {
    constraints_lowerbound[i] = 0;
    constraints_upperbound[i] = 0;
  }
  // Set init state lower and upper limits
  constraints_lowerbound[x_start] = x;
  constraints_lowerbound[y_start] = y;
  constraints_lowerbound[psi_start] = psi;
  constraints_lowerbound[v_start] = v;
  constraints_lowerbound[cte_start] = vars[cte_start];
  constraints_lowerbound[epsi_start] = vars[epsi_start];

  constraints_upperbound[x_start] = x;
  constraints_upperbound[y_start] = y;
  constraints_upperbound[psi_start] = psi;
  constraints_upperbound[v_start] = v;
  constraints_upperbound[cte_start] = vars[cte_start];
  constraints_upperbound[epsi_start] = vars[epsi_start];

  nlp = CreateIpoptProblem(n_vars, vars_lowerbound, vars_upperbound, n_constraints, 
                           constraints_lowerbound, constraints_upperbound, 
                           nele_jac, nele_hess, index_style,
                           &eval_f, &eval_g, &eval_grad_f,
                           &eval_jac_g, &eval_h);

  if( !nlp ){
    if(TERMLOG_EN) printf("(!) Error creaing IPOPT problem!\n");
    return 1;
  }
  else
    if(TERMLOG_EN) printf("(*) Created IPOPT problem.\n");

  // Set NLP options
  AddIpoptNumOption(nlp, "max_cpu_time", 60.0);
  AddIpoptIntOption(nlp, "print_level", 0); // verbosity
  AddIpoptIntOption(nlp, "max_iter", 3000); 
  AddIpoptStrOption(nlp, "jacobian_approximation", "finite-difference-values"); // So that we need not implement eval_jac_g
  AddIpoptStrOption(nlp, "hessian_approximation", "limited-memory"); // So that we need not implement eval_h

  // Solve problem
  status = IpoptSolve(nlp, vars, NULL, &obj, NULL, NULL, NULL, NULL);

  // Print solution
  if(TERMLOG_EN){
    printf("(*) Solved nlp [%d]. Optimal objective evaluation: %.2f\n", status, obj);
    printf("(*) Optimal delta: %.2f\n", vars[delta_start]);
    printf("(*) Optimal accel: %.2f\n", vars[a_start]);
  }
 
  // Set solution
  ctrl->delta = vars[delta_start];
  ctrl->throttle = vars[a_start];

  return 0;
}

int ctrl_mpc_deinit(){

  if( nlp )
    FreeIpoptProblem(nlp);

  return 0;
}

static void get_cte_and_epsi(double x, double y, double yaw, double v,double * cte, double * epsi){
  double yaw_path, yaw_diff, crosstrack_error;
  double yaw_cross_track, yaw_path2ct;

  Env_KBM env = { .x=x, .y=y, .yaw=yaw, .v=v };
  int n_i = get_nearest_ind(&env, waypoints);
  // printf("(%.2f,%.2f) | n_i=%d\n",x,y,n_i);
  int len = MPC_LOOKAHEAD_PTS;

  // Calculate heading error
  yaw_path = atan2(*(waypoints+2*(len-1)+1)-*(waypoints+1), *(waypoints+2*(len-1))-*waypoints);
  yaw_diff = yaw_path - yaw;
  if(yaw_diff > M_PI)
    yaw_diff -= 2*M_PI;
  if(yaw_diff < -M_PI)
    yaw_diff += 2*M_PI;

  // Calculate crosstrack error
  crosstrack_error = distance(x, y, *(waypoints+2*n_i), *(waypoints+2*n_i+1));
  yaw_cross_track = atan2(y-*(waypoints+1), x-*waypoints);
  yaw_path2ct = yaw_path - yaw_cross_track;
  if(yaw_path2ct > M_PI)
    yaw_path2ct -= 2*M_PI;
  if(yaw_path2ct < -M_PI)
    yaw_path2ct += 2*M_PI;
  if(yaw_path2ct > 0)
    crosstrack_error = fabs(crosstrack_error);
  else
    crosstrack_error = -1*fabs(crosstrack_error);

  *cte = crosstrack_error;
  *epsi = yaw_diff;
}


