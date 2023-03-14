#include "IpStdCInterface.h"
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <math.h>

// Set the timestep length and duration
const size_t N = 15;
const double dt = 0.1;

// Set cost factors
const int cost_cte_factor = 3000;
const int cost_epsi_factor = 500; // made initial portion etc much less snaky
const int cost_v_factor = 1;
const int cost_current_delta_factor = 1;
const int cost_diff_delta_factor = 200;
const int cost_current_a_factor = 1;
const int cost_diff_a_factor = 1;

const double Lf = 2.67;

// Reference cross-track error and orientation error = 0
double ref_cte = 0;
double ref_epsi = 0;
double ref_v = 40;

const size_t x_start = 0;
const size_t y_start = x_start + N;
const size_t psi_start = y_start + N;
const size_t v_start = psi_start + N;
const size_t cte_start = v_start + N;
const size_t epsi_start = cte_start + N;
const size_t delta_start = epsi_start + N;
const size_t a_start = delta_start + N - 1;

//
// Needful functions
//

static
bool eval_f(
   ipindex     n,
   ipnumber*   x,
   bool        new_x,
   ipnumber*   obj_value,
   UserDataPtr user_data
)
{
   // assert(n == 4);
   // (void) n;

   // (void) new_x;
   // (void) user_data;

   // *obj_value = x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2];

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
  // struct MyUserData* my_data = user_data;

  // assert(n == 4);
  // (void) n;
  // assert(m == 2);
  // (void) m;

  // (void) new_x;

  // g[0] = x[0] * x[1] * x[2] * x[3] + my_data->g_offset[0];
  // g[1] = x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3] + my_data->g_offset[1];

  // printf("(*) In eval_g\n");
  
  // Set initial state 
  g[x_start] = x[x_start];
  g[y_start] = x[y_start];
  g[psi_start] = x[psi_start];
  g[v_start] = x[v_start];
  g[cte_start] = x[cte_start];
  g[epsi_start] = x[epsi_start];

  // Update based on model (placeholder: do nothing)
  for (size_t i = 1; i < N; i++) {
    g[x_start+i] = x[x_start+i-1];
    g[y_start+i] = x[y_start+i-1];
    g[psi_start+i] = x[psi_start+i-1];
    g[v_start+i] = x[v_start+i-1];
    g[cte_start+i] = x[cte_start+i-1];
    g[epsi_start+i] = x[epsi_start+i-1];
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

    for (size_t i = 0; i < N; i++) {
      for (size_t j = 0; j < 6; j++) {
        iRow[cnt] = 6*i+j;
        jCol[cnt] = x_start+i;
        cnt++;

        iRow[cnt] = 6*i+j;
        jCol[cnt] = y_start+i;
        cnt++;

        iRow[cnt] = 6*i+j;
        jCol[cnt] = psi_start+i;
        cnt++;

        iRow[cnt] = 6*i+j;
        jCol[cnt] = v_start+i;
        cnt++;

        iRow[cnt] = 6*i+j;
        jCol[cnt] = cte_start+i;
        cnt++;

        iRow[cnt] = 6*i+j;
        jCol[cnt] = epsi_start+i;
        cnt++;

        if(i != 0){
          iRow[cnt] = 6*i+j;
          jCol[cnt] = delta_start+i-1;
          cnt++;

          iRow[cnt] = 6*i+j;
          jCol[cnt] = a_start+i-1;
          cnt++;
        }
      }
    }

    return true; 
  }

  //return false;
  return true;
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

int main(){
  enum ApplicationReturnStatus status;
  IpoptProblem nlp = NULL;
  ipindex nele_jac;                    /* number of nonzeros in the Jacobian of the constraints */
  ipindex nele_hess;                   /* number of nonzeros in the Hessian of the Lagrangian (lower or upper triangular part only) */
  ipindex index_style;                 /* indexing style for matrices */
  ipnumber obj;

  // Initialize problem
  
  // State: [x,y,psi,v,cte,epsi]
  // Actuators: [delta,a]
  size_t n_vars = 6 * N + 2 * (N - 1);
  
  // Set the number of constraints
  size_t n_constraints = 6 * N;

  // Initial value of the independent variables.
  // SHOULD BE 0 besides initial state.
  double vars[n_vars];
  for (size_t i = 0; i < n_vars; i++) {
    vars[i] = 0.0;
  }
  // Get init state (set random values for now)
  const double x = 100.0;
  const double y = -100.0;
  const double psi = 0.23;
  const double v = 10.0;
  const double cte = 12.3;
  const double epsi = 1.5;
  vars[x_start] = x;
  vars[y_start] = y;
  vars[psi_start] = psi;
  vars[v_start] = v;
  vars[cte_start] = cte;
  vars[epsi_start] = epsi;

  // Setting the lower and upper limits for variables
  double vars_lowerbound[n_vars];
  double vars_upperbound[n_vars];
  for (size_t i = 0; i < delta_start; i++) {
    vars_upperbound[i] = 1.0e19;
    vars_lowerbound[i] = -1.0e19;
  }

  // Steering angle (deltas)
  for (size_t i = delta_start; i < a_start; i++)
  {
    vars_upperbound[i] = M_PI/8; // max values allowed in simulator
    vars_lowerbound[i] = -M_PI/8;
  }

  // Acceleration
  for (size_t i = a_start; i < n_vars; i++)
  {
    vars_upperbound[i] = 1.0;
    vars_lowerbound[i] = -1.0;
  }

  
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
  constraints_lowerbound[cte_start] = cte;
  constraints_lowerbound[epsi_start] = epsi;

  constraints_upperbound[x_start] = x;
  constraints_upperbound[y_start] = y;
  constraints_upperbound[psi_start] = psi;
  constraints_upperbound[v_start] = v;
  constraints_upperbound[cte_start] = cte;
  constraints_upperbound[epsi_start] = epsi;

  // nlp = CreateIpoptProblem(n, x_L, x_U, m, g_L, g_U, nele_jac, nele_hess, index_style,
  //                           &eval_f, &eval_g, &eval_grad_f,
  //                           &eval_jac_g, &eval_h);

  /* set the number of nonzeros in the Jacobian and Hessian */
  nele_jac = 8 * 6*(N-1) + 6*6*(1);
  nele_hess = 0; // Not using this

  /* set the indexing style to C-style (start counting of rows and column indices at 0) */
  index_style = 0;

  nlp = CreateIpoptProblem(n_vars, vars_lowerbound, vars_upperbound, n_constraints, 
                           constraints_lowerbound, constraints_upperbound, 
                           nele_jac, nele_hess, index_style,
                           &eval_f, &eval_g, &eval_grad_f,
                           &eval_jac_g, &eval_h);

  if( !nlp ){
    printf("(!) Error creaing IPOPT problem!\n");
    return 1;
  }
  else
    printf("(*) Created IPOPT problem.\n");

  // Set NLP options
  AddIpoptNumOption(nlp, "max_cpu_time", 0.5);
  AddIpoptIntOption(nlp, "print_level", 12); // verbosity
  
  // So that we need not implement eval_jac_g
  if(!AddIpoptStrOption(nlp, "jacobian_approximation", "finite-difference-values"))
    printf("(!) Error setting jacobian_approximation!\n");
  AddIpoptStrOption(nlp, "hessian_approximation", "limited-memory"); // So that we need not implement eval_h
  AddIpoptStrOption(nlp, "derivative_test", "first-order");

  // Solve problem
  status = IpoptSolve(nlp, vars, NULL, &obj, NULL, NULL, NULL, NULL);

  // Print solution
  printf("(*) Solved nlp. Optimal objective evaluation: %.2f\n", obj);
  printf("(*) Optimal delta: %.2f\n", vars[delta_start]);
  printf("(*) Optimal accel: %.2f\n", vars[a_start]);
  
  // Needful
  FreeIpoptProblem(nlp);

	return 0;
}
