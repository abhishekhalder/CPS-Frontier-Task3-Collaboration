This directory contains the code for the KBM path-following controller simulator. 


----- Simulator Structure -----
The simulator works as follows. The file "sim_config.h" contains all high-level simulation parameters, to be configured as desired, within reason.
Of particular note are the following #defines:

RUN_TIME, in seconds, defines the duration to simulate. It is not currently the case that "main.ex" will actually run for RUN_TIME seconds, but rather that RUN_TIME seconds will be simulated.

DT, in seconds, is the timestep used by both the environment and the controller.

TERMLOG_EN, enable printing of status/debug information to terminal while simulation
is running.

FILELOG_EN, enable writing system state to OUTFILE as the simulation is running.

TS_DELAY_MS, in milliseconds, sets the length of the artificial delay between
controller loops. A positive value, or zero, works as expected. A negative value
will cause the control loop runtime to be subtracted from the delay time for 
every loop cycle. E.g. if TS_DELAY_MS = -100, and the controller took 12ms to run,
the control loop will delay for 88ms.

CONTROLLER_TYPE defines the type of controller used (currently, only PID + Stanley || MPC).

ENV_xx_INIT define the KBM initial conditions. 

Controller-specific definitions are located in the respective controllers' header files ("./controllers_kbm/ctrl_xx.h"), and do not normally need to be modified.

"main.c" initializes the KBM environment ("environment_kbm.c") and the controller ("controller_kbm.c"). It is within the latter that the specific controller (set by CONTROLLER_TYPE) is selected and initialized.
Following initialization, main enters the following loop until the runtime expires:

 - Get controller output
 - Apply control and update environment
 - Log state for timestep
 - Repeat
 

----- Dependencies -----
The MPC controller uses the IPOPT library for solution of the relevant optimization problem at each timestep. This library, along with its own dependencies, will need to be installed on the machine running the simulator. Installation instructions can be found here: https://coin-or.github.io/Ipopt/INSTALL.html


----- Compilation -----
Assuming dependencies are installed, simply running 'make' in this directory will build the executable "main.ex".
Running 'make clean' will delete the executable, as well as all object files created during the build process.
See "makefile" for further details.


----- Execution -----
Once built, the simulator can be started by running './main.ex' from within this directory. Once the simulation completes, the state at each timestep will be written to "run_data.txt" from within this directory. Once the simulation completes, the state at each timestep will be written to "run_data.txt"


----- Plots -----
The python3 script "plot_lbm_data.py" will read the file "run_data.txt" and generate three figures for visualization of the system state during simulation. These figures are saved in the "./plots/" subdirectory.


 - Georgiy A. Bondar (03-14-2023)
