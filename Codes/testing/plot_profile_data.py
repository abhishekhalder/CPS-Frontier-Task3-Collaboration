#
# Plot KBM simulator profiling data
#
# Author: Georgiy Antonovich Bondar
# Date  : 04-26-2023
#
import matplotlib.pyplot as plt
import pdb
import re

INPATH = "../../Data/kbm_sim_profile_MCP/"
INFILE = "kbm_sim_15_1224_perf_3.txt"

instructions = [[],[]]
llc_loads = [[],[]]
llc_stores = [[],[]]
llc_loads_misses = [[],[]]

i = 0
for line in open(INPATH + INFILE, 'r'):
    i = i + 1

    # Process data for each line
    # print(line)

    try:
        splt = line.split()
        msm_type = splt[2]
        print(splt[2])

        if msm_type == "instructions:u":
            instructions[0].append((splt[0]))
            instructions[1].append(int(re.sub(",", "", splt[1])))
        elif msm_type == "LLC-loads":
            llc_loads[0].append(float(splt[0]))
            llc_loads[1].append(int(re.sub(",", "", splt[1])))
        elif msm_type == "LLC-stores":
            llc_stores[0].append(float(splt[0]))
            llc_stores[1].append(int(re.sub(",", "", splt[1])))
        elif msm_type == "LLC-loads-misses":
            llc_loads_misses[0].append(float(splt[0]))
            llc_loads_misses[1].append(int(re.sub(",", "", splt[1])))
        # pdb.set_trace()
    except:
        continue

    if i > 1000:
        break

f1 = plt.figure("Figure 1")
plt.plot(instructions[0],instructions[1])
plt.title("Instructions")

f2 = plt.figure("Figure 2")
plt.plot(llc_loads[0],llc_loads[1])
plt.title("LLC Loads")

f3 = plt.figure("Figure 3")
plt.plot(llc_stores[0],llc_stores[1])
plt.title("LLC Stores")

f4 = plt.figure("Figure 4")
plt.plot(llc_loads_misses[0],llc_loads_misses[1])
plt.title("LLC Loads Misses")

plt.show()

