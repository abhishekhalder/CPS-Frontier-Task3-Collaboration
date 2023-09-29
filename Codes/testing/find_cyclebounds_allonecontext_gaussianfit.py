#
# Given a specific context [Cache,MemBW,PathNo] find the time at which each control cycle ends for all 500 profiles.
# Construct and plot distributions for each control cycle end time.
#
# Author: Georgiy Antonovich Bondar
# Date  : 08-23-2023
#
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import math
import pdb
import re

## Configuration ##
###################
# [CACHE, MEMBW] = [1, 72]
# [CACHE, MEMBW] = [31, 360]
# [CACHE, MEMBW] = [1023, 720]
[CACHE, MEMBW] = [32767, 1080]
# [CACHE, MEMBW] = [1048575, 1440]
PATHNO = 0                                  # Valid values 0-11
###################
NUM_RUNS = 500                              # Valid values 1-500
NUM_STDS = 1
INPATH   = "../../Data/kbm_sim_profile/"
OUTPATH  = "./halder_outfiles_0824/"
###################

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

allfiles_data = []

## Read in data from all relevant files ##
##########################################
for j in range(NUM_RUNS):
    i = 0
    data_cnt   = 0
    data_found = False
    curr_file = INPATH + "kbm_sim_"+str(CACHE)+"_"+str(MEMBW)+"_profile_"+str(PATHNO)+"_"+str(j+1)+".txt"

    # print("Parsing file " + curr_file)

    for line in open(curr_file, 'r'):
        i = i + 1

        # Process data for each line
        splt = line.split()

        # Find lines of correct format
        if( (len(splt) != 2) or not(is_number(splt[0])) or not(is_number(splt[1])) ):
            if( data_found ):
                break
            else:
                continue

        if( not(data_found) ):
            data_found = True;

        data_cnt = data_cnt + 1

        if( len(allfiles_data) < data_cnt ):
            allfiles_data.append([[float(splt[0])], [float(splt[1])]])
        else:
            allfiles_data[data_cnt-1][0].append(float(splt[0]))
            allfiles_data[data_cnt-1][1].append(float(splt[1]))

        if i > 1000:
            break

##########################################

# pdb.set_trace()

## Plot all data ##
###################
tN_stats = []
f1, axs = plt.subplots(math.ceil(len(allfiles_data)/2), 2)
f1.suptitle("Control Cycle Durations | c=[%d, %d, %d]" % (CACHE, MEMBW, PATHNO))
for i in range(len(allfiles_data)):

    mu, std = norm.fit(allfiles_data[i][0])

    axs[math.floor(i/2), i%2].hist(allfiles_data[i][0], bins=25, density=True, alpha=0.6, color='g', edgecolor='k')
    xmin, xmax = axs[math.floor(i/2), i%2].get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    axs[math.floor(i/2), i%2].plot(x, p, 'k', linewidth=2)
    title = "t_%d (mu = %.4f,  std = %.4f)" % (i+1, mu, std)
    axs[math.floor(i/2), i%2].set_title(title)
    # axs[math.floor(i/2), i%2].set_xlabel("t(s)")

for ax in axs.flat[len(allfiles_data):]:
    ax.remove()

#####

f1, axs = plt.subplots(math.ceil(len(allfiles_data)/2), 2)
f1.suptitle("Control Cycle End Times | c=[%d, %d, %d]" % (CACHE, MEMBW, PATHNO))
for i in range(len(allfiles_data)):

    mu, std = norm.fit(allfiles_data[i][1])
    tN_stats.append([mu, std])

    axs[math.floor(i/2), i%2].hist(allfiles_data[i][1], bins=25, density=True, alpha=0.6, color='g', edgecolor='k')
    xmin, xmax = axs[math.floor(i/2), i%2].get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    axs[math.floor(i/2), i%2].plot(x, p, 'k', linewidth=2)
    title = "t_%d (mu = %.4f,  std = %.4f)" % (i+1, mu, std)
    axs[math.floor(i/2), i%2].set_title(title)
    # axs[math.floor(i/2), i%2].set_xlabel("t(s)")

for ax in axs.flat[len(allfiles_data):]:
    ax.remove()

#####

f3 = plt.figure("Figure 3")
for i in range(len(allfiles_data)):

    mu, std = norm.fit(allfiles_data[i][1])

    plt.hist(allfiles_data[i][1], bins=25, density=True, alpha=0.6, color='g', edgecolor='k')
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)

plt.xlabel("t(s)")
plt.title("All Control Cycles | c=[%d, %d, %d]" % (CACHE, MEMBW, PATHNO))


###################


## Use statistics (mu,std) from plotting to determine t_N values ##
###################################################################
tN_vals = []

print("\nWe subtract %d standard deviations from the mean to obtain each t_N:\n" % (NUM_STDS))

print("t_0 = 0.0")

for i in range(len(tN_stats)):
    tN_vals.append(tN_stats[i][0] - NUM_STDS*tN_stats[i][1])
    print("n=%i | (mu,std) = (%.4f,%.4f) => t_%d = %0.4f" % (i+1, tN_stats[i][0], tN_stats[i][1], i+1, tN_vals[i]))
###################################################################


## For each of the NUM_RUNS profiles, find the sample \xi matching the t_N values ##
####################################################################################
# xi_vals = [[]] * (len(tN_vals)+1)
xi_vals = [[] for i in range(len(tN_vals)+1)]

for j in range(NUM_RUNS):
    i = 0
    last_line = []
    curr_file = INPATH + "kbm_sim_"+str(CACHE)+"_"+str(MEMBW)+"_perf_"+str(PATHNO)+"_"+str(j+1)+"_clean.txt"

    print("Parsing file " + curr_file)

    for line in open(curr_file, 'r'):
        i = i + 1

        # Process data for each line
        splt = line.split()

        if(len(splt) != 6):
            continue

        xi = [ float(splt[2]), float(splt[3]) ]                   # The relevant elements (instr. ret., LLC Loads)

        # If reading the first line, save xi for t_0
        if( last_line == [] ):
            xi_vals[0].append(xi)
            last_line = splt
            # print("Found %f ~= 0 (t_0)" % float(splt[0]))
            continue

        xi = [ float(last_line[2]), float(last_line[3]) ]         # The relevant elements (instr. ret., LLC Loads)

        # Check if last line matches any of the t_N
        for k in range(len(tN_vals)):
            if( (float(last_line[0]) < tN_vals[k]) and (tN_vals[k] <= float(splt[0])) ):
                xi_vals[k+1].append(xi)
                # print("Found %f ~= %f (t_%d)" % (float(last_line[0]), tN_vals[k], k))
                break

        last_line = splt

        if i > 1000:
            break


####################################################################################


## Plot xi_vals at each t_N ##
##############################
ms = 2
f4, axs = plt.subplots(1, 2)
f4.suptitle("Stored Values of \\xi at Values of t_N | c=[%d, %d, %d]" % (CACHE, MEMBW, PATHNO))

for j in range(len(xi_vals[0])):
    axs[0].scatter(0, xi_vals[0][j][0], s=ms)
    axs[1].scatter(0, xi_vals[0][j][1], s=ms)

for i in range(1,len(xi_vals),1):
    for j in range(len(xi_vals[i])):
        axs[0].scatter(tN_vals[i-1], xi_vals[i][j][0], s=ms)
        axs[1].scatter(tN_vals[i-1], xi_vals[i][j][1], s=ms)

axs[0].set_title("Instructions Retired")
axs[1].set_title("LLC Loads")
##############################

## Write the needful to file ##
###############################
for i in range(len(xi_vals)):
    outfile = OUTPATH + "kbm_sim_"+str(CACHE)+"_"+str(MEMBW)+"_"+str(PATHNO)+"_t"+str(i)+".txt"

    print("Writing file " + outfile)

    with open(outfile, 'w') as f:
        for j in range(len(xi_vals[i])):
            f.write("%f %f\n" % (xi_vals[i][j][0], xi_vals[i][j][1]))
###############################


## Display Plots ##
plt.show()
print("Fertig.")
###################
















