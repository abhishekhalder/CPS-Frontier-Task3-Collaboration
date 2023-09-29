#
# Given a specific context [Cache,MemBW,PathNo] plot all 500 profiles superimposed on 4 plots (1 for each state variable).
#
# Author: Georgiy Antonovich Bondar
# Date  : 08-23-2023
#
import matplotlib.pyplot as plt
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
INPATH = "../../Data/kbm_sim_profile/"
###################

allfiles_data = []

## Read in data from all relevant files ##
##########################################
for j in range(NUM_RUNS):
    i = 0
    curr_file = INPATH + "kbm_sim_"+str(CACHE)+"_"+str(MEMBW)+"_perf_"+str(PATHNO)+"_"+str(j+1)+"_clean.txt"

    # Create new data set
    curr_file_data = [[],[],[],[],[]]

    print("Parsing file " + curr_file)

    for line in open(curr_file, 'r'):
        i = i + 1

        # Process data for each line
        splt = line.split()
        if(len(splt) != 6):
            continue

        curr_file_data[0].append(float(splt[0]))
        curr_file_data[1].append(int(splt[2]))
        curr_file_data[2].append(int(splt[3]))
        curr_file_data[3].append(int(splt[4]))
        curr_file_data[4].append(int(splt[5]))

        if i > 1000:
            break

    allfiles_data.append(curr_file_data)

##########################################


## Plot all data ##
###################
fig, axs = plt.subplots(2, 2)

for j in range(NUM_RUNS):
    axs[0,0].plot(allfiles_data[j][0],allfiles_data[j][1])
    axs[0,1].plot(allfiles_data[j][0],allfiles_data[j][2])
    axs[1,0].plot(allfiles_data[j][0],allfiles_data[j][3])
    axs[1,1].plot(allfiles_data[j][0],allfiles_data[j][4])

axs[0, 0].set_title('Instructions Retired')
axs[0, 1].set_title('LLC Loads')
axs[1, 0].set_title('LLC Stores')
axs[1, 1].set_title('LLC Loads Misses')

plt.show()
###################

