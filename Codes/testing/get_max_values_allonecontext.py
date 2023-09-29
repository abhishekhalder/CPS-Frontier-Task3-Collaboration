#
# Given a specific context [Cache,MemBW,PathNo] get the max value of all states (\xi_{i\in[4]}) for all 500 profiles, for all times.
#
# Author: Georgiy Antonovich Bondar
# Date  : 09-18-2023
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
max_statevals = [[],[],[],[]]

## Read in data from all relevant files ##
##########################################
for j in range(NUM_RUNS):
    i = 0
    curr_file = INPATH + "kbm_sim_"+str(CACHE)+"_"+str(MEMBW)+"_perf_"+str(PATHNO)+"_"+str(j+1)+"_clean.txt"

    # Create new data set
    curr_file_data = [[],[],[],[],[]]

    # print("Parsing file " + curr_file)

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

    for i in range(len(curr_file_data)-1):
        max_statevals[i].append(max(curr_file_data[i+1]))

    allfiles_data.append(curr_file_data)

##########################################


## Print max values ##
######################
print("--------Max Values--------")
for i in range(len(max_statevals)):
    print("max(\\xi_{%d}) = %d" % (i+1,max(max_statevals[i])))
print("--------------------------")

###################

