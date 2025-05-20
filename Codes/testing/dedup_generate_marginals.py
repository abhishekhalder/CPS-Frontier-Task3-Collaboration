#!/usr/bin/python3
#
# For the dedup profiles provided on 09-18, generate a set of marginals incorporating all profiles, with
# context as part of \xi.
# Given a specific context [Cache,MemBW] and a list of timestamps at which we desire to place marginals,
# construct and plot distributions at each timestamp, as well as at some equispaced times between 
# these end times.
# In this code, we combine LLC loads and stores into "LLC requests", thereby reducing the dimension of \xi.
#
# Author: Georgiy Antonovich Bondar
# Date  : 10-03-2024
#
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import numpy as np
import math
import pdb
import re

## Configuration ##
###################
VALID_CACHE = [ 0b1, 0b11, 0b111, 0b1111, 0b11111, 0b111111, 0b1111111, 0b11111111, 0b111111111, 
               0b1111111111, 0b11111111111, 0b111111111111, 0b1111111111111, 0b11111111111111,
               0b111111111111111, 0b1111111111111111, 0b11111111111111111, 0b111111111111111111,
               0b1111111111111111111 ,0b11111111111111111111 ]
VALID_CACHE = VALID_CACHE[:5]               # Contract to first 5 cache allocations
VALID_MEMBW = [0]
# [CACHE, MEMBW] = [7, 72]
###################
NUM_RUNS = 300                               # Valid values 1-300
NUM_CPUS = 4                                # Number of concurrent CPUs to expect 
# NUM_INTERMEDIATE = 7                        # number of marginals to get between requested times 
NUM_INTERMEDIATE = 0                        # number of marginals to get between requested times 
INPATH   = "../../Data/dedup_profile_mt/"
OUTPATH  = "./dedup_outfiles_1015/marginals/"
###################
# REQ_TIMES = [0.03, 1.0]                 # A list of times at which we would like to place marginals
REQ_TIMES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]                 # A list of times at which we would like to place marginals

#====================================================
# Make plots beautiful
#====================================================

pts_per_inch = 72.27
# write "\the\textwidth" (or "\showthe\columnwidth" for a 2 collumn text)
text_width_in_pts = 300.0
# inside a figure environment in latex, the result will be on the
# dvi/pdf next to the figure. See url above.
text_width_in_inches = text_width_in_pts / pts_per_inch
# figure.png or figure.eps will be intentionally larger, because it is prettier
inverse_latex_scale = 2
fig_proportion = (3.0 / 3.0)
csize = inverse_latex_scale * fig_proportion * text_width_in_inches
# always 1.0 on the first argument
fig_size = (1.0 * csize, 0.85 * csize)
# find out the fontsize of your latex text, and put it here
text_size = inverse_latex_scale * 9
label_size = inverse_latex_scale * 10
tick_size = inverse_latex_scale * 8
# learn how to configure:
# http://matplotlib.sourceforge.net/users/customizing.html
params = {
          'axes.labelsize': 16,
          'legend.fontsize': tick_size,
          'legend.handlelength': 2.5,
          'legend.borderaxespad': 0,
          'axes.labelsize': label_size,
          'xtick.labelsize': tick_size,
          'ytick.labelsize': tick_size,
          'font.family': 'serif',
          'font.size': text_size,
          'font.serif': ['Computer Modern Roman'],
          'ps.usedistiller': 'xpdf',
          'text.usetex': True,
          'figure.figsize': fig_size,
          # include here any neede package for latex
          'text.latex.preamble': [r'\usepackage{amsmath}',
                                  r'\usepackage{bm}'],
          }
plt.rcParams.update(params)

#====================================================
# END Make plots beautiful
#====================================================

## For each of the NUM_RUNS profiles, find the sample \xi matching the t_N values ##
####################################################################################

temp = []
# for i in range(len(tN_vals)):
for i in range(len(REQ_TIMES)):
    if( i>0 ):
        prev = REQ_TIMES[i-1]
        for j in range(1,NUM_INTERMEDIATE+1):
            temp.append(prev + (j/(NUM_INTERMEDIATE+1))*(REQ_TIMES[i]-prev))

    temp.append(REQ_TIMES[i])

tN_vals = temp

print("\n(*) Execution times at which marginals are to be generated:")
print(tN_vals)
print("-----------------------------------------------------------\n")
# exit()

for cache in VALID_CACHE:
    for membw in VALID_MEMBW:
        for cpuno in range(NUM_CPUS):
            ctxt = [0, 0, 0, 0]
            ctxt[cpuno] = cache
            xi_vals = [[[] for j in range(NUM_CPUS)] for i in range(len(tN_vals)+1)]
            for j in range(NUM_RUNS):
                i = 0
                last_line = [ [] for i in range(NUM_CPUS) ]
                curr_file = INPATH + "dedup_"+str(cache)+"_"+str(membw)+"_"+str(cpuno)+"_perf_mt_"+str(j+1)+"_clean.txt"

                print("Parsing file " + curr_file)

                for line in open(curr_file, 'r'):
                    i = i + 1

                    # Process data for each line
                    splt = line.split()

                    if(len(splt) != 7):
                        continue

                    ccpu = int(splt[1][3:]) - 1
                    xi = [ float(splt[3]), float(splt[4])+float(splt[5]), float(splt[6]) ] # Combining LLC loads and stores

                    # If reading the first line, save xi for t_0
                    if( last_line[ccpu] == [] ):
                        xi_vals[0][ccpu].append(xi)
                        last_line[ccpu] = splt
                        print("Found %f ~= 0 (t_0)" % float(splt[0]))
                        continue

                    # ccpu = int(last_line[1][3:]) - 1
                    xi = [ float(last_line[ccpu][3]), float(last_line[ccpu][4])+float(last_line[ccpu][5]), float(last_line[ccpu][6]) ]

                    # Check if last line matches any of the t_N
                    for k in range(len(tN_vals)):
                        if( (float(last_line[ccpu][0]) < tN_vals[k]) and (tN_vals[k] <= float(splt[0])) ):
                            xi_vals[k+1][ccpu].append(xi)
                            print("Found %f ~= %f (t_%d)" % (float(last_line[ccpu][0]), tN_vals[k], k))
                            break

                    last_line[ccpu] = splt

                    if i > 10000:
                        break

            ## Write the needful to file ##
            ###############################
            for i in range(len(xi_vals)):
                for k in range(NUM_CPUS):
                    outfile = (OUTPATH + "dedup_%d_%d_%d_%d_MARG"+str(i)+"."+str(k+1)+"_3dim.txt") % (ctxt[0], ctxt[1], ctxt[2], ctxt[3])

                    print("Writing file " + outfile)

                    with open(outfile, 'w') as f:
                        for j in range(len(xi_vals[i][k])):
                            # if( j>=200 ):
                            #     break
                            # f.write("%f %f %f %f\n" % (xi_vals[i][j][0], xi_vals[i][j][1], xi_vals[i][j][2], xi_vals[i][j][3]))
                            f.write("%f %f %f\n" % (xi_vals[i][k][j][0], xi_vals[i][k][j][1], xi_vals[i][k][j][2]))
            ###############################

            # exit()



####################################################################################

