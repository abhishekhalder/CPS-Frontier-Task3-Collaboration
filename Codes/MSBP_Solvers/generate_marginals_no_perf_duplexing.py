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
# Date  : 04-22-2025
#
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import numpy as np
import math
import pdb
import re

## Configuration ##
###############################################################################
BENCHMARK_NAME = "streamcluster"
###############################################################################
NUM_CACHE_PARTS = 20
NUM_MEMBW_PARTS = 20
VALID_CACHE     = [ (1<<k)-1    for k in range(1,NUM_CACHE_PARTS+1) ]
VALID_MEMBW     = [ 72*k        for k in range(1,NUM_MEMBW_PARTS+1) ]
###############################################################################
NUM_RUNS        = 100
NUM_CPUS        = 1                        # Do not change this
INPATH          = "../../Data/single_core_profiles_no_perf_duplexing/%s_profile_no_perf_duplexing/" % BENCHMARK_NAME
OUTPATH         = "./%s_no_perf_duplexing_outfiles/marginals/" % BENCHMARK_NAME
OUTPATH_AVG     = "./%s_no_perf_duplexing_outfiles/avg_profiles/" % BENCHMARK_NAME
###############################################################################
REQ_TIMES        = [ 0.05*(k+1) for k in range(60)  ] # Times at which to place marginals
REQ_TIMES_AVG    = [ 0.01*k     for k in range(301) ] # Times at which to place marginals
NUM_INTERMEDIATE = 0                                # Number of marginals to place between pairs of requested times 
###############################################################################

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
plt.ticklabel_format(style='sci', axis='y')
#====================================================
# END Make plots beautiful
#====================================================

tN_vals = REQ_TIMES
print("\n(*) Execution times at which marginals are to be generated:")
print(tN_vals)
print("-----------------------------------------------------------\n")
# exit()

ccpu = 0
for cache in VALID_CACHE:
    for membw in VALID_MEMBW:
        # for cpuno in range(NUM_CPUS):
        ctxt = [cache, membw]
        xi_vals     = [[[] for j in range(NUM_CPUS)] for i in range(len(tN_vals)+1)]
        xi_vals_avg = [[[] for j in range(NUM_CPUS)] for i in range(len(REQ_TIMES_AVG)+1)]
        for j in range(NUM_RUNS):
            i = 0
            last_line = [ [] for i in range(NUM_CPUS) ]
            curr_file = INPATH + ("%s_%d_%d_perf_%d_clean.txt" % (BENCHMARK_NAME,cache,membw,j+1))

            print("Parsing file " + curr_file)

            for line in open(curr_file, 'r'):
                i = i + 1

                # Process data for each line
                splt = line.split()

                if(len(splt) != 5):
                    continue

                xi = [ float(splt[2]), float(splt[3]), float(splt[4]) ]

                # If reading the first line, save xi for t_0
                if( last_line[ccpu] == [] ):
                    xi_vals[0][ccpu].append(xi)
                    xi_vals_avg[0][ccpu].append(xi)
                    last_line[ccpu] = splt
                    print("Found %f ~= 0 (t_0)" % float(splt[0]))
                    continue

                xi = [ float(last_line[ccpu][2]), float(last_line[ccpu][3]), float(last_line[ccpu][4]) ]

                # Check if last line matches any of the t_N_avg
                for k in range(len(REQ_TIMES_AVG)):
                    if( (float(last_line[ccpu][0]) < REQ_TIMES_AVG[k]) and (REQ_TIMES_AVG[k] <= float(splt[0])) ):
                        xi_vals_avg[k+1][ccpu].append(xi)
                        break

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
            k = 0
            outfile = (OUTPATH + "%s_%d_%d_MARG%d_3dim.txt") % (BENCHMARK_NAME, ctxt[0], ctxt[1], i)

            print("Writing file " + outfile)

            with open(outfile, 'w') as f:
                for j in range(len(xi_vals[i][k])):
                    f.write("%f %f %f\n" % (xi_vals[i][k][j][0], xi_vals[i][k][j][1], xi_vals[i][k][j][2]))
        ###############################

        ## Write the (avg) needful to file ##
        #####################################
        outfile = (OUTPATH_AVG + "%s_avgprofile_%d_%d.txt") % (BENCHMARK_NAME, ctxt[0], ctxt[1])
        with open(outfile, 'w') as f:
            for i in range(len(REQ_TIMES_AVG)):
                avg_xi = [0, 0, 0];
                for j in range(len(xi_vals_avg[i][0])):
                    avg_xi[0] = avg_xi[0] + xi_vals_avg[i][0][j][0]
                    avg_xi[1] = avg_xi[1] + xi_vals_avg[i][0][j][1]
                    avg_xi[2] = avg_xi[2] + xi_vals_avg[i][0][j][2]
                avg_xi = [ xi / NUM_RUNS for xi in avg_xi ]
                f.write("%f %f %f %f\n" % (REQ_TIMES_AVG[i], avg_xi[0], avg_xi[1], avg_xi[2]))
        #####################################

