#!/usr/bin/python3
#
# For all contexts [Cache,MemBW] plot all profiles superimposed on 3 plots 
# (1 for each state variable) in the format requested by Dr. Halder for 
# publication, combining LLC loads and stores into one variable, LLC requests.
#
# Author: Georgiy Antonovich Bondar
# Date  : 11-03-2024
#
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from operator import add
import pdb
import re

## Configuration ##
###############################################################################
BENCHMARK_NAME  = "canneal"
###############################################################################
NUM_CACHE_PARTS = 20
NUM_MEMBW_PARTS = 20
VALID_CACHE     = [ (1<<k)-1    for k in range(1,NUM_CACHE_PARTS+1) ]
VALID_MEMBW     = [ 72*k        for k in range(1,NUM_MEMBW_PARTS+1) ]
###############################################################################
CPU_START_INDEX = 0
NUM_RUNS        = 100
NUM_CPUS        = 1    # Do not change this
INPATH          = "../../Data/single_core_profiles/%s_profile/" % BENCHMARK_NAME
OUTPATH         = "./%s_outfiles/profile_plots/" % BENCHMARK_NAME
###############################################################################
PLOTS_LINECOLOR = 'grey'
PLOTS_LINEWIDTH = 0.3
PLOTS_LINEALPHA = 0.1

allfiles_data = []
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
params = {# 'backend': 'ps',
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
          'text.latex.preamble': [r'\usepackage{amsmath}'],
          'savefig.dpi': 300
          }
plt.rcParams.update(params)

#====================================================
# END Make plots beautiful
#====================================================

## Read in data from all relevant files ##
##########################################
ccpu = 0
for cache in VALID_CACHE:
    for membw in VALID_MEMBW:
        ctxt = [cache, membw]
        allfiles_data = []
        for j in range(NUM_RUNS):
            i = 0
            curr_file = INPATH + ("%s_%d_%d_perf_%d_clean.txt" % (BENCHMARK_NAME,cache,membw,j+1))

            # Create new data set
            curr_file_data = [ [[],[],[],[],[]] for k in range(1) ]

            print("Parsing file " + curr_file)

            for line in open(curr_file, 'r'):
                i = i + 1

                # Process data for each line
                splt = line.split()
                if(len(splt) != 6):
                    continue

                curr_file_data[ccpu][0].append(float(splt[0]))
                curr_file_data[ccpu][1].append(  int(splt[2]))
                curr_file_data[ccpu][2].append(  int(splt[3]))
                curr_file_data[ccpu][3].append(  int(splt[4]))
                curr_file_data[ccpu][4].append(  int(splt[5]))

                if i > 10000:
                    break

            allfiles_data.append(curr_file_data)

        ## Plot all data ##
        ###################
        fig, axs = plt.subplots(NUM_CPUS, 3)
        fig.suptitle(r'Measured states $\mathbf{\xi}$ for context $\mathbf{c}=\left[%d, %d\right]^{\top}$' % (ctxt[0], ctxt[1]))


        for j in range(NUM_RUNS):
            for k in range(NUM_CPUS):
                axs[0].plot(allfiles_data[j][k][0],allfiles_data[j][k][1], linewidth=PLOTS_LINEWIDTH, color=PLOTS_LINECOLOR, alpha=PLOTS_LINEALPHA)
                axs[1].plot(allfiles_data[j][k][0],list( map(add, allfiles_data[j][k][2], allfiles_data[j][k][3]) ), linewidth=PLOTS_LINEWIDTH, color=PLOTS_LINECOLOR, alpha=PLOTS_LINEALPHA)
                axs[2].plot(allfiles_data[j][k][0],allfiles_data[j][k][4], linewidth=PLOTS_LINEWIDTH, color=PLOTS_LINECOLOR, alpha=PLOTS_LINEALPHA)

                axs[0].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
                axs[1].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
                axs[2].ticklabel_format(axis='y', style='sci', scilimits=(0,0))

        axs[0].set_title('Instructions Retired')
        axs[1].set_title('LLC Requests')
        axs[2].set_title('LLC Loads Misses')

        axs[0].set_xlabel(r'$t$ [s]')
        axs[1].set_xlabel(r'$t$ [s]')
        axs[2].set_xlabel(r'$t$ [s]')

        fig.set_size_inches(15,14)
        plt.savefig(OUTPATH + "%s_allprofiles_%d_%d.png" % (BENCHMARK_NAME, ctxt[0], ctxt[1]), dpi=300)
        fig.clear()
        plt.close(fig)
        ###################


