#!/usr/bin/python3
#
# Plot all profiles 
# superimposed on 3x4=12 plots (1 for each state variable, and 1 for each CPU),
# in the format requested by Dr. Halder for publication, combining LLC loads and
# stores into one variable, LLC requests.
#
# Author: Georgiy Antonovich Bondar
# Date  : 10-03-2024
#
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.ticker import MaxNLocator
from operator import add
from frechetdist import frdist
from math import log
import pdb
import re

## Configuration ##
###############################################################################
BENCHMARK_NAME  = "radiosity" # Choices are "dedup", "fft", "canneal" "radiosity"
###############################################################################
NUM_CACHE_PARTS = 20
NUM_MEMBW_PARTS = 20
VALID_CACHE     = [ (1<<k)-1    for k in range(1,NUM_CACHE_PARTS+1) ]
VALID_MEMBW     = [ 72*k        for k in range(1,NUM_MEMBW_PARTS+1) ]
###############################################################################
CPU_START_INDEX = 0
NUM_RUNS        = 100                              # Valid values 1-300
NUM_CPUS        = 1    # Do not change this
INPATH          = "../../Data/single_core_profiles/%s_profile/" % BENCHMARK_NAME
INPATH_AVG      = "./%s_outfiles/avg_profiles/" % BENCHMARK_NAME
INPATH_SYNTH    = "./Context_Interp_SingleCore/data_out/%s_synth_profiles/" % BENCHMARK_NAME
OUTPATH         = "./%s_outfiles/synth_profile_plots/" % BENCHMARK_NAME
OUTPATH_ERRS    = "./%s_outfiles/error_plots/" % BENCHMARK_NAME
###############################################################################
PLOTS_LINECOLOR = 'grey'
PLOTS_LINEWIDTH = 0.3
PLOTS_LINEALPHA = 0.05
###############################################################################

allfiles_data = []
profile_errors_avg  = [ [ [ 0 for j in range(NUM_MEMBW_PARTS) ] for i in range(NUM_CACHE_PARTS) ] for k in range(3) ]
profile_errors_maxl = [ [ [ 0 for j in range(NUM_MEMBW_PARTS) ] for i in range(NUM_CACHE_PARTS) ] for k in range(3) ]

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
inverse_latex_scale = 3
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
ccpu = 0; cind = 0;
for cache in VALID_CACHE:
    mind = 0;
    for membw in VALID_MEMBW:
        fig, axs = plt.subplots(1,3)
        plt.subplots_adjust(wspace=0.3)
        axs[0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        axs[1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        axs[2].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

        axs[0].set_xlabel(r'$t$ (s)'); axs[0].set_ylabel(r'$\xi_1$')
        axs[1].set_xlabel(r'$t$ (s)'); axs[1].set_ylabel(r'$\xi_2$')
        axs[2].set_xlabel(r'$t$ (s)'); axs[2].set_ylabel(r'$\xi_3$')

        axs[0].yaxis.set_major_locator(MaxNLocator(integer=True))
        axs[1].yaxis.set_major_locator(MaxNLocator(integer=True))
        axs[2].yaxis.set_major_locator(MaxNLocator(integer=True))

        # v Temporary for RTAS Paper and for fft & radiosity v #
        axs[0].set_xlim([0,1])
        axs[1].set_xlim([0,1])
        axs[2].set_xlim([0,1])
        fig.suptitle(r'$\mathbf{\xi}(t)\mid\beta$ for $\beta=(%d,%d)^\top$, benchmark $\mathsf{%s}$' % (int(log(cache+1,2)),int(membw/72),BENCHMARK_NAME), y=1)

        allfiles_data = []
        for j in range(NUM_RUNS):
            i = 0
            curr_file = INPATH + ("%s_%d_%d_perf_%d_clean.txt" % (BENCHMARK_NAME,cache,membw,j+1))
            curr_file_data = [ [[],[],[],[],[]] for k in range(NUM_CPUS) ]

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

            axs[0].plot(allfiles_data[j][0][0], allfiles_data[j][0][1], linewidth=PLOTS_LINEWIDTH, color=PLOTS_LINECOLOR, alpha=PLOTS_LINEALPHA)
            axs[1].plot(allfiles_data[j][0][0], list( map(add, allfiles_data[j][0][2], allfiles_data[j][0][3]) ), linewidth=PLOTS_LINEWIDTH, color=PLOTS_LINECOLOR, alpha=PLOTS_LINEALPHA)
            axs[2].plot(allfiles_data[j][0][0], allfiles_data[j][0][4], linewidth=PLOTS_LINEWIDTH, color=PLOTS_LINECOLOR, alpha=PLOTS_LINEALPHA)

        # Load emperical mean profile
        avg_profile_file = INPATH_AVG + ("%s_avgprofile_%d_%d.txt" % (BENCHMARK_NAME,cache,membw))
        curr_avg_profile = [[],[],[],[]]
        for line in open(avg_profile_file, 'r'):
            splt = line.split(" ")
            if(len(splt) != 4):
                continue

            curr_avg_profile[0].append(float(splt[0]))
            curr_avg_profile[1].append(float(splt[1]))
            curr_avg_profile[2].append(float(splt[2]))
            curr_avg_profile[3].append(float(splt[3]))

        # Load synthetic profile
        synth_profile_file = INPATH_SYNTH + ("%s-synth_c%d_%d.txt" % (BENCHMARK_NAME,cache,membw))
        curr_synth_profile = [[],[],[],[],[],[],[]]
        for line in open(synth_profile_file, 'r'):
            splt = line.split(",")
            if(len(splt) != 7):
                continue

            curr_synth_profile[0].append(float(splt[0]))
            curr_synth_profile[1].append(float(splt[1]))
            curr_synth_profile[2].append(float(splt[2]))
            curr_synth_profile[3].append(float(splt[3]))
            curr_synth_profile[4].append(float(splt[4]))
            curr_synth_profile[5].append(float(splt[5]))
            curr_synth_profile[6].append(float(splt[6]))



        # Plot the emperical mean trajectory (BLACK)
        axs[0].plot(curr_avg_profile[0],curr_avg_profile[1], linewidth=4, color='black', alpha=0.5)
        axs[1].plot(curr_avg_profile[0],curr_avg_profile[2], linewidth=4, color='black', alpha=0.5)
        axs[2].plot(curr_avg_profile[0],curr_avg_profile[3], linewidth=4, color='black', alpha=0.5)

        # Plot the mean trajectory (RED)
        axs[0].plot(curr_synth_profile[0],curr_synth_profile[1], linewidth=3, color='red', alpha=0.5)
        axs[1].plot(curr_synth_profile[0],curr_synth_profile[2], linewidth=3, color='red', alpha=0.5)
        axs[2].plot(curr_synth_profile[0],curr_synth_profile[3], linewidth=3, color='red', alpha=0.5)
            
        # Plot the max likelihood trajectory (BLUE)
        axs[0].plot(curr_synth_profile[0],curr_synth_profile[4], linewidth=2, color='blue', alpha=0.5)
        axs[1].plot(curr_synth_profile[0],curr_synth_profile[5], linewidth=2, color='blue', alpha=0.5)
        axs[2].plot(curr_synth_profile[0],curr_synth_profile[6], linewidth=2, color='blue', alpha=0.5)

        # axs[0].set_title(r'$\mathbf{\xi}_1$')
        # axs[1].set_title(r'$\mathbf{\xi}_2$')
        # axs[2].set_title(r'$\mathbf{\xi}_3$')

        fig.set_size_inches(15,7)
        # plt.savefig(OUTPATH + ("%s_synth-comp-c%d_%d.png" % (BENCHMARK_NAME,cache,membw)), dpi=300)
        plt.savefig(OUTPATH + ("%s_synth-comp-c%d_%d.pdf" % (BENCHMARK_NAME,cache,membw)), bbox_inches="tight")
        fig.clear()
        plt.close(fig)

        # Get Frechet distance between average profile and predictions
        minlen = min(len(curr_avg_profile[0]),len(curr_synth_profile[0]))
        for j in range(3):
            profile_errors_avg[j][cind][mind]  = frdist([[curr_avg_profile[0][i],curr_avg_profile[1+j][i]] for i in range(minlen)], [[curr_synth_profile[0][i],curr_synth_profile[1+j][i]] for i in range(minlen)])
            profile_errors_maxl[j][cind][mind] = frdist([[curr_avg_profile[0][i],curr_avg_profile[1+j][i]] for i in range(minlen)], [[curr_synth_profile[0][i],curr_synth_profile[4+j][i]] for i in range(minlen)])

        mind = mind + 1
    cind = cind + 1
################################################################################

fig, axs = plt.subplots(1,3,subplot_kw={"projection": "3d"})
fig.suptitle(r'Fr\'echet Distance Between Emperical and Synthetic Profiles for `%s` for all $\beta$' % BENCHMARK_NAME)
fig.set_size_inches(18,7)
X = np.array(range(1,NUM_CACHE_PARTS+1))
Y = np.array(range(1,NUM_MEMBW_PARTS+1))
X,Y = np.meshgrid(X, Y)
for j in range(3):
    # Write errors to file
    outfile_avg  = OUTPATH_ERRS + ("%s_avg-errs_xi%d.txt" % (BENCHMARK_NAME,j+1))
    outfile_mxlk = OUTPATH_ERRS + ("%s_mxlk-errs_xi%d.txt" % (BENCHMARK_NAME,j+1))

    with open(outfile_avg,'w') as f:
        for errors in profile_errors_avg[j]:
            for error in errors:
                f.write("%f " % error)
            f.write("\n")

    with open(outfile_mxlk,'w') as f:
        for errors in profile_errors_avg[j]:
            for error in errors:
                f.write("%f " % error)
            f.write("\n")

    # Plot errors as a surface
    Z = np.array(profile_errors_maxl[j])

    axs[j].set_title(r'$\mathbf{\xi}_%d$' % (j+1))
    axs[j].set_xlabel(r'$\beta_1$')
    axs[j].set_ylabel(r'$\beta_2$')
    axs[j].plot_surface(X, Y, Z, cmap=cm.Blues)
    axs[j].set_xticks(range(1,NUM_CACHE_PARTS+1))
    axs[j].set_yticks(range(1,NUM_MEMBW_PARTS+1))

plt.savefig(OUTPATH_ERRS + ("%s_mxlk-errs.png" % (BENCHMARK_NAME)), dpi=300)

# plt.show()

