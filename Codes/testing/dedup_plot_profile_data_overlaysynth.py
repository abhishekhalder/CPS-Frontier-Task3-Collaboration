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
import matplotlib
matplotlib.use("TkAgg")
# import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from operator import add
import pdb
import re

## Configuration ##
###################
VALID_CACHE = [ 0b1, 0b11, 0b111, 0b1111, 0b11111, 0b111111, 0b1111111, 0b11111111, 0b111111111,
               0b1111111111, 0b11111111111, 0b111111111111, 0b1111111111111, 0b11111111111111,
               0b111111111111111, 0b1111111111111111, 0b11111111111111111, 0b111111111111111111,
               0b1111111111111111111 ,0b11111111111111111111 ]
VALID_CACHE = VALID_CACHE[0:5]               # Contract to first 5 cache allocations
VALID_MEMBW = [0]
# [CACHE, MEMBW] = [1, 0] # For testing
###################
CPU_START_INDEX = 0
# STRT_IND = 100 # This doesn't do anything, maybe someday it will
NUM_RUNS = 300                              # Valid values 1-300
NUM_CPUS = 4                               # Number of concurrent CPUs to expect
INPATH = "../../Data/dedup_profile_mt/"
INPATH_SYNTH = "./Context_Interp_SBP/data_out/synthetic_profiles_1022/"
OUTPATH  = "./dedup_outfiles_1022/plots/"
###################
PLOTS_LINECOLOR = 'grey'
PLOTS_LINEWIDTH = 0.3
PLOTS_LINEALPHA = 0.05
# PLOTS_LINEALPHA = 1

allfiles_data = []

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
membw = 0;
for xicomp in range(1,4):
    fig, axs = plt.subplots(NUM_CPUS, len(VALID_CACHE))
    fig.suptitle(r'Measured vs. predicted states $\mathbf{\xi}_%d$' % (xicomp))

    for cache in VALID_CACHE:
        for cpuno in range(NUM_CPUS):
            ctxt = [0, 0, 0, 0]
            ctxt[cpuno] = cache
            allfiles_data = []
            for j in range(NUM_RUNS):
                i = 0
                curr_file = INPATH + "dedup_"+str(cache)+"_"+str(membw)+"_"+str(cpuno)+"_perf_mt_"+str(j+1)+"_clean.txt"

                # Create new data set
                curr_file_data = [ [[],[],[],[],[]] for k in range(NUM_CPUS) ]

                print("Parsing file " + curr_file)

                for line in open(curr_file, 'r'):
                    i = i + 1

                    # Process data for each line
                    splt = line.split()
                    if(len(splt) != 7):
                        continue

                    # Determine CPU of current line
                    ccpu = int(splt[1][3:]) - CPU_START_INDEX

                    curr_file_data[ccpu][0].append(float(splt[0]))
                    curr_file_data[ccpu][1].append(int(splt[3]))
                    curr_file_data[ccpu][2].append(int(splt[4]))
                    curr_file_data[ccpu][3].append(int(splt[5]))
                    curr_file_data[ccpu][4].append(int(splt[6]))

                    if i > 10000:
                        break


                allfiles_data.append(curr_file_data)

                # if xicomp == 3:
                if xicomp == 1:
                    axs[cpuno][VALID_CACHE.index(cache)].plot(allfiles_data[j][cpuno][0],allfiles_data[j][cpuno][1], linewidth=PLOTS_LINEWIDTH, color=PLOTS_LINECOLOR, alpha=PLOTS_LINEALPHA)
                # elif xicomp == 1:
                elif xicomp == 2:
                    axs[cpuno][VALID_CACHE.index(cache)].plot(allfiles_data[j][cpuno][0],list( map(add, allfiles_data[j][cpuno][2], allfiles_data[j][cpuno][3]) ), linewidth=PLOTS_LINEWIDTH, color=PLOTS_LINECOLOR, alpha=PLOTS_LINEALPHA)
                else:
                    axs[cpuno][VALID_CACHE.index(cache)].plot(allfiles_data[j][cpuno][0],allfiles_data[j][cpuno][4], linewidth=PLOTS_LINEWIDTH, color=PLOTS_LINECOLOR, alpha=PLOTS_LINEALPHA)

            # Load and plot synthetic profile
            synth_profile_file = INPATH_SYNTH + "dedup-synth_CPU"+str(cpuno+1)+"_c"+str(VALID_CACHE.index(cache)+1)+".txt"
            if cpuno == 0:
                synth_profile_file = INPATH_SYNTH + "dedup-synth_CPU"+str(4)+"_c"+str(VALID_CACHE.index(cache)+1)+".txt"
            elif cpuno == 2:
                synth_profile_file = INPATH_SYNTH + "dedup-synth_CPU"+str(1)+"_c"+str(VALID_CACHE.index(cache)+1)+".txt"
            elif cpuno == 3:
                synth_profile_file = INPATH_SYNTH + "dedup-synth_CPU"+str(3)+"_c"+str(VALID_CACHE.index(cache)+1)+".txt"

            curr_synth_profile = [[],[]]
            for line in open(synth_profile_file, 'r'):
                splt = line.split(",")
                if(len(splt) != 7):
                    continue

                curr_synth_profile[0].append(float(splt[0]))
                curr_synth_profile[1].append(float(splt[xicomp]))

            if xicomp == 3:
                pdb.set_trace()

            axs[cpuno][VALID_CACHE.index(cache)].plot(curr_synth_profile[0],curr_synth_profile[1], linewidth=1, color='red', alpha=0.5)
            

    for j in range(len(VALID_CACHE)):
        axs[0][j].set_title('Cache = %d' % VALID_CACHE[j])
        axs[NUM_CPUS-1][j].set_xlabel(r'$t$ [s]')

    for j in range(NUM_CPUS):
        axs[j][0].set_ylabel(r'CPU%d' % (j+1))

    fig.set_size_inches(15,18)
    plt.savefig(OUTPATH + "dedup_syntheticcomp_xi%d.png" % (xicomp), dpi=300)
    # plt.show()
    fig.clear()
    plt.close(fig)
    # exit()
    ###################


# plt.show()

