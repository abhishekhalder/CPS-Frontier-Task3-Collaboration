#!/usr/bin/python3
#
# Given a specific context [Cache,MemBW] plot all 300 profiles 
# superimposed on 3x4=12 plots (1 for each state variable, and 1 for eeach CPU),
# in the format requested by Dr. Halder for publication, combining LLC loads and
# stores into one variable, LLC requests.
#
# Author: Georgiy Antonovich Bondar
# Date  : 12-18-2023
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
[CACHE, MEMBW] = [0, 0]

# [CACHE, MEMBW] = [1, 72]
# [CACHE, MEMBW] = [1, 144]
# [CACHE, MEMBW] = [1, 216]
# [CACHE, MEMBW] = [1, 288]
# [CACHE, MEMBW] = [1, 360]

# [CACHE, MEMBW] = [3, 72]
# [CACHE, MEMBW] = [3, 144]
# [CACHE, MEMBW] = [3, 216]
# [CACHE, MEMBW] = [3, 288]
# [CACHE, MEMBW] = [3, 360]

# [CACHE, MEMBW] = [7, 72]
# [CACHE, MEMBW] = [7, 144]
# [CACHE, MEMBW] = [7, 216]
# [CACHE, MEMBW] = [7, 288]
# [CACHE, MEMBW] = [7, 360]

# [CACHE, MEMBW] = [15, 72]
# [CACHE, MEMBW] = [15, 144]
# [CACHE, MEMBW] = [15, 216]
# [CACHE, MEMBW] = [15, 288]
# [CACHE, MEMBW] = [15, 360]

# [CACHE, MEMBW] = [31, 72]
# [CACHE, MEMBW] = [31, 144]
# [CACHE, MEMBW] = [31, 216]
# [CACHE, MEMBW] = [31, 288]
# [CACHE, MEMBW] = [31, 360]
###################
CPU_START_INDEX = 0
# STRT_IND = 100 # This doesn't do anything, maybe someday it will
NUM_RUNS = 400                              # Valid values 1-300
NUM_CPUS = 4                               # Number of concurrent CPUs to expect
# INPATH = "../../Data/synthetic_profile_mt/"
# INPATH = "../../Data/synthetic_profile_mt_020824/"
# INPATH = "../../Data/canneal_profile_mt/"
INPATH = "../../Data/canneal_profile_mt_040424/"
OUTPATH  = "./halder_outfiles_0307/"
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
# for j in range(STRT_IND, NUM_RUNS):
for j in range(NUM_RUNS):
    i = 0
    # curr_file = INPATH + "canneal_"+str(CACHE)+"_"+str(MEMBW)+"_perf_mt_"+str(j+1)+"_clean.txt"
    curr_file = INPATH + "canneal_1__perf_mt_"+str(j+1)+"_clean.txt"
    # curr_file = INPATH + "synthetic_1__perf_mt_"+str(j+1)+"_clean.txt"

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

##########################################

## Plot all data ##
###################
fig, axs = plt.subplots(NUM_CPUS, 3)
fig.suptitle(r'Measured states $\mathbf{\xi}$ for context $\mathbf{c}=\left[%d, %d\right]^{\top}$' % (CACHE, MEMBW))


for j in range(NUM_RUNS):
# for j in range(100,400):
# for j in range(300,400):
    for k in range(NUM_CPUS):
        axs[k][0].plot(allfiles_data[j][k][0],allfiles_data[j][k][1], linewidth=PLOTS_LINEWIDTH, color=PLOTS_LINECOLOR, alpha=PLOTS_LINEALPHA)
        axs[k][1].plot(allfiles_data[j][k][0],list( map(add, allfiles_data[j][k][2], allfiles_data[j][k][3]) ), linewidth=PLOTS_LINEWIDTH, color=PLOTS_LINECOLOR, alpha=PLOTS_LINEALPHA)
        axs[k][2].plot(allfiles_data[j][k][0],allfiles_data[j][k][4], linewidth=PLOTS_LINEWIDTH, color=PLOTS_LINECOLOR, alpha=PLOTS_LINEALPHA)

        axs[k][0].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        axs[k][1].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        axs[k][2].ticklabel_format(axis='y', style='sci', scilimits=(0,0))

axs[0][0].set_title('Instructions Retired')
axs[0][1].set_title('LLC Requests')
axs[0][2].set_title('LLC Loads Misses')

axs[NUM_CPUS-1][0].set_xlabel(r'$t$ [s]')
axs[NUM_CPUS-1][1].set_xlabel(r'$t$ [s]')
axs[NUM_CPUS-1][2].set_xlabel(r'$t$ [s]')

# axs[0][0].set_ylabel(r'$\xi_{1}$')
# axs[0][1].set_ylabel(r'$\xi_{2}$')
# axs[0][2].set_ylabel(r'$\xi_{3}$')

for j in range(NUM_CPUS):
    axs[j][0].set_ylabel(r'CPU%d' % (j+1))

# axs[0][2].set_ylim([-5, 100])

# fig.tight_layout()

# plt.savefig(OUTPATH + 'all_measured.png', dpi=300)
plt.show()
###################

