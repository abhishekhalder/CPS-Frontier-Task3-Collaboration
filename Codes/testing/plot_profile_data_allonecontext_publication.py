#
# Given a specific context [Cache,MemBW,PathNo] plot all 500 profiles superimposed on 4 plots (1 for each state variable), in the format requested by Dr. Halder for publication.
#
# Author: Georgiy Antonovich Bondar
# Date  : 09-05-2023
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
OUTPATH  = "./halder_outfiles_0824/"
###################
PLOTS_LINECOLOR = 'grey'
PLOTS_LINEWIDTH = 0.3
PLOTS_LINEALPHA = 0.05

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
params = {'backend': 'ps',
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
          'text.latex.preamble': [r'\usepackage{amsmath}'],
          }
plt.rcParams.update(params)

#====================================================
# END Make plots beautiful
#====================================================

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

    allfiles_data.append(curr_file_data)

##########################################


## Plot all data ##
###################
fig, axs = plt.subplots(1, 4)
fig.suptitle(r'Measured states $\mathbf{\xi}$ for context $\mathbf{c}=\left(%d, %d, y_{\rm{des}}^{%d}(x)\right)^{\top}$' % (CACHE, MEMBW, (PATHNO+1)), fontsize=30)


for j in range(NUM_RUNS):
    axs[0].plot(allfiles_data[j][0],allfiles_data[j][1], linewidth=PLOTS_LINEWIDTH, color=PLOTS_LINECOLOR, alpha=PLOTS_LINEALPHA)
    axs[1].plot(allfiles_data[j][0],allfiles_data[j][2], linewidth=PLOTS_LINEWIDTH, color=PLOTS_LINECOLOR, alpha=PLOTS_LINEALPHA)
    axs[2].plot(allfiles_data[j][0],allfiles_data[j][3], linewidth=PLOTS_LINEWIDTH, color=PLOTS_LINECOLOR, alpha=PLOTS_LINEALPHA)
    axs[3].plot(allfiles_data[j][0],allfiles_data[j][4], linewidth=PLOTS_LINEWIDTH, color=PLOTS_LINECOLOR, alpha=PLOTS_LINEALPHA)

    '''
    axs[0].semilogy(allfiles_data[j][0],allfiles_data[j][1], linewidth=PLOTS_LINEWIDTH, color=PLOTS_LINECOLOR, alpha=PLOTS_LINEALPHA)
    axs[1].semilogy(allfiles_data[j][0],allfiles_data[j][2], linewidth=PLOTS_LINEWIDTH, color=PLOTS_LINECOLOR, alpha=PLOTS_LINEALPHA)
    axs[2].semilogy(allfiles_data[j][0],allfiles_data[j][3], linewidth=PLOTS_LINEWIDTH, color=PLOTS_LINECOLOR, alpha=PLOTS_LINEALPHA)
    axs[3].semilogy(allfiles_data[j][0],allfiles_data[j][4], linewidth=PLOTS_LINEWIDTH, color=PLOTS_LINECOLOR, alpha=PLOTS_LINEALPHA)
    '''

axs[0].set_title('Instructions Retired', fontsize=25)
axs[1].set_title('LLC Loads', fontsize=25)
axs[2].set_title('LLC Stores', fontsize=25)
axs[3].set_title('LLC Loads Misses', fontsize=25)

axs[0].set_xlabel(r'$t$ [s]')
axs[1].set_xlabel(r'$t$ [s]')
axs[2].set_xlabel(r'$t$ [s]')
axs[3].set_xlabel(r'$t$ [s]')

axs[0].set_ylabel(r'$\xi_{1}$')
axs[1].set_ylabel(r'$\xi_{2}$')
axs[2].set_ylabel(r'$\xi_{3}$')
axs[3].set_ylabel(r'$\xi_{4}$')

'''
axs[0].spines[['top', 'right']].set_visible(False)
axs[1].spines[['top', 'right']].set_visible(False)
axs[2].spines[['top', 'right']].set_visible(False)
axs[3].spines[['top', 'right']].set_visible(False)
'''

axs[3].set_ylim([-5, 100])

# plt.savefig(OUTPATH + 'all_measured.png', dpi=300)
plt.show()
###################

