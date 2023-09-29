#
# Plot all sim paths
#
# Author: Georgiy Antonovich Bondar
# Date  : 09-08-2023
#
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import numpy as np
import pandas as pd
import math
import csv
import pdb
import re

## Configuration ##
###################
NUM_PATHS = 12
INPATH   = "../../Codes/kbm_sim/sim_paths/"
OUTPATH  = "./halder_outfiles_0824/"
###################

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
          'text.latex.preamble': [r'\usepackage{amsmath}',
                                  r'\usepackage{bm}'],
          }
plt.rcParams.update(params)

#====================================================
# END Make plots beautiful
#====================================================

allfiles_data = []

## Read in paths from relevant files ##
##########################################
for j in range(NUM_PATHS):
    pathdata = [];

    data = pd.read_csv(INPATH + "path_wypts_%d.txt" % (j), header=None)
    data.columns = ["x", "y", "v"]
    pathdata = [ list(data.x), list(data.y) ]

    allfiles_data.append( pathdata )

# pdb.set_trace()

##########################################

## Plot all paths ##
####################
f1 = plt.figure("Figure 1")
ax = plt.subplot(111)
for i in range(len(allfiles_data)):
    ax.plot(allfiles_data[i][0], allfiles_data[i][1])

ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y_{\rm{des}}^i(x)$')
ax.set_title(r'All simulated paths  $\mid$ $y_{\rm{des}}^{i}(x)\:\forall i\in\{1,2,\dots,12\}$')
ax.spines[['top', 'right']].set_visible(False)

# plt.savefig(OUTPATH + 'hist_endtimes.png', dpi=300)

###################

plt.show()
exit()


