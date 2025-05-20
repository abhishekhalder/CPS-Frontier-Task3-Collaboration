#
# Plot all figures to be used in in ACC paper
#
# Author: Georgiy Antonovich Bondar
# Date  : 09-26-2023
#
#!/usr/bin/python3
import matplotlib
# matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from matplotlib import ticker as tck
import numpy as np
import pandas as pd
import math
import csv
import pdb
import re
import os

## Configuration ##
###################
NUM_XIS       = 3
NUM_CPUS      = 4
NUM_TAUS      = 5
NUM_MARGINALS = 26
BC_DATAOUT_PATH = "./Barycenter_Graph_SBP/data_out/"
SP_DATAOUT_PATH = "./SeriesParallel_Graph_SBP/data_out/"
OUTPATH  = "./halder_outfiles_0403/"
###################

#==============================================================================
# Make plots beautiful
#==============================================================================
pts_per_inch = 72.27
# write "\the\textwidth" (or "\showthe\columnwidth" for a 2 collumn text)
text_width_in_pts = 300.0
# inside a figure environment in latex, the result will be on the
# dvi/pdf next to the figure. See url above.
text_width_in_inches = text_width_in_pts / pts_per_inch
# figure.png or figure.eps will be intentionally larger, because it is prettier
inverse_latex_scale = 4
fig_proportion = (3.0 / 3.0)
csize = inverse_latex_scale * fig_proportion * text_width_in_inches
# always 1.0 on the first argument
fig_size = (1.0 * csize, 0.85 * csize)
# find out the fontsize of your latex text, and put it here
text_size = inverse_latex_scale * 12 #9
label_size = inverse_latex_scale * 10
tick_size = inverse_latex_scale * 8
# learn how to configure:
# http://matplotlib.sourceforge.net/users/customizing.html
params = {#'backend': 'ps',
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
          'text.latex.preamble': "\n".join([r"\usepackage{amsmath}",
                                  r"\usepackage{stmaryrd}",
                                  r"\usepackage{bm}"]),
          'savefig.dpi': 300
          }
plt.rcParams.update(params)
#==============================================================================
# END Make plots beautiful
#==============================================================================

'''
## Figure 1: Convergence of Sinkhorn errors
#==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2);

# Plot Barycentric convergence
for file in os.listdir(BC_DATAOUT_PATH):
    if file.startswith("f1_err"):
        data = pd.read_csv(BC_DATAOUT_PATH + file, header=None)
        data = data.T
        data.columns = ["i", "err"]
        ax1.semilogy(list(data.i), list(data.err))

ax1.set_xlabel(r'Iteration index $(k)$', fontsize=30)
ax1.set_ylabel(r'$d_{\rm{H}}\Bigr(u_{\sigma}^{(k)}, u_{\sigma}^{(k-1)}\Bigr)$', fontsize=30)
ax1.spines[['top', 'right']].set_visible(False)

# Plot Series-parallel convergence
for file in os.listdir(SP_DATAOUT_PATH):
    if file.startswith("f1_err"):
        data = pd.read_csv(SP_DATAOUT_PATH + file, header=None)
        data = data.T
        data.columns = ["i", "err"]
        ax2.semilogy(list(data.i), list(data.err))

ax2.set_xlabel(r'Iteration index $(k)$', fontsize=30)
ax2.spines[['top', 'right']].set_visible(False)

#==============================================================================
'''

## Figure 2: Predicted vs. measured distributions, all CPUs
#==============================================================================
for j in range(NUM_CPUS):
# for j in range(1):
    fig, ax = plt.subplots(NUM_XIS, NUM_TAUS);
    for i in range(NUM_TAUS):
        for k in range(NUM_XIS):

            data_ms = pd.read_csv(BC_DATAOUT_PATH + "measured_t" + str(i+1) + "_xi" + str(k+1) + "_CPU" + str(j+1) + ".txt", header=None)
            data_bc = pd.read_csv(BC_DATAOUT_PATH + "interpolated_t" + str(i+1) + "_xi" + str(k+1) + "_CPU" + str(j+1) + ".txt", header=None)
            data_sp = pd.read_csv(SP_DATAOUT_PATH + "interpolated_t" + str(i+1) + "_xi" + str(k+1) + "_CPU" + str(j+1) + ".txt", header=None)
            # data_bc = pd.read_csv(BC_DATAOUT_PATH + "interpolated_t" + str(i+1) + "_CPU" + str(j+1) + ".txt", header=None)
            # data_sp = pd.read_csv(SP_DATAOUT_PATH + "interpolated_t" + str(i+1) + "_CPU" + str(j+1) + ".txt", header=None)

            # print(str(i) + "," + str(j))
            data_ms.columns = ["x", "marg"]
            data_bc.columns = ["x", "marg"]
            data_sp.columns = ["x", "marg"]

            ax[k,i].plot(list(data_ms.x), list(data_ms.marg), 'k', linewidth=5.0, alpha=0.5)
            ax[k,i].plot(list(data_bc.x), list(data_bc.marg), 'r', linewidth=5.0, alpha=0.8)
            ax[k,i].plot(list(data_sp.x), list(data_sp.marg), 'b', linewidth=2.5, alpha=1.0)

            if not any(list(data_ms.x)):
                ax[k,i].axvline(x=0, color='k', linestyle='solid', linewidth=4.0, alpha=0.5)
            if not any(list(data_bc.x)):
                ax[k,i].axvline(x=0, color='r', linestyle='--', linewidth=5.0, alpha=0.8)
            if not any(list(data_sp.x)):
                ax[k,i].axvline(x=0, color='b', linestyle=':', linewidth=2.5, alpha=1.0)

            ax[k,i].set_xticks([])
            ax[k,i].set_yticks([])

            if k == 0:
                ax[k,i].set_title("$$\widehat{\sigma}=" + str(i+1) + "$$", fontsize=30)
            if i == 0:
                ax[k,i].set_ylabel("$$\\mathbf{\\xi}_" + str(k+1) + "$$")

    plt.suptitle("CPU" + str(j+1))

plt.show()
exit()


