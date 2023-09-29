#
# Given a specific context [Cache,MemBW,PathNo] find the time at which each control cycle ends for all 500 profiles.
# Construct and plot distributions for each control cycle end time, as well as at some equispaced times between 
# these end times.
#
# Author: Georgiy Antonovich Bondar
# Date  : 09-21-2023
#
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import math
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
NUM_STDS = 1
NUM_INTERMEDIATE = 2                        # number of marginals to get between cycle end times
INPATH   = "../../Data/kbm_sim_profile/"
OUTPATH  = "./halder_outfiles_0921/"
###################
KDE_KERNEL = "gaussian"
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

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

allfiles_data = []

## Read in data from all relevant files ##
##########################################
for j in range(NUM_RUNS):
    i = 0
    data_cnt   = 0
    data_found = False
    curr_file = INPATH + "kbm_sim_"+str(CACHE)+"_"+str(MEMBW)+"_profile_"+str(PATHNO)+"_"+str(j+1)+".txt"

    # print("Parsing file " + curr_file)

    for line in open(curr_file, 'r'):
        i = i + 1

        # Process data for each line
        splt = line.split()

        # Find lines of correct format
        if( (len(splt) != 2) or not(is_number(splt[0])) or not(is_number(splt[1])) ):
            if( data_found ):
                break
            else:
                continue

        if( not(data_found) ):
            data_found = True;

        data_cnt = data_cnt + 1

        if( len(allfiles_data) < data_cnt ):
            allfiles_data.append([[float(splt[0])], [float(splt[1])]])
        else:
            allfiles_data[data_cnt-1][0].append(float(splt[0]))
            allfiles_data[data_cnt-1][1].append(float(splt[1]))

        if i > 1000:
            break

##########################################

# pdb.set_trace()

## Plot all data ##
###################
tN_stats = []
f1, axs = plt.subplots(math.ceil(len(allfiles_data)/2), 2)
f1.suptitle(r'Control Cycle Durations $\mid$ c=[%d, %d, %d]' % (CACHE, MEMBW, PATHNO))
for i in range(len(allfiles_data)):

    # KDE
    kde = KernelDensity(kernel=KDE_KERNEL, bandwidth=0.001)
    kde_fit = kde.fit(np.array(allfiles_data[i][0])[:, np.newaxis])
    
    # Plot histogram 
    axs[math.floor(i/2), i%2].hist(allfiles_data[i][0], bins=25, density=True, alpha=0.6, color='g', edgecolor='k')
    xmin, xmax = axs[math.floor(i/2), i%2].get_xlim()

    # Plot KDE fit
    x = np.linspace(xmin, xmax, 100)
    log_dens = kde_fit.score_samples(x[:, np.newaxis])
    # Mean and std calculation
    pdf = lambda x : np.exp(kde_fit.score_samples([[x]]))[0]
    mu = quad(lambda x: x * pdf(x), a=-np.inf, b=np.inf)[0]
    std = np.sqrt(quad(lambda x: (x ** 2) * pdf(x), a=-np.inf, b=np.inf)[0] - mu ** 2)
    axs[math.floor(i/2), i%2].fill(x, np.exp(log_dens), fc="#AAAAFF")
    title = r'$t_%d\:(\mu = %.4f,  \sigma = %.4f)$' % (i+1, mu, std)
    axs[math.floor(i/2), i%2].set_title(title)

for ax in axs.flat[len(allfiles_data):]:
    ax.remove()

#####

f1, axs = plt.subplots(math.ceil(len(allfiles_data)/2), 2)
f1.suptitle(r'Control Cycle End Times $\mid$ c=[%d, %d, %d]' % (CACHE, MEMBW, PATHNO))
for i in range(len(allfiles_data)):

    # KDE
    kde = KernelDensity(kernel=KDE_KERNEL, bandwidth=0.001)
    kde_fit = kde.fit(np.array(allfiles_data[i][1])[:, np.newaxis])
    
    # Plot histogram 
    axs[math.floor(i/2), i%2].hist(allfiles_data[i][1], bins=25, density=True, alpha=0.6, color='g', edgecolor='k')
    xmin, xmax = axs[math.floor(i/2), i%2].get_xlim()

    # Plot KDE fit
    x = np.linspace(xmin, xmax, 100)
    log_dens = kde_fit.score_samples(x[:, np.newaxis])
    # Mean and std calculation
    pdf = lambda x : np.exp(kde_fit.score_samples([[x]]))[0]
    mu = quad(lambda x: x * pdf(x), a=xmin, b=xmax)[0]
    std = np.sqrt(quad(lambda x: (x ** 2) * pdf(x), a=xmin, b=xmax)[0] - mu ** 2)
    axs[math.floor(i/2), i%2].fill(x, np.exp(log_dens), fc="#AAAAFF")
    title = r'$t_%d\:(\mu = %.4f,  \sigma = %.4f)$' % (i+1, mu, std)
    axs[math.floor(i/2), i%2].set_title(title)

    # tN_stats.append([mu, std])

for ax in axs.flat[len(allfiles_data):]:
    ax.remove()

#####

f3 = plt.figure("Figure 3")
for i in range(len(allfiles_data)):

    ax = plt.subplot(111)

    tmin = np.min(np.array(allfiles_data[i][1]))
    tmax = np.max(np.array(allfiles_data[i][1]))
    print("%.4f, %.4f" % (tmin, tmax))

    # Get optimal bandwidth
    grid = GridSearchCV(KernelDensity(),
                    {'bandwidth': np.linspace(0.0001, 0.2, 100)},
                    cv=20) # 20-fold cross-validation
    grid.fit(np.array(allfiles_data[i][1])[:, np.newaxis])
    print(grid.best_params_)

    # KDE
    # kde = KernelDensity(kernel=KDE_KERNEL, bandwidth=0.001)
    kde = KernelDensity(kernel=KDE_KERNEL, bandwidth=grid.best_params_['bandwidth'])
    kde_fit = kde.fit(np.array(allfiles_data[i][1])[:, np.newaxis])
    
    # Plot histogram
    ## ax.hist(allfiles_data[i][1], bins=25, density=True, alpha=0.6, color='g', edgecolor='k')
    ax.hist(allfiles_data[i][1], bins=25, density=True, fc='gray', histtype='stepfilled', alpha=0.5)
    ## ax.hist(allfiles_data[i][1], bins=25, density=False, fc='gray', histtype='stepfilled', alpha=0.5, weights=np.full(len(allfiles_data[i][1]), 1/len(allfiles_data[i][1])))
    xmin, xmax = plt.xlim()

    # Plot KDE fit
    # x = np.linspace(xmin, xmax, 100)
    x = np.linspace(tmin, tmax, 100)
    log_dens = kde_fit.score_samples(x[:, np.newaxis])
    # Mean and std calculation
    pdf = lambda x : np.exp(kde_fit.score_samples([[x]]))[0]
    mu = quad(lambda x: x * pdf(x), a=xmin, b=xmax)[0]
    std = np.sqrt(quad(lambda x: (x ** 2) * pdf(x), a=xmin, b=xmax)[0] - mu ** 2)
    # print("(*) (mu,std)=(%.4f, %.4f)" % (mu, std))
    # plt.fill(x, np.exp(log_dens), fc="#AAAAFF")
    ## plt.plot(x, np.exp(log_dens)/len(allfiles_data[i][1]), linewidth=2, alpha=1, color='blue')
    plt.plot(x, np.exp(log_dens), linewidth=2, alpha=1, color='blue')
    ax.spines[['top', 'right']].set_visible(False)
    # ax.yaxis.set_major_formatter(tck.PercentFormatter(xmax=1))
    ax.yaxis.set_ticks(np.arange(0, 100, 20))
    ax.yaxis.set_ticklabels(np.arange(0, 100/len(allfiles_data[i][1]), 20/len(allfiles_data[i][1])))

    tN_stats.append([mu, std])

ax.set_xlabel(r'$t$ [s]')
ax.set_title(r'End times for all control cycles $\mid$ $\mathbf{c}=\left(%d, %d, y_{\rm{des}}^{%d}(x)\right)^{\top}$' % (CACHE, MEMBW, (PATHNO+1)))

plt.savefig(OUTPATH + 'hist_endtimes.png', dpi=300)

###################

# plt.show()
# exit()


## Use statistics (mu,std) from plotting to determine t_N values ##
###################################################################
tN_vals = []

print("\nWe subtract %d standard deviations from the mean to obtain each t_N:\n" % (NUM_STDS))

print("t_0 = 0.0")

for i in range(len(tN_stats)):
    tN_vals.append(tN_stats[i][0] - NUM_STDS*tN_stats[i][1])
    print("n=%i | (mu,std) = (%.4f,%.4f) => t_%d = %0.4f" % (i+1, tN_stats[i][0], tN_stats[i][1], i+1, tN_vals[i]))
###################################################################


## For each of the NUM_RUNS profiles, find the sample \xi matching the t_N values ##
####################################################################################

# needful 09-21
temp = []
for i in range(len(tN_vals)):

    if( i==0 ):
        prev = 0;
    else:
        prev = tN_vals[i-1]

    for j in range(1,NUM_INTERMEDIATE+1):
        temp.append(prev + (j/(NUM_INTERMEDIATE+1))*(tN_vals[i]-prev))

    temp.append(tN_vals[i])

tN_vals = temp
print(tN_vals)

xi_vals = [[] for i in range(len(tN_vals)+1)]

for j in range(NUM_RUNS):
    i = 0
    last_line = []
    curr_file = INPATH + "kbm_sim_"+str(CACHE)+"_"+str(MEMBW)+"_perf_"+str(PATHNO)+"_"+str(j+1)+"_clean.txt"

    # print("Parsing file " + curr_file)

    for line in open(curr_file, 'r'):
        i = i + 1

        # Process data for each line
        splt = line.split()

        if(len(splt) != 6):
            continue

        # xi = [ float(splt[2]), float(splt[3]) ]                   # The relevant elements (instr. ret., LLC Loads)
        xi = [ float(splt[2]), float(splt[3]), float(splt[4]), float(splt[5]) ]

        # If reading the first line, save xi for t_0
        if( last_line == [] ):
            xi_vals[0].append(xi)
            last_line = splt
            # print("Found %f ~= 0 (t_0)" % float(splt[0]))
            continue

        # xi = [ float(last_line[2]), float(last_line[3]) ]         # The relevant elements (instr. ret., LLC Loads)
        xi = [ float(last_line[2]), float(last_line[3]), float(last_line[4]), float(last_line[5]) ]

        # Check if last line matches any of the t_N
        for k in range(len(tN_vals)):
            if( (float(last_line[0]) < tN_vals[k]) and (tN_vals[k] <= float(splt[0])) ):
                xi_vals[k+1].append(xi)
                # print("Found %f ~= %f (t_%d)" % (float(last_line[0]), tN_vals[k], k))
                break

        last_line = splt

        if i > 1000:
            break


####################################################################################


## Plot xi_vals at each t_N ##
##############################
ms = 2
f4, axs = plt.subplots(1, 4)
f4.suptitle(r'Stored Values of $\xi$ at Values of $t_N$ $\mid$ c=[%d, %d, %d]' % (CACHE, MEMBW, PATHNO))

for j in range(len(xi_vals[0])):
    axs[0].scatter(0, xi_vals[0][j][0], s=ms)
    axs[1].scatter(0, xi_vals[0][j][1], s=ms)
    axs[2].scatter(0, xi_vals[0][j][2], s=ms)
    axs[3].scatter(0, xi_vals[0][j][3], s=ms)

for i in range(1,len(xi_vals),1):
    for j in range(len(xi_vals[i])):
        axs[0].scatter(tN_vals[i-1], xi_vals[i][j][0], s=ms)
        axs[1].scatter(tN_vals[i-1], xi_vals[i][j][1], s=ms)
        axs[2].scatter(tN_vals[i-1], xi_vals[i][j][2], s=ms)
        axs[3].scatter(tN_vals[i-1], xi_vals[i][j][3], s=ms)

axs[0].set_title("Instructions Retired")
axs[1].set_title("LLC Loads")
axs[2].set_title("LLC Stores")
axs[3].set_title("LLC Misses")
##############################

## Write the needful to file ##
###############################
for i in range(len(xi_vals)):
    outfile = OUTPATH + "kbm_sim_"+str(CACHE)+"_"+str(MEMBW)+"_"+str(PATHNO)+"_t"+str(i)+"_allstates.txt"

    print("Writing file " + outfile)

    with open(outfile, 'w') as f:
        for j in range(len(xi_vals[i])):
            f.write("%f %f %f %f\n" % (xi_vals[i][j][0], xi_vals[i][j][1], xi_vals[i][j][2], xi_vals[i][j][3]))
###############################


## Display Plots ##
# plt.show()
print("Fertig.")
###################


