#
# Plot control loop execution times
#
# Author: Georgiy Antonovich Bondar
# Date  : 05-23-2023
#
import matplotlib.pyplot as plt

INFILE = "maintimes_fixed.txt"

t = []
lt = []
for line in open(INFILE, 'r'):
	lines = [i for i in line.split()]
	lt.append(float(lines[0]))
	t.append(float(lines[1]))
	

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
	
# Loop time vs. time
figure3, axis3 = plt.subplots(1,1, figsize=(1.5 * csize, 1.1 * csize))
figure3.suptitle("Figure 3: Control loop time vs. time")
axis3.plot(t, lt, marker = 'None', c = 'r', linewidth=5)
axis3.set_xlabel('t (s)')
axis3.set_ylabel('loop t (s)')

figure3.savefig('./plots/ex_times_fixed.png', dpi=300)

plt.show()
exit()


