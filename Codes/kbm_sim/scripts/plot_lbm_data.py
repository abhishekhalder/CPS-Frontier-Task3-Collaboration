#
# Plot data output by env/controller program for Kinematic Bicycle Model
#
# Author: Georgiy Antonovich Bondar
# Date  : 03-14-2023
#
import matplotlib.pyplot as plt

INFILE = "run_data.txt"
TRAJ_FILE = "racetrack_waypoints.txt"

t = []
x = []
x_tr = []
y = []
y_tr = []
yaw = []
v = []
v_tr = []
throttle = []
delta = []
for line in open(INFILE, 'r'):
	lines = [i for i in line.split()]
	t.append(float(lines[0]))
	x.append(float(lines[1]))
	y.append(float(lines[2]))
	yaw.append(float(lines[3]))
	v.append(float(lines[4]))
	throttle.append(float(lines[5]))
	delta.append(float(lines[6]))
	
for line in open(TRAJ_FILE, 'r'):
	lines = [i for i in line.split(",")]
	x_tr.append(float(lines[0]))
	y_tr.append(float(lines[1]))
	v_tr.append(float(lines[2]))


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
	
# 1st figure, vehicle states vs. time
figure1, axis1 = plt.subplots(1,4, figsize=(3.6 * csize, 0.85 * csize))
figure1.suptitle("Figure 1: Vehicle states vs. time")
axis1[0].plot(t, x, marker = 'None', c = 'b', linewidth=2)
axis1[0].set_title("Vehicle x position (m)")
axis1[0].set_xlabel('t (s)')
axis1[0].set_ylabel('x (m)')

axis1[1].plot(t, y, marker = 'None', c = 'b', linewidth=2)
axis1[1].set_title("Vehicle y position (m)")
axis1[1].set_xlabel('t (s)')
axis1[1].set_ylabel('y (m)')

axis1[2].plot(t,yaw, marker = 'None', c = 'b')
axis1[2].set_title("Vehicle yaw (rads)")
axis1[2].set_xlabel('t (s)')
axis1[2].set_ylabel('yaw (rads)')

axis1[3].plot(t,v, marker = 'None', c = 'b')
axis1[3].set_title("Vehicle speed (m/s)")
axis1[3].set_xlabel('t (s)')
axis1[3].set_ylabel('v (m/s)')

figure1.savefig('./plots/figure_1.png', dpi=300)

# 2nd figure, control outputs vs. time
figure2, axis2 = plt.subplots(1,2, figsize=(2.5 * csize, 0.85 * csize))

figure2.suptitle("Figure 2: Control outputs vs. time")
axis2[0].plot(t, throttle, marker = 'None', c = 'b', linewidth=2)
axis2[0].set_title("Vehicle throttle")
axis2[0].set_xlabel('t (s)')
axis2[0].set_ylabel('throttle ($m/s^2$)')

axis2[1].plot(t, delta, marker = 'None', c = 'b', linewidth=2)
axis2[1].set_title("Vehicle steering")
axis2[1].set_xlabel('t (s)')
axis2[1].set_ylabel('steering (rad/s)')

figure2.savefig('./plots/figure_2.png', dpi=300)

# 3rd figure, desired vs. actual trajectory
figure3, axis3 = plt.subplots(1,1, figsize=(1.5 * csize, 1.1 * csize))
figure3.suptitle("Figure 3: Desired vs. actual trajectory")
axis3.plot(x, y, marker = 'None', c = 'r', linewidth=5)
axis3.plot(x_tr, y_tr, marker = 'None', c = 'b', linewidth=1)
axis3.set_xlabel('x (m)')
axis3.set_ylabel('y (m)')

figure3.savefig('./plots/figure_3.png', dpi=300)

plt.show()
exit()


