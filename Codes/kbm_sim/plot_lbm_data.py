#
# Plot data output by env/controller program for Linear Bicycle Model
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
	
# 1st figure, vehicle states vs. time
figure1, axis1 = plt.subplots(1,4)
figure1.suptitle("Figure 1: Vehicle states vs. time")
axis1[0].plot(t, x, marker = 'None', c = 'r', linewidth=2)
axis1[0].set_title("Vehicle x position")
axis1[0].set_xlabel('t')
axis1[0].set_ylabel('x')

axis1[1].plot(t, y, marker = 'None', c = 'r', linewidth=2)
axis1[1].set_title("Vehicle y position")
axis1[1].set_xlabel('t')
axis1[1].set_ylabel('y')

axis1[2].plot(t,yaw, marker = 'None', c = 'r')
axis1[2].set_title("Vehicle yaw")
axis1[2].set_xlabel('t')
axis1[2].set_ylabel('yaw (rads)')

axis1[3].plot(t,v, marker = 'None', c = 'r')
axis1[3].set_title("Vehicle speed")
axis1[3].set_xlabel('t')
axis1[3].set_ylabel('v')

# 2nd figure, control outputs vs. time
figure2, axis2 = plt.subplots(1,2)
figure2.suptitle("Figure 2: Control outputs vs. time")
axis2[0].plot(t, throttle, marker = 'None', c = 'r', linewidth=2)
axis2[0].set_title("Vehicle throttle")
axis2[0].set_xlabel('t')
axis2[0].set_ylabel('throttle')

axis2[1].plot(t, delta, marker = 'None', c = 'r', linewidth=2)
axis2[1].set_title("Vehicle steering")
axis2[1].set_xlabel('t')
axis2[1].set_ylabel('steering')

# 3rd figure, desired vs. actual trajectory
figure3, axis3 = plt.subplots(1,1)
figure3.suptitle("Figure 3: Desired vs. actual trajectory")
axis3.plot(x, y, marker = 'None', c = 'k', linewidth=2)
axis3.plot(x_tr, y_tr, marker = 'None', c = 'b', linewidth=1)
axis3.set_xlabel('x')
axis3.set_ylabel('y')

plt.show()



