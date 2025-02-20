import numpy as np
import matplotlib.pyplot as plt
from hurst import compute_Hc, random_walk
from levy_walk import levy_walk_simulation
from levy_walk_2 import levy_walk
#from levy_walk_3 import levy_walk
from brownian_motion import brownian_motion_2d

# Simulation 
STEPS = 10000

# Distribution 
DIST_TYPE = 3
A = 1.5         
B = 1
random_changes = 1. + np.random.randn(99999) / 1000.
series = np.cumprod(random_changes)  # create a random walk from random changes
#trajectory = levy_walk_simulation(STEPS, DIST_TYPE, A, B)
#trajectory = levy_walk(STEPS, A)
trajectory = brownian_motion_2d(STEPS)

persistent = random_walk(99999, proba=0.7)
# Evaluate Hurst equation
H_x, c_x, data_x = compute_Hc(trajectory[:,0], kind='random_walk', simplified=False)
H_y, c_y, data_y = compute_Hc(trajectory[:,1], kind='random_walk', simplified=False)
H_p, c_p, data_p = compute_Hc(persistent, kind='random_walk', simplified=False)


print(H_x, H_y)
print(H_p)

H = np.average([H_x, H_y])
c = np.average([c_x, c_y])

# Plot
f, ax = plt.subplots()
ax.plot(data_x[0], c_x*data_x[0]**H_x, color="deepskyblue")
ax.scatter(data_x[0], data_x[1], color="purple")
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Time interval')
ax.set_ylabel('R/S ratio')
ax.grid(True)
plt.show()

f, ax = plt.subplots()
ax.plot(data_y[0], c_x*data_y[0]**H_y, color="deepskyblue")
ax.scatter(data_y[0], data_y[1], color="purple")
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Time interval')
ax.set_ylabel('R/S ratio')
ax.grid(True)
plt.show()

print("H={:.4f}, c={:.4f}".format(H,c))