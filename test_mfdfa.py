from random_walk_types.brownian_motion import brownian_motion_2d_without_sigma
from random_walk_types.levy_walk import levy_walk  
from random_walk_types.correlated_random_walk import correlated_random_walk_2d
from metrics.hurst_exponent_mfdfa import run_for_trajectory
import numpy as np
    
t_final = 2000
delta_t = 0.01

time = np.arange(0, t_final, delta_t)

# Distribution 
DIST_TYPE = 3
A = 2        
B = 1
steps = int(t_final / delta_t)
print(f"STEPS: {steps}")
#trajectory = levy_walk_simulation(steps, DIST_TYPE, A, B)
trajectory_l = levy_walk(steps, A)
trajectory_b = brownian_motion_2d_without_sigma(steps)
trajectory_c = correlated_random_walk_2d(steps)

print("Lévy walk")
run_for_trajectory(trajectory_l)

print("Brownian motion")
run_for_trajectory(trajectory_b)

print("CRW")
run_for_trajectory(trajectory_c)