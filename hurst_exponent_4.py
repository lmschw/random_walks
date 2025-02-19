import numpy as np
import matplotlib.pyplot as plt
import hurst as hs
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
def Calculate_Hurst(trajectory, window_size, ser_type):
    """
    Calculates the hurst exponent for the var:variable in df:dataset for given rolling window size
    ser_type: gives the type of series. It can be of three types:
        'change':      a series is just random values (i.e. np.random.randn(...))
        'random_walk': a series is a cumulative sum of changes (i.e. np.cumsum(np.random.randn(...)))
        'price':       a series is a cumulative product of changes (i.e. np.cumprod(1+epsilon*np.random.randn(...))
    """
    hursts = [np.nan] * (window_size-1)
    #var_values = list(df[var])
    for i in range(0, len(trajectory)-(window_size-1)):
        H, c, data = hs.compute_Hc(trajectory[i:i+window_size], kind=ser_type, simplified=True)
        hursts.append(H)
    return hursts

trajectory = levy_walk(STEPS, A)
trajectory = trajectory[:,0]
hurst_windows = [100, 500, 1000]
hurst_values = []
for i in range(0, len(hurst_windows)):
    hurst_values.append(Calculate_Hurst(trajectory, hurst_windows[i], 'random_walk'))

print(hurst_values)

print(np.average(hurst_values))
print(hs.compute_Hc(trajectory))