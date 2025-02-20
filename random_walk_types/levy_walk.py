

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import levy_stable
from hurst import compute_Hc
from scipy.optimize import curve_fit

# Simulation 
STEPS = 1000

# Distribution 
DIST_TYPE = 3
A = 2         
B = 1

# Flight range 
MIN_LENGTH = 1
MAX_LENGTH = 50
N_POINTS = 100

def project_flight(r, theta):
    """
    Projects the fly lenght in the given direction

    """
    x_coord = r * np.cos(theta)
    y_coord = r * np.sin(theta)
    return np.array([x_coord, y_coord])


def single_levy_step(L, A, B):
    """
    Samples a fly lenght from the given distribution

    """
    t = np.linspace(MIN_LENGTH, MAX_LENGTH, N_POINTS)

    t = np.linspace(levy_stable.ppf(0.2, A, B),
                levy_stable.ppf(0.99, A, B), 100)
    
    if L == 1:    # Lévy-Smirnov distribution
        s = (1/np.sqrt(2*np.pi))*(t-0.5)**(-3/2)*(np.exp(-1/(2*(t-0.5))))
    elif L == 2:  # Cauchy distribution
        s = (1/np.pi)*(1/(0.01+(t-1)**2))
    elif L == 3:  # Levy stable distribution using scipy
        s = levy_stable.pdf(t, A, B)
    elif L == 4:  # Custom Levy distribution from the research paper
        r = levy_stable.rvs(A, B, size=1000)[0]
        
        fly_length = r
        return fly_length
    

    fly_length = np.random.choice(t, size=1, p=s/s.sum())[0]
    fly_length = MIN_LENGTH if fly_length < MIN_LENGTH else MAX_LENGTH if fly_length > MAX_LENGTH else fly_length
    
    return fly_length

def levy_walk_simulation(N, L, A, B):
    """
    Simulate a Levy walk and visualize the trajectory and probability density distribution

    """
    pos = np.array([0, 0])
    pos_track = pos
    
    for _ in range(N):
        if L == 5:
            direction = np.random.rand() * np.pi 
        else:
            direction = np.random.uniform(-np.pi, np.pi)

        step_size = single_levy_step(L, A, B)
        step = project_flight(step_size, direction)

        # Update position
        pos = pos + step
        pos_track = np.vstack((pos_track, pos))
    
    return pos_track

def plot_levy_simulation(pos_track):
    """
    Plot the trajectory and probability density distribution

    """
    if DIST_TYPE == 1:
        pdf = "Levy-Smirnoff"
    elif DIST_TYPE == 2:
        pdf = "Cauchy"
    elif DIST_TYPE == 3:
        pdf = "Levy Scipy"
    elif DIST_TYPE == 4:
        pdf = "Levy Paper"

    plt.figure(figsize=(12, 6))

    # Trajectory
    plt.subplot(1, 2, 1)
    plt.plot(pos_track[:, 0], pos_track[:, 1], marker='o', linestyle='-', color='b')
    plt.title( pdf + ' Trajectory')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid(True)

    # Pdf
    plt.subplot(1, 2, 2)
    step_sizes = np.linalg.norm(pos_track[1:] - pos_track[:-1], axis=1)  
    plt.hist(step_sizes, bins=20, density=True, alpha=0.7, color='green', edgecolor='black')
    plt.title('Probability Density Distribution of Step Sizes')
    plt.xlabel('Step Size')
    plt.ylabel('Probability Density')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    trajectory = levy_walk_simulation(STEPS, DIST_TYPE, A, B)
    plot_levy_simulation(trajectory)
