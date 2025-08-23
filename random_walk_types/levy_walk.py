

import numpy as np
from scipy.stats import levy_stable

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

def levy_walk(n_steps, alpha, x_range=[-50000, 50000], y_range=[-50000, 50000]):    
    positions = np.zeros((n_steps, 2))

    x, y = np.random.uniform(x_range[0], x_range[1]), np.random.uniform(y_range[0], y_range[1])
    for j in range(n_steps):
        # Generate step length from power-law distribution
        step_length = np.random.pareto(alpha)

        # Generate random direction
        theta = np.random.uniform(0, 2 * np.pi)

        # Update position
        new_x = x + step_length * np.cos(theta)
        new_y = y + step_length * np.sin(theta)

        # Check boundary conditions
        if x_range[0] <= new_x <= x_range[1] and y_range[0] <= new_y <= y_range[1]:
            x, y = new_x, new_y

        positions[j] = [x, y]

    return positions

def levy_walk_2(num_steps, alpha=1.5):
    # Generate Lévy-distributed step sizes using a power-law distribution
    # Lévy distribution has the form of p(x) ~ x^(-1-α)
    step_sizes = np.random.standard_t(df=alpha, size=num_steps)
    
    # Initialize the starting position
    x, y = 0, 0
    positions = [(x, y)]
    
    for step in step_sizes:
        # Choose a random angle for the direction (uniform distribution between 0 and 2π)
        angle = np.random.uniform(0, 2 * np.pi)
        
        # Update position
        x += step * np.cos(angle)
        y += step * np.sin(angle)
        
        positions.append((x, y))
    
    return np.array(positions)

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

