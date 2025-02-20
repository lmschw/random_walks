import numpy as np

def levy_walk_3(n_steps, alpha, x_range=[-50000, 50000], y_range=[-50000, 50000]):    
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