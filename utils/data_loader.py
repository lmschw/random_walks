import os
import pickle
import numpy as np

def calculate_angles(positions):
    """Calculate angles from positions."""
    d_theta = (np.arctan2(positions[:, 1], positions[:, 0]) + np.pi) / (2 * np.pi)
    return d_theta

def load_all_data_in_directory(directory_path, random_walk_type):
    """ Load all pickle files in a directory and return an array of angles from all runs."""
    trajectories = []  # Array to hold angles from all runs

    for filename in os.listdir(directory_path):
        if filename.endswith('.pickle') and filename.startswith(random_walk_type):
            print(f"Reading File {filename}")
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'rb') as f:
                loaded_data = pickle.load(f)

            trajectory = np.array(loaded_data["trajectory"])
            trajectories.append(trajectory)
    return trajectories

def load_data(file_path):
    """ Load a pickle file and return an array of angles from the trajectory."""
    with open(file_path, 'rb') as f:
        loaded_data = pickle.load(f)
    return np.array(loaded_data["trajectory"])

def main():
    """ Main function to load data and process it."""
    curr_dir = os.path.dirname(__file__)
    exp_results_dir = os.path.join(curr_dir, 'brown_free_25')
    load_data(exp_results_dir)
