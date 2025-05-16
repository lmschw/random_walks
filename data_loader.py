import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def calculate_angles(positions):
    """Calculate angles from positions."""
    d_theta = (np.arctan2(positions[:, 1], positions[:, 0]) + np.pi) / (2 * np.pi)
    return d_theta

def load_all_data_in_directory(directory_path, random_walk_type):
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
    with open(file_path, 'rb') as f:
        loaded_data = pickle.load(f)
    return np.array(loaded_data["trajectory"])

def main():
    curr_dir = os.path.dirname(__file__)
    exp_results_dir = os.path.join(curr_dir, 'brown_free_25')
    load_data(exp_results_dir)
