import matplotlib.pyplot as plt
import pickle
import numpy as np

import metrics.vacf as vacf 

"""
Evaluate the velocity autocorrelation function (VACF) for different random walk types and implementations.
Saves plots of the VACF for each run and averaged results.
"""

sample_types = ["brown", "levy", "correlated"]
leader_types = ["free"]
num_agents_options = [1, 5, 7, 10]
steps_per_run = 25000
num_runs = 30
dimension = "2D"
implementation_types = ["old", "new"]

basepath = "J:/leader_emergence/results/2D/res/"
save_location = "vacf_results/"

for num_agents in num_agents_options:
    for sample_type in sample_types:
        for leader_type in leader_types:
            comp_results = []
            for implementation_type in implementation_types:
                if implementation_type == "new":
                    impl_prefix = "rtm_"
                else:
                    impl_prefix = ""
                results = []
                for run in range(num_runs):
                    filename = f"{impl_prefix}{sample_type}_{leader_type}_{num_agents**2}_run{run + 1}"
                    print(filename)
                    filepath = f"{basepath}{filename}.pickle"
                    with open(filepath, "rb") as input_file:
                        trajectory = pickle.load(input_file)["trajectory"]

                        result = vacf.velocity_autocorrelation(trajectory)
                        results.append(result)

                        # Plot the VACF
                        plt.figure(figsize=(8, 6))
                        plt.plot(np.arange(1, len(result) + 1), result)
                        plt.xlabel('Time Lag (τ)', fontsize=14)
                        plt.ylabel('VACF (C_v(τ))', fontsize=14)
                        plt.title(f'Velocity Autocorrelation Function (VACF) - {num_agents} agents, {sample_type}, {leader_type}, {implementation_type}, run={run}', fontsize=16)
                        plt.grid(True)
                        plt.savefig(f"{save_location}{filename}.svg")
                        plt.savefig(f"{save_location}{filename}.png")
                        plt.savefig(f"{save_location}{filename}.pdf")
                filename = f"{impl_prefix}{sample_type}_{leader_type}_{num_agents**2}"
                results_arr = np.array(results)
                plt.figure(figsize=(8, 6))
                plt.plot(np.arange(1, len(results_arr[0]) + 1), np.average(results_arr.T, axis=1))
                plt.xlabel('Time Lag (τ)', fontsize=14)
                plt.ylabel('VACF (C_v(τ))', fontsize=14)
                plt.title(f'Velocity Autocorrelation Function (VACF) - {num_agents} agents, {sample_type}, {leader_type}, {implementation_type}', fontsize=16)
                plt.grid(True)
                plt.savefig(f"{save_location}{filename}.svg")
                plt.savefig(f"{save_location}{filename}.png")
                plt.savefig(f"{save_location}{filename}.pdf")
                comp_results.append(np.average(results_arr.T, axis=1))
            filename = f"comp_{sample_type}_{leader_type}_{num_agents**2}"
            plt.figure(figsize=(8, 6))
            plt.plot(np.arange(1, len(results[0]) + 1), results[0])
            plt.plot(np.arange(1, len(results[0]) + 1), results[1])
            plt.legend(implementation_types, fontsize=12)

            plt.xlabel('Time Lag (τ)', fontsize=14)
            plt.ylabel('VACF (C_v(τ))', fontsize=14)
            plt.title(f'Velocity Autocorrelation Function (VACF) - {num_agents} agents, {sample_type}, {leader_type}', fontsize=16)
            plt.grid(True)
            plt.savefig(f"{save_location}{filename}.svg")
            plt.savefig(f"{save_location}{filename}.png")
            plt.savefig(f"{save_location}{filename}.pdf")