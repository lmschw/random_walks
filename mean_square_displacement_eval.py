import matplotlib.pyplot as plt
import pickle
import numpy as np

import mean_square_displacement as msd

def plot_results(result, name, save_location, save_path):
    times = np.array([i for i in range(1, len(result)+1)])

    print(f"{name}: {result[-1]}")

    f, ax = plt.subplots()
    ax.plot(times, result, color="deepskyblue")
    ax.scatter(times, result, color="purple")
    ax.set_xlabel('Time interval')
    ax.set_ylabel(f'msd {name}')
    ax.grid(True)
    plt.savefig(f"{save_location}msd_{save_path}.svg")
    plt.savefig(f"{save_location}msd_{filename}.png")
    plt.savefig(f"{save_location}msd_{filename}.pdf")

    plt.figure()
    plt.plot(range(1, len(result) + 1), result, color='purple', label='MSD')
    plt.xlabel('Time interval')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('Mean Squared Displacement (MSD)')
    plt.title(f'Mean Squared Displacement (MSD) Analysis {name}')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{save_location}msd_log_{save_path}.svg")
    plt.savefig(f"{save_location}msd_log_{filename}.png")
    plt.savefig(f"{save_location}msd_log_{filename}.pdf")

    popt, pcov = msd.curve_fit(msd.power_law, range(1, len(result) + 1), result)
    alpha = popt[0]
    print(f"Scaling exponent (α) {name}: {alpha}")
    print(f"MSD at last step {name}: {result[-1]}")

sample_types = ["brown", "levy", "correlated"]
leader_types = ["free"]
num_agents_options = [1, 5, 7, 10]
steps_per_run = 25000
num_runs = 2
dimension = "2D"
implementation_types = ["old", "new"]

basepath = "J:/leader_emergence/results/2D/res/"
save_location = "msd_results/"

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
                    filepath = f"{basepath}{filename}.pickle"
                    with open(filepath, "rb") as input_file:
                        trajectory = np.array(pickle.load(input_file)["trajectory"])

                        result = msd.msd(trajectory,1)
                        results.append(result)
                        plot_results(result=result, name=f"{num_agents} agents, {sample_type}, {leader_type}, {implementation_type}, run={run}", save_location=save_location, save_path=filename)
                        print(f"msd last timestep: {filename}: {result[-1]}")

                filename = f"{impl_prefix}{sample_type}_{leader_type}_{num_agents**2}"
                results = np.array(results)
                results = np.average(results.T, axis=1)
                plot_results(result=results, name=f"{num_agents} agents, {sample_type}, {leader_type}, {implementation_type}", save_location=save_location, save_path=filename)
                print(f"average msd last timestep: {filename}: {results[-1]}")

                comp_results.append(np.average(results.T, axis=1))
            filename = f"comp_{sample_type}_{leader_type}_{num_agents**2}"
            plot_results(result=comp_results, name=f"{num_agents} agents, {sample_type}, {leader_type}", save_location=save_location, save_path=filename)
            print(f"average msd last timestep total: {filename}: old: {comp_results[0][-1]} - new: {comp_results[1][-1]}")
