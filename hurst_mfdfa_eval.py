import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd

import hurst_exponent_mfdfa as mdfdfa 

sample_types = ["brown", "levy", "correlated"]
leader_types = ["free"]
num_agents_options = [1, 5, 7, 10]
steps_per_run = 25000
num_runs = 30
dimension = "2D"
implementation_types = ["old", "new"]

basepath = "J:/leader_emergence/results/2D/res/"
save_location = "mdfdfa_results/"

data = []
for num_agents in num_agents_options:
    for sample_type in sample_types:
        for leader_type in leader_types:
            comp_hx = []
            comp_hy = []
            comp_h = []
            for implementation_type in implementation_types:
                if implementation_type == "new":
                    impl_prefix = "rtm_"
                else:
                    impl_prefix = ""
                hxs = []
                hys = []
                hs = []
                
                for run in range(num_runs):
                    filename = f"{impl_prefix}{sample_type}_{leader_type}_{num_agents**2}_run{run + 1}"
                    print(filename)
                    filepath = f"{basepath}{filename}.pickle"
                    with open(filepath, "rb") as input_file:
                        trajectory = pickle.load(input_file)["trajectory"]

                        hx, hy, h = mdfdfa.run_for_trajectory(trajectory)

                        hxs.append(hx)
                        hys.append(hy)
                        hs.append(h)
                        data.append([sample_type, leader_type, num_agents, implementation_type, run, hx, hy, h])

                filename = f"{impl_prefix}{sample_type}_{leader_type}_{num_agents**2}"

                comp_hx.append(np.average(hxs))
                comp_hy.append(np.average(hys)) 
                comp_h.append(np.average(hs))
            filename = f"comp_{sample_type}_{leader_type}_{num_agents**2}"
            print(f"{filename}: avg hx: {np.average(comp_hx)}, min hx: {np.min(comp_hx)}, max hx: {np.max(comp_hx)}, std: {np.std(comp_hx)}")
            print(f"{filename}: avg hy: {np.average(comp_hy)}, min hy: {np.min(comp_hy)}, max hy: {np.max(comp_hy)}, std: {np.std(comp_hy)}")
            print(f"{filename}: avg h: {np.average(comp_h)}, min h: {np.min(comp_h)}, max h: {np.max(comp_h)}, std: {np.std(comp_h)}")


df = pd.DataFrame(data, columns=["sample_type", "leader_type", "num_agents", "implementation_type", "run", "hx", "hy", "h"])
df.to_csv(f"{save_location}mfdfa_results.csv", index=False)

