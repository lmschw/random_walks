import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd

import mdea 

sample_types = ["brown", "levy", "correlated"]
leader_types = ["free"]
num_agents_options = [1, 5, 7, 10]
steps_per_run = 25000
num_runs = 30
dimension = "2D"
implementation_types = ["old", "new"]

basepath = "J:/leader_emergence/results/2D/res/"
save_location = "mdea_results/"

data = []
for num_agents in num_agents_options:
    for sample_type in sample_types:
        for leader_type in leader_types:
            comp_deltas = []
            comp_mu1s = []
            comp_mu2s = []
            for implementation_type in implementation_types:
                if implementation_type == "new":
                    impl_prefix = "rtm_"
                else:
                    impl_prefix = ""
                deltas = []
                mu1s = []
                mu2s = []
                
                for run in range(num_runs):
                    filename = f"{impl_prefix}{sample_type}_{leader_type}_{num_agents**2}_run{run + 1}"
                    print(filename)
                    filepath = f"{basepath}{filename}.pickle"
                    with open(filepath, "rb") as input_file:
                        trajectory = pickle.load(input_file)["trajectory"]

                        dea_engine = mdea.DeaEngine(trajectory)
                        dea_engine.analyze_with_stripes(fit_start=0.1, fit_stop=0.9, n_stripes=60)

                        deltas.append(dea_engine.delta)
                        mu1s.append(dea_engine.mu1)
                        mu2s.append(dea_engine.mu2)
                        data.append([sample_type, leader_type, num_agents, implementation_type, run, dea_engine.delta, dea_engine.mu1, dea_engine.mu2])

                        #print(f"{filename}: delta: {dea_engine.delta}, mu1: {dea_engine.mu1}, mu2: {dea_engine.mu2}")
                filename = f"{impl_prefix}{sample_type}_{leader_type}_{num_agents**2}"
                

                # print(f"{filename}: avg delta: {np.average(deltas)}, min delta: {np.min(deltas)}, max delta: {np.max(deltas)}, std: {np.std(deltas)}")
                # print(f"{filename}: avg mu1: {np.average(mu1s)}, min mu1: {np.min(mu1s)}, max mu1: {np.max(mu1s)}, std: {np.std(mu1s)}")
                # print(f"{filename}: avg mu2: {np.average(mu2s)}, min mu2: {np.min(mu2s)}, max mu2: {np.max(mu2s)}, std: {np.std(mu2s)}")

                comp_deltas.append(np.average(deltas))
                comp_mu1s.append(np.average(mu1s)) 
                comp_mu2s.append(np.average(mu2s))
            filename = f"comp_{sample_type}_{leader_type}_{num_agents**2}"
            print(f"{filename}: avg delta: {np.average(comp_deltas)}, min delta: {np.min(comp_deltas)}, max delta: {np.max(comp_deltas)}, std: {np.std(comp_deltas)}")
            print(f"{filename}: avg mu1: {np.average(comp_mu1s)}, min mu1: {np.min(comp_mu1s)}, max mu1: {np.max(comp_mu1s)}, std: {np.std(comp_mu1s)}")
            print(f"{filename}: avg mu2: {np.average(comp_mu2s)}, min mu2: {np.min(comp_mu2s)}, max mu2: {np.max(comp_mu2s)}, std: {np.std(comp_mu2s)}")


df = pd.DataFrame(data, columns=["sample_type", "leader_type", "num_agents", "implementation_type", "run", "delta", "mu1", "mu2"])
df.to_csv(f"{save_location}mdea_results.csv", index=False)

