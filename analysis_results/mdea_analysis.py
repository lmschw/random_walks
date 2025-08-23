import utils.data_loader as dl


def analyse_mdea(path, headers=["sample_type", "leader_type", "num_agents", "implementation_type", "run", "delta", "mu1", "mu2"]):
    df = dl.load_csv_data(path, headers)

    # Group by sample_type, leader_type, num_agents, implementation_type and calculate average and std
    summary = df.groupby(["sample_type", "leader_type", "num_agents", "implementation_type"]).agg(
        avg_delta=("delta", "mean"),
        std_delta=("delta", "std"),
        avg_mu1=("mu1", "mean"),
        std_mu1=("mu1", "std"),
        avg_mu2=("mu2", "mean"),
        std_mu2=("mu2", "std")
    ).reset_index()

    print(summary)

    return summary
# average per type


