import numpy as np

def get_hurst_exponent(time_series, max_lag=20):
    """Returns the Hurst Exponent of the time series"""
    
    lags = range(2, max_lag)

    # variances of the lagged differences
    tau = [np.std(np.subtract(time_series[lag:], time_series[:-lag])) for lag in lags]

    # calculate the slope of the log plot -> the Hurst Exponent
    reg = np.polyfit(np.log(lags), np.log(tau), 1)

    return reg[0]

trajectory = np.array([i for i in range(1, 2001)])

for lag in [20, 100, 300, 500, 1000]:
    hurst_exp = get_hurst_exponent(trajectory, lag)
    print(f"Hurst exponent with {lag} lags: {hurst_exp:.4f}")