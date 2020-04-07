import pandas as pd
import numpy as np


def counts_data(mean=100, size=1000, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # generate counts data
    samples = np.random.poisson(mean, size=size)

    return pd.Series(data=samples)


def counts_data_nan(mean=100, size=1000, seed=None):
    samples = counts_data(mean, size, seed)
    samples = samples.astype(np.float64)
    samples.ravel()[np.random.choice(samples.size, 100, replace=False)] = np.nan

    return pd.Series(data=samples)
