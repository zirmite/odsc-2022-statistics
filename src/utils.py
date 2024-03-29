"""blah."""
import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns

# Lesson 1 data functions


def wknd_add(x):
    if x.dayofweek in (5, 6):
        return 1
    else:
        return 0


def tweets_data(mean=50, size=1000, seed=None, wknd_offset=0.1):
    """Generate simple column of counts data as a Pandas Series."""
    if seed is not None:
        np.random.seed(seed)

    # generate counts data
    samples = np.random.poisson(mean, size=size)
    dates = pd.date_range('2016-01-01', periods=size)

    # more on the weekend
    wknd_samples = np.random.poisson(mean * wknd_offset, size=size) * dates.map(
        wknd_add
    )

    return pd.Series(data=samples + wknd_samples, index=dates)


def add_nan(data, frac=0.05, seed=None):
    """
    Given a column of data, return that column with (at most) the specified
    fraction of NaN values.
    """
    if seed is not None:
        np.random.seed(seed)

    data.ravel()[
        np.random.choice(
            data.size,
            np.int(data.size * frac * np.random.random()),
            replace=False,
        )
    ] = np.nan

    return pd.Series(data=data)


def add_noise(data, sn=5.0):
    data = data + np.random.normal(
        loc=0, scale=data.mean() / sn, size=data.size
    )
    return data


def plot_anscombe():
    # Load the example dataset for Anscombe's quartet
    df = sns.load_dataset("anscombe")

    # Show the results of a linear regression within each dataset
    sns.lmplot(
        x="x",
        y="y",
        col="dataset",
        hue="dataset",
        data=df,
        col_wrap=2,
        ci=None,
        palette="muted",
        height=4,
        scatter_kws={"s": 50, "alpha": 1},
    )


# Lesson 2 data functions


def continuous_data_one(mean=100, std=25.0, size=1000, seed=None):
    if seed is not None:
        np.random.seed(seed)

    samples = np.random.normal(loc=mean, scale=std, size=size)

    return pd.Series(data=samples)


def cumulative_d(data):
    data = pd.Series(data)
    sorted_data = data.sort_values()
    n = data.size
    y = np.arange(1, n + 1) / n

    return (sorted_data, y)


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
