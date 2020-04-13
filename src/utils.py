"""blah."""
import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns

# Lesson 1 data functions


def tweets_data(mean=50, size=1000, seed=None):
    """Generate simple column of counts data as a Pandas Series."""
    if seed is not None:
        np.random.seed(seed)

    # generate counts data
    samples = np.random.poisson(mean, size=size)

    return pd.Series(data=samples)


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
    data = data + np.random.normal(loc=0, scale=data.mean()/sn, size=data.size)
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
