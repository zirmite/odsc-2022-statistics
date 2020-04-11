import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns

## Lesson 1 data functions


def counts_data(mean=100, size=1000, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # generate counts data
    samples = np.random.poisson(mean, size=size)

    return pd.Series(data=samples)


def add_nan(data, frac=0.05, seed=None):
    if seed is not None:
        np.random.seed(seed)

    data.ravel()[
        np.random.choice(
            data.size, np.int(data.size * frac * np.random.random()), replace=False
        )
    ] = np.nan

    return pd.Series(data=data)


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


## Lesson 2 data functions


def continuous_data(mean=100, std=25.0, size=1000, seed=None):
    if seed is not None:
        np.random.seed(seed)

    samples = np.random.normal(loc=mean, scale=std, size=size)

    return pd.Series(data=samples)
