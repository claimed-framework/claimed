# Copyright contributors to the geobench project
# modified from geobench (https://github.com/ServiceNow/geo-bench/blob/main/geobench/plot_tools.py)


import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
import json
from scipy.stats import trim_mean

sns.set_style("dark", {"grid.color": "0.98", "axes.facecolor": "(0.95, 0.95, 0.97)"})
GEO_BENCH_DIR = "geobench"


def biqm(scores):
    """Return a bootstram sample of iqm."""
    b_scores = np.random.choice(scores, size=len(scores), replace=True)
    return trim_mean(b_scores, proportiontocut=0.25, axis=None)


def iqm(scores):
    """Interquantile mean."""
    return trim_mean(scores, proportiontocut=0.25, axis=None)


def bootstrap_iqm(
    df,
    group_keys=("model", "dataset", "partition_name"),
    metric="test_metric",
    repeat=100,
):
    """Boostram of seeds for all model and all datasets to comput iqm score distribution."""
    df_list = []
    for i in range(repeat):
        series = df.groupby(list(group_keys))[metric].apply(biqm)
        df_list.append(series.to_frame().reset_index())

    return pd.concat(df_list)


def bootstrap_iqm_aggregate(df, metric="test_metric", repeat=100):
    """Stratified bootstrap (by dataset) of all seeds to compute iqm score distribution for each model."""

    group = df.groupby(["model", "dataset", "partition_name"])

    df_list = []
    for i in range(repeat):
        new_df = group.sample(frac=1, replace=True)
        series = new_df.groupby(["model", "partition_name"])[metric].apply(iqm)
        df_list.append(series.to_frame().reset_index())

    new_df = pd.concat(df_list)
    new_df.loc[:, "dataset"] = "aggregated"
    return new_df


def average_seeds(
    df, group_keys=("model", "dataset", "partition_name"), metric="test metric"
):
    """Average seeds for all model and all datasets."""
    df_avg = df.groupby(list(group_keys))[metric].mean()
    df_avg = df_avg.unstack(level="dataset")

    df_avg = df_avg.round(3)
    return df_avg


def extract_1x_data(df_all):
    """Extract only resutls trained on 100% of the data"""
    return df_all[
        (df_all["partition_name"] == "1.00x train")
        | (df_all["partition_name"] == "default")
    ].copy()


def normalize_bootstrap_and_plot(
    df,
    metric,
    benchmark_name,
    model_order,
    model_colors=None,
    repeat=100,
    fig_size=None,
    n_legend_rows=2,
):
    """Add aggregated data as a new dataset."""

    # normalize all the scores based on the benchmark name.
    # the normalizing data is expected to be found in the benchmark directory under normalizer.json
    if benchmark_name:
        normalizer = load_normalizer(benchmark_name=benchmark_name)
        new_metric = normalizer.normalize_data_frame(df, metric)
    else:
        new_metric = metric

    # create a new df containing bootstrapped samples of iqm
    bootstrapped_iqm = pd.concat(
        (
            bootstrap_iqm_aggregate(
                df, metric=new_metric, repeat=repeat
            ),  # stratified bootstrap across all datasets
            bootstrap_iqm(
                df, metric=new_metric, repeat=repeat
            ),  # bootstrapped iqm for each dataset
        )
    )

    # plot results per dataset (aggregated results is an extra dataset)
    plot_per_dataset(
        bootstrapped_iqm,
        model_order,
        model_colors=model_colors,
        metric=new_metric,
        fig_size=fig_size,
        n_legend_rows=n_legend_rows,
    )


class Normalizer:
    """Class used to normalize results beween min and max for each dataset."""

    def __init__(self, range_dict):
        """Initialize a new instance of Normalizer class."""
        self.range_dict = range_dict

    def __call__(self, ds_name, values, scale_only=False):
        """Call the Normalizer class."""
        mn, mx = self.range_dict[ds_name]
        range = mx - mn
        if scale_only:
            return values / range
        else:
            return (values - mn) / range

    def from_row(self, row, scale_only=False):
        """Normalize from row."""
        return [
            self(ds_name, val, scale_only=scale_only) for ds_name, val in row.items()
        ]

    def normalize_data_frame(self, df, metric):
        """Normalize the entire dataframe."""
        new_metric = f"normalized {metric}"
        df[new_metric] = df.apply(
            lambda row: self.__call__(row["dataset"], row[metric]), axis=1
        )
        return new_metric

    def save(self, benchmark_name):
        """Save normalizer to json file."""

        with open(f"{benchmark_name}/normalizer.json", "w") as f:
            json.dump(self.range_dict, f, indent=2)


def load_normalizer(benchmark_name):
    """Load normalizer from json file."""
    with open(f"{benchmark_name}/normalizer.json", "r") as f:
        range_dict = json.load(f)
    return Normalizer(range_dict)


def make_normalizer(data_frame, metrics=("test metric",), benchmark_name=None):
    """Extract min and max from data_frame to build Normalizer object for all datasets."""
    datasets = data_frame["dataset"].unique()
    range_dict = {}

    for dataset in datasets:
        sub_df = data_frame[data_frame["dataset"] == dataset]
        data = []
        for metric in metrics:
            data.append(sub_df[metric].to_numpy())
        range_dict[dataset] = (np.min(data), np.max(data))

    normalizer = Normalizer(range_dict)

    if benchmark_name:
        normalizer.save(benchmark_name)

    return normalizer


def remove_violin_outline(ax):
    """Remove the outline of the violin plot."""
    for pc in ax.collections:
        pc.set_edgecolor("none")


def plot_per_dataset(
    df,
    model_order,
    metric="test metric",
    aggregated_name="aggregated",
    sharey=True,
    inner="box",
    fig_size=None,
    n_legend_rows=1,
    model_colors=None,
):
    """Violin plots for each datasets and each models.

    If a dataset is named `aggregated_name` it will be the first and will be highlighted in light blue.

    """
    datasets = sorted(df["dataset"].unique())

    if fig_size is None:
        fig_width = len(datasets) * 2
        fig_size = (fig_width, 3)
    fig, axes = plt.subplots(1, len(datasets), sharey=sharey, figsize=fig_size)

    if model_colors is None:
        colors = sns.color_palette("colorblind", n_colors=len(model_order))
        model_colors = dict(zip(model_order, colors))

    for dataset, ax in zip(datasets, axes):
        sub_df = df[df["dataset"] == dataset]
        sns.violinplot(
            x="dataset",
            y=metric,
            hue="model",
            data=sub_df,
            hue_order=model_order,
            linewidth=0.5,
            saturation=1,
            scale="count",
            inner=inner,
            palette=model_colors,
            ax=ax,
        )
        remove_violin_outline(ax)
        ax.tick_params(axis="y", labelsize=8)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.grid(axis="y")

        if dataset == aggregated_name:
            ax.set_facecolor("#cff6fc")

        ax.set(xlabel=None)

        if dataset != datasets[int((len(datasets) - 1) / 2)]:
            ax.get_legend().remove()
        else:
            ncols = int(np.ceil(len(model_order) / n_legend_rows))
            sns.move_legend(
                ax, loc="lower center", bbox_to_anchor=(0.5, 1), ncol=ncols, title=""
            )

        if dataset != datasets[0]:
            ax.set(ylabel=None)

    if sharey:
        fig.subplots_adjust(wspace=0.02)
    else:
        fig.subplots_adjust(wspace=0.3)
