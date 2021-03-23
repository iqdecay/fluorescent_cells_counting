import os
import sys
from time import time, strftime, gmtime
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch


def progressbar(to_progress: Iterable, n_steps=100, length=60):
    """Display a progress bar when iterating `to_progress`."""

    def show(k: int, cumulative_time: int):
        """Display the k-th state of a progress bar."""
        x = int(length * k / n_steps)

        current_step_time = strftime("%H:%M:%S", gmtime(cumulative_time))
        approx_total_time = strftime(
            "%H:%M:%S", gmtime((n_steps * cumulative_time) / (k + 1))
        )
        sys.stdout.write(
            f"[{'=' * x}{'>' * int(x != length)}{'.' * (length - x - 1)}]"
            + f"{k}/{n_steps} ETA: {current_step_time}/{approx_total_time}\r",
        )
        sys.stdout.flush()

    cumulative_time = 0
    show(0, cumulative_time)
    for k, item in enumerate(to_progress):
        t0 = time()
        yield item
        cumulative_time += time() - t0
        show(k + 1, cumulative_time)
    sys.stdout.write("\n")
    sys.stdout.flush()


def plot_training_curves(metric: str, history: pd.DataFrame, path: str):
    """Plot the evolution of a train and validation metric over the epochs.

    :param metric: name of the metric to plot.
    :param history: history of the training (accuracy and loss).
    :param path: if provided save figure at this path.
    """
    sns.lineplot(
        x="epochs",
        y=metric,
        data=history,
        label="train",
        color="#e41a1c",
    )
    sns.lineplot(
        x="epochs",
        y="val_" + metric,
        data=history,
        label="validation",
        color="#377eb8",
    )
    plt.title(f"{metric} over training epochs")

    if path:
        directory_name = os.path.dirname(path)
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)
        plt.savefig(path)

    plt.show()


def get_intersection_over_union(
    outputs: torch.Tensor, labels: torch.Tensor, mean=True
) -> float:
    """Compute accuracy from output probabiliies."""
    outputs = (outputs > 0.5).int()
    labels = labels.int()

    intersection = (outputs & labels).float().sum((1, 2, 3))
    union = (outputs | labels).float().sum((1, 2, 3))

    EPS = 1e-6
    iou = (intersection + EPS) / (union + EPS)

    if mean:
        iou = iou.mean()

    return iou
