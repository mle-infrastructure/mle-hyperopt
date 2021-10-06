import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import List, Union

import seaborn as sns

# Set overall plots appearance sns style
sns.set(
    context="poster",
    style="white",
    palette="Paired",
    font="sans-serif",
    font_scale=1.05,
    color_codes=True,
    rc=None,
)


def visualize_2D_grid(
    hyper_df: pd.core.frame.DataFrame,
    fixed_params: Union[None, dict] = None,
    params_to_plot: list = [],
    target_to_plot: str = "target",
    plot_title: str = "Temp Title",
    plot_subtitle: Union[None, str] = None,
    xy_labels: Union[None, List[str]] = ["x-label", "y-label"],
    variable_name: Union[None, str] = "Var Label",
    every_nth_tick: int = 1,
    plot_colorbar: bool = True,
    text_in_cell: bool = False,
    max_heat: Union[None, float] = None,
    min_heat: Union[None, float] = None,
    norm_cols: bool = False,
    norm_rows: bool = False,
    return_array: bool = False,
    round_ticks: int = 1,
    fig=None,
    ax=None,
    figsize: tuple = (10, 8),
    cmap="magma",
    fname: Union[None, str] = None,
):
    """Fix certain params & visualize grid target value over other two."""
    assert len(params_to_plot) == 2, "You can only plot 2 variables!"

    # Select the data to plot - max. fix 2 other vars
    p_to_plot = params_to_plot + [target_to_plot]
    try:
        sub_log = hyper_df.hyper_log.copy()
    except Exception:
        sub_log = hyper_df.copy()
    if fixed_params is not None:
        for k, v in fixed_params.items():
            if type(v) == float or type(v) == int:
                sub_log = sub_log[sub_log[k].astype(float) == v]
            elif type(v) == str:
                sub_log = sub_log[sub_log[k].astype(str) == v]

    # Subselect the desired params from the pd df
    temp_df = sub_log[p_to_plot]

    # Construct the 2D array using helper function
    range_x = np.unique(temp_df[p_to_plot[0]])
    range_y = np.unique(temp_df[p_to_plot[1]])
    heat_array = get_heatmap_array(
        range_x, range_y, temp_df.to_numpy(), norm_cols, norm_rows
    )

    if return_array:
        return heat_array, range_x, range_y
    else:
        # Construct the plot
        fig, ax = plot_2D_heatmap(
            range_x,
            range_y,
            heat_array,
            plot_title,
            plot_subtitle,
            xy_labels,
            variable_name,
            every_nth_tick,
            plot_colorbar,
            text_in_cell,
            max_heat,
            min_heat,
            round_ticks,
            figsize=figsize,
            fig=fig,
            ax=ax,
            cmap=cmap,
        )

        # Save the figure if a filename was provided
        if fname is not None:
            fig.savefig(fname, dpi=300)
        else:
            return fig, ax


def get_heatmap_array(
    range_x: np.ndarray,
    range_y: np.ndarray,
    results_df: np.ndarray,
    norm_cols: bool = False,
    norm_rows: bool = False,
):
    """Construct the 2D array to plot the heat."""
    bring_the_heat = np.zeros((len(range_y), len(range_x)))
    for i, val_x in enumerate(range_x):
        for j, val_y in enumerate(range_y):
            case_at_hand = np.where(
                (results_df[:, 0] == val_x) & (results_df[:, 1] == val_y)
            )
            results_temp = results_df[case_at_hand, 2]
            # Reverse index so that small in bottom left corner
            bring_the_heat[len(range_y) - 1 - j, i] = results_temp

    # Normalize the rows and/or columns by the maximum
    if norm_cols:
        bring_the_heat /= bring_the_heat.max(axis=0)
    if norm_rows:
        bring_the_heat /= bring_the_heat.max(axis=1)[:, np.newaxis]
    return bring_the_heat


def plot_2D_heatmap(
    range_x: np.ndarray,
    range_y: np.ndarray,
    heat_array: np.ndarray,
    title: str = "Placeholder Title",
    subtitle: Union[None, str] = None,
    xy_labels: list = ["x-label", "y-label"],
    variable_name: Union[None, str] = None,
    every_nth_tick: int = 1,
    plot_colorbar: bool = True,
    text_in_cell: bool = False,
    max_heat: Union[None, float] = None,
    min_heat: Union[None, float] = None,
    round_ticks: int = 1,
    fig=None,
    ax=None,
    figsize: tuple = (10, 8),
    cmap="magma",
):
    """Plot the 2D heatmap."""
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    if max_heat is None and min_heat is None:
        im = ax.imshow(
            heat_array, cmap=cmap, vmax=np.max(heat_array), vmin=np.min(heat_array)
        )
    elif max_heat is not None and min_heat is None:
        im = ax.imshow(heat_array, cmap=cmap, vmax=max_heat)
    elif max_heat is None and min_heat is not None:
        im = ax.imshow(heat_array, cmap=cmap, vmin=min_heat)
    else:
        im = ax.imshow(heat_array, cmap=cmap, vmin=min_heat, vmax=max_heat)

    ax.set_yticks(np.arange(len(range_y)))
    if len(range_y) != 0:
        if type(range_y[-1]) is not str:
            if round_ticks != 0:
                yticklabels = [
                    str(round(float(label), round_ticks)) for label in range_y[::-1]
                ]
            else:
                yticklabels = [str(int(label)) for label in range_y[::-1]]
        else:
            yticklabels = [str(label) for label in range_y[::-1]]
    else:
        yticklabels = []
    ax.set_yticklabels(yticklabels)

    for n, label in enumerate(ax.yaxis.get_ticklabels()):
        if n % every_nth_tick != 0:
            label.set_visible(False)

    ax.set_xticks(np.arange(len(range_x)))
    if len(range_x) != 0:
        if type(range_x[-1]) is not str:
            if round_ticks != 0:
                xticklabels = [
                    str(round(float(label), round_ticks)) for label in range_x
                ]
            else:
                xticklabels = [str(int(label)) for label in range_x]
        else:
            xticklabels = [str(label) for label in range_x]
    else:
        xticklabels = []
    ax.set_xticklabels(xticklabels)

    for n, label in enumerate(ax.xaxis.get_ticklabels()):
        if n % every_nth_tick != 0:
            label.set_visible(False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    if subtitle is None:
        ax.set_title(title)
    else:
        ax.set_title(title + "\n" + str(subtitle))
    if len(range_x) != 0:
        ax.set_xlabel(xy_labels[0])
    if len(range_y) != 0:
        ax.set_ylabel(xy_labels[1])

    if plot_colorbar:
        # fig.subplots_adjust(right=0.8)
        # cbar_ax = fig.add_axes([0.85, 0.25, 0.05, 0.5])
        # cbar = fig.colorbar(im, cax=cbar_ax)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="7%", pad=0.15)
        cbar = fig.colorbar(im, cax=cax)
        if variable_name is not None:
            cbar.set_label(variable_name, rotation=270, labelpad=30)
        fig.tight_layout()

    if text_in_cell:
        for y in range(heat_array.shape[0]):
            for x in range(heat_array.shape[1]):
                ax.text(
                    x,
                    y,
                    "%.2f" % heat_array[y, x],
                    horizontalalignment="center",
                    verticalalignment="center",
                )
    return fig, ax
