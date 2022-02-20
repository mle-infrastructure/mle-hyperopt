import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import List, Union
import seaborn as sns
from mle_logging import load_config


def load_search_log(log_fname: str) -> pd.core.frame.DataFrame:
    """Reload the stored log yaml file.

    Args:
        log_fname (str): Filename to load

    Returns:
        pd.core.frame.DataFrame: Reloaded log as pandas dataframe.
    """
    log_dict = load_config(log_fname)

    log_list = []
    for k in log_dict.keys():
        log_list.append(log_dict[k])
    # Load in json format for nested dictionaries
    df = pd.json_normalize(log_list)
    # Rename columns and get rid of 'params.'
    new_cols = [df.columns[i].split(".")[-1] for i in range(len(df.columns))]
    df.columns = new_cols
    return df


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
    target_to_plot: str = "objective",
    plot_title: str = "Temp Title",
    plot_subtitle: Union[None, str] = None,
    xy_labels: Union[None, List[str]] = ["x-label", "y-label"],
    variable_name: Union[None, str] = "Performance",
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
    """Fix certain params & visualize grid target value over two selected ones.

    Args:
        hyper_df (pd.core.frame.DataFrame):
            Dataframe with variable values and target values.
        fixed_params (Union[None, dict], optional):
            Dictionary of key, value pairs to fix/slice by. Defaults to None.
        params_to_plot (list, optional):
            List of two variables to plot on x/y axis of heatmap. Defaults to [].
        target_to_plot (str, optional):
            Target variable name to plot. Defaults to "target".
        plot_title (str, optional):
            Title of the plot. Defaults to "Temp Title".
        plot_subtitle (Union[None, str], optional):
            Subtitle of the plot. Defaults to None.
        xy_labels (Union[None, List[str]], optional):
            List of x/y labels. Defaults to ["x-label", "y-label"].
        variable_name (Union[None, str], optional):
            Variable name shown in heatmap colorbar. Defaults to "Var Label".
        every_nth_tick (int, optional):
            Spacing between x/y ticks. Defaults to 1.
        plot_colorbar (bool, optional):
            Option to plot colorbar. Defaults to True.
        text_in_cell (bool, optional):
            Option to plot text in heat cells. Defaults to False.
        max_heat (Union[None, float], optional):
            Heat clipping max value. Defaults to None.
        min_heat (Union[None, float], optional):
            Heat clipping min value. Defaults to None.
        norm_cols (bool, optional):
            Option to normalize columns to max 1. Defaults to False.
        norm_rows (bool, optional):
            Option to normalize rows to max 1. Defaults to False.
        return_array (bool, optional):
            Option to return extracted heat array. Defaults to False.
        round_ticks (int, optional):
            Decimals to round ticks to. Defaults to 1.
        fig (_type_, optional):
            Figure object to manipulate. Defaults to None.
        ax (_type_, optional):
            Axis object to manipulate. Defaults to None.
        figsize (tuple, optional):
            Size of figure. Defaults to (10, 8).
        cmap (str, optional):
            Choice of colormap. Defaults to "magma".
        fname (Union[None, str], optional):
            Optional filename to store figure in. Defaults to None.

    Returns:
        _type_: Heat arrays or figure and axis matplotlib objects.
    """
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
) -> np.ndarray:
    """Construct the 2D array to plot in heatmap.

    Args:
        range_x (np.ndarray): Discrete range on x-axis.
        range_y (np.ndarray): Discrete range on y-axis.
        results_df (np.ndarray): Flat array with results [x, y, target].
        norm_cols (bool, optional):
            Option to normalize columns to max 1. Defaults to False.
        norm_rows (bool, optional):
            Option to normalize rows to max 1. Defaults to False.

    Returns:
        np.ndarray: 2D array of shape [|X|, |Y|] containing target values.
    """
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
    """Plot the 2D heatmap.

    Args:
        range_x (np.ndarray): Discrete range on x-axis.
        range_y (np.ndarray): Discrete range on y-axis.
        heat_array (np.ndarray): 2D array of shape [|X|,|Y|] containing targets.
        title (str, optional):
            Title of the plot. Defaults to "Temp Title".
        subtitle (Union[None, str], optional):
            Subtitle of the plot. Defaults to None.
        xy_labels (Union[None, List[str]], optional):
            List of x/y labels. Defaults to ["x-label", "y-label"].
        variable_name (Union[None, str], optional):
            Variable name shown in heatmap colorbar. Defaults to "Var Label".
        every_nth_tick (int, optional):
            Spacing between x/y ticks. Defaults to 1.
        plot_colorbar (bool, optional):
            Option to plot colorbar. Defaults to True.
        text_in_cell (bool, optional):
            Option to plot text in heat cells. Defaults to False.
        max_heat (Union[None, float], optional):
            Heat clipping max value. Defaults to None.
        min_heat (Union[None, float], optional):
            Heat clipping min value. Defaults to None.
        round_ticks (int, optional):
            Decimals to round ticks to. Defaults to 1.
        fig (_type_, optional):
            Figure object to manipulate. Defaults to None.
        ax (_type_, optional):
            Axis object to manipulate. Defaults to None.
        figsize (tuple, optional):
            Size of figure. Defaults to (10, 8).
        cmap (str, optional):
            Choice of colormap. Defaults to "magma".

    Returns:
        _type_: Figure and axis matplotlib objects.
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    if max_heat is None and min_heat is None:
        im = ax.imshow(
            heat_array,
            cmap=cmap,
            vmax=np.max(heat_array),
            vmin=np.min(heat_array),
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
                    str(round(float(label), round_ticks))
                    for label in range_y[::-1]
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
    plt.setp(
        ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
    )

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
