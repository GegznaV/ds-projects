import pandas as pd
import seaborn as sns
from scipy.stats import skew

from typing import Union

# Generalizable --------------------------------------

# Hypothesis testing
def format_p(p):
    """Format p values at 3 decimal places.

    Args:
        p (float): p value (number between 0 and 1).
    """
    if p < 0.001:
        return "p < 0.001"
    elif p > 0.999:
        return "p > 0.999"
    else:
        return f"p = {p:.3f}"


# For data wrangling ---------------------------------------------
def as_df(
    data: pd.Series,
    index_name: Union[str, list[str], tuple[str, ...]],
    name: str,
) -> pd.DataFrame:
    """Convert Series to DataFrame with desired column names.

    Used to display data in more user-friendly way.

    Args:
        data (pandas.Series): Pandas Series object.
        index_name (Union[str, list[str], tuple[str, ...]]):
                    New name for Series index
                    (applied before conversion to DataFrame).
        name (str): Name for series values
                    (applied before conversion to DataFrame).

    Returns:
        pandas.DataFrame: DataFrame.
    """
    return data.rename_axis(index_name).rename(name).reset_index()


# Merge all tables
def merge_all(df_list, on="user_id", how="outer"):
    """Merge multiple data frames.

    Args:
        df_list (list of pandas.dataframes): Data frames to join.
        on (str, optional): Column names to join on.
             See details in pandas.merge(). Defaults to "user_id".
        how (str, optional): {'left', 'right', 'outer', 'inner', 'cross'}.
             Type of merge to be performed. See details in pandas.merge().
             Defaults to "outer".

    Returns:
        pandas.dataframe: merged data frame.

    Note:
       Function is based on https://stackoverflow.com/a/71886035/4783029
    """
    merged = df_list[0]
    for to_merge in df_list[1:]:
        merged = pd.merge(left=merged, right=to_merge, how=how, on=on)
    return merged


# Descriptive statistics ----------------------------------------------------
def calc_summaries(x, ndigits=1):
    """Calculate some common summary statistics.

    Args:
        x (pandas.Series): Numeric variable to summarize.
        ndigits (int, optional): Number of decimal digits to round to.
                Defaults to 1.
    Return:
       pandas.DataFrame with summary statistics.
    """
    res = x.agg(["count", "min", "mean", "median", "std", "mad", "max", "skew"])
    summary = pd.DataFrame(round(res, ndigits=ndigits)).T
    # Present count data as integer:
    summary = summary.assign(count=lambda d: d["count"].astype(int))

    return summary


def format_percent(x: float):
    """Round percentages to 1 decimal place and format as string

    Values between 0 and 0.1 are printed as <0.1%
    Values between 99.9 and 100 are printed as >100%

    Args:
        x (float): A sequence of percentage values ranging from 0 to 100.

    Returns:
        pd.Series[str]: Pandas series of formatted values.

    Author: Vilmantas Gėgžna
    """
    return pd.Series(
        [
            "0%"
            if i == 0
            else "<0.1%"
            if i < 0.1
            else ">99.9%"
            if 99.9 < i < 100
            else f"{i:.1f}%"
            for i in x
        ],
        index=x.index,
    )


def counts_to_percentages(x, name="percent"):
    """Express counts as percentages.

    The sum of count values is treated as 100%.

    Args:
        x (int, float): Counts data as pandas.Series.
        name (str, optional): The name for output pandas.Series with percentage
             values. Defaults to "percent".

    Returns:
        str: pandas.Series object with `x` values expressed as percentages
             and rounded to 1 decimal place, e.g., "0.2%".
             Values equal to 0 are formatted as "0%", values between
             0 and 0.1 are formatted as "<0.1%", values between 99.9 and 100
             are formatted as ">99.9%".

    Examples:
    >>> import pandas as pd
    >>> counts_to_percentages(pd.Series([1, 0, 1000, 2000, 1000, 5000, 1000]))
    >>> counts_to_percentages(pd.Series([1, 0, 10000]))

    Author: Vilmantas Gėgžna
    """
    return format_percent(x / x.sum() * 100).rename(name)
    # .replace("0.0%", "<0.1%") # FIXME: why is this row needed???


def calc_counts_and_percentages(
    group, data, sort=True, weight=None, n_label="n", perc_label="percent"
):
    """Create frequency table that contains counts and percentages.

    Args:
        group (str): Variable that defines the groups. Column name from `data`.
        data (pandas.DataFrame): data frame.
        sort (bool or "index", optional): Way to sort values:
             - True - sort by count descending.
             - "index" - sort by index ascending.
             Defaults to True.
        weight (str, optional): Frequency weights. Column name from `data`.
                Defaults to None: no weights are used.
        n_label (str, optional): Name for output column with counts.
        perc_label (str, optional): Name for output column with percentage.

    Return: pandas.DataFrame with 3 columns:
            - column with unique values of `x`,
            - column `n_label` (defaults to "n") with counts as int, and
            - column `perc_label` (defaults to "percent") with percentage
              values formatted as str.

    Author: Vilmantas Gėgžna
    """

    vsort = sort == True

    if weight is None:
        counts = data[group].value_counts(sort=vsort)
        if sort == "index":
            counts = counts.sort_index()
    else:
        counts = data.groupby(group)[weight].sum()

    percent = counts_to_percentages(counts)

    return (
        pd.concat([counts.rename(n_label), percent.rename(perc_label)], axis=1)
        .rename_axis(group)
        .reset_index()
    )


# Plot counts ---------------------------------------------------------------
def plot_counts_with_labels(
    counts,
    title="",
    x=None,
    y="n",
    x_lab=None,
    y_lab="Count",
    label="percent",
    label_rotation=0,
    title_fontsize=13,
    legend=False,
    ec="black",
    y_lim_max=None,
    ax=None,
    **kwargs,
):
    """Plot count data as barplots with labels.

    Args:
        counts (pandas.DataFrame): Data frame with counts data.
        title (str, optional): Figure title. Defaults to "".
        x (str, optional): Column name from `counts` to plot on x axis.
                Defaults to None: first column.
        y (str, optional): Column name from `counts` to plot on y axis.
                Defaults to "n".
        x_lab (str, optional): X axis label.
              Defaults to value of `x` with capitalized first letter.
        y_lab (str, optional): Y axis label. Defaults to "Count".
        label (str, optional): Column name from `counts` for value labels.
                Defaults to "percent".
        label_rotation (int, optional): Angle of label rotation. Defaults to 0.
        legend (bool, optional): Should legend be shown?. Defaults to False.
        ec (str, optional): Edge color. Defaults to "black".
        y_lim_max (float, optional): Upper limit for Y axis.
                Defaults to None: do not change.
        ax (matplotlib.axes.Axes, optional): Axes object. Defaults to None.
        **kwargs: further arguments to pandas.DataFrame.plot.bar()

    Returns:
        matplotlib.axes.Axes: Axes object of the generate plot.

    Author: Vilmantas Gėgžna
    """
    if x is None:
        x = counts.columns[0]

    if x_lab is None:
        x_lab = x.capitalize()

    if y_lim_max is None:
        y_lim_max = counts[y].max() * 1.15

    ax = counts.plot.bar(x=x, y=y, legend=legend, ax=ax, ec=ec, **kwargs)
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel(x_lab)
    ax.set_ylabel(y_lab)
    ax_add_value_labels_ab(ax, labels=counts[label], rotation=label_rotation)
    ax.set_ylim(0, y_lim_max)

    return ax


def plot_crosstab_as_heatmap(
    x: str,
    y: str,
    data: pd.DataFrame,
    title: str = None,
    vmax: int = None,
    cbar: bool = True,
):
    """Plot a Cross-Tabulation

    Args:
        x (str): Column name in 'data'.
        data (pandas.DataFrame): Table with data.
        title (str, optional): Title of the plot.
        vmax (int, optional): Maximum value for color scale.
                              Defaults to None (calculated the maximum frequency
                              in cross-tabulation).
        cbar (bool, optional): Should color bar be shown?. Defaults to True.

    Returns:
        Axes object.
    """

    # Create cross-tabulation
    crosstab = pd.crosstab(data[x], data[y])

    # Visualize
    sns.set(font_scale=1)

    if vmax is None:
        vmax = crosstab.max().max()

    ax = sns.heatmap(
        crosstab,
        linewidths=0.5,
        annot=True,
        fmt="1d",
        annot_kws={"size": 10},
        vmin=0,
        vmax=vmax,
        cmap="PiYG",
        cbar=cbar,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")

    if title is not None:
        ax.set(title=title)

    return ax


def crosstab(data, x: str, y: str, **kwargs):
    """Create a cross-tabulation

    Args:
        data (pandas.DataFrame): Data frame
        x (str): Column name in `data`
        y (str): Column name in `data`
        **kwargs: other arguments to crosstab.crosstab()

    Returns:
        pandas.DataFrame: Cross-tabulation.

    Author: Vilmantas Gėgžna
    """
    return pd.crosstab(data[x], data[y], **kwargs)


def plot_crosstab_as_barplot(
    data: pd.DataFrame,
    x: str = None,
    y: str = None,
    title: str = None,
    xlabel="",
    ylabel="Percentage",
    rot=0,
    **kwargs,
):
    """Plot a Cross-Tabulation as Barplot

    Args:
        data (pandas.DataFrame): Either cross-tabulation or data frame with
            categorical (or discrete) data.
        x (str, optional): Column name in 'data'.
        y (str, optional): Column name in 'data'.
        title (str, optional): Title of the plot.
        xlabel: see details in pandas.DataFrame.plot.bar()
                Defaults to "".
        ylabel: see details in pandas.DataFrame.plot.bar()
                Defaults to "Percentage".
        rot: see details in pandas.DataFrame.plot.bar()
              Defaults to 0
        **kwargs: other args to pandas.DataFrame.plot.bar()

    Returns:
        Axes object.

    Author: Vilmantas Gėgžna

    """

    if (x is None) & (y is None):
        cross_t = data
    else:
        # Create cross-tabulation
        cross_t = pd.crosstab(data[x], data[y])

    # Row percentage
    cross_p = round(cross_t.div(cross_t.sum(axis=1), axis=0) * 100, ndigits=1)

    # Visualize
    ax = cross_p.plot.bar(
        ec="black", title=title, rot=rot, xlabel=xlabel, ylabel=ylabel, **kwargs
    )
    return ax


def ax_add_value_labels_lr(ax, labels=None, spacing=25, size=7, weight="bold"):
    """Add value labels left/right to each bar in a bar chart.

    Arguments:
        ax (matplotlib.axes.Axes): Plot (axes) to annotate.
        label (str or similar): Values to be used as labels.
        spacing (int): Number of points between bar and label.
        size (int): font size.
        weight (str): font weight.

    Source:
        This function is based on https://stackoverflow.com/a/48372659/4783029
    """

    # For each bar: Place a label
    for rect, label in zip(ax.patches, labels):
        # Get X and Y placement of label from rect.
        y_value = rect.get_y() + rect.get_height() / 2
        x_value = rect.get_width()

        space = spacing

        # Horizontal alignment for positive values
        ha = "right"

        # If the value of a bar is negative: Place left to the bar
        if x_value < 0:
            # Invert space to place label on the left
            space *= -1
            # Horizontal alignment
            ha = "left"

        # Use X value as label and format number with one decimal place
        if labels is None:
            label = "{:.1f}".format(x_value)

        # Create annotation
        ax.annotate(
            label,
            (x_value, y_value),
            xytext=(space, 0),
            textcoords="offset points",
            ha=ha,
            va="center",
            fontsize=size,
            fontweight=weight,
        )


def ax_add_value_labels_ab(ax, labels=None, spacing=2, size=9, weight="bold", **kwargs):
    """Add value labels above/below each bar in a bar chart.

    Arguments:
        ax (matplotlib.Axes): Plot (axes) to annotate.
        label (str or similar): Values to be used as labels.
        spacing (int): Number of points between bar and label.
        size (int): font size.
        weight (str): font weight.
        **kwargs: further arguments to axis.annotate.

    Source:
        This function is based on https://stackoverflow.com/a/48372659/4783029
    """

    # For each bar: Place a label
    for rect, label in zip(ax.patches, labels):
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        space = spacing

        # Vertical alignment for positive values
        va = "bottom"

        # If the value of a bar is negative: Place label below the bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertical alignment
            va = "top"

        # Use Y value as label and format number with one decimal place
        if labels is None:
            label = "{:.1f}".format(y_value)

        # Create annotation
        ax.annotate(
            label,
            (x_value, y_value),
            xytext=(0, space),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=size,
            fontweight=weight,
            **kwargs,
        )
