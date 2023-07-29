# OS module
import os

# Enable ability to run R code in Python
os.environ["R_HOME"] = "C:/PROGRA~1/R/R-4.3.1"
import rpy2.robjects as r_obj
from rpy2.robjects.conversion import localconverter

# Other Python libraries and modules
from typing import Union
from IPython.display import display, HTML
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.stats.api as sms
import scipy.stats as sps
from scipy.stats import median_abs_deviation
from matplotlib.ticker import MaxNLocator


# Exploratory analysis
def count_unique(data: pd.DataFrame):
    """Get number and percentage of unique values

    Args:
        data (pd.DataFrame): Data frame to analyze.

    Return: data frame with columns `n_unique` (int) and `percent_unique` (str)
    """
    n_unique = data.nunique()
    return pd.concat(
        [
            n_unique.rename("n_unique"),
            format_percent((n_unique / data.shape[0]).multiply(100)).rename(
                "percent_unique"
            ),
        ],
        axis=1,
    )


# Display in ipynb
def display_collapsible(x, summary: str = "", sep=" "):
    """Display data frame or other object surrounded by `<details>` tags

    (I.e., display in collapsible way)

    Args:
        x (pd.DataDrame, str, list): Object to display
        summary (str, optional): Collapsed section name. Defaults to "".
        sep (str, optional): Symbol used to join strings (when x is a list).
             Defaults to " ".
    """
    if hasattr(x, "to_html") and callable(x.to_html):
        html_str = x.to_html()
    elif type(x) == str:
        html_str = x
    else:
        html_str = sep.join([str(i) for i in x])

    display(
        HTML(f"<details><summary>{summary}</summary>" + html_str + "</details>")
    )


# Inferential statistics -----------------------------------------------------
def ci_proportion_multinomial(
    counts,
    method: str = "goodman",
    n_label: str = "n",
    percent_label: str = "percent",
) -> pd.DataFrame:
    """Calculate  simultaneous confidence intervals for multinomial proportion.

    More information in documentation of statsmodels'
    multinomial_proportions_confint.

    Args:
        x (int): ps.Series, list or tuple with count data.
        method (str, optional): Method. Defaults to "goodman".
       n_label (str, optional): Name for column for counts.
       percent_label (str, optional): Name for column for percentage values.

    Returns:
        pd.DataFrame: _description_

    Examples:
    >>> ci_proportion_multinomial([62, 33, 55])
    """
    assert type(counts) in [pd.Series, list, tuple]
    if type(counts) is not pd.Series:
        counts = pd.Series(counts)

    return pd.concat(
        [
            (counts).rename(n_label),
            (counts / sum(counts)).rename(percent_label) * 100,
            pd.DataFrame(
                sms.multinomial_proportions_confint(counts, method=method),
                index=counts.index,
                columns=["ci_lower", "ci_upper"],
            )
            * 100,
        ],
        axis=1,
    )


def test_chi_square_gof(
    f_obs: list[int], f_exp: Union[str, list[float]] = "all equal"
) -> str:
    """Chi squared (χ²) goodness-of-fit (gof) test

    Args:
        f_obs (list[int]): Observed frequencies
        f_exp str, list[int]: List of expected frequencies or "all equal" if
              all frequencies are equal to the mean of observed frequencies.
              Defaults to "all equal".

    Returns:
        str: test p value.
    """
    k = len(f_obs)
    n = sum(f_obs)
    exp = n / k
    dof = k - 1
    if f_exp == "all equal":
        f_exp = [exp for _ in range(k)]
    stat, p = sps.chisquare(f_obs=f_obs, f_exp=f_exp)
    # May also be formatted this way:
    return (
        f"Chi square test, χ²({dof}, n = {n}) = {round(stat, 2)}, {format_p(p)}"
    )


def pairwise_chisq_gof_test(x: pd.Series):
    """Post-hoc Pairwise chi-squared Test

    Interface to R function `rstatix::pairwise_chisq_gof_test()`.

    Args:
        x (pandas.Series): data with group counts

    Returns:
        pandas.DataFrame: DataFrame with CLD results.
    """
    # Loading R package
    rstatix = r_obj.packages.importr("rstatix")
    dplyr = r_obj.packages.importr("dplyr")

    # Converting Pandas obj to R obj
    with localconverter(r_obj.default_converter + r_obj.pandas2ri.converter):
        x_in_r = r_obj.conversion.py2rpy(x)

    # Invoking the R function and getting the result
    df_result_r = rstatix.pairwise_chisq_gof_test(x_in_r)
    df_result_r = dplyr.relocate(df_result_r, "group1", "group2")

    # Converting the result to a Pandas dataframe
    return r_obj.pandas2ri.rpy2py(df_result_r)


def convert_pairwise_p_to_cld(
    data,
    group1: str = "group1",
    group2: str = "group2",
    p_name: str = "p.adj",
    output_gr_var: str = "group",
):
    """Convert p values from pairwise comparisons to CLD

    CLD - compact letter display: shared letter shows that difference
    is not significant. Interface to R function `convert_pairwise_p_to_cld()`.

    Args:
        data (pandas.DataFrame): Data frame with at least 3 columns:
              the first 2 columns contain names of both groups, one more
              column should contain p values.
        group1 (str, optional): Name of the  first column with group names.
               Defaults to "group1".
        group2 (str, optional): Name of the  first column with group names.
               Defaults to "group2".
        p_name (str, optional): Name of column with p values.
               Defaults to "p.adj".
        output_gr_var (str, optional): Name of column in output dataset
               with group names. Defaults to "group".

    Returns:
        pandas.DataFrame: DataFrame with CLD results.
    """
    # Loading R function from file
    r_obj.r["source"]("functions.R")
    convert_pairwise_p_to_cld = r_obj.globalenv["convert_pairwise_p_to_cld"]

    # Converting Pandas data frame to R data frame
    with localconverter(r_obj.default_converter + r_obj.pandas2ri.converter):
        df_in_r = r_obj.conversion.py2rpy(data)

    # Invoking the R function and getting the result
    df_result_r = convert_pairwise_p_to_cld(
        df_in_r,
        group1=group1,
        group2=group2,
        p_name=p_name,
        output_gr_var=output_gr_var,
    )

    # Converting the result back to a Pandas dataframe
    return r_obj.pandas2ri.rpy2py(df_result_r)


# Helper functions to work with R in Python -----------------------------------
def r_to_python(obj: str):
    """Import object from R environment to Python

    Import object from R environment created in ipynb cells via `rpy2` package.

    Args:
        obj (str): Object name in R global environment.

    Returns:
        Analogous Python object (NOTE: tested with data frames only).
    """
    return r_obj.pandas2ri.rpy2py(r_obj.globalenv[obj])


# Format values --------------------------------------
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


def format_percent(x: float):
    """Round percentages to 1 decimal place and format as strings

    Values between 0 and 0.05 are printed as <0.1%
    Values between 99.95 and 100 are printed as >100%

    Args:
        x (float): A sequence of percentage values ranging from 0 to 100.

    Returns:
        pd.Series[str]: Pandas series of formatted values.
        Values equal to 0 are formatted as "0%", values between
        0 and 0.05 are formatted as "<0.1%", values between 99.95 and 100
        are formatted as ">99.9%", and values equal to 100 are formatted
        as "100%".

    Author: Vilmantas Gėgžna
    """
    return pd.Series(
        [
            "0%"
            if i == 0
            else "<0.1%"
            if i < 0.05
            else ">99.9%"
            if 99.95 <= i < 100
            else f"{i:.1f}%"
            for i in x
        ],
        index=x.index,
    )


# For data wrangling ---------------------------------------------
def as_df(
    data: pd.Series,
    index_name: Union[str, list[str], tuple[str, ...]] = "value",
    name: str = "count",
) -> pd.DataFrame:
    """Convert Series to DataFrame with desired column names.

    Used to display data in more user-friendly way.

    Args:
        data (pandas.Series): Pandas Series object.
        index_name (str or sequence of str):
                    New name for Series index
                    (applied before conversion to DataFrame).
                    Defaults to "value".
        name (str): Name for series values
                    (applied before conversion to DataFrame).
                    Defaults "count".

    Returns:
        pandas.DataFrame: DataFrame.
    """
    return data.rename_axis(index_name).rename(name).reset_index()


# Merge all tables
def merge_all(df_list, on, how="outer"):
    """Merge multiple data frames.

    Args:
        df_list (list of pandas.dataframes): Data frames to join.
        on (str): Column names to join on.
             See details in pandas.DataFrame.merge().
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
def calc_summaries(x, ndigits=None):
    """Calculate some common summary statistics.

    Args:
        x (pandas.Series): Numeric variable to summarize.
        ndigits (int, None, optional): Number of decimal digits to round to.
                Defaults to None.
    Return:
       pandas.DataFrame with summary statistics.
    """

    def mad(x):
        return median_abs_deviation(x)

    def range(x):
        return x.max() - x.min()

    res = x.agg(
        ["count", "min", "max", range, "mean", "median", "std", mad, "skew"]
    )

    if ndigits is not None:
        summary = pd.DataFrame(round(res, ndigits=ndigits)).T
    else:
        summary = pd.DataFrame(res).T
    # Present count data as integer:
    summary = summary.assign(count=lambda d: d["count"].astype(int))

    return summary


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
    """Plot count data as bar plots with labels.

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
        label (str, None, optional): Column name from `counts` for value labels.
                Defaults to "percent".
                If None, label is not added.
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
    if label is not None:
        ax_add_value_labels_ab(
            ax, labels=counts[label], rotation=label_rotation
        )
    ax.set_ylim(0, y_lim_max)

    return ax


def ax_xaxis_integer_ticks(min_n_ticks: int, rot: int = 0):
    """Ensure that x axis ticks has integer values

    Args:
        min_n_ticks (int): Minimal number of ticks to use.
        rot (int, optional): Rotation angle of x axis tick labels.
        Defaults to 0.
    """
    ax = plt.gca()
    ax.xaxis.set_major_locator(
        MaxNLocator(min_n_ticks=min_n_ticks, integer=True)
    )
    plt.xticks(rotation=rot)


def ax_axis_comma_format(axis: str = "xy", ax=None):
    """Write values of X axis ticks with comma as thousands separator

    Args:
        axis (str, optional): which axis should be formatted:
           "x" X axis, "y" Y axis or "xy" (default) both axes.
        ax (axis object, None, optional):Axis of plot.
            Defaults to None: current axis.
    """

    if ax is None:
        ax = plt.gca()

    fmt = "{x:,.0f}"
    formatter = plt.matplotlib.ticker.StrMethodFormatter(fmt)
    if "x" in axis:
        ax.xaxis.set_major_formatter(formatter)

    if "y" in axis:
        ax.yaxis.set_major_formatter(formatter)


def ax_add_value_labels_ab(
    ax, labels=None, spacing=2, size=9, weight="bold", **kwargs
):
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
