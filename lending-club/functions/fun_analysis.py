"""Functions and classes to perform statistical analysis and output the results."""

from typing import Optional, Union
from IPython.display import display

import humanize
import math
import numpy as np
import pandas as pd
import pingouin as pg
import statsmodels.stats.api as sms
import scipy.stats as sps
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go

import functions.fun_utils as my  # Custom module
import functions.cld as cld  # Custom module; CLD calculations


from statsmodels.graphics.mosaicplot import mosaic
from statsmodels.stats.multitest import multipletests as p_adjust

from pandas.api.types import is_integer_dtype
from sklearn.feature_selection import mutual_info_classif


# Functions =================================================================

# Exploratory analysis ------------------------------------------------------


def count_unique(data: pd.DataFrame) -> pd.DataFrame:
    """Get number and percentage of unique values

    Args:
        data (pd.DataFrame): Data frame to analyze.

    Return: data frame with columns `n_unique` (int) and `percent_unique` (str)
    """
    n_unique = data.nunique()
    return pd.concat(
        [
            n_unique.rename("n_unique"),
            (
                my.format_percent((n_unique / data.shape[0]).multiply(100)).rename(
                    "percent_unique"
                )
            ),
        ],
        axis=1,
    )


def summarize_numeric(x, ndigits=None):
    """Calculate some common summary statistics.

    Args:
        x (pandas.Series): Numeric variable to summarize.
        ndigits (int, None, optional): Number of decimal digits to round to.
            Defaults to None.
    Return:
        pandas.DataFrame with summary statistics.
    """

    def mad(x):
        return sps.median_abs_deviation(x)

    def range(x):
        return x.max() - x.min()

    res = x.agg(["count", "min", "max", range, "mean", "median", "std", mad, "skew"])

    if ndigits is not None:
        summary = pd.DataFrame(round(res, ndigits=ndigits)).T
    else:
        summary = pd.DataFrame(res).T
    # Present count data as integers:
    summary = summary.assign(count=lambda d: d["count"].astype(int))

    return summary


def frequency_table(
    group: str,
    data: pd.DataFrame,
    sort: Union[bool, str] = True,
    weight: Optional[str] = None,
    n_label: str = "n",
    perc_label: str = "percent",
) -> pd.DataFrame:
    """Create frequency table that contains counts and percentages.

    Args:
        group (str): Variable that defines the groups. Column name from `data`.
        data (pandas.DataFrame): data frame.
        sort (bool or "index", optional): Way to sort values:
            - True or "count" - sort by count descending.
            - "index" - sort by index ascending.
            - False - no sorting.
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

    vsort = sort == True or sort == "count"

    if weight is None:
        counts = data[group].value_counts(sort=vsort)
        if sort == "index":
            counts = counts.sort_index()
    else:
        counts = data.groupby(group)[weight].sum()

    percent = my.counts_to_percentages(counts)

    return (
        pd.concat([counts.rename(n_label), percent.rename(perc_label)], axis=1)
        .rename_axis(group)
        .reset_index()
    )


def summarize_discrete(data: pd.DataFrame, max_n_unique: int = 20, **kwargs) -> None:
    """Create and display frequency tables for columns with a low number of unique values.

    This function generates and displays frequency tables for columns in the
    input DataFrame where the number of unique values is less than the specified
    threshold (`max_n_unique`).

    Args:
        data (pd.DataFrame): The input data frame.

        max_n_unique (int, optional): The maximum number of unique values
            allowed for a column to be considered for generating a frequency
            table.
            Defaults to 20.

        **kwargs: Additional keyword arguments to pass to
            `frequency_table()`.

    Returns:
        None: This function displays the frequency tables using the
        `display()` function.

    Example:
        To create and display frequency tables for columns with 10 or less
        unique values in a DataFrame `df`, you can use:

        >>> summarize_discrete(df, max_n_unique=10)
    """
    n = data.shape[0]

    tables_to_display = [
        frequency_table(column_name, data, **kwargs)
        for column_name in data.columns
        if data[column_name].nunique() <= max_n_unique
    ]

    for table in tables_to_display:
        display(
            table.style.hide(axis="index")
            .bar(subset=["n"], color="grey", width=60, vmin=0, vmax=n)
            .set_properties(**{"width": "12em"})
            # See https://stackoverflow.com/questions/40990700/pandas-dataframes-in-jupyter-columns-of-equal-width-and-centered
        )


def col_info(
    data: pd.DataFrame,
    style: bool = False,
    n_unique_max: int = 20,
    p_missing_threshold: float = 50.0,
    p_dominant_threshold: float = 90.0,
) -> pd.DataFrame:
    """Get overview of data frame columns: data types, number of unique values,
    and number of missing values.

    Args:
        data (pd.DataFrame): Data frame to analyze.
        style (bool, optional): Flag to return styled data frame.
            if True, styled data frame is returned:
            - index is hidden;
            - `memory_size` is formatted as human-readable string;
            - `data_type` is highlighted:
                in blue for columns with numeric data types (integer or float)
                in green for columns with "category" data type,
                in purple for columns with words "date" or "time" in data type;
                in khaki for columns with boolean data type;
            - `p_unique` and `p_missing` are formatted as percentages (str);
            - data_type is highlighted for columns with numeric data types
                ('int' or 'float' in the name);
            - `n_missing` and `p_missing` are grayed out for columns with no missing
                values;
            - `n_unique` is highlighted for binary variables (in green) and
                columns with more than `n_unique_max` (usually 20) unique values
                are in blue;
            - `p_unique` is highlighted in red for columns with only one unique
                value and in orange for columns with more than
                `p_dominant_threshold` (usually 90%) of unique values.
            - `p_dominant` and `p_dom_excl_na` is highlighted:
                in orange for more than `p_dominant_threshold` (usually 90%) of
                unique values;
                in dark orange, for values 99.9% or >99.9%;
                in red for 100% (that means a single unique value in a column).
        n_unique_max (int, optional): Maximum number of unique values to
            treat them as categorical. When `style=True`, values > n_unique_max
            are highlighted in the same way as numeric data types.
            Defaults to 20.
        p_dominant_threshold (float, optional): Threshold for percentage of
            dominant value. When `style=True`, values > p_dominant_threshold
            are highlighted in orange. Default is 90.0.

    Returns:
        pd.DataFrame: Data frame with columns:
        `column` (column name),
        `data_type` (data type of the column),
        `memory_size` (memory size of the column in bytes),
        `n_unique` (number of unique values),
        `p_unique` (percentage of unique values, number from 0 to 100),
        `n_missing` (number of missing values),
        `p_pissing` (percentage of missing values, number from 0 to 100),
        `n_dominant` (count of the most frequent, i.e. dominant, value),
        `p_dominant` (percentage of the most frequent value);
        `p_dom_excl_na` (percentage of the most frequent value excluding missing
            values),
        `dominant` (the most frequent value).
    """

    def safe_division(a, b):
        """In case of zero"""
        try:
            result = a / b
        except ZeroDivisionError:
            result = 0
        return result

    def safe_max(x):
        x_max = x.max()
        if math.isnan(x_max):
            return 0
        else:
            return x_max

    def safe_idxmax(x):
        try:
            return str(x.idxmax())
        except:
            return ""

    def get_dominant(x):
        counts = x.value_counts()
        return safe_max(counts), safe_idxmax(counts)

    n = data.shape[0]

    dominant_df = data.apply(get_dominant).T
    n_dominant = dominant_df.iloc[:, 0].astype(int)
    dominant = dominant_df.iloc[:, 1]

    # n_dominant = data.apply(lambda x: safe_max(x.value_counts()))
    # dominant = data.apply(lambda x: safe_idxmax(x.value_counts()))
    n_unique = data.nunique()

    data_na = data.isna()
    n_na = data_na.sum()

    info = pd.DataFrame({
        "column": data.columns,
        "data_type": data.dtypes,
        "memory_size": data.memory_usage(deep=True, index=False),
        "n_unique": n_unique,
        "p_unique": (n_unique / n * 100),
        "n_missing": n_na,
        "p_missing": (data_na.mean() * 100),
        "n_dominant": n_dominant,
        "p_dominant": (n_dominant / n * 100),  # NOTE: modin warning comes from here
        "p_dom_excl_na": safe_division(n_dominant, n - n_na) * 100,
        "dominant": dominant,
    }).pipe(my.use_numeric_index)

    if style:
        return style_col_info(
            info, n_unique_max, p_missing_threshold, p_dominant_threshold
        )
    else:
        return info


def style_col_info_before_pd210(
    info: pd.DataFrame,
    n_unique_max: int = 20,
    p_missing_threshold: float = 50.0,
    p_dominant_threshold: float = 90.0,
) -> pd.DataFrame:
    """Format and style the result of `col_info()` function.

    NOTE: This function is for compatibility with older versions of (Pandas < 2.1.0)

    style_col_info is applied when col_info(style=True) is called.

    For more details, see the documentation of `style` argument of
    `col_info()` function.

    To describe the meaning of formatted results, the following text might be used:

    ***
    In this type of `.col_info()` tables:

    - `dominant` is the most frequent value;
    - `p_` is a percentage of certain values;
    -  `n_` is a number of certain values;
    - in `data_type`, numeric data types (float and integer) are highlighted
        in *blue*, and the "category" data type is in *green*;
    - in `n_unique`, binary variables are in different *green*, and
        columns with a high number of unique values ($> 10$)
        are highlighted in *blue*;
    - in `n_missing` and `p_missing`, zero values are in grey;
    - in `p_missing`, percentages above `p_missing_threshold` (usually 50%) are
        highlighted in *orange*;
    - in `p_dominant` and `p_dom_excl_na`, 100% are in *red*, >99.9% and 99.9%
        are in *dark orange*, and percentages above 90% are in *orange*;
    - errors or extremely suspicious values (if any) are highlighted in *red*.
    ***
    """

    color_category = "limegreen"
    color_binary = color_category
    color_bool = "darkkhaki"
    color_numeric = "deepskyblue"
    color_dt = "violet"
    color_prewarning = "sienna"
    color_warning = "orange"
    color_danger = "red"
    color_fade = "gray"
    return (
        info.assign(
            memory_size=lambda d: d["memory_size"].apply(humanize.naturalsize),
            p_unique=lambda d: my.format_percent(d["p_unique"]),
            p_missing=lambda d: my.format_percent(d["p_missing"]),
            p_dominant=lambda d: my.format_percent(d["p_dominant"]),
            p_dom_excl_na=lambda d: my.format_percent(d["p_dom_excl_na"]),
        )
        .style.applymap(
            my.highlight_int_float_text, color=color_numeric, subset=["data_type"]
        )
        .applymap(
            my.highlight_category_text, color=color_category, subset=["data_type"]
        )
        .applymap(my.highlight_bool_text, color=color_bool, subset=["data_type"])
        .applymap(my.highlight_date_or_time_text, color=color_dt, subset=["data_type"])
        .applymap(
            my.highlight_between,
            min=0,
            max=1,
            color=color_danger,
            subset=["n_unique"],
        )
        .applymap(
            my.highlight_between,
            min=2,
            max=2,
            color=color_binary,
            subset=["n_unique"],
        )
        .applymap(
            my.highlight_above,
            min=n_unique_max,
            color=color_numeric,
            subset=["n_unique"],
        )
        .applymap(my.highlight_value, when=0, color=color_fade, subset=["n_missing"])
        .applymap(
            my.highlight_above_str,
            min=p_missing_threshold,
            color=color_warning,
            subset=["p_missing"],
        )
        .applymap(
            my.highlight_above_str,
            min=90,
            color=color_prewarning,
            subset=["p_missing"],
        )
        .applymap(my.highlight_value, when="0%", color=color_fade, subset=["p_missing"])
        .applymap(
            my.highlight_value,
            when="100.0%",
            color=color_danger,
            subset=["p_missing"],
        )
        .applymap(
            my.highlight_value, when="0%", color=color_danger, subset=["p_unique"]
        )
        .applymap(my.highlight_value, when=0, color=color_danger, subset=["n_dominant"])
        .applymap(
            my.highlight_above_str,
            min=p_dominant_threshold,
            color=color_warning,
            subset=["p_dominant", "p_dom_excl_na"],
        )
        .applymap(
            my.highlight_above_str,
            min=99.8,
            color=color_prewarning,
            subset=["p_dominant", "p_dom_excl_na"],
        )
        .applymap(
            my.highlight_value,
            when="0%",
            color=color_danger,
            subset=["p_dominant", "p_dom_excl_na"],
        )
        .applymap(
            my.highlight_value,
            when="100.0%",
            color=color_danger,
            subset=["p_dominant", "p_dom_excl_na"],
        )
    )


def style_col_info(
    info: pd.DataFrame,
    n_unique_max: int = 20,
    p_missing_threshold: float = 50.0,
    p_dominant_threshold: float = 90.0,
) -> pd.DataFrame:
    """Format and style the result of `col_info()` function.

    style_col_info is applied when col_info(style=True) is called.

    For more details, see the documentation of `style` argument of
    `col_info()` function.

    To describe the meaning of formatted results, the following text might be used:

    ***
    In this type of `.col_info()` tables:

    - `dominant` is the most frequent value;
    - `p_` is a percentage of certain values;
    -  `n_` is a number of certain values;
    - in `data_type`, numeric data types (float and integer) are highlighted
        in *blue*, and the "category" data type is in *green*;
    - in `n_unique`, binary variables are in different *green*, and
        columns with a high number of unique values ($> 10$)
        are highlighted in *blue*;
    - in `n_missing` and `p_missing`, zero values are in grey;
    - in `p_missing`, percentages above `p_missing_threshold` (usually 50%) are
        highlighted in *orange*;
    - in `p_dominant` and `p_dom_excl_na`, 100% are in *red*, >99.9% and 99.9%
        are in *dark orange*, and percentages above 90% are in *orange*;
    - errors or extremely suspicious values (if any) are highlighted in *red*.
    ***
    """

    color_category = "limegreen"
    color_binary = color_category
    color_bool = "darkkhaki"
    color_numeric = "deepskyblue"
    color_dt = "violet"
    color_prewarning = "sienna"
    color_warning = "orange"
    color_danger = "red"
    color_fade = "gray"
    return (
        info.assign(
            memory_size=lambda d: d["memory_size"].apply(humanize.naturalsize),
            p_unique=lambda d: my.format_percent(d["p_unique"]),
            p_missing=lambda d: my.format_percent(d["p_missing"]),
            p_dominant=lambda d: my.format_percent(d["p_dominant"]),
            p_dom_excl_na=lambda d: my.format_percent(d["p_dom_excl_na"]),
        )
        .style.map(
            my.highlight_int_float_text, color=color_numeric, subset=["data_type"]
        )
        .map(my.highlight_category_text, color=color_category, subset=["data_type"])
        .map(my.highlight_bool_text, color=color_bool, subset=["data_type"])
        .map(my.highlight_date_or_time_text, color=color_dt, subset=["data_type"])
        .map(
            my.highlight_between,
            min=0,
            max=1,
            color=color_danger,
            subset=["n_unique"],
        )
        .map(
            my.highlight_between,
            min=2,
            max=2,
            color=color_binary,
            subset=["n_unique"],
        )
        .map(
            my.highlight_above,
            min=n_unique_max,
            color=color_numeric,
            subset=["n_unique"],
        )
        .map(my.highlight_value, when=0, color=color_fade, subset=["n_missing"])
        .map(
            my.highlight_above_str,
            min=p_missing_threshold,
            color=color_warning,
            subset=["p_missing"],
        )
        .map(
            my.highlight_above_str,
            min=90,
            color=color_prewarning,
            subset=["p_missing"],
        )
        .map(my.highlight_value, when="0%", color=color_fade, subset=["p_missing"])
        .map(
            my.highlight_value,
            when="100.0%",
            color=color_danger,
            subset=["p_missing"],
        )
        .map(my.highlight_value, when="0%", color=color_danger, subset=["p_unique"])
        .map(my.highlight_value, when=0, color=color_danger, subset=["n_dominant"])
        .map(
            my.highlight_above_str,
            min=p_dominant_threshold,
            color=color_warning,
            subset=["p_dominant", "p_dom_excl_na"],
        )
        .map(
            my.highlight_above_str,
            min=99.8,
            color=color_prewarning,
            subset=["p_dominant", "p_dom_excl_na"],
        )
        .map(
            my.highlight_value,
            when="0%",
            color=color_danger,
            subset=["p_dominant", "p_dom_excl_na"],
        )
        .map(
            my.highlight_value,
            when="100.0%",
            color=color_danger,
            subset=["p_dominant", "p_dom_excl_na"],
        )
    )


# Inferential statistics -----------------------------------------------------
def ci_proportion_binomial(
    counts,
    method: str = "wilson",
    n_label: str = "n",
    percent_label: str = "percent",
    **kwargs,
) -> pd.DataFrame:
    """Calculate confidence intervals for binomial proportion.

    Calculates confidence intervals for each category separately on
    "category counts / total counts" basis.

    Wrapper around statsmodels.stats.proportion.proportion_confint()

    More information in documentation of statsmodels's proportion_confint().

    Args:
        x (int): ps.Series, list or tuple with count data.
            method (str, optional): Method. Defaults to "wilson".
        n_label (str, optional): Name for column for counts.
        percent_label (str, optional): Name for column for percentage values.
        **kwargs: Additional arguments passed to proportion_confint().

    Returns:
        pd.DataFrame: Data frame with group names, absolute counts, percentages
        and their confidence intervals.

    Examples:
    >>> ci_proportion_binomial([62, 55])
    """
    assert isinstance(counts, (pd.Series, list, tuple))
    if not isinstance(counts, pd.Series):
        counts = pd.Series(counts)

    return pd.concat(
        [
            (counts).rename(n_label),
            (counts / sum(counts)).rename(percent_label) * 100,
            pd.DataFrame(
                [
                    sms.proportion_confint(
                        count_i, sum(counts), method=method, **kwargs
                    )
                    for count_i in counts
                ],
                index=counts.index,
                columns=[f"ci_lower", "ci_upper"],
            )
            * 100,
        ],
        axis=1,
    )


def ci_proportion_multinomial(
    counts,
    method: str = "goodman",
    n_label: str = "n",
    percent_label: str = "percent",
    **kwargs,
) -> pd.DataFrame:
    """Calculate  simultaneous confidence intervals for multinomial proportion.

    Wrapper around statsmodels.stats.proportion.multinomial_proportions_confint()

    More information in documentation of statsmodels's
    multinomial_proportions_confint().

    Args:
        x (int): ps.Series, list or tuple with count data.
            method (str, optional): Method. Defaults to "goodman".
        n_label (str, optional): Name for column for counts.
        percent_label (str, optional): Name for column for percentage values.
        **kwargs: Additional arguments passed to multinomial_proportions_confint().

    Returns:
        pd.DataFrame: Data frame with group names, absolute counts, percentages
        and their confidence intervals.

    Examples:
    >>> ci_proportion_multinomial([62, 33, 55])
    """
    assert isinstance(counts, (pd.Series, list, tuple))
    if not isinstance(counts, pd.Series):
        counts = pd.Series(counts)

    return pd.concat(
        [
            (counts).rename(n_label),
            (counts / sum(counts)).rename(percent_label) * 100,
            pd.DataFrame(
                sms.multinomial_proportions_confint(counts, method=method, **kwargs),
                index=counts.index,
                columns=[f"ci_lower", "ci_upper"],
            )
            * 100,
        ],
        axis=1,
    )


def test_chi_square_gof(
    f_obs: list[int],
    f_exp: Union[str, list[float]] = "all equal",
    output: str = "long",
) -> str:
    """Chi-squared (χ²) goodness-of-fit (gof) test

    Args:
        f_obs (list[int]): Observed frequencies
        f_exp str, list[int]: List of expected frequencies or "all equal" if
            all frequencies are equal to the mean of observed frequencies.
            Defaults to "all equal".
        output (str, optional): Output format (available options:
        "short", "long"). Defaults to "long".

    Returns:
        str: formatted test results including p value.
    """
    k = len(f_obs)
    n = sum(f_obs)
    exp = n / k
    dof = k - 1
    if f_exp == "all equal":
        f_exp = [exp for _ in range(k)]
    stat, p = sps.chisquare(f_obs=f_obs, f_exp=f_exp)
    # May also be formatted this way:
    if output == "short":
        result = f"chi-square test, {my.format_p(p)}"
    else:
        result = (
            f"chi-square test, χ²({dof}, n = {n}) = {round(stat, 2)}, {my.format_p(p)}"
        )

    return result


# Classes ===================================================================


# Analyze count data --------------------------------------------------------
class AnalyzeCounts:
    """The class to analyze count data.

    - Performs omnibus chi-squared and post-hoc pair-wise chi-squared test.
    - Compactly presents results of post-hoc test as compact letter display, CLD
        (Shared CLD letter show no significant difference between groups).
    - Calculates percentages and their confidence intervals by using Goodman's
    method.
    - Creates summary of grouped values (group counts and percentages).
    - Plots results as bar plots with percentage labels.
    """

    def __init__(self, counts, by=None, counts_of=None):
        """
        Object initialization function.

        Args:
            counts (pandas.Series[int]): Count data to analyze.
            by (str, optional): Grouping variable name. Used to create labels.
                If None, defaults to "Group"
            counts_of (str, optional): The thing that was counted.
                This name is used for labels in plots and tables.
                Defaults to `counts.name`.
        """
        assert isinstance(counts, pd.Series)

        # Set defaults
        if by is None:
            by = "Group"

        if counts_of is None:
            counts_of = counts.name

        # Set attributes: user inputs or defaults
        self.counts = counts
        self.counts_of = counts_of
        self.by = by

        # Set attributes: created/calculated
        self.n_label = f"n_{counts_of}"  # Create label for counts

        # Set attributes: results to be calculated
        self.results_are_calculated = False
        self.omnibus = None
        self.n_ci_and_cld = None
        self.descriptive_stats = None

        # Alias attributes
        counts = self.counts
        by = self.by
        n_label = self.n_label

        # Omnibus test: perform and save the results
        self.omnibus = test_chi_square_gof(counts)

        # Post-hoc (pairwise chi-square): perform
        posthoc_p = my.pairwise_chisq_gof_test(counts)
        posthoc_cld = cld.make_cld(posthoc_p, output_gr_var=by)

        # Confidence interval: calculate
        ci = (
            ci_proportion_multinomial(counts, method="goodman", n_label=n_label)
            .rename_axis(by)
            .reset_index()
        )

        # Make sure datasets are mergeable
        ci[by] = ci[by].astype(str)
        posthoc_cld[by] = posthoc_cld[by].astype(str)

        # Merge results
        n_ci_and_cld = pd.merge(ci, posthoc_cld, on=by)

        # Format percentages and counts
        vars = ["percent", "ci_lower", "ci_upper"]
        n_ci_and_cld[vars] = n_ci_and_cld[vars].apply(my.format_percent)

        # Save results
        self.n_ci_and_cld = n_ci_and_cld

        # Descriptive statistics: calculate
        to_format = ["min", "max", "range", "mean", "median", "std", "mad"]

        def format_0f(x):
            return [f"{i:,.0f}" for i in x]

        summary_count = my.summarize_numeric(ci[n_label])
        summary_count[to_format] = summary_count[to_format].apply(format_0f)

        summary_perc = my.summarize_numeric(ci["percent"])
        summary_perc[to_format] = summary_perc[to_format].apply(my.format_percent)
        # Save results
        self.descriptive_stats = pd.concat([summary_count, summary_perc])

        # Initialization status
        self.results_are_calculated = True

        # Output
        return self

    def print(
        self,
        omnibus: bool = True,
        posthoc: bool = True,
        descriptives: bool = True,
    ):
        """Print numeric results.

        Args:
            omnibus (bool, optional): Flag to print omnibus test results.
                                    Defaults to True.
            posthoc (bool, optional): Flag to print post-hoc test results.
                                    Defaults to True.
            descriptives (bool, optional): Flag to print descriptive statistics.
                                    Defaults to True.

        Raises:
            Exception: if calculations with `.fit()` method were not performed.
        """
        if not self.results_are_calculated:
            raise Exception("No results. Run `.fit()` first.")

        # Omnibus test
        if omnibus:
            print("Omnibus (chi-squared) test results:")
            print(self.omnibus, "\n")

        # Post-hoc and CI
        if posthoc:
            print(
                f"Counts of {self.counts_of} with 95% CI "
                "and post-hoc (pairwise chi-squared) test results:"
            )
            print(self.n_ci_and_cld, "\n")

        # Descriptive statistics: display
        if descriptives:
            print(f"Descriptive statistics of group ({self.by}) counts:")
            print(self.descriptive_stats, "\n")

    def display(
        self,
        omnibus: bool = True,
        posthoc: bool = True,
        descriptives: bool = True,
    ):
        """Display numeric results in Jupyter Notebooks.

        Args:
            omnibus (bool, optional): Flag to print omnibus test results.
                                      Defaults to True.
            posthoc (bool, optional): Flag to print post-hoc test results.
                                      Defaults to True.
            descriptives (bool, optional): Flag to print descriptive statistics.
                                      Defaults to True.

        Raises:
            Exception: if calculations with `.analyze()` method were
            not performed.
        """
        if not self.results_are_calculated:
            raise Exception("No results. Run `.fit()` first.")

        # Omnibus test
        if omnibus:
            my.display_collapsible(self.omnibus, "Omnibus (chi-squared) test results")

        # Post-hoc and CI
        if posthoc:
            my.display_collapsible(
                self.n_ci_and_cld.style.format({self.n_label: "{:,.0f}"}),
                f"Counts of {self.counts_of} with 95% CI and post-hoc "
                " (pairwise chi-squared) test results",
            )

        # Descriptive statistics: display
        if descriptives:
            my.display_collapsible(
                self.descriptive_stats,
                f"Descriptive statistics of group ({self.by}) counts",
            )

    def plot(self, xlabel=None, ylabel=None, **kwargs):
        """Plot analysis results.

        Args:
            xlabel (str, None, optional): X axis label.
                    Defaults to None: autogenerated label.
            ylabel (str, None, optional): Y axis label.
                    Defaults to None: autogenerated label.
            **kwargs: further arguments passed to `my.plot_counts_with_labels()`

        Raises:
            Exception: if calculations with `.fit()` method were
            not performed.

        Returns:
            matplotlib.axes object
        """
        if not self.results_are_calculated:
            raise Exception("No results. Run `.fit()` first.")

        # Plot
        if xlabel is None:
            xlabel = self.by.capitalize()

        if ylabel is None:
            ylabel = f"Number of {self.counts_of}"

        ax = my.plot_counts_with_labels(
            self.n_ci_and_cld,
            x_lab=xlabel,
            y_lab=ylabel,
            y=self.n_label,
            **kwargs,
        )

        my.ax_axis_comma_format("y")

        return ax


# Analyze numeric groups ------------------------------------------------------
class AnalyzeNumericGroups:
    """Class to analyze numeric/continuous data by groups.

    - Calculates mean ratings per group and their confidence intervals using
        t distribution.
    - Performs omnibus (Kruskal-Wallis) and post-hoc (Conover-Iman) tests.
    - Compactly presents results of post-hoc test as compact letter display, CLD
      NOTE: for CLD calculations, R is required.
      (Shared CLD letter show no significant difference between groups).
    - Creates summary of grouped values (group counts and percentages).
    - Plots results as points with 95% confidence interval error bars.
    """

    def __init__(self, data, y: str, by: str):
        """Initialize the class.

        Args:
            y (str): Name of numeric/continuous (dependent) variable.
            by (str): Name of grouping (independent) variable.
            data (pandas.DataFrame): data frame with variables indicated in
                `y` and `by`.
        """
        assert isinstance(data, pd.DataFrame)

        # Set attributes: user inputs
        self.data = data
        self.y = y
        self.by = by

        # Set attributes: results to be calculated
        self.results_are_calculated = False
        self.omnibus = None
        self.ci_and_cld = None
        self.descriptive_stats = None

    def fit(self):
        # Aliases:
        data = self.data
        y = self.y
        by = self.by

        # Omnibus test: Kruskal-Wallis test
        omnibus = pg.kruskal(data=data, dv=y, between=by)
        omnibus["p-unc"] = my.format_p(omnibus["p-unc"][0])

        self.omnibus = omnibus

        # Confidence intervals
        ci_raw = data.groupby(by)[y].apply(
            lambda x: [np.mean(x), *sms.DescrStatsW(x).tconfint_mean()]
        )
        ci = pd.DataFrame(
            list(ci_raw),
            index=ci_raw.index,
            columns=["mean", "ci_lower", "ci_upper"],
        ).reset_index()

        # Post-hoc test: Conover-Iman test
        posthoc_p_matrix = sp.posthoc_conover(
            data, val_col=y, group_col=by, p_adjust="holm"
        )
        posthoc_p_df = posthoc_p_matrix.stack().to_df("p.adj", ["group1", "group2"])
        posthoc_cld = my.convert_pairwise_p_to_cld(posthoc_p_df, output_gr_var=by)

        # Make sure datasets are mergeable
        ci[by] = ci[by].astype(str)
        posthoc_cld[by] = posthoc_cld[by].astype(str)

        self.ci_and_cld = pd.merge(posthoc_cld, ci, on=by)

        # Descriptive statistics of means
        self.descriptive_stats = my.summarize_numeric(ci["mean"])

        # Results are present
        self.results_are_calculated = True

        # Output:
        return self

    def print(
        self,
        omnibus: bool = True,
        posthoc: bool = True,
        descriptives: bool = True,
    ):
        """Print numeric results.

        Args:
            omnibus (bool, optional): Flag to print omnibus test results.
                                      Defaults to True.
            posthoc (bool, optional): Flag to print post-hoc test results.
                                      Defaults to True.
            descriptives (bool, optional): Flag to print descriptive statistics.
                                      Defaults to True.

        Raises:
            Exception: if calculations with `.fit()` method were
            not performed.
        """
        if not self.results_are_calculated:
            raise Exception("No results. Run `.fit()` first.")

        # Omnibus test
        if omnibus:
            print("Omnibus (Kruskal-Wallis) test results:")
            print(self.omnibus, "\n")

        # Post-hoc and CI
        if posthoc:
            print(
                "Post-hoc (Conover-Iman) test results as CLD and "
                "Confidence intervals (CI):",
            )
            print(self.ci_and_cld, "\n")

        # Descriptive statistics
        if descriptives:
            print(f"Descriptive statistics of group ({self.by}) means:")
            print(self.descriptive_stats, "\n")

    def display(
        self,
        omnibus: bool = True,
        posthoc: bool = True,
        descriptives: bool = True,
    ):
        """Display numeric results in Jupyter Notebooks.

        Args:
            omnibus (bool, optional): Flag to print omnibus test results.
                                      Defaults to True.
            posthoc (bool, optional): Flag to print post-hoc test results.
                                      Defaults to True.
            descriptives (bool, optional): Flag to print descriptive statistics.
                                      Defaults to True.

        Raises:
            Exception: if calculations with `.fit()` method were
            not performed.
        """
        if not self.results_are_calculated:
            raise Exception("No results. Run `.fit()` first.")

        # Omnibus test
        if omnibus:
            my.display_collapsible(
                self.omnibus, "Omnibus (Kruskal-Wallis) test results"
            )

        # Post-hoc and CI
        if posthoc:
            my.display_collapsible(
                self.ci_and_cld,
                "Post-hoc (Conover-Iman) test results as CLD and "
                "Confidence intervals (CI)",
            )

        # Descriptive statistics of means
        if descriptives:
            my.display_collapsible(
                self.descriptive_stats,
                f"Descriptive statistics of group ({self.by}) means",
            )

    def plot(self, title=None, xlabel=None, ylabel=None, **kwargs):
        """Plot the results

        Args:

            xlabel (str, None, optional): X axis label.
                    Defaults to None: capitalized value of `by`.
            ylabel (str, None, optional): Y axis label.
                    Defaults to None: capitalized value of `y`.
            title (str, None, optional): The title of the plot.
                    Defaults to None.

        Returns:
            Tuple with matplotlib figure and axis objects (fig, ax).
        """
        if not self.results_are_calculated:
            raise Exception("No results. Run `.fit()` first.")

        # Aliases:
        ci = self.ci_and_cld
        by = self.by
        y = self.y

        # Create figure and axes
        fig, ax = plt.subplots()

        # Construct plot
        x = ci.iloc[:, 0]

        ax.errorbar(
            x=x,
            y=ci["mean"],
            yerr=[ci["mean"] - ci["ci_lower"], ci["ci_upper"] - ci["mean"]],
            mfc="red",
            ms=2,
            mew=1,
            fmt="ko",
            zorder=3,
        )

        if xlabel is None:
            xlabel = by.capitalize()

        if ylabel is None:
            ylabel = y.capitalize()

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_ylim([0, None])
        ax.set_title(title)

        # Output
        return (fig, ax)


# Cross-tabulation ------------------------------------------------------------
class CrossTab:
    """Class for cross-tabulation analysis.

    Author: Vilmantas Gėgžna

    """

    def __init__(self, x=None, y=None, data=None, **kwargs):
        """Create a cross-tabulation from data frame.

        Args (option 1):
            data (pandas.DataFrame): Data frame
            x (str): Column name in `data`
            y (str): Column name in `data`

        Args (option 2):
            x (pandas.Series): Variable with numeric values.
            y (pandas.Series): Variable with numeric values.

        Args (common):
            **kwargs: other arguments to pandas.crosstab()

        """
        if data is None:
            # x and y are series objects
            self.xlabel = x.name
            self.ylebel = y.name
        else:
            # x and y are column names in data
            self.xlabel = x
            self.ylabel = y
            x = data[x]
            y = data[y]

        counts = pd.crosstab(x, y, **kwargs)

        self.counts = counts
        self.row_percentage = round(
            counts.div(counts.sum(axis=1), axis=0) * 100, ndigits=1
        )
        self.column_percentage = round(
            counts.div(counts.sum(axis=0), axis=1) * 100, ndigits=1
        )
        self.total_percentage = round(counts.div(counts.sum().sum()) * 100, ndigits=1)

        self.chi_square_omnibus = None
        self.chi_square_posthoc = None

    def __call__(self) -> pd.DataFrame:
        """Return cross-tabulation."""
        return self.counts

    def print(self):
        """Print cross-tabulation."""
        print(self.counts)

    def display(self):
        """Display cross-tabulation in Jupyter Notebook."""
        display(self.counts)

    def assign(self, *args, **kwargs):
        """The same as Pandas .assign() method applied on `counts` attribute.

        Returns a copy of data frame with .assign() applied.
        Does not change internal `counts` attribute.
        """
        return self.counts.assign(*args, **kwargs)

    def chisq_test(self, output="full", correction: bool = True) -> pd.DataFrame:
        """Perform chi-squared test of independence.

        Args:
            output (str, optional): Output format. Available options:
                "short" - string with test name and p value;
                "full" - string with test, sample size and degrees of freedom, and
                    p value;
                None - tuple with the results).
                Defaults to "full".
            correction (bool, optional): Whether to apply Yates' correction
                for continuity. Defaults to True.

        Returns:
            pd.DataFrame: Data frame with test results.
        """
        rez = sps.chi2_contingency(self.counts, correction=correction)
        self.chi_square_omnibus = rez

        if output == "full":
            out = f"chi-square test, χ²({rez.dof}, "
            f"n = {self.counts.sum().sum()}) = {round(rez.statistic, 2)}, "
            f"{my.format_p(rez.pvalue)}"
        elif output == "short":
            out = f"chi-square test, {my.format_p(rez.pvalue)}"
        else:
            out = rez

        return out

    def heatmap(
        self,
        title: Optional[str] = None,
        xlabel: str = None,
        ylabel: str = None,
        vmax: Optional[int] = None,
        vmin: int = 0,
        cbar: bool = True,
        fmt: Union[str, dict] = "1d",
        annot: bool = True,
        annot_kws: dict = {"size": 10},
        cmap: str = "RdYlBu",
        **kwargs,
    ) -> plt.Axes:
        """Plot a Cross-Tabulation as a heatmap.

        Args:
            title (str, optional): Title of the plot.
            xlabel (str, optional): Label for the x-axis. If None (default),
                                    the name will be used.
            ylabel (str, optional): Label for the y-axis. If None (default),
                                    the name will be used.
            vmax (int, optional): Maximum value for color scale.
                                If not provided, the maximum frequency in the
                                cross-tabulation will be calculated.
            vmin (int, optional): Minimum value for color scale. Defaults to 0.
            cbar (bool, optional): Whether to show the color bar. Defaults to True.
            fmt (str or dict, optional): String formatting code for annotations.
                                        Defaults to "1d". Can also be a dictionary
                                        of format codes for specific columns.
            annot (bool, optional): Whether to show annotations. Defaults to True.
            annot_kws (dict, optional): Additional keyword arguments for annotations.
                                        Defaults to {"size": 10}.
            cmap (str, optional): Colormap to use. Defaults to "RdYlBu".
            **kwargs: Additional keyword arguments to be passed to the underlying
                    seaborn heatmap function.

        Returns:
            plt.Axes: The created Axes object.
        """
        crosstab = self.counts

        # Visualize
        sns.set(font_scale=1)

        if vmax is None:
            vmax = crosstab.max().max()

        ax = sns.heatmap(
            crosstab,
            linewidths=0.5,
            annot=annot,
            fmt=fmt,
            annot_kws=annot_kws,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            cbar=cbar,
            **kwargs,
        )

        # NOTE: xlabel of plot is not self.xlabel
        if xlabel is not None:
            ax.set_xlabel(xlabel)

        # NOTE: ylabel og plot is self.ylabel
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        if title is not None:
            ax.set(title=title)

        return ax

    def mosaic(
        self,
        title: str = None,
        xlabel: str = None,
        ylabel: str = None,
        x_rot: int = 0,
        x_ha="center",
        **kwargs,
    ):
        """Plot a Cross-Tabulation as a mosaic plot.

        Args:
            title (str, optional): Title of the plot.
            xlabel (str, optional): Label for the x-axis. If None (default),
                the name will be used.
            ylabel (str, optional): Label for the y-axis. If None (default),
                the name will be used.
            x_rot (int, optional): Rotation angle for x-axis labels. Defaults to 0.
            x_ha (str, optional): Horizontal alignment for x-axis labels.
                Defaults to "center".
            **kwargs: Additional keyword arguments to be passed to the underlying
                statsmodels.graphics.mosaicplot.mosaic() function.

        Returns:
            plt.figure: The created Axes object.
        """
        crosstab = self.counts

        if xlabel is None:
            xlabel = self.xlabel

        if ylabel is None:
            ylabel = self.ylabel

        fig, _ = mosaic(crosstab.T.unstack().to_dict(), title=title, **kwargs)
        ax = fig.axes[0]
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        ax.set_xticklabels(ax.get_xticklabels(), rotation=x_rot, ha=x_ha)

        return fig

    def barplot(
        self,
        title: str = None,
        xlabel: str = None,
        ylabel: str = "AUTO",
        rot: int = 0,
        normalize: str = "none",
        **kwargs,
    ) -> plt.Axes:
        """
        Plot a cross-tabulation as a barplot.

        Args:
            title (str, optional): Title of the plot.
            xlabel (str, optional): Label for the x-axis.
                Defaults to None (use the default value).
            ylabel (str, optional): Label for the y-axis.
                Defaults to "AUTO" (results in either "Count" or "Percentage").
            rot (int, optional): Rotation angle for x-axis labels. Defaults to 0.
            normalize (str, optional): Whether to show absolute counts ("none"),
                row percentages ("rows"), column percentages ("cols"), or
                overall percentages ("all"). Defaults to "none".
            **kwargs: Additional arguments to be passed to the underlying
                pandas.DataFrame.plot.bar() function.

        Returns:
            plt.Axes: The created Axes object.
        """
        cross_t = self.counts

        if xlabel is None:
            xlabel = self.xlabel

        if normalize == "rows":
            # Row percentage
            cross_p = round(cross_t.div(cross_t.sum(axis=1), axis=0) * 100, ndigits=1)
        elif normalize == "cols":
            # Column percentage
            cross_p = round(cross_t.div(cross_t.sum(axis=0), axis=1) * 100, ndigits=1)
        elif normalize == "all":
            # Overall percentage
            cross_p = round(cross_t.div(cross_t.sum().sum()) * 100, ndigits=1)
        else:
            # Absolute counts
            cross_p = cross_t

        if ylabel == "AUTO":
            if normalize in ("rows", "cols", "all"):
                ylabel = "Percentage"
            else:
                ylabel = "Count"

        # Visualize
        ax = cross_p.plot.bar(
            ec="black", title=title, rot=rot, xlabel=xlabel, ylabel=ylabel, **kwargs
        )
        return ax

    def mosaic_dict(self):
        return self.counts.T.unstack().to_dict()

    def mosaic_go(
        self, title: str = None, xlabel: str = None, ylabel: str = None, **kwargs
    ):
        """Plot a Cross-Tabulation as a merimekko (mosaic) chart using Plotly.

        Args:
            title (str, optional): Title of the plot.
            xlabel (str, optional): Label for the x-axis. If None (default),
                the name will be used.
            ylabel (str, optional): Label for the y-axis. If None (default),
                the name will be used.
            **kwargs: Additional keyword arguments to be passed to the Plotly
                go.Figure().

        Returns:
            go.Figure: The created merimekko chart Figure.
        """
        crosstab = self.counts

        if xlabel is None:
            xlabel = self.xlabel

        if ylabel is None:
            ylabel = self.ylabel

        years = list(crosstab.columns)
        marker_colors = {
            col: "rgb({}, {}, {})".format(*np.random.randint(0, 256, 3))
            for col in crosstab.index
        }

        fig = go.Figure()

        for idx in crosstab.index:
            dff = crosstab.loc[idx:idx]
            widths = np.array(dff.values[0])

            fig.add_trace(
                go.Bar(
                    x=np.cumsum(widths) - widths,
                    y=dff.values[0],
                    width=widths,
                    marker_color=marker_colors[idx],
                    text=["{:.2f}%".format(x) for x in dff.values[0]],
                    name=idx,
                )
            )

        fig.update_xaxes(
            tickvals=np.cumsum(widths) - widths,
            ticktext=["%s<br>%d" % (l, w) for l, w in zip(years, widths)],
        )

        fig.update_xaxes(range=[0, sum(widths)])
        fig.update_yaxes(range=[0, 100])

        fig.update_layout(
            title_text=title if title else "Merimekko Chart",
            barmode="stack",
            uniformtext=dict(mode="hide", minsize=10),
            **kwargs,
        )

        return fig


def get_mutual_information(
    data: pd.DataFrame,
    target: str,
    drop: list[str] = [],
    precision: int = 3,
    random_state: int = None,
):
    """Get mutual information scores for classification problem.

    Args:
        data (pd.DataFrame): Dataframe with target column.
        target (str): Target column name for classification task.
        drop (list[str], optional): Columns to drop. Defaults to [].
        precision (int, optional): Number of decimal places for rounding.
            Defaults to 3.
        random_state (int, optional): Random state for mutual information
            calculation. It is needed as some random noise is added to break ties. Defaults to None.

    Returns:
        pd.DataFrame: Mutual information scores.
    """
    # Copy the data and drop unnecessary columns
    X = data.dropna().drop(columns=drop)
    y = X.pop(target)

    # Label encoding for categorical columns
    for colname in X.select_dtypes(["object", "category"]):
        X[colname], _ = X[colname].factorize()

    # Identify discrete features
    discrete_features = [is_integer_dtype(i) for i in X.dtypes]

    # Calculate mutual information scores
    mutual_info = mutual_info_classif(
        X, y, discrete_features=discrete_features, random_state=random_state
    )

    # Create a DataFrame for the scores
    mi_scores = pd.DataFrame(
        {"variable_1": target, "variable_2": X.columns, "mutual_info": mutual_info}
    )

    # Style the DataFrame and return
    styled_scores = (
        mi_scores.sort_values("mutual_info", ascending=False).pipe(my.use_numeric_index)
        #   .reset_index(drop=True)
        .style.format({"mutual_info": f"{{:.{precision}f}}"})
    )
    styled_scores.bar(cmap="BrBG", subset=["mutual_info"])

    return styled_scores


def get_pointbiserial_corr_scores(data, target: str):
    """Get point-biserial correlation scores for numeric variables.

    Pairwise missing values are removed.

    Args:
        data (pd.DataFrame): Dataframe with target column.
        target (str): Target column name.

    Returns:
        pd.DataFrame: Point-biserial correlation scores.
    """
    non_target_numeric = data.select_dtypes("number").drop(columns=target).columns

    def pointbiserialr_wo_pairwise_na(x, y):
        df = pd.DataFrame(zip(x, y)).dropna()
        return df.shape[0], *sps.pointbiserialr(df.iloc[:, 0], df.iloc[:, 1])

    cor_data = [
        (i, *pointbiserialr_wo_pairwise_na(data[target], data[i]))
        for i in non_target_numeric
    ]
    cor_data = pd.DataFrame(cor_data, columns=["variable_2", "n", "r_pb", "p"])
    cor_data.insert(0, "variable_1", target)

    return (
        cor_data.sort_values("r_pb", ascending=False, key=abs)
        .assign(p_adj=lambda x: [my.format_p(i, add_p=False) for i in p_adjust(x.p)[1]])
        .assign(p=lambda x: x.p.apply(my.format_p, add_p=False))
        .pipe(my.use_numeric_index)
        .style.format({"r_pb": "{:.3f}"})
        .background_gradient(cmap="Blues", subset=["n"])
        .bar(vmin=-1, vmax=1, cmap="BrBG", subset=["r_pb"])
    )
