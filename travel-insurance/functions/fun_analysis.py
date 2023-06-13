"""Functions to perform statistical analysis and output the results."""

import pandas as pd
import statsmodels.stats.api as sms
import scipy.stats as sps
import scikit_posthocs as sp

import functions.fun_utils as my  # Custom module

from typing import Optional, Union

# Functions =================================================================

# Exploratory analysis ------------------------------------------------------
def get_columns_overview(data: pd.DataFrame) -> pd.DataFrame:
    """Get overview of data frame columns: data types, number of unique values,
    and number of missing values.

    Args:
        data (pd.DataFrame): Data frame to analyze.

    Returns:
        pd.DataFrame: Data frame with columns `column`, `data_type`, `n_unique`,
        and `n_missing`.
    """
    return pd.DataFrame(
        {
            "column": data.columns,
            "data_type": data.dtypes,
            "n_unique": data.nunique(),
            "n_missing": data.isna().sum(),
        }
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
        result = f"chi-square test, χ²({dof}, n = {n}) = {round(stat, 2)}, {my.format_p(p)}"

    return result
