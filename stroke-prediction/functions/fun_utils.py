"""Various functions for data pre-processing, analysis and plotting."""


# Other Python libraries and modules
import re
import pathlib
import joblib
import functools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union
from IPython.display import display, HTML
from matplotlib.ticker import MaxNLocator


# Utilities ==================================================================
# Check/Assert
def index_has_names(obj: Union[pd.Series, pd.DataFrame]) -> bool:
    """Check if index of a Pandas object (Series of DataFrame) has names.

    Args:
        obj: Object that has `.index` attribute.

    Returns:
        bool: True if index has names, False otherwise.

    Examples:
        >>> import pandas as pd

        >>> series1 = pd.Series([1, 2], index=pd.Index(['A', 'B'], name='Letters'))
        >>> index_has_names(series1)
        True

        >>> series2 = pd.Series([1, 2], index=pd.Index(['A', 'B']))
        >>> index_has_names(series2)
        False

        >>> dataframe1 = pd.DataFrame(
        ...    [[1, 2], [3, 4]],
        ...    index=pd.Index(['A', 'B'], name='Rows'),
        ...    columns=['X', 'Y']
        ... )
        >>> index_has_names(dataframe1)
        True

        >>> dataframe2 = pd.DataFrame([[1, 2], [3, 4]])
        >>> index_has_names(dataframe2)
        False
    """
    return None not in list(obj.index.names)


def assert_values(df: pd.DataFrame, expected_values: list[str]) -> None:
    """Assert that the values of each column in a Pandas DataFrame are among
      the expected values.

    Args:
        df (pd.DataFrame): The input DataFrame to check for expected values.
        expected_values (list[str]): The list of expected values.

    Raises:
        AssertionError: If any column in the DataFrame contains values not
        present in the expected values.

    Examples:
        >>> data = pd.DataFrame({
        >>>     'col1': ['Yes', 'No', 'Yes'],
        >>>     'col2': ['Yes', 'Yes', 'Yes']
        >>> })
        >>> assert_values(data, ['Yes', 'No'])
        # No AssertionError is raised

        >>> data = pd.DataFrame({
        >>>     'col1': ['Yes', 'No', 'Yes'],
        >>>     'col2': ['Yes', 'Maybe', 'no']
        >>> })
        >>> assert_values(data, ['Yes', 'No'])
        AssertionError:
        Only ['Yes', 'No'] values were expected in the following columns
        (Column name [unexpected values]):
        col2: ['Maybe', 'no']

    """
    non_matching_values = {}
    for column in df.columns:
        non_matching = df[~df[column].isin(expected_values)][column].tolist()
        if non_matching:
            non_matching_values[column] = non_matching
    if non_matching_values:
        error_message = (
            f"\nOnly {expected_values} values were expected in the following "
            "columns\n(Column name [unexpected values]):\n"
        )
        for col_name, unexpected_values in non_matching_values.items():
            error_message += f"{col_name}: {unexpected_values}\n"
        raise AssertionError(error_message)


# Display in Jupyter notebook
def display_collapsible(x, summary: str = "", sep=" ", is_open: bool = False):
    """Display data frame or other object surrounded by `<details>` tags

    (I.e., display in collapsible way)

    Args:
        x (pd.DataFrame, str, list[str]): Object to display.
        summary (str, optional): Collapsed section name. Defaults to "".
        sep (str, optional): Symbol used to join strings (when x is a list).
             Defaults to " ".
        is_open (bool, optional): Should the section be open by default
            Defaults to False.
    """
    if is_open:
        is_open = " open"
    else:
        is_open = ""

    if hasattr(x, "to_html") and callable(x.to_html):
        html_str = x.to_html()
    elif isinstance(x, str):
        html_str = x
    else:
        html_str = sep.join([str(i) for i in x])

    display(
        HTML(
            f"<details{is_open}><summary>{summary}</summary>" + html_str + "</details>"
        )
    )


# Cache
def cache_results(file: str, force: bool = False):
    """Decorator to cache results of a function and save them to a file in
    the pickle format.

    Args.:
        file (str): File name.
        force (bool, optional): Should the function be run even if the file
            exists? Defaults to False.
    """

    def decorator_cache(fun):
        @functools.wraps(fun)
        def wrapper(*args, **kwargs):
            if pathlib.Path(file).is_file() and not force:
                with open(file, "rb") as f:
                    results = joblib.load(f)
            else:
                results = fun(*args, **kwargs)
                with open(file, "wb") as f:
                    joblib.dump(results, f)
            return results

        return wrapper

    return decorator_cache


# Format values --------------------------------------------------------------
def to_snake_case(text: str):
    """Convert a string to the snake case.

    Args:
        text (str): The input string to change to the snake case.

    Returns:
        str: The string converted to the snake case.

    Examples:
        >>> to_snake_case("Some Text")
        'some_text'
        >>> to_snake_case("SomeText2")
        'some_text_2'
    """
    assert isinstance(text, str), "Input must be a string."

    return (
        pd.Series(text)
        .str.replace("(?<=[a-z])(?=[A-Z0-9])", "_", regex=True)
        .str.replace(r"[ ]", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.lower()
        .to_string(index=False)
    )


def format_p(p: float, digits: int = 3, add_p: bool = True) -> str:
    """Format p values at 3 decimal places.

    Args:
        p (float): p value (number between 0 and 1).
        digits (int, optional): Number of decimal places to round to.
            Defaults to 3.
        add_p (bool, optional): Should the string start with "p"?

    Examples:
        >>> format_p(1)
        'p > 0.999'

        >>> format_p(0.12345)
        'p = 0.123'

        >>> format_p(0.00001)
        'p < 0.001'

        >>> format_p(0.00001, digits=2)
        'p < 0.01'

        >>> format_p(1, digits=5)
        'p > 0.99999'
    """

    precision = 10 ** (-digits)
    if add_p:
        prefix = ["p < ", "p > ", "p = "]
    else:
        prefix = ["<", ">", ""]

    if p < precision:
        return f"{prefix[0]}{precision}"
    elif p > (1 - precision):
        return f"{prefix[1]}{1 - precision}"
    else:
        return f"{prefix[2]}{p:.{digits}f}"


def format_percent(x: Union[float, list[float], pd.Series]) -> pd.Series:
    """Round percentages to 1 decimal place and format as strings

    Values between 0 and 0.05 are printed as <0.1%
    Values between 99.95 and 100 are printed as >99.9%

    Args:
        x: (A sequence of) percentage values ranging from 0 to 100.

    Returns:
        pd.Series[str]: Pandas series of formatted values.
        Values equal to 0 are formatted as "0%", values between
        0 and 0.05 are formatted as "<0.1%", values between 99.95 and 100
        are formatted as ">99.9%", and values equal to 100 are formatted
        as "100%".

    Examples:
        >>> format_percent(0)
        ['0%']

        >>> format_percent(0.01)
        ['<0.1%']

        >>> format_percent(1)
        ['1.0%']

        >>> format_percent(10)
        ['10.0%']

        >>> format_percent(99.986)
        ['>99.9%']

        >>> format_percent(100)
        ['100.0%']

        >>> format_percent([100, 0, 0.2])
        ['100.0%', '0%', '0.2%']

    Author: Vilmantas Gėgžna
    """
    if not isinstance(x, (list, pd.Series)):
        x = [x]
    # fmt: off
    x_formatted = [
        "0%" if i == 0
        else "<0.1%" if i < 0.05
        else ">99.9%" if 99.95 <= i < 100
        else f"{i:.1f}%"
        for i in x
    ]
    # fmt: on

    if isinstance(x, pd.Series):
        return pd.Series(x_formatted, index=x.index)
    else:
        return x_formatted


def counts_to_percentages(x: pd.Series, name: str = "percent") -> pd.Series:
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
        0    <0.1%
        1       0%
        2    10.0%
        3    20.0%
        4    10.0%
        5    50.0%
        6    10.0%
        Name: percent, dtype: object

        >>> counts_to_percentages(pd.Series([1, 0, 10000]))
        0     <0.1%
        1        0%
        2    >99.9%
        Name: percent, dtype: object

    Author: Vilmantas Gėgžna
    """
    return format_percent(x / x.sum() * 100).rename(name)


def extract_number(text: str) -> float:
    return float(re.findall(r"-?\d+[.]?\d*", str(text))[0])


# Display -------------------------------------------------------------------
def highlight_max(s, color="green"):
    """Helper function to highlight the maximum in a Series or DataFrame

    Args:
        s: Numeric values one of which will be highlighted.
        color (str, optional): Text highlight color. Defaults to 'green'.
    Examples:
    >>> import pandas as pd
    >>> data_frame = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    >>> data_frame.style.apply(highlight_max)

    >>> data_frame.style.format(precision=2).apply(highlight_max)
    """
    is_max = s == s.max()
    return [f"color: {color}" if cell else "" for cell in is_max]


def highlight_rows_by_index(x, values, color="green"):
    """Highlight rows/columns with certain index/column name.

    Args.:
        x (pandas.DataFrame): Dataframe to highlight.
        values (list): List of index/column names to highlight.
        color (str, optional): Text highlight color. Defaults to 'green'.

    Examples:
    >>> iris.head(10).style.apply(highlight_rows, values=[8, 9], axis=1)
    """
    return [f"color: {color}" if (x.name in values) else "" for i in x]


# Function to change text color for data types with 'int' or 'float'
def highlight_int_float_text(value, color="deepskyblue"):
    if "int" in str(value) or "float" in str(value):
        return f"color: {color}"
    else:
        return ""


# Function to change text color for data types with 'object'
def highlight_category_text(value, color="limegreen"):
    if "category" in str(value):
        return f"color: {color}"
    else:
        return ""


def highlight_value(value, when, color="grey"):
    if value == when:
        return f"color: {color}"
    else:
        return ""


def highlight_between(value, min: float = None, max: float = None, color="yellow"):
    if min <= value <= max:
        return f"color: {color}"
    else:
        return ""


def highlight_above(value, min: float = None, color="yellow"):
    if min < value:
        return f"color: {color}"
    else:
        return ""


def highlight_above_str(value, min: float = None, color="yellow"):
    if min < extract_number(value):
        return f"color: {color}"
    else:
        return ""


def highlight_below(value, max: float = None, color="yellow"):
    if value < max:
        return f"color: {color}"
    else:
        return ""


def highlight_below_str(value, max: float = None, color="yellow"):
    if extract_number(value) < max:
        return f"color: {color}"
    else:
        return ""


# For data wrangling --------------------------------------------------------
# Index
def use_numeric_index(self, start=1):
    """Create a new sequential index that starts at indicated number.

    Args.:
        self (pd.DataFrame):
            The object the method is applied to.
        start_at (int):
            The start of an index

    Return:
        pandas.DataFrame
    """
    i = self.index
    self.index = range(start, len(i) + start)
    return self


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


# Data types
# Function to convert 0/1 to No/Yes
def convert_01_to_no_yes(x):
    dtype_no_yes = pd.CategoricalDtype(categories=["No", "Yes"], ordered=True)
    return x.replace({0: "No", 1: "Yes"}).astype(dtype_no_yes)


# Function to convert False/True to No/Yes
def convert_bool_to_no_yes(x):
    return x.astype(int).pipe(convert_01_to_no_yes)


# Function to convert No/Yes to 0/1
def convert_no_yes_to_01(x):
    return x.replace({"No": 0, "Yes": 1}).astype(np.int8)


# Function to convert False/True to 0/1
def convert_bool_to_01(x):
    return x.astype(np.int8)


# Function to convert No/Yes to -1/1
def convert_no_yes_to_mp1(x):
    return x.replace({"No": -1, "Yes": 1}).astype(np.int8)


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
        ax_add_value_labels_ab(ax, labels=counts[label], rotation=label_rotation)
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
    ax.xaxis.set_major_locator(MaxNLocator(min_n_ticks=min_n_ticks, integer=True))
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


if __name__ == "__main__":
    # doctest
    import doctest

    doctest.testmod()
