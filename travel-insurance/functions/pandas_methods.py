"""Additional methods for Pandas Series and DataFrames"""

# Setup -----------------------------------------------------------------
import pandas as pd
import pandas_flavor as pf
from typing import Union

import functions.fun_utils as my  # Custom module

# Series methods --------------------------------------------------------
@pf.register_series_method
def to_df(
    self: pd.Series,
    values_name: str = None,
    key_name: Union[str, list[str], tuple[str, ...]] = None,
) -> pd.DataFrame:
    """Convert Series to DataFrame with desired or default column names.

    Similar to `pandas.Series.to_frame()`, but the main purpose of this method
    is to be used with the result of `.value_counts()`. So appropriate default
    column names are pre-defined. And index is always reset.

    Args:
        self (pandas.Series):
            The object the method is applied to.
        values_name (str):
            Name for series values (applied before conversion to DataFrame).
            Defaults "count".
        key_name (str or sequence of str):
            New name for the columns, that are created from Series index
            that was present before the conversion to DataFrame.
            Defaults to `self.index.names`, if index has names,
            to `self.name` if index has no names but series has name,
            or to "value" otherwise.

    Return:
        pandas.DataFrame

    Examples:
    >>> import pandas as pd
    >>> df = pd.Series({'right': 138409, 'left': 44733}).rename("foot")

    >>> df.to_df()

    >>> # Compared to .to_frame()
    >>> df.to_frame()
    """

    k_name = None
    v_name = None

    # Check if defaults can be set based on non-missing attribute values
    if my.index_has_names(self):
        k_name = self.index.names
        if self.name is not None:
            v_name = self.name
    else:
        k_name = self.name

    # Set user-defined values or defaults
    if key_name is not None:
        k_name = key_name
    elif k_name is None:
        k_name = "value"  # Default

    if values_name is not None:
        v_name = values_name
    elif v_name is None:
        v_name = "count"  # Default

    # Output
    return self.rename_axis(k_name).rename(v_name).reset_index()


# DataFrame methods --------------------------------------------------------
@pf.register_dataframe_method
def index_start_at(self, start=1):
    """Create a new sequential index that starts at indicated number.

    Args.:
        self (pd.DataFrame):
            The object the method is applied to.
        start (int):
            The start of an index

    Return:
        pandas.DataFrame
    """
    i = self.index
    self.index = range(start, len(i) + start)
    return self
