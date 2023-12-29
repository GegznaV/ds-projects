"""Ad-hoc functions and classes for the project.

NOTE: The following functions are on a separate file as this is required for
correct pickling (saving the final pre-processing pipeline and the model).
Otherwise, the pipeline and the model cannot be loaded from the pickle file on
the cloud."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import set_config

set_config(transform_output="pandas")
# Pre-processing pipeline ====================================================


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Keeps only the indicated DataFrame columns
    and drops the rest.

    Attributes:
        feature_names (list): List of column names to keep.

    Methods:
        fit(X, y=None):
            Fit method (Returns self).
        transform(X):
            Transform method to select columns of interest.
            Returns a DataFrame with the selected columns only.
    """

    def __init__(self, keep):
        """Constructor

        Args.:
            keep (list): List of column names to keep.
        """
        self.keep = keep

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Select the indicated features from the input DataFrame X
        selected_features = X[self.keep]
        return pd.DataFrame(selected_features, columns=self.keep)
