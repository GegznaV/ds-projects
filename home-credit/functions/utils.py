"""Ad-hoc functions and classes for the project.

NOTE: The following functions are on a separate file as this is required for
correct pickling (saving the final pre-processing pipeline and the model).
Otherwise, the pipeline and the model cannot be loaded from the pickle file on
the cloud."""

import numpy as np
import pandas as pd
import re
from sklearn.base import BaseEstimator, TransformerMixin


# Common variables ============================================================


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

    def __init__(self, keep, strict=True):
        """Constructor

        Args.:
            keep (list): List of column names to keep.
            strict (bool): If True, raises an error if the indicated columns
                are not found in the input DataFrame.
        """
        self.keep = keep
        self.strict = strict

    def fit(self, X, y=None):
        if self.strict:
            # Check that the indicated columns are in the input DataFrame
            assert set(self.keep).issubset(X.columns)
        else:
            # Remove columns not in the input DataFrame
            self.keep = [col for col in self.keep if col in X.columns]

        return self

    def transform(self, X):
        # Select the indicated features from the input DataFrame X
        selected_features = X[self.keep]
        return pd.DataFrame(selected_features, columns=self.keep)

    def get_feature_names_out(self):
        pass


# Custom transformer to clean column names after one-hot encoding
class CleanColumnNames(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X.columns = [
            re.sub(r"_+", "_", "".join(c if c.isalnum() else "_" for c in str(col)))
            for col in X.columns
        ]
        return X

    def get_feature_names_out(self):
        pass
