"""Ad-hoc functions and classes for the project.

NOTE: The following functions are on a separate file as this is required for
correct pickling (saving the final pre-processing pipeline and the model).
Otherwise, the pipeline and the model cannot be loaded from the pickle file on
the cloud."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# Common variables ============================================================
work_categories = [
    "< 1 year",
    "1 year",
    "2 years",
    "3 years",
    "4 years",
    "5 years",
    "6 years",
    "7 years",
    "8 years",
    "9 years",
    "10+ years",
]

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

    def get_feature_names_out(self):
        pass


class PreprocessorForGrades(BaseEstimator, TransformerMixin):
    """Transformer for the loan grade prediction."""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.assign(
            # Extract features
            issue_d=lambda df: pd.to_datetime(df["issue_d"], format="%b-%Y"),
            issue_month=lambda df: df["issue_d"].dt.month.astype("Int8"),
        )
        return X.drop(columns=["issue_d"])

    def get_feature_names_out(self):
        pass


class PreprocessorForSubgrades(BaseEstimator, TransformerMixin):
    """Transformer for the loan subgrade prediction."""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.assign(
            # Extract features
            title_len=lambda df: df["title"].str.len().fillna(0).astype("Int16"),
            issue_d=lambda df: pd.to_datetime(df["issue_d"], format="%b-%Y"),
            issue_month=lambda df: df["issue_d"].dt.month.astype("Int8"),
        )
        return X.drop(columns=["issue_d", "title"])

    def get_feature_names_out(self):
        pass


class PreprocessorForInterestRates(BaseEstimator, TransformerMixin):
    """Transformer for the loan interest rate prediction."""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.assign(
            # Extract features
            issue_d=lambda df: pd.to_datetime(df["issue_d"], format="%b-%Y"),
            issue_month=lambda df: df["issue_d"].dt.month.astype("Int8"),
        )
        return X.drop(columns=["issue_d"])

    def get_feature_names_out(self):
        pass
