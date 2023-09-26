"""Various functions for machine learning related tasks."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import functions.fun_utils as my  # Custom module

from typing import Union
from IPython.display import display
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error as mse,
    r2_score,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    cohen_kappa_score,
    recall_score,
    precision_score,
    classification_report,
    confusion_matrix,
)


# Helpers
def as_formula(
    target: str = None,
    include: Union[list[str], pd.DataFrame] = None,
    exclude: list[str] = None,
    add: str = "",
):
    """
    Generates the R style formula for statsmodels (patsy) given
    the dataframe, dependent variable and optional excluded columns
    as strings.

    Args:
        target (str): name of target variable.
        include (pandas.DataFrame or list[str]):
            dataframe of column names to include.
        exclude (list[str], optional):
            columns to exclude.
        add (str): string to add to formula, e.g., "+ 0"

    Return:
        String with R style formula for `patsy` (e.g., "target ~ x1 + x2").

    See also: https://stackoverflow.com/a/44866142/4783029
    """
    if isinstance(include, pd.DataFrame):
        include = list(include.columns.values)

    if target in include:
        include.remove(target)

    if exclude is not None:
        for col in exclude:
            include.remove(col)

    return target + " ~ " + " + ".join(include) + add


def get_columns_by_purpose(data, target: str):
    """Split data frame into 3 data frames: for target, numeric, and remaining
    variables.

    Examples:
    >>> # Split
    >>> d_target, d_num, d_other = get_columns_by_purpose(data, "class")

    >>> # Merge back
    >>> pd.concat([d_target, d_num, d_other], axis=1)
    """
    d_num = data.drop(columns=target).select_dtypes("number")
    d_other = data.drop(columns=[target, *d_num.columns.values])

    return data[target].to_frame(), d_num, d_other


def get_metric_abbreviation_and_sign(scoring: str):
    """Internal function to parse scoring string and return
    abbreviation and sign.
    """
    sign = -1 if scoring.startswith("neg") else 1

    if scoring == "neg_root_mean_squared_error":
        metric = "RMSE"
    elif scoring == "balanced_accuracy":
        metric = "BAcc"
    elif scoring == "r2":
        metric = "R²"
    elif scoring == "f1":
        metric = "F1"
    else:
        metric = scoring
    return sign, metric


# Feature selection
def sfs(estimator, est_type, k_features="parsimonious", forward=True):
    """Create SFS instance for classification

    Args.:
        est_type (str): classification or regression
        other arguments: see mlextend.SequentialFeatureSelector()
    """
    if est_type == "regression":
        scoring = "neg_root_mean_squared_error"
    elif est_type == "classification":
        scoring = "balanced_accuracy"
    else:
        raise Exception(f"Unrecognized learner/estimator type: {type}")

    return SequentialFeatureSelector(
        estimator,
        k_features=k_features,  # "parsimonious",
        forward=forward,
        floating=False,
        scoring=scoring,
        verbose=1,
        cv=5,
        n_jobs=-1,
    )


def sfs_get_score(sfs_object, k_features):
    """Return performance score achieved with certain number of features.

    Args.:
        sfs_object: result of function do_sfs_lin_reg()
        k_features (int): number of features.
    """
    md = round(np.median(sfs_object.get_metric_dict()[k_features]["cv_scores"]), 3)
    return {
        "k_features": k_features,
        "mean_score": round(sfs_object.get_metric_dict()[k_features]["avg_score"], 3),
        "median_score": md,
        "sd_score": round(sfs_object.get_metric_dict()[k_features]["std_dev"], 3),
    }


def sfs_plot_results(sfs_object, sub_title="", ref_y=None):
    """Plot results from SFS object

    Args.:
      sfs_object: object with SFS results.
      sub_title (str): second line of title.
      ref_y (float): Y coordinate of reference line.
    """

    scoring = sfs_object.get_params()["scoring"]

    sign, metric = get_metric_abbreviation_and_sign(scoring)

    if sfs_object.forward:
        sfs_plot_title = "Forward Feature Selection"
    else:
        sfs_plot_title = "Backward Feature Elimination"

    fig, ax = plt.subplots(1, 2, sharey=True)

    xlab = "Number of predictors included"

    if ref_y is not None:
        ax[0].axhline(y=ref_y, color="darkred", linestyle="--", lw=0.5)
        ax[1].axhline(y=ref_y, color="darkred", linestyle="--", lw=0.5)

    avg_score = [
        (int(i), sign * c["avg_score"]) for i, c in sfs_object.subsets_.items()
    ]

    averages = pd.DataFrame(avg_score, columns=["k_features", "avg_score"])

    (
        averages.plot.scatter(
            x="k_features",
            y="avg_score",
            xlabel=xlab,
            ylabel=metric,
            title=f"Average {metric}",
            ax=ax[0],
        )
    )

    cv_scores = {int(i): sign * c["cv_scores"] for i, c in sfs_object.subsets_.items()}
    (
        pd.DataFrame(cv_scores).plot.box(
            xlabel=xlab,
            title=f"{metric} in CV splits",
            ax=ax[1],
        )
    )

    ax[0].xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax[1].xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    if not sfs_object.forward:
        ax[1].invert_xaxis()

    main_title = f"{sfs_plot_title} with {sfs_object.cv}-fold CV " + f"\n{sub_title}"

    fig.suptitle(main_title)
    plt.tight_layout()
    plt.show()

    # Print results
    if not sfs_object.interrupted_:
        if sfs_object.is_parsimonious:
            note = "[Parsimonious]"
            k_selected = f"k = {len(sfs_object.k_feature_names_)}"
            score_at_k = f"avg. {metric} = {sign * sfs_object.k_score_:.3f}"
            note_2 = "Smallest number of predictors at best ± 1 SE score"
        else:
            note = "[Best]"
            if sign < 0:
                best = averages.nsmallest(1, "avg_score")
            else:
                best = averages.nlargest(1, "avg_score")
            k_selected = f"k = {int(best.k_features.values)}"
            score_at_k = f"avg. {metric} = {float(best.avg_score.values):.3f}"
            note_2 = "Number of predictors at best score"

        print(f"{k_selected}, {score_at_k} {note}\n({note_2})")


def sfs_list_features(sfs_result):
    """List features by order when they were added.
    Current implementation correctly works with forward selection only.

    Args:
        sfs_result (SFS object)
    """

    scoring = sfs_result.get_params()["scoring"]

    sign, metric = get_metric_abbreviation_and_sign(scoring)

    feature_dict = sfs_result.get_metric_dict()
    lst = [[*feature_dict[i]["feature_names"]] for i in feature_dict]
    feature = []

    if sfs_result.forward:
        for x, y in zip(lst[0::], lst[1::]):
            feature.append(*set(y).difference(x))
        res = pd.DataFrame({
            "added_feature": [*lst[0], *feature],
            "metric": metric,
            "score": [sign * feature_dict[i]["avg_score"] for i in feature_dict],
        }).pipe(my.use_numeric_index)
    else:
        for x, y in zip(lst[0::], lst[1::]):
            feature.append(*set(x).difference(y))
        res = pd.DataFrame({
            "feature": [*feature, *lst[-1]],
            "metric": metric,
            "score": [sign * feature_dict[i]["avg_score"] for i in feature_dict],
        }).pipe(my.use_numeric_index)[::-1]

    return (
        res.assign(score_improvement=lambda x: sign * x.score.diff())
        .assign(score_percentage_change=lambda x: sign * x.score.pct_change() * 100)
        .rename_axis("step")
    )


# Functions for regression


def get_regression_performance(y_true, y_pred, name=""):
    """Evaluate regression model performance

    Calculate R², RMSE, and SD of predicted variable

    Args.:
      y_true, y_pred: true and predicted numeric values.
      name (str): the name of investigated set.
    """
    return (
        pd.DataFrame({
            "set": name,
            "n": len(y_true),
            "SD": [float(np.std(y_true))],
            "RMSE": [float(mse(y_true, y_pred, squared=False))],
            "R²": [float(r2_score(y_true, y_pred))],
        })
        .eval("RMSE_SD_ratio = RMSE/SD")
        .eval("SD_RMSE_ratio = SD/RMSE")
    )


# Functions for classification
def get_classification_scores(model, X, y):
    """Calculate scores of classification performance for a model.

    The following metrics are calculated:

    - No information rate
    - Accuracy
    - Balanced Accuracy (BAcc)
    - Balanced Accuracy adjusted to be between 0 and 1 (BAcc_01)
    - F1 Score
    - F1 macro average (F1_macro),
    - F1 weighted macro average (F1_weighted),
    - ROC AUC Value
    - Cohen's Kappa

    Args:
        model (object): Scikit-learn classifier.
        X (array-like): Predictor variables.
        y (array-like): Target variable.

    Returns:
        dict: Dictionary with classification scores.
    """

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    return {
        "n": len(y),
        "No_info_rate": max(y.mean(), 1 - y.mean()),
        "Accuracy": accuracy_score(y, y_pred),
        "BAcc": balanced_accuracy_score(y, y_pred),
        "BAcc_01": balanced_accuracy_score(y, y_pred, adjusted=True),
        "F1": f1_score(y, y_pred, pos_label=1),
        "F1_neg": f1_score(y, y_pred, pos_label=0),
        "TPR": recall_score(y, y_pred, pos_label=1),
        "TNR": recall_score(y, y_pred, pos_label=0),
        "PPV": precision_score(y, y_pred, pos_label=1),
        "NPV": precision_score(y, y_pred, pos_label=0),
        "Kappa": cohen_kappa_score(y, y_pred),
        "ROC_AUC": roc_auc_score(y, y_proba),
    }


def print_classification_scores(
    models,
    X,
    y,
    title="--- All data ---",
    color="green",
    precision=3,
    sort_by="No_info_rate",
):
    """Print classification scores for a set of models.

    Args:
        models (dictionary): A dictionary of models.
        X: predictor variables.
        y: target variable.
        title (str, optional): Title to print. Defaults to "--- All data ---".
        color (str, optional): Text highlight color. Defaults to "green".
        precision (int, optional): Number of digits after the decimal point.
        sort_by (str, optional): Column name to sort by.
            Defaults to "No_info_rate".
    """
    print(title)
    print(
        "No information rate: ",
        round(pd.Series(y).value_counts(normalize=True).max(), precision),
    )
    display(
        pd.DataFrame.from_dict(
            {
                name: get_classification_scores(model, X, y)
                for name, model in models.items()
            },
            orient="index",
        )
        .sort_values(sort_by, ascending=False)
        .style.format(precision=precision)
        .apply(my.highlight_max, color=color)
    )


# Classification ===========================================================


def classification_report_with_confusion_matrix(best_models, X, y, labels=[1, 0]):
    """
    Function to print a classification report with a confusion matrix.

    Reference to interpret confusion matrix:

    - rows: true class
    - columns: predicted class
    - cells contain counts. Usually, the meaning is as follows:
    ```
    TN FP
    FN TP
    ```
    (T - true, F - false, P - positive, N - negative)

    Args:
        best_models (dict): Dictionary containing the best models.
        X (pd.DataFrame): Dataframe containing the features.
        y (pd.Series): Series containing the target.
        labels (list): List containing the labels for confusion matrix.
    """
    [
        [
            y_pred := model.predict(X),
            print(
                "<<< " + name + " >>>",
                "\n",
                classification_report(y, y_pred),
                "\n",
                "Confusion matrix (order of labels is " + str(labels) + "): \n",
                "rows = true labels, columns = predicted labels \n",
                confusion_matrix(y, y_pred, labels=labels),
                "\n",
                "\n\n",
            ),
        ]
        for name, model in best_models.items()
    ]


# PCA ---------------------------------------------------------------------------
def pca_screeplot(data, n_components=30):
    """Plot PCA screeplot

    Args:
        data (pandas.Dataframe): Numeric data
        n_components (int, optional):
            Max number of principal components to extract.
            Defaults to 30.

    Returns:
        3 objects: plot (fig and ax) and pca object.
    """
    scale = StandardScaler()
    pca = PCA(n_components=n_components)

    scaled_data = scale.fit_transform(data)
    pca.fit(scaled_data)

    pct_explained = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(pct_explained, "-o", color="tab:green")

    ax.set_xlabel("Number of components")
    ax.set_ylabel("% of explained variance")

    return fig, ax, pca


def do_pca(data, target: str, n_components: int = 10, scale=None, pca=None):
    """Do PCA on numeric non-target variables

    Args:
        data (pandas.Dataframe): data
        target (str): Target variable name
        n_components (int, optional):
            Number of PCA components to extract.
            Defaults to 10.
            n_components is ignored if `pca` is not None.
        scale (instance of sklearn.preprocessing.StandardScaler or None):
            Fitted object to scale data.
        pca (instance of sklearn.decomposition.PCA or None):
            Fitted PCA object.

    Returns:
        tuple with 6 elements:
          - 4 data frames: d_target, d_num, d_other, d_pca
          - fitted instance of sklearn.preprocessing.StandardScaler.
          - fitted instance of sklearn.decomposition.PCA.
    """
    d_target, d_num, d_other = get_columns_by_purpose(data, target)

    if scale is None:
        scale = StandardScaler()
        sc_data = scale.fit_transform(d_num)
    else:
        sc_data = scale.transform(d_num)

    if pca is None:
        pca = PCA(n_components=n_components)
        pc_num = pca.fit_transform(sc_data)
    else:
        pc_num = pca.transform(sc_data)
        n_components = pc_num.shape[1]

    # Convert to DataFrame and name columns (pc_1, pc_2, etc.)
    d_pca = pd.DataFrame(
        pc_num,
        index=d_num.index,
        columns=[f"pc_{i}" for i in np.arange(1, n_components + 1)],
    )

    return (d_target, d_num, d_other, d_pca, scale, pca)
