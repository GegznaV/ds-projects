"""Various functions for machine learning related tasks."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import functions.fun_utils as my  # Custom module

from IPython.display import display

from sklearn.metrics import (
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    cohen_kappa_score,
    recall_score,
    precision_score,
)


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
