"""Various functions for machine learning related tasks."""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Machine learning
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    cohen_kappa_score,
)

import functions.fun_utils as my  # Custom module

# Helpers
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

    cv_scores = {
        int(i): sign * c["cv_scores"] for i, c in sfs_object.subsets_.items()
    }
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

    main_title = (
        f"{sfs_plot_title} with {sfs_object.cv}-fold CV " + f"\n{sub_title}"
    )

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
        res = pd.DataFrame(
            {
                "added_feature": [*lst[0], *feature],
                "metric": metric,
                "score": [
                    sign * feature_dict[i]["avg_score"] for i in feature_dict
                ],
            }
        ).index_start_at(1)
    else:
        for x, y in zip(lst[0::], lst[1::]):
            feature.append(*set(x).difference(y))
        res = pd.DataFrame(
            {
                "feature": [*feature, *lst[-1]],
                "metric": metric,
                "score": [
                    sign * feature_dict[i]["avg_score"] for i in feature_dict
                ],
            }
        ).index_start_at(1)[::-1]

    return (
        res.assign(score_improvement=lambda x: sign * x.score.diff())
        .assign(
            score_percentage_change=lambda x: sign * x.score.pct_change() * 100
        )
        .rename_axis("step")
    )


# Functions for regression/classification
def get_classification_scores(model, X, y):
    """Calculate scores of classification performance for a model.
    The following metrics are calculated:
    - accuracy
    - balanced accuracy
    - Cohen's kappa
    - F1 score
    - ROC AUC value

    Args:
        model: Scikit-learn classifier.
        X: predictor variables.
        y: target variable.

    Returns:
        dictionary with classification scores.
    """

    y_pred = model.predict(X)
    return {
        "Accuracy": accuracy_score(y, y_pred),
        "BAcc": balanced_accuracy_score(y, y_pred),
        "Kappa": cohen_kappa_score(y, y_pred),
        "F1": f1_score(y, y_pred),
        "ROC AUC": roc_auc_score(y, y_pred),
    }


def print_classification_scores(
    models, X, y, title="--- All data ---", color="green", precision=3
):
    """Print classification scores for a set of models.

    Args:
        models (dictionary): A dictionary of models.
        X: predictor variables.
        y: target variable.
        title (str, optional): Title to print. Defaults to "--- All data ---".
        color (str, optional): Text highlight color. Defaults to "green".
        precision (int, optional): Number of digits after the decimal point.
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
        .style.format(precision=precision)
        .apply(my.highlight_max, color=color)
    )
