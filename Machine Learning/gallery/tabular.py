"""Common splits and diagnostics for the supervised examples."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics as met
from sklearn.datasets import load_breast_cancer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)

SEED = 17
BLUE = "#28618a"
RUST = "#b64f24"


def concrete_split():
    data = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "concrete.csv")
    X = data.drop(columns="strength")
    y = data.strength
    # Repeated measurements of one mix at different ages belong together.
    groups = X.drop(columns="age").round(1).groupby(list(X.columns[:-1]), sort=False).ngroup()
    split = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    train, test = next(split.split(X, y, groups))
    assert set(groups.iloc[train]).isdisjoint(groups.iloc[test])
    return X.iloc[train], X.iloc[test], y.iloc[train], y.iloc[test], groups.iloc[train]


def diagnostic_split():
    ds = load_breast_cancer(as_frame=True)
    y = (ds.target == list(ds.target_names).index("malignant")).astype(int)
    return train_test_split(ds.data, y, test_size=0.25, stratify=y, random_state=SEED)


def regression_report(model, X, X_test, y, y_test, groups):
    folds = GroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = cross_validate(
        model,
        X,
        y,
        groups=groups,
        cv=folds,
        scoring=["r2", "neg_mean_absolute_error"],
        error_score="raise",
    )
    print(
        f"Training: {len(y)} observations across {groups.nunique()} recipes. Test: {len(y_test)} observations."
    )
    print(
        f"Training folds, mean R² {scores['test_r2'].mean():.3f}, MAE {-scores['test_neg_mean_absolute_error'].mean():.2f} MPa"
    )
    model.fit(X, y)
    pred = np.asarray(model.predict(X_test)).ravel()
    mean_pred = DummyRegressor().fit(X, y).predict(X_test)

    def measures(p):
        return {
            "R²": met.r2_score(y_test, p),
            "MAE (MPa)": met.mean_absolute_error(y_test, p),
            "RMSE (MPa)": met.root_mean_squared_error(y_test, p),
            "Median AE (MPa)": met.median_absolute_error(y_test, p),
            "Bias, pred − obs (MPa)": np.mean(p - y_test),
        }

    print(
        pd.DataFrame([measures(pred), measures(mean_pred)], index=["Model", "Training mean"])
        .round(3)
        .to_string()
    )
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.2, 3.15), constrained_layout=True)
    a.scatter(y_test, pred, s=13, alpha=0.62, color=BLUE)
    lo, hi = min(y_test.min(), pred.min()), max(y_test.max(), pred.max())
    a.plot([lo, hi], [lo, hi], "--", lw=1, color=RUST)
    a.set(
        xlabel="Measured strength (MPa)",
        ylabel="Predicted strength (MPa)",
        title="Held-out concrete mixes",
    )
    b.scatter(pred, y_test - pred, s=13, alpha=0.62, color=BLUE)
    b.axhline(0, ls="--", lw=1, color=RUST)
    b.set(xlabel="Predicted strength (MPa)", ylabel="Measured − predicted (MPa)", title="Residuals")
    return pred


def classification_report(model, X, X_test, y, y_test):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = cross_validate(
        model, X, y, cv=cv, scoring=["balanced_accuracy", "roc_auc"], error_score="raise"
    )
    print(f"Training: {len(y)} records. Test: {len(y_test)} records. Positive class: malignant.")
    print(
        f"Training folds, mean balanced accuracy {scores['test_balanced_accuracy'].mean():.3f}, ROC AUC {scores['test_roc_auc'].mean():.3f}"
    )
    model.fit(X, y)
    probability = model.predict_proba(X_test)[:, 1]
    labels = model.predict(X_test)
    baseline = DummyClassifier(strategy="prior").fit(X, y)

    def measures(label, prob):
        return {
            "Accuracy": met.accuracy_score(y_test, label),
            "Balanced accuracy": met.balanced_accuracy_score(y_test, label),
            "Precision": met.precision_score(y_test, label, zero_division=0),
            "Recall": met.recall_score(y_test, label),
            "F1": met.f1_score(y_test, label),
            "MCC": met.matthews_corrcoef(y_test, label),
            "ROC AUC": met.roc_auc_score(y_test, prob),
            "Average precision": met.average_precision_score(y_test, prob),
            "Log loss": met.log_loss(y_test, prob),
            "Brier score": met.brier_score_loss(y_test, prob),
        }

    print(
        pd.DataFrame(
            {
                "Model": measures(labels, probability),
                "Training prior": measures(
                    baseline.predict(X_test), baseline.predict_proba(X_test)[:, 1]
                ),
            }
        )
        .round(3)
        .to_string()
    )
    print(
        met.classification_report(
            y_test, labels, target_names=["Benign", "Malignant"], digits=3, zero_division=0
        )
    )
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(9.1, 2.9), constrained_layout=True)
    met.ConfusionMatrixDisplay.from_predictions(
        y_test, labels, display_labels=["Benign", "Malignant"], cmap="Blues", colorbar=False, ax=a
    )
    a.set_title("Held-out counts")
    fpr, tpr, _ = met.roc_curve(y_test, probability)
    b.plot(fpr, tpr, color=BLUE)
    b.plot([0, 1], [0, 1], "--", color="0.6", lw=1)
    b.set(
        title="ROC",
        xlabel="False positive rate",
        ylabel="True positive rate",
        xlim=(0, 1),
        ylim=(0, 1.03),
    )
    precision, recall, _ = met.precision_recall_curve(y_test, probability)
    c.step(recall, precision, where="post", color=RUST)
    c.axhline(np.mean(y_test), ls="--", color="0.6", lw=1)
    c.set(
        title="Precision–recall", xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.03)
    )
    return probability
