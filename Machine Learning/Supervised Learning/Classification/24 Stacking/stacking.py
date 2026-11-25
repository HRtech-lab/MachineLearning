"""Stacking

A second logistic model learns from out-of-fold predictions made by a linear model and a forest. Its training inputs exclude in-sample base predictions.

scikit-learn, StackingClassifier
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html
Wolberg, Mangasarian, Street and Street, Breast Cancer Wisconsin (Diagnostic), UCI, CC BY 4.0
https://doi.org/10.24432/C5DW2B"""

from pathlib import Path
import sys

start = Path(__file__).resolve().parent
for parent in (start, *start.parents):
    if (parent / "gallery" / "__init__.py").is_file():
        sys.path.insert(0, str(parent))
        break
else:
    raise RuntimeError("Open this example inside the Machine Learning collection.")


import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from gallery.tabular import diagnostic_split, classification_report

model = StackingClassifier(
    [
        ("logit", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1500))),
        ("forest", RandomForestClassifier(n_estimators=60, random_state=17)),
    ],
    final_estimator=LogisticRegression(),
    cv=4,
)


def main():
    X, X_test, y, y_test = diagnostic_split()

    probability = classification_report(model, X, X_test, y, y_test)
    plt.show()


if __name__ == "__main__":
    main()
