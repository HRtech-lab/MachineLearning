"""Stochastic Gradient Classifier

Online-style updates optimise logistic loss. Coefficient averaging reduces sensitivity to the last few observations in an epoch.

scikit-learn, SGDClassifier
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
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
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from gallery.tabular import diagnostic_split, classification_report

model = make_pipeline(
    StandardScaler(),
    SGDClassifier(loss="log_loss", alpha=0.001, average=True, max_iter=2500, random_state=17),
)


def main():
    X, X_test, y, y_test = diagnostic_split()

    probability = classification_report(model, X, X_test, y, y_test)
    plt.show()


if __name__ == "__main__":
    main()
