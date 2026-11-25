"""Gradient Boosting

Small boosted trees successively improve the current log-odds estimate. Two split levels permit limited feature interactions.

scikit-learn, GradientBoostingClassifier
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
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
from sklearn.ensemble import GradientBoostingClassifier
from gallery.tabular import diagnostic_split, classification_report

model = GradientBoostingClassifier(
    n_estimators=110, learning_rate=0.06, max_depth=2, random_state=17
)


def main():
    X, X_test, y, y_test = diagnostic_split()

    probability = classification_report(model, X, X_test, y, y_test)
    plt.show()


if __name__ == "__main__":
    main()
