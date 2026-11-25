"""XGBoost

Here XGBoost optimises binary logistic loss using shallow regularised trees. All tuning constants are fixed before splitting the data.

Chen and Guestrin, XGBoost (2016)
https://doi.org/10.1145/2939672.2939785
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
from xgboost import XGBClassifier
from gallery.tabular import diagnostic_split, classification_report

model = XGBClassifier(
    n_estimators=110,
    max_depth=3,
    learning_rate=0.08,
    eval_metric="logloss",
    n_jobs=1,
    random_state=17,
)


def main():
    X, X_test, y, y_test = diagnostic_split()

    probability = classification_report(model, X, X_test, y, y_test)
    plt.show()


if __name__ == "__main__":
    main()
