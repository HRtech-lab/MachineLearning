"""Bagging

Bootstrap aggregation averages trees fitted to resampled training mixes. Individual trees can disagree sharply near a split boundary.

scikit-learn, BaggingRegressor
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html
I-Cheng Yeh, Concrete Compressive Strength (UCI), CC BY 4.0
https://doi.org/10.24432/C5PK67"""

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
from sklearn.ensemble import BaggingRegressor
from gallery.tabular import concrete_split, regression_report

model = BaggingRegressor(n_estimators=80, max_samples=0.85, random_state=17, n_jobs=1)


def main():
    X, X_test, y, y_test, groups = concrete_split()

    predictions = regression_report(model, X, X_test, y, y_test, groups)
    plt.show()


if __name__ == "__main__":
    main()
