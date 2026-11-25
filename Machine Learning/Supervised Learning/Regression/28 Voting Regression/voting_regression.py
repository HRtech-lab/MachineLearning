"""Voting Regression

This ensemble averages a linear estimate with two tree-based estimates. Its weights are fixed and each component sees the same training data.

scikit-learn, VotingRegressor
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingRegressor.html
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
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from gallery.tabular import concrete_split, regression_report

model = VotingRegressor(
    [
        ("ridge", make_pipeline(StandardScaler(), Ridge(alpha=3))),
        ("forest", RandomForestRegressor(n_estimators=70, min_samples_leaf=2, random_state=17)),
        ("boost", HistGradientBoostingRegressor(max_iter=90, random_state=17)),
    ]
)


def main():
    X, X_test, y, y_test, groups = concrete_split()

    predictions = regression_report(model, X, X_test, y, y_test, groups)
    plt.show()


if __name__ == "__main__":
    main()
