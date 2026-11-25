"""Stochastic Gradient Regression

Stochastic gradient descent updates a linear model a few observations at a time. Averaging the iterates reduces noise in the final coefficients.

scikit-learn, SGDRegressor
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html
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
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from gallery.tabular import concrete_split, regression_report

model = make_pipeline(
    StandardScaler(), SGDRegressor(max_iter=2500, tol=1e-5, average=True, random_state=17)
)


def main():
    X, X_test, y, y_test, groups = concrete_split()

    predictions = regression_report(model, X, X_test, y, y_test, groups)
    plt.show()


if __name__ == "__main__":
    main()
