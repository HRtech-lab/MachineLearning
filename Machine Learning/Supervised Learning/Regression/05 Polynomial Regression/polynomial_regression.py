"""Polynomial Regression

Second-order terms allow the effect of water to change with cement content or curing age. Ridge limits the size of the expanded fit.

scikit-learn, PolynomialFeatures
https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
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
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from gallery.tabular import concrete_split, regression_report

model = make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False), Ridge(alpha=3))


def main():
    X, X_test, y, y_test, groups = concrete_split()

    predictions = regression_report(model, X, X_test, y, y_test, groups)
    plt.show()


if __name__ == "__main__":
    main()
