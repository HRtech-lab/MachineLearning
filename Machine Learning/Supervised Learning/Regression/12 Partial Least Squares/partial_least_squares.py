"""Partial Least Squares

Partial least squares builds five directions that carry information about both the predictors and strength. The final regression uses these directions.

scikit-learn, PLSRegression
https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html
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
from sklearn.cross_decomposition import PLSRegression
from gallery.tabular import concrete_split, regression_report

model = PLSRegression(n_components=5)


def main():
    X, X_test, y, y_test, groups = concrete_split()

    predictions = regression_report(model, X, X_test, y, y_test, groups)
    plt.show()


if __name__ == "__main__":
    main()
