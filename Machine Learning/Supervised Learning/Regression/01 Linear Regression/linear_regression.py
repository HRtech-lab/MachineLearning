"""Linear Regression

How much of the variation in strength can a straight-line model explain? Each ingredient gets one coefficient, with age entering as another additive term.

scikit-learn, LinearRegression
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
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
from sklearn.linear_model import LinearRegression
from gallery.tabular import concrete_split, regression_report

model = LinearRegression()


def main():
    X, X_test, y, y_test, groups = concrete_split()

    predictions = regression_report(model, X, X_test, y, y_test, groups)
    import pandas as pd

    print(
        pd.Series(model.coef_, index=X.columns, name="MPa per original unit").round(3).to_string()
    )
    plt.show()


if __name__ == "__main__":
    main()
