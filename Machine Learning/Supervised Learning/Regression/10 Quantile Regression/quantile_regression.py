"""Quantile Regression

Median regression estimates the conditional middle of the strength distribution. Absolute and pinball losses are natural ways to judge it.

scikit-learn, QuantileRegressor
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.QuantileRegressor.html
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
from sklearn.linear_model import QuantileRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from gallery.tabular import concrete_split, regression_report

model = make_pipeline(
    StandardScaler(), QuantileRegressor(quantile=0.5, alpha=0.025, solver="highs")
)


def main():
    X, X_test, y, y_test, groups = concrete_split()

    predictions = regression_report(model, X, X_test, y, y_test, groups)
    from sklearn.metrics import mean_pinball_loss

    print(f"Median pinball loss: {mean_pinball_loss(y_test, predictions, alpha=0.5):.3f} MPa")
    print(f"Fraction below fitted median: {(y_test <= predictions).mean():.3f}")
    plt.show()


if __name__ == "__main__":
    main()
