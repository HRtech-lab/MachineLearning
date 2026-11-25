"""XGBoost

XGBoost fits regularised trees using gradient and curvature information from the loss. This run uses squared error in MPa.

Chen and Guestrin, XGBoost (2016)
https://doi.org/10.1145/2939672.2939785
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
from xgboost import XGBRegressor
from gallery.tabular import concrete_split, regression_report

model = XGBRegressor(
    n_estimators=140,
    max_depth=4,
    learning_rate=0.07,
    objective="reg:squarederror",
    n_jobs=1,
    random_state=17,
)


def main():
    X, X_test, y, y_test, groups = concrete_split()

    predictions = regression_report(model, X, X_test, y, y_test, groups)
    plt.show()


if __name__ == "__main__":
    main()
