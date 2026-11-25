"""Gaussian Process

A Gaussian process gives a predictive mean and standard deviation for each recipe. The length scale and observation noise are fixed here.

scikit-learn, GaussianProcessRegressor
https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html
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
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from gallery.tabular import concrete_split, regression_report

model = make_pipeline(
    StandardScaler(),
    GaussianProcessRegressor(kernel=RBF(2.5) + WhiteKernel(0.09), optimizer=None, normalize_y=True),
)


def main():
    X, X_test, y, y_test, groups = concrete_split()

    predictions = regression_report(model, X, X_test, y, y_test, groups)
    from scipy.stats import norm

    centre, sd = model.predict(X_test, return_std=True)
    radius = norm.ppf(0.95) * sd
    covered = (abs(y_test - centre) <= radius).mean()
    print(f"90% interval coverage: {covered:.3f}; mean width: {2 * radius.mean():.2f} MPa")
    plt.show()


if __name__ == "__main__":
    main()
