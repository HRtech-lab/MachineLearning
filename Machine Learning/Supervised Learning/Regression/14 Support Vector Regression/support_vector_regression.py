"""Support Vector Regression

An RBF kernel lets the fitted surface bend. The epsilon tube tolerates small deviations in strength before adding a loss.

scikit-learn, SVR
https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from gallery.tabular import concrete_split, regression_report

model = make_pipeline(StandardScaler(), SVR(C=45, epsilon=2.0, gamma="scale"))


def main():
    X, X_test, y, y_test, groups = concrete_split()

    predictions = regression_report(model, X, X_test, y, y_test, groups)
    plt.show()


if __name__ == "__main__":
    main()
