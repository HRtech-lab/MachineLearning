"""Perceptron

The perceptron updates when a training point is misclassified. A sigmoid fitted inside the training folds supplies probabilities for the evaluation.

scikit-learn, Perceptron
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html
Wolberg, Mangasarian, Street and Street, Breast Cancer Wisconsin (Diagnostic), UCI, CC BY 4.0
https://doi.org/10.24432/C5DW2B"""

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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import Perceptron
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from gallery.tabular import diagnostic_split, classification_report

model = CalibratedClassifierCV(
    make_pipeline(
        StandardScaler(), Perceptron(alpha=0.0002, penalty="l2", max_iter=1500, random_state=17)
    )
)


def main():
    X, X_test, y, y_test = diagnostic_split()

    probability = classification_report(model, X, X_test, y, y_test)
    plt.show()


if __name__ == "__main__":
    main()
