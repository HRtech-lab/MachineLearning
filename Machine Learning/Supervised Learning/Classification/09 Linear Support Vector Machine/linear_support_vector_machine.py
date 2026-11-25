"""Linear Support Vector Machine

A linear support vector machine seeks a wide margin between the two diagnoses. Sigmoid calibration uses only training folds.

scikit-learn, LinearSVC
https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from gallery.tabular import diagnostic_split, classification_report

model = CalibratedClassifierCV(
    make_pipeline(StandardScaler(), LinearSVC(C=0.4, max_iter=12000, random_state=17))
)


def main():
    X, X_test, y, y_test = diagnostic_split()

    probability = classification_report(model, X, X_test, y, y_test)
    plt.show()


if __name__ == "__main__":
    main()
