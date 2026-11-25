"""Gaussian Naive Bayes

Gaussian naive Bayes models each measurement separately within a diagnosis. Combining those densities assumes conditional independence.

scikit-learn, GaussianNB
https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
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
from sklearn.naive_bayes import GaussianNB
from gallery.tabular import diagnostic_split, classification_report

model = GaussianNB()


def main():
    X, X_test, y, y_test = diagnostic_split()

    probability = classification_report(model, X, X_test, y, y_test)
    plt.show()


if __name__ == "__main__":
    main()
