"""Isolation Forest

An isolation forest gives a high anomaly score to observations that are separated with few random cuts.

scikit-learn, IsolationForest
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
scikit-learn, novelty and outlier detection
https://scikit-learn.org/stable/modules/outlier_detection.html"""

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
from sklearn.ensemble import IsolationForest
from gallery.unsupervised import anomaly_data, anomaly_report


def main():
    X_train, X_test, y_test = anomaly_data()
    model = IsolationForest(n_estimators=110, contamination=0.05, random_state=17).fit(X_train)

    anomaly_report(model, X_test, y_test)
    plt.show()


if __name__ == "__main__":
    main()
