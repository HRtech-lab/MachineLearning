"""Local Outlier Factor

Local outlier factor compares the density near each new observation with the densities around its training neighbours.

scikit-learn, LocalOutlierFactor
https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html
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
from sklearn.neighbors import LocalOutlierFactor
from gallery.unsupervised import anomaly_data, anomaly_report


def main():
    X_train, X_test, y_test = anomaly_data()
    model = LocalOutlierFactor(n_neighbors=24, novelty=True, contamination=0.05).fit(X_train)

    anomaly_report(model, X_test, y_test)
    plt.show()


if __name__ == "__main__":
    main()
