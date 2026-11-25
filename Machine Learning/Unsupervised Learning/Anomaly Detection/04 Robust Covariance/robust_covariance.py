"""Robust Covariance

A robust covariance estimate defines an elliptical boundary. This geometry is well matched to the clean training distribution.

scikit-learn, EllipticEnvelope
https://scikit-learn.org/stable/modules/generated/sklearn.covariance.EllipticEnvelope.html
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
from sklearn.covariance import EllipticEnvelope
from gallery.unsupervised import anomaly_data, anomaly_report


def main():
    X_train, X_test, y_test = anomaly_data()
    model = EllipticEnvelope(contamination=0.05, random_state=17).fit(X_train)

    anomaly_report(model, X_test, y_test)
    plt.show()


if __name__ == "__main__":
    main()
