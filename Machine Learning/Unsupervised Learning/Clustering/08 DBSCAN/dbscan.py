"""DBSCAN

A chain of densely packed neighbours can follow a curved crescent. DBSCAN tests that idea directly on two noisy moons.

scikit-learn, DBSCAN
https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
scikit-learn, make_moons
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html
scikit-learn, clustering algorithms and evaluation
https://scikit-learn.org/stable/modules/clustering.html"""

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
from sklearn.cluster import DBSCAN
from gallery.unsupervised import cluster_data, cluster_report


def main():
    X, reference = cluster_data("moons")
    model = DBSCAN(eps=0.17, min_samples=7)
    labels = model.fit_predict(X)

    cluster_report(X, reference, labels)
    plt.show()


if __name__ == "__main__":
    main()
