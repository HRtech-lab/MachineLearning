"""Bisecting K-Means

Bisecting k-means builds a hierarchy by repeatedly splitting one current cluster into two.

scikit-learn, BisectingKMeans
https://scikit-learn.org/stable/modules/generated/sklearn.cluster.BisectingKMeans.html
scikit-learn, make_blobs
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html
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
from sklearn.cluster import BisectingKMeans
from gallery.unsupervised import cluster_data, cluster_report


def main():
    X, reference = cluster_data("blobs")
    model = BisectingKMeans(n_clusters=3, n_init=8, random_state=17)
    labels = model.fit_predict(X)

    cluster_report(X, reference, labels)
    plt.show()


if __name__ == "__main__":
    main()
