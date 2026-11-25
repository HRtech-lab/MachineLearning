"""Spectral Clustering

A Gaussian affinity graph gives nearby points on each ring much larger edge weights. Spectral clustering uses the graph eigenvectors to separate the inner and outer circles.

scikit-learn, SpectralClustering
https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html
scikit-learn, make_circles
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_circles.html
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
from sklearn.cluster import SpectralClustering
from gallery.unsupervised import cluster_data, cluster_report


def main():
    X, reference = cluster_data("rings")
    model = SpectralClustering(
        n_clusters=2, affinity="rbf", gamma=32, assign_labels="cluster_qr", random_state=17
    )
    labels = model.fit_predict(X)

    cluster_report(X, reference, labels)
    plt.show()


if __name__ == "__main__":
    main()
