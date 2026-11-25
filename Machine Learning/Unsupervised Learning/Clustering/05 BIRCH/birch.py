"""BIRCH

BIRCH summarises nearby observations in a tree of small subclusters before assigning the final three labels.

scikit-learn, Birch
https://scikit-learn.org/stable/modules/generated/sklearn.cluster.Birch.html
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
from sklearn.cluster import Birch
from gallery.unsupervised import cluster_data, cluster_report


def main():
    X, reference = cluster_data("blobs")
    model = Birch(threshold=0.38, branching_factor=25, n_clusters=3)
    labels = model.fit_predict(X)

    cluster_report(X, reference, labels)
    plt.show()


if __name__ == "__main__":
    main()
