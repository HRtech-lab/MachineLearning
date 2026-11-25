"""Mean Shift

Mean shift follows local averages towards density peaks. The bandwidth chooses the spatial scale of a peak.

scikit-learn, MeanShift
https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html
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
from sklearn.cluster import MeanShift
from gallery.unsupervised import cluster_data, cluster_report


def main():
    X, reference = cluster_data("blobs")
    model = MeanShift(bandwidth=1.25, bin_seeding=True)
    labels = model.fit_predict(X)

    cluster_report(X, reference, labels)
    plt.show()


if __name__ == "__main__":
    main()
