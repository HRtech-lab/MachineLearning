"""OPTICS

OPTICS builds a reachability ordering. A cut at a specified radius extracts the crescent-shaped groups from that ordering.

scikit-learn, OPTICS
https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html
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
from sklearn.cluster import OPTICS
from gallery.unsupervised import cluster_data, cluster_report


def main():
    X, reference = cluster_data("moons")
    model = OPTICS(min_samples=8, cluster_method="dbscan", eps=0.17)
    labels = model.fit_predict(X)

    cluster_report(X, reference, labels)
    import numpy as np

    reach = model.reachability_[model.ordering_]
    plt.figure(figsize=(6, 2.1))
    plt.plot(np.where(np.isfinite(reach), reach, np.nan), linewidth=0.8)
    plt.axhline(0.17, color="tab:red", linestyle="--")
    plt.xlabel("OPTICS ordering")
    plt.ylabel("Reachability")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
