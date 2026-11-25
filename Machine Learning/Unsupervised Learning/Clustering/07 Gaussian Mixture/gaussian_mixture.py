"""Gaussian Mixture

A Gaussian mixture models a density with elliptical components. Soft posterior responsibilities become hard labels for the cluster comparison.

scikit-learn, GaussianMixture
https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
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
from sklearn.mixture import GaussianMixture
from gallery.unsupervised import cluster_data, cluster_report


def main():
    X, reference = cluster_data("ellipses")
    model = GaussianMixture(n_components=3, covariance_type="full", n_init=5, random_state=17)
    labels = model.fit_predict(X)

    cluster_report(X, reference, labels)
    print(f"BIC: {model.bic(X):.1f} | AIC: {model.aic(X):.1f}")
    plt.show()


if __name__ == "__main__":
    main()
