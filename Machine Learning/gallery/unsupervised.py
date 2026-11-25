"""Small datasets with known geometry and diagnostics for unsupervised fits.

scikit-learn, clustering and evaluation
https://scikit-learn.org/stable/modules/clustering.html
Alpaydin and Kaynak, Optical Recognition of Handwritten Digits
https://doi.org/10.24432/C50P49
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics as met
from sklearn.datasets import load_digits, make_blobs, make_circles, make_moons
from sklearn.manifold import trustworthiness
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors


def cluster_data(shape):
    if shape == "moons":
        return make_moons(n_samples=550, noise=0.065, random_state=17)
    if shape == "rings":
        return make_circles(n_samples=530, noise=0.035, factor=0.40, random_state=17)
    X, labels = make_blobs(
        n_samples=[150, 190, 170],
        centers=[[-2, 0], [0, 3], [3, 0]],
        cluster_std=[0.48, 0.60, 0.64],
        random_state=17,
    )
    if shape == "ellipses":
        X = X @ np.array([[1.3, 0.48], [0.25, 0.85]])
    return X, labels


def cluster_report(X, truth, labels):
    keep = labels != -1
    clusters = len(np.unique(labels[keep]))
    print(f"Clusters: {clusters} | Assigned: {keep.mean():.1%} | Noise points: {(~keep).sum()}")
    print(f"Adjusted Rand index (all): {met.adjusted_rand_score(truth,labels):.3f}")
    print(f"Adjusted mutual information (all): {met.adjusted_mutual_info_score(truth,labels):.3f}")
    if 1 < clusters < keep.sum():
        print(f"Silhouette (assigned): {met.silhouette_score(X[keep],labels[keep]):.3f}")
        print(f"Davies–Bouldin (assigned): {met.davies_bouldin_score(X[keep],labels[keep]):.3f}")
        print(
            f"Calinski–Harabasz (assigned): {met.calinski_harabasz_score(X[keep],labels[keep]):.1f}"
        )
    fig, axes = plt.subplots(
        1, 2, figsize=(7.7, 3.1), constrained_layout=True, sharex=True, sharey=True
    )
    axes[0].scatter(*X.T, s=10, alpha=0.65, c="#426573")
    axes[1].scatter(*X[keep].T, s=11, alpha=0.78, c=labels[keep], cmap="tab10")
    if (~keep).any():
        axes[1].scatter(*X[~keep].T, c="0.33", s=13, marker="x", label="Noise")
        axes[1].legend()
    for ax, title in zip(axes, ("Input points", "Fitted groups")):
        ax.set(title=title, xlabel="Coordinate 1", aspect="equal")
    axes[0].set_ylabel("Coordinate 2")


def digit_sample():
    X, y = load_digits(return_X_y=True)
    X, _, y, _ = train_test_split(X / 16, y, train_size=650, stratify=y, random_state=17)
    return X, y


def embedding_report(model, X, y, Z):
    k = 10
    original = NearestNeighbors(n_neighbors=k).fit(X).kneighbors(return_distance=False)
    projected = NearestNeighbors(n_neighbors=k).fit(Z).kneighbors(return_distance=False)
    overlap = np.mean([len(set(a) & set(b)) / k for a, b in zip(original, projected)])
    print(f"Trustworthiness at {k}: {trustworthiness(X,Z,n_neighbors=k):.3f}")
    print(f"Mean neighbour overlap at {k}: {overlap:.3f}")
    print(f"Neighbour label agreement (evaluation only): {(y[projected] == y[:, None]).mean():.3f}")
    if hasattr(model, "inverse_transform"):
        print(
            f"Pixel reconstruction MSE: {met.mean_squared_error(X,model.inverse_transform(Z)):.4f}"
        )
    if hasattr(model, "explained_variance_ratio_"):
        print(f"Explained variance fraction: {model.explained_variance_ratio_.sum():.3f}")
    if hasattr(model, "kl_divergence_"):
        print(f"Final KL divergence: {model.kl_divergence_:.3f}")
    fig, ax = plt.subplots(figsize=(6.1, 3.5), constrained_layout=True)
    points = ax.scatter(*Z.T, c=y, cmap="tab10", s=13, alpha=0.8, vmin=-0.5, vmax=9.5)
    fig.colorbar(points, ax=ax, ticks=range(10), label="Digit")
    ax.set(
        xlabel="Coordinate 1", ylabel="Coordinate 2", title="Digit neighbourhoods in two dimensions"
    )


def anomaly_data():
    rng = np.random.default_rng(17)
    centre = [0, 0]
    covariance = [[1, 0.65], [0.65, 1]]
    train = rng.multivariate_normal(centre, covariance, 360)
    normal = rng.multivariate_normal(centre, covariance, 230)
    # The three small shifted groups include some hard cases near the training cloud.
    out = np.vstack(
        [rng.normal(loc, 0.45, (12, 2)) for loc in [(1.9, -1.3), (-1.8, 1.4), (3.4, 2.4)]]
    )
    return (
        train,
        np.vstack([normal, out]),
        np.r_[np.zeros(len(normal), int), np.ones(len(out), int)],
    )


def anomaly_report(model, X, y):
    score = -model.score_samples(X)
    flagged = model.predict(X) == -1
    print(
        f"ROC AUC: {met.roc_auc_score(y,score):.3f} | Average precision: {met.average_precision_score(y,score):.3f} | Prevalence: {y.mean():.3f}"
    )
    print(
        f"Precision: {met.precision_score(y,flagged,zero_division=0):.3f} | Recall: {met.recall_score(y,flagged):.3f} | F1: {met.f1_score(y,flagged):.3f}"
    )
    print("Rows = actual inlier/outlier, columns = predicted inlier/outlier")
    print(met.confusion_matrix(y, flagged))
    x, ygrid = np.meshgrid(np.linspace(-4, 5, 110), np.linspace(-4, 4, 100))
    boundary = model.decision_function(np.c_[x.ravel(), ygrid.ravel()]).reshape(x.shape)
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.8, 3.1), constrained_layout=True)
    a.contour(x, ygrid, boundary, levels=[0], colors=["#a74b28"])
    for value, colour, title in [(0, "#45718a", "Inlier"), (1, "#bb572b", "Injected")]:
        a.scatter(*X[y == value].T, s=11, color=colour, alpha=0.65, label=title)
    a.legend(fontsize=8)
    a.set(xlabel="Coordinate 1", ylabel="Coordinate 2", title="Held-out cases and threshold")
    for value, colour in [(0, "#45718a"), (1, "#bb572b")]:
        b.hist(score[y == value], bins=16, alpha=0.63, color=colour, density=True)
    b.set(xlabel="Anomaly score (higher is stranger)", ylabel="Density", title="Score separation")
