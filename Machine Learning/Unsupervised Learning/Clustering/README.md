# Clustering

Blobs, ellipses, rings and two moons, with known generating labels for evaluation.

| Example | Question or method |
|---|---|
| [K-Means](01%20K-Means/k_means.ipynb) · [Python](01%20K-Means/k_means.py) | K-means draws a straight boundary between nearby centroids. The three compact groups make the nearest-centre assumption visible. |
| [Mini-Batch K-Means](02%20Mini-Batch%20K-Means/mini_batch_k_means.ipynb) · [Python](02%20Mini-Batch%20K-Means/mini_batch_k_means.py) | Small batches update the centroids incrementally. Compare the group assignments with full-batch k-means on the same cloud. |
| [Bisecting K-Means](03%20Bisecting%20K-Means/bisecting_k_means.ipynb) · [Python](03%20Bisecting%20K-Means/bisecting_k_means.py) | Bisecting k-means builds a hierarchy by repeatedly splitting one current cluster into two. |
| [Agglomerative Clustering](04%20Agglomerative%20Clustering/agglomerative_clustering.ipynb) · [Python](04%20Agglomerative%20Clustering/agglomerative_clustering.py) | Ward linkage merges the pair of clusters that causes the smallest increase in within-group squared error. |
| [BIRCH](05%20BIRCH/birch.ipynb) · [Python](05%20BIRCH/birch.py) | BIRCH summarises nearby observations in a tree of small subclusters before assigning the final three labels. |
| [Mean Shift](06%20Mean%20Shift/mean_shift.ipynb) · [Python](06%20Mean%20Shift/mean_shift.py) | Mean shift follows local averages towards density peaks. The bandwidth chooses the spatial scale of a peak. |
| [Gaussian Mixture](07%20Gaussian%20Mixture/gaussian_mixture.ipynb) · [Python](07%20Gaussian%20Mixture/gaussian_mixture.py) | A Gaussian mixture models a density with elliptical components. Soft posterior responsibilities become hard labels for the cluster comparison. |
| [DBSCAN](08%20DBSCAN/dbscan.ipynb) · [Python](08%20DBSCAN/dbscan.py) | A chain of densely packed neighbours can follow a curved crescent. DBSCAN tests that idea directly on two noisy moons. |
| [HDBSCAN](09%20HDBSCAN/hdbscan.ipynb) · [Python](09%20HDBSCAN/hdbscan.py) | HDBSCAN follows connected groups over a range of density levels. Short-lived groups can be labelled as noise. |
| [OPTICS](10%20OPTICS/optics.ipynb) · [Python](10%20OPTICS/optics.py) | OPTICS builds a reachability ordering. A cut at a specified radius extracts the crescent-shaped groups from that ordering. |
| [Spectral Clustering](11%20Spectral%20Clustering/spectral_clustering.ipynb) · [Python](11%20Spectral%20Clustering/spectral_clustering.py) | A Gaussian affinity graph gives nearby points on each ring much larger edge weights. Spectral clustering uses the graph eigenvectors to separate the inner and outer circles. |
