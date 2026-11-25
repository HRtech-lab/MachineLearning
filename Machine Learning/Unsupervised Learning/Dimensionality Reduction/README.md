# Dimensionality Reduction

Neighbour preservation and reconstruction on handwritten digits.

| Example | Question or method |
|---|---|
| [Principal Component Analysis](01%20Principal%20Component%20Analysis/principal_component_analysis.ipynb) · [Python](01%20Principal%20Component%20Analysis/principal_component_analysis.py) | PCA retains the two orthogonal directions with the largest sample variance. |
| [Kernel PCA](02%20Kernel%20PCA/kernel_pca.ipynb) · [Python](02%20Kernel%20PCA/kernel_pca.py) | Kernel PCA extracts principal directions in a nonlinear similarity space. The inverse mapping is learned separately. |
| [Truncated SVD](03%20Truncated%20SVD/truncated_svd.ipynb) · [Python](03%20Truncated%20SVD/truncated_svd.py) | Truncated SVD factors the uncentred pixel matrix. Average brightness can therefore influence its first direction. |
| [Nonnegative Matrix Factorisation](04%20Nonnegative%20Matrix%20Factorisation/nonnegative_matrix_factorisation.ipynb) · [Python](04%20Nonnegative%20Matrix%20Factorisation/nonnegative_matrix_factorisation.py) | Nonnegative matrix factorisation approximates each digit as an additive combination of two nonnegative patterns. |
| [Independent Component Analysis](05%20Independent%20Component%20Analysis/independent_component_analysis.ipynb) · [Python](05%20Independent%20Component%20Analysis/independent_component_analysis.py) | ICA looks for statistically independent coordinates. Independence and explained variance answer different questions. |
| [Isomap](06%20Isomap/isomap.ipynb) · [Python](06%20Isomap/isomap.py) | Isomap approximates distances along a neighbour graph before embedding the resulting distances in two dimensions. |
| [t-SNE](07%20t-SNE/t_sne.ipynb) · [Python](07%20t-SNE/t_sne.py) | t-SNE emphasises local neighbours. The distance between two well-separated islands has no simple global interpretation. |
