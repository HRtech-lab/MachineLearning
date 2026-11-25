# Anomaly Detection

Novelty scores for independent inliers and injected outliers.

| Example | Question or method |
|---|---|
| [Isolation Forest](01%20Isolation%20Forest/isolation_forest.ipynb) · [Python](01%20Isolation%20Forest/isolation_forest.py) | An isolation forest gives a high anomaly score to observations that are separated with few random cuts. |
| [Local Outlier Factor](02%20Local%20Outlier%20Factor/local_outlier_factor.ipynb) · [Python](02%20Local%20Outlier%20Factor/local_outlier_factor.py) | Local outlier factor compares the density near each new observation with the densities around its training neighbours. |
| [One-Class SVM](03%20One-Class%20SVM/one_class_svm.ipynb) · [Python](03%20One-Class%20SVM/one_class_svm.py) | One-class SVM learns a soft kernel boundary around the training distribution. |
| [Robust Covariance](04%20Robust%20Covariance/robust_covariance.ipynb) · [Python](04%20Robust%20Covariance/robust_covariance.py) | A robust covariance estimate defines an elliptical boundary. This geometry is well matched to the clean training distribution. |
