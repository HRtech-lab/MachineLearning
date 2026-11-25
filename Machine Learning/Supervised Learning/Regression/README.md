# Regression

The concrete examples share a test set of 206 observations from unseen recipes. Their errors are measured in MPa. Poisson regression uses a separate hourly count forecast. Every notebook also includes baseline errors and residual plots.

| Example | Dataset | MAE | RMSE | R² |
|---|---|---:|---:|---:|
| [Linear Regression](01%20Linear%20Regression/linear_regression.ipynb) · [Python](01%20Linear%20Regression/linear_regression.py) | Concrete (MPa) | 8.509 | 10.453 | 0.633 |
| [Ridge Regression](02%20Ridge%20Regression/ridge_regression.ipynb) · [Python](02%20Ridge%20Regression/ridge_regression.py) | Concrete (MPa) | 8.552 | 10.466 | 0.632 |
| [Lasso Regression](03%20Lasso%20Regression/lasso_regression.ipynb) · [Python](03%20Lasso%20Regression/lasso_regression.py) | Concrete (MPa) | 8.596 | 10.527 | 0.627 |
| [Elastic Net](04%20Elastic%20Net/elastic_net.ipynb) · [Python](04%20Elastic%20Net/elastic_net.py) | Concrete (MPa) | 8.989 | 10.977 | 0.595 |
| [Polynomial Regression](05%20Polynomial%20Regression/polynomial_regression.ipynb) · [Python](05%20Polynomial%20Regression/polynomial_regression.py) | Concrete (MPa) | 6.378 | 8.222 | 0.773 |
| [Bayesian Ridge](06%20Bayesian%20Ridge/bayesian_ridge.ipynb) · [Python](06%20Bayesian%20Ridge/bayesian_ridge.py) | Concrete (MPa) | 8.525 | 10.450 | 0.633 |
| [ARD Regression](07%20ARD%20Regression/ard_regression.ipynb) · [Python](07%20ARD%20Regression/ard_regression.py) | Concrete (MPa) | 8.523 | 10.408 | 0.636 |
| [Huber Regression](08%20Huber%20Regression/huber_regression.ipynb) · [Python](08%20Huber%20Regression/huber_regression.py) | Concrete (MPa) | 8.278 | 10.347 | 0.640 |
| [RANSAC Regression](09%20RANSAC%20Regression/ransac_regression.ipynb) · [Python](09%20RANSAC%20Regression/ransac_regression.py) | Concrete (MPa) | 8.015 | 11.111 | 0.585 |
| [Quantile Regression](10%20Quantile%20Regression/quantile_regression.ipynb) · [Python](10%20Quantile%20Regression/quantile_regression.py) | Concrete (MPa) | 8.771 | 10.958 | 0.596 |
| [Stochastic Gradient Regression](11%20Stochastic%20Gradient%20Regression/stochastic_gradient_regression.ipynb) · [Python](11%20Stochastic%20Gradient%20Regression/stochastic_gradient_regression.py) | Concrete (MPa) | 8.616 | 10.528 | 0.627 |
| [Partial Least Squares](12%20Partial%20Least%20Squares/partial_least_squares.ipynb) · [Python](12%20Partial%20Least%20Squares/partial_least_squares.py) | Concrete (MPa) | 8.494 | 10.363 | 0.639 |
| [Nearest Neighbours](13%20Nearest%20Neighbours/nearest_neighbours.ipynb) · [Python](13%20Nearest%20Neighbours/nearest_neighbours.py) | Concrete (MPa) | 7.439 | 10.291 | 0.644 |
| [Support Vector Regression](14%20Support%20Vector%20Regression/support_vector_regression.ipynb) · [Python](14%20Support%20Vector%20Regression/support_vector_regression.py) | Concrete (MPa) | 5.366 | 7.328 | 0.819 |
| [Kernel Ridge](15%20Kernel%20Ridge/kernel_ridge.ipynb) · [Python](15%20Kernel%20Ridge/kernel_ridge.py) | Concrete (MPa) | 5.503 | 7.628 | 0.804 |
| [Gaussian Process](16%20Gaussian%20Process/gaussian_process.ipynb) · [Python](16%20Gaussian%20Process/gaussian_process.py) | Concrete (MPa) | 5.324 | 7.182 | 0.827 |
| [Decision Tree](17%20Decision%20Tree/decision_tree.ipynb) · [Python](17%20Decision%20Tree/decision_tree.py) | Concrete (MPa) | 6.771 | 9.520 | 0.695 |
| [Bagging](18%20Bagging/bagging.ipynb) · [Python](18%20Bagging/bagging.py) | Concrete (MPa) | 5.134 | 7.589 | 0.806 |
| [Random Forest](19%20Random%20Forest/random_forest.ipynb) · [Python](19%20Random%20Forest/random_forest.py) | Concrete (MPa) | 5.275 | 7.724 | 0.799 |
| [Extra Trees](20%20Extra%20Trees/extra_trees.ipynb) · [Python](20%20Extra%20Trees/extra_trees.py) | Concrete (MPa) | 4.663 | 6.756 | 0.847 |
| [AdaBoost](21%20AdaBoost/adaboost.ipynb) · [Python](21%20AdaBoost/adaboost.py) | Concrete (MPa) | 7.132 | 9.117 | 0.721 |
| [Gradient Boosting](22%20Gradient%20Boosting/gradient_boosting.ipynb) · [Python](22%20Gradient%20Boosting/gradient_boosting.py) | Concrete (MPa) | 4.894 | 6.724 | 0.848 |
| [Histogram Gradient Boosting](23%20Histogram%20Gradient%20Boosting/histogram_gradient_boosting.ipynb) · [Python](23%20Histogram%20Gradient%20Boosting/histogram_gradient_boosting.py) | Concrete (MPa) | 4.974 | 7.377 | 0.817 |
| [XGBoost](24%20XGBoost/xgboost_regression.ipynb) · [Python](24%20XGBoost/xgboost_regression.py) | Concrete (MPa) | 4.708 | 6.507 | 0.858 |
| [LightGBM](25%20LightGBM/lightgbm_regression.ipynb) · [Python](25%20LightGBM/lightgbm_regression.py) | Concrete (MPa) | 5.125 | 7.496 | 0.811 |
| [CatBoost](26%20CatBoost/catboost_regression.ipynb) · [Python](26%20CatBoost/catboost_regression.py) | Concrete (MPa) | 4.625 | 6.299 | 0.867 |
| [Multilayer Perceptron](27%20Multilayer%20Perceptron/multilayer_perceptron.ipynb) · [Python](27%20Multilayer%20Perceptron/multilayer_perceptron.py) | Concrete (MPa) | 4.521 | 6.192 | 0.871 |
| [Voting Regression](28%20Voting%20Regression/voting_regression.ipynb) · [Python](28%20Voting%20Regression/voting_regression.py) | Concrete (MPa) | 5.353 | 7.383 | 0.817 |
| [Stacking Regression](29%20Stacking%20Regression/stacking_regression.ipynb) · [Python](29%20Stacking%20Regression/stacking_regression.py) | Concrete (MPa) | 4.780 | 6.947 | 0.838 |
| [Poisson Regression](30%20Poisson%20Regression/poisson_regression.ipynb) · [Python](30%20Poisson%20Regression/poisson_regression.py) | Arrivals (counts) | 2.616 | 3.379 | 0.754 |
