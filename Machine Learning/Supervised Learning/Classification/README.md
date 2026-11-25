# Classification

The first 24 examples use the same 143 held-out diagnostic records. Malignant is the positive class. The last three examples use a ten-class split of 450 handwritten digits. Log loss measures how much probability was assigned to the true label.

| Example | Test data | Balanced accuracy | MCC | Log loss |
|---|---|---:|---:|---:|
| [Logistic Regression](01%20Logistic%20Regression/logistic_regression.ipynb) · [Python](01%20Logistic%20Regression/logistic_regression.py) | Diagnostics | 0.962 | 0.941 | 0.066 |
| [Ridge Classifier](02%20Ridge%20Classifier/ridge_classifier.ipynb) · [Python](02%20Ridge%20Classifier/ridge_classifier.py) | Diagnostics | 0.957 | 0.925 | 0.105 |
| [Stochastic Gradient Classifier](03%20Stochastic%20Gradient%20Classifier/stochastic_gradient_classifier.ipynb) · [Python](03%20Stochastic%20Gradient%20Classifier/stochastic_gradient_classifier.py) | Diagnostics | 0.966 | 0.940 | 0.075 |
| [Perceptron](04%20Perceptron/perceptron.ipynb) · [Python](04%20Perceptron/perceptron.py) | Diagnostics | 0.953 | 0.926 | 0.126 |
| [Gaussian Naive Bayes](05%20Gaussian%20Naive%20Bayes/gaussian_naive_bayes.ipynb) · [Python](05%20Gaussian%20Naive%20Bayes/gaussian_naive_bayes.py) | Diagnostics | 0.923 | 0.865 | 0.601 |
| [Linear Discriminant Analysis](06%20Linear%20Discriminant%20Analysis/linear_discriminant_analysis.ipynb) · [Python](06%20Linear%20Discriminant%20Analysis/linear_discriminant_analysis.py) | Diagnostics | 0.953 | 0.926 | 0.083 |
| [Quadratic Discriminant Analysis](07%20Quadratic%20Discriminant%20Analysis/quadratic_discriminant_analysis.ipynb) · [Python](07%20Quadratic%20Discriminant%20Analysis/quadratic_discriminant_analysis.py) | Diagnostics | 0.943 | 0.912 | 0.091 |
| [Nearest Neighbours](08%20Nearest%20Neighbours/nearest_neighbours.ipynb) · [Python](08%20Nearest%20Neighbours/nearest_neighbours.py) | Diagnostics | 0.962 | 0.941 | 0.111 |
| [Linear Support Vector Machine](09%20Linear%20Support%20Vector%20Machine/linear_support_vector_machine.ipynb) · [Python](09%20Linear%20Support%20Vector%20Machine/linear_support_vector_machine.py) | Diagnostics | 0.953 | 0.926 | 0.117 |
| [Kernel Support Vector Machine](10%20Kernel%20Support%20Vector%20Machine/kernel_support_vector_machine.ipynb) · [Python](10%20Kernel%20Support%20Vector%20Machine/kernel_support_vector_machine.py) | Diagnostics | 0.966 | 0.940 | 0.100 |
| [Gaussian Process](11%20Gaussian%20Process/gaussian_process.ipynb) · [Python](11%20Gaussian%20Process/gaussian_process.py) | Diagnostics | 0.953 | 0.926 | 0.173 |
| [Decision Tree](12%20Decision%20Tree/decision_tree.ipynb) · [Python](12%20Decision%20Tree/decision_tree.py) | Diagnostics | 0.898 | 0.819 | 2.337 |
| [Bagging](13%20Bagging/bagging.ipynb) · [Python](13%20Bagging/bagging.py) | Diagnostics | 0.942 | 0.895 | 0.115 |
| [Random Forest](14%20Random%20Forest/random_forest.ipynb) · [Python](14%20Random%20Forest/random_forest.py) | Diagnostics | 0.951 | 0.910 | 0.115 |
| [Extra Trees](15%20Extra%20Trees/extra_trees.ipynb) · [Python](15%20Extra%20Trees/extra_trees.py) | Diagnostics | 0.942 | 0.895 | 0.110 |
| [AdaBoost](16%20AdaBoost/adaboost.ipynb) · [Python](16%20AdaBoost/adaboost.py) | Diagnostics | 0.934 | 0.897 | 0.369 |
| [Gradient Boosting](17%20Gradient%20Boosting/gradient_boosting.ipynb) · [Python](17%20Gradient%20Boosting/gradient_boosting.py) | Diagnostics | 0.932 | 0.880 | 0.114 |
| [Histogram Gradient Boosting](18%20Histogram%20Gradient%20Boosting/histogram_gradient_boosting.ipynb) · [Python](18%20Histogram%20Gradient%20Boosting/histogram_gradient_boosting.py) | Diagnostics | 0.934 | 0.897 | 0.135 |
| [XGBoost](19%20XGBoost/xgboost_classification.ipynb) · [Python](19%20XGBoost/xgboost_classification.py) | Diagnostics | 0.904 | 0.835 | 0.120 |
| [LightGBM](20%20LightGBM/lightgbm_classification.ipynb) · [Python](20%20LightGBM/lightgbm_classification.py) | Diagnostics | 0.910 | 0.852 | 0.129 |
| [CatBoost](21%20CatBoost/catboost_classification.ipynb) · [Python](21%20CatBoost/catboost_classification.py) | Diagnostics | 0.953 | 0.926 | 0.095 |
| [Multilayer Perceptron](22%20Multilayer%20Perceptron/multilayer_perceptron.ipynb) · [Python](22%20Multilayer%20Perceptron/multilayer_perceptron.py) | Diagnostics | 0.951 | 0.910 | 0.495 |
| [Soft Voting](23%20Soft%20Voting/soft_voting.ipynb) · [Python](23%20Soft%20Voting/soft_voting.py) | Diagnostics | 0.938 | 0.896 | 0.092 |
| [Stacking](24%20Stacking/stacking.ipynb) · [Python](24%20Stacking/stacking.py) | Diagnostics | 0.966 | 0.940 | 0.087 |
| [Multinomial Logistic Regression](25%20Multinomial%20Logistic%20Regression/multinomial_logistic_regression.ipynb) · [Python](25%20Multinomial%20Logistic%20Regression/multinomial_logistic_regression.py) | Digits | 0.969 | 0.965 | 0.119 |
| [Multinomial Naive Bayes](26%20Multinomial%20Naive%20Bayes/multinomial_naive_bayes.ipynb) · [Python](26%20Multinomial%20Naive%20Bayes/multinomial_naive_bayes.py) | Digits | 0.900 | 0.889 | 1.946 |
| [Bernoulli Naive Bayes](27%20Bernoulli%20Naive%20Bayes/bernoulli_naive_bayes.ipynb) · [Python](27%20Bernoulli%20Naive%20Bayes/bernoulli_naive_bayes.py) | Digits | 0.873 | 0.859 | 0.610 |
