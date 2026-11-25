"""Stacking Regression

A final ridge model learns how to combine a linear estimate, a forest and a boosted tree estimate. Each training prediction for the final model comes from an unseen recipe.

scikit-learn, StackingRegressor
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html
I-Cheng Yeh, Concrete Compressive Strength (UCI), CC BY 4.0
https://doi.org/10.24432/C5PK67"""

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
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from gallery.tabular import concrete_split, regression_report


class RecipeStack(RegressorMixin, BaseEstimator):
    def fit(self, X, y):
        ingredients = X.drop(columns="age").round(1)
        group = ingredients.groupby(list(ingredients.columns), sort=False).ngroup()
        folds = list(GroupKFold(4, shuffle=True, random_state=17).split(X, y, group))
        estimators = [
            ("linear", make_pipeline(StandardScaler(), Ridge(alpha=3))),
            (
                "forest",
                RandomForestRegressor(
                    n_estimators=60, min_samples_leaf=2, random_state=17, n_jobs=1
                ),
            ),
            ("boost", HistGradientBoostingRegressor(max_iter=100, random_state=17)),
        ]
        self.model_ = StackingRegressor(
            estimators=estimators, final_estimator=Ridge(alpha=2), cv=folds
        )
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)


def main():
    X, X_test, y, y_test, groups = concrete_split()
    model = RecipeStack()
    prediction = regression_report(model, X, X_test, y, y_test, groups)
    print(
        "Meta-model coefficients, linear/forest/boost:",
        model.model_.final_estimator_.coef_.round(3),
    )
    plt.show()


if __name__ == "__main__":
    main()
