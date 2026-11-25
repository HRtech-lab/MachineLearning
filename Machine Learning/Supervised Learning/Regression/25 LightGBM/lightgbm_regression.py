"""LightGBM

LightGBM grows a leaf with a large estimated gain at each step. Limiting the number of leaves controls how finely it partitions the recipes.

Ke et al., LightGBM (2017)
https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html
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
from lightgbm import LGBMRegressor
from gallery.tabular import concrete_split, regression_report

model = LGBMRegressor(
    n_estimators=140, num_leaves=16, learning_rate=0.06, n_jobs=1, verbosity=-1, random_state=17
)


def main():
    X, X_test, y, y_test, groups = concrete_split()

    predictions = regression_report(model, X, X_test, y, y_test, groups)
    plt.show()


if __name__ == "__main__":
    main()
