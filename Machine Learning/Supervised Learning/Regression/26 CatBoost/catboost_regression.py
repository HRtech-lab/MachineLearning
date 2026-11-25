"""CatBoost

CatBoost can also fit entirely numerical data. Its symmetric trees use the same split condition at each level.

Prokhorenkova et al., CatBoost (2018)
https://arxiv.org/abs/1706.09516
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
from catboost import CatBoostRegressor
from gallery.tabular import concrete_split, regression_report

model = CatBoostRegressor(
    iterations=160,
    depth=5,
    learning_rate=0.07,
    verbose=False,
    allow_writing_files=False,
    thread_count=1,
    random_seed=17,
)


def main():
    X, X_test, y, y_test, groups = concrete_split()

    predictions = regression_report(model, X, X_test, y, y_test, groups)
    plt.show()


if __name__ == "__main__":
    main()
