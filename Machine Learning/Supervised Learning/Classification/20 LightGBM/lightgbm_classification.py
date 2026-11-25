"""LightGBM

LightGBM grows trees leaf by leaf to reduce binary log loss. A modest leaf budget suits this small diagnostic dataset.

Ke et al., LightGBM (2017)
https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html
Wolberg, Mangasarian, Street and Street, Breast Cancer Wisconsin (Diagnostic), UCI, CC BY 4.0
https://doi.org/10.24432/C5DW2B"""

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
from lightgbm import LGBMClassifier
from gallery.tabular import diagnostic_split, classification_report

model = LGBMClassifier(
    n_estimators=110, num_leaves=12, learning_rate=0.06, n_jobs=1, verbosity=-1, random_state=17
)


def main():
    X, X_test, y, y_test = diagnostic_split()

    probability = classification_report(model, X, X_test, y, y_test)
    plt.show()


if __name__ == "__main__":
    main()
