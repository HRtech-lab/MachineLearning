"""CatBoost

Symmetric boosted trees estimate the malignant-class probability. Each tree applies one shared splitting rule per level.

Prokhorenkova et al., CatBoost (2018)
https://arxiv.org/abs/1706.09516
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
from catboost import CatBoostClassifier
from gallery.tabular import diagnostic_split, classification_report

model = CatBoostClassifier(
    iterations=150,
    depth=4,
    learning_rate=0.07,
    verbose=False,
    allow_writing_files=False,
    thread_count=1,
    random_seed=17,
)


def main():
    X, X_test, y, y_test = diagnostic_split()

    probability = classification_report(model, X, X_test, y, y_test)
    plt.show()


if __name__ == "__main__":
    main()
