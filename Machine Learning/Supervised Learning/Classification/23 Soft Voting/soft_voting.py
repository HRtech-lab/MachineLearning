"""Soft Voting

Soft voting averages three probability estimates before choosing a class. The component models represent quite different assumptions.

scikit-learn, VotingClassifier
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html
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
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from gallery.tabular import diagnostic_split, classification_report

model = VotingClassifier(
    [
        ("logit", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1500))),
        ("forest", RandomForestClassifier(n_estimators=70, random_state=17)),
        ("nb", GaussianNB()),
    ],
    voting="soft",
)


def main():
    X, X_test, y, y_test = diagnostic_split()

    probability = classification_report(model, X, X_test, y, y_test)
    plt.show()


if __name__ == "__main__":
    main()
