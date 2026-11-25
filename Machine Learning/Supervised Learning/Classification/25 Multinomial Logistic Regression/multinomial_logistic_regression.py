"""Multinomial Logistic Regression

A joint ten-class softmax model estimates a distribution over handwritten digits. The class with the highest probability wins.

scikit-learn, LogisticRegression
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
Alpaydin and Kaynak, Optical Recognition of Handwritten Digits, UCI, CC BY 4.0
https://doi.org/10.24432/C50P49"""

from pathlib import Path
import sys

start = Path(__file__).resolve().parent
for parent in (start, *start.parents):
    if (parent / "gallery" / "__init__.py").is_file():
        sys.path.insert(0, str(parent))
        break
else:
    raise RuntimeError("Open this example inside the Machine Learning collection.")


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from gallery.vision import digits_split, class_scores, confusion


def main():
    train, test, y, y_test = digits_split()
    X = train.reshape(len(train), -1) * 16
    T = test.reshape(len(test), -1) * 16
    model = LogisticRegression(C=1, max_iter=1500, random_state=17)
    model.fit(X, y)
    prob = model.predict_proba(T)
    pred = prob.argmax(1)
    class_scores(y_test, pred, prob)
    prior = np.bincount(y) / len(y)
    print(
        f"Training-prior baseline accuracy {(y_test==prior.argmax()).mean():.3f}, log loss {-np.log(prior[y_test]).mean():.3f}"
    )
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.8, 3.2), constrained_layout=True)
    confusion(a, y_test, pred)
    by_class = [np.mean(pred[y_test == d] == d) for d in range(10)]
    b.bar(np.arange(10), by_class, color="#42778e")
    b.set(
        xticks=range(10),
        ylim=(0, 1.05),
        title="Recall by digit",
        xlabel="True digit",
        ylabel="Recall",
    )
    plt.show()


if __name__ == "__main__":
    main()
