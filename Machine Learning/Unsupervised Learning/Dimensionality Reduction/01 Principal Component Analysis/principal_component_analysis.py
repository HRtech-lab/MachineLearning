"""Principal Component Analysis

PCA retains the two orthogonal directions with the largest sample variance.

scikit-learn, PCA
https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
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


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from gallery.unsupervised import digit_sample, embedding_report


def main():
    X, y = digit_sample()
    model = PCA(n_components=2, random_state=17)
    Z = model.fit_transform(X)

    embedding_report(model, X, y, Z)
    plt.show()


if __name__ == "__main__":
    main()
