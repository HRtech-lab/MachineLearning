"""t-SNE

t-SNE emphasises local neighbours. The distance between two well-separated islands has no simple global interpretation.

scikit-learn, TSNE
https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
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
from sklearn.manifold import TSNE
from gallery.unsupervised import digit_sample, embedding_report


def main():
    X, y = digit_sample()
    model = TSNE(
        n_components=2,
        perplexity=27,
        init="pca",
        learning_rate="auto",
        max_iter=800,
        random_state=17,
    )
    Z = model.fit_transform(X)

    embedding_report(model, X, y, Z)
    plt.show()


if __name__ == "__main__":
    main()
