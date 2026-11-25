"""Convolutional Digit Classifier

Two convolutional layers learn local stroke features directly from the original eight-pixel-wide digits.

LeCun et al., Gradient-based learning applied to document recognition
https://doi.org/10.1109/5.726791
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
import torch
from torch import nn
from gallery.vision import digits_split, class_scores, confusion
from gallery.neural import seed_cpu, train_classifier


def digit_net():
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(32 * 4 * 4, 10),
    )


def main():
    seed_cpu()
    train, test, y, y_test = digits_split()
    net = digit_net()
    loss = train_classifier(net, train[:, None], y, epochs=18)
    with torch.no_grad():
        prob = net(torch.tensor(test[:, None])).softmax(1).numpy()
    pred = prob.argmax(1)
    class_scores(y_test, pred, prob)
    fig, (a, b) = plt.subplots(1, 2, figsize=(7, 3), constrained_layout=True)
    a.plot(range(1, len(loss) + 1), loss)
    a.set(xlabel="Epoch", ylabel="Cross-entropy", title="Training loss")
    confusion(b, y_test, pred)
    plt.show()


if __name__ == "__main__":
    main()
