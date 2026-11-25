"""3D Convolutional Motion Classifier

A short clip shows a disk moving right, left, down or up. Start position and disk size are randomised independently of the class.

Tran et al., Learning Spatiotemporal Features with 3D Convolutional Networks (2015)
https://arxiv.org/abs/1412.0767"""

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
from gallery.vision import moving_clips, class_scores, confusion
from gallery.neural import seed_cpu, train_classifier


def motion_net():
    return nn.Sequential(
        nn.Conv3d(1, 12, (3, 3, 3)),
        nn.ReLU(),
        nn.MaxPool3d((1, 2, 2)),
        nn.Conv3d(12, 24, (3, 3, 3)),
        nn.ReLU(),
        nn.AdaptiveAvgPool3d(1),
        nn.Flatten(),
        nn.Linear(24, 4),
    )


def main():
    seed_cpu()
    train, y = moving_clips(640, 17)
    test, y_test = moving_clips(160, 89)
    net = motion_net()
    history = train_classifier(net, train[:, None], y, epochs=16, lr=0.003)
    with torch.no_grad():
        prob = net(torch.tensor(test[:, None])).softmax(1).numpy()
    class_scores(y_test, prob.argmax(1), prob)
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.5, 3), constrained_layout=True)
    a.imshow(torch.tensor(test[0]).permute(1, 0, 2).reshape(16, -1), cmap="gray", aspect="auto")
    a.axis("off")
    a.set_title("Six frames in time order")
    confusion(b, y_test, prob.argmax(1))
    plt.show()


if __name__ == "__main__":
    main()
