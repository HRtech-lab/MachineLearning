"""DDPM on Two Moons

A noise predictor learns how to undo a forward diffusion process on two crescents. Running the full stochastic reverse chain turns normal noise into a new point cloud.

Ho, Jain and Abbeel, Denoising Diffusion Probabilistic Models (2020)
https://arxiv.org/abs/2006.11239
Nichol and Dhariwal, Improved Denoising Diffusion Probabilistic Models (2021)
https://arxiv.org/abs/2102.09672
scikit-learn, make_moons
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html"""

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
import torch
from gallery.neural import seed_cpu
from gallery.diffusion import (
    NoisePredictor,
    NoiseSchedule,
    train_noise,
    heldout_noise,
    sample,
    distribution_scores,
    training_bandwidth,
)

from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler


def main():
    seed_cpu()
    train, _ = make_moons(1800, noise=0.055, random_state=17)
    reference, _ = make_moons(450, noise=0.055, random_state=73)
    split_ref, _ = make_moons(450, noise=0.055, random_state=117)
    scaler = StandardScaler().fit(train)
    X = torch.tensor(scaler.transform(train), dtype=torch.float32)
    reference = scaler.transform(reference)
    split_ref = scaler.transform(split_ref)
    schedule = NoiseSchedule(48)
    net = NoisePredictor(2, 96)
    history = train_noise(net, schedule, X, updates=1700)
    heldout_noise(net, schedule, torch.tensor(reference, dtype=torch.float32))
    bandwidth = training_bandwidth(X.numpy())
    initial = torch.randn((450, 2), generator=torch.Generator().manual_seed(89))

    generated = sample(net, schedule, initial, "ddpm", limit=2.6)
    sets = [
        ("Independent real sample", split_ref),
        ("Initial noise", initial.numpy()),
        ("DDPM, 48 steps", generated),
    ]
    panels = [
        ("Held-out moons", reference),
        ("Initial noise", initial.numpy()),
        ("DDPM", generated),
    ]

    for title, cloud in sets:
        mmd, sw = distribution_scores(reference, cloud, bandwidth)
        print(f"{title}: MMD² {mmd:.4f} | sliced Wasserstein {sw:.3f}")
    fig, axes = plt.subplots(
        1, 3, figsize=(9.4, 3), sharex=True, sharey=True, constrained_layout=True
    )
    for ax, (label, cloud) in zip(axes, panels):
        ax.scatter(*cloud.T, s=7, alpha=0.6, color="#30718c")
        ax.set(xlim=(-3, 3), ylim=(-3, 3), aspect="equal", title=label, xlabel="Scaled x")
    axes[0].set_ylabel("Scaled y")
    plt.show()


if __name__ == "__main__":
    main()
