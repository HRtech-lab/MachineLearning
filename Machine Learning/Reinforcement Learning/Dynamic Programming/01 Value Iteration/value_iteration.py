"""Value Iteration

The Bellman optimality backup replaces each state value with its best expected one-step return. The full transition model of a slippery grid is known.

Sutton and Barto, Reinforcement Learning: An Introduction (2018)
https://mitpress.mit.edu/9780262039246/reinforcement-learning/"""

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
from gallery.gridworld import (
    S,
    A,
    GAMMA,
    START,
    choose,
    step,
    q_from_v,
    exact_values,
    policy_report,
)


def main():
    v = np.zeros(S)
    residuals = []
    for sweep in range(500):
        updated = q_from_v(v).max(axis=1)
        difference = abs(updated - v).max()
        residuals.append(difference)
        v = updated
        if difference < 1e-10:
            break
    policy = q_from_v(v).argmax(axis=1)
    print(
        f"Sweeps {len(residuals)} | Bellman optimality residual {abs(v-q_from_v(v).max(1)).max():.3g}"
    )

    policy_report([("exact", policy)], [residuals], "Bellman residual", "Sweep")
    plt.show()


if __name__ == "__main__":
    main()
