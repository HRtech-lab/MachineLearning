"""Policy Iteration

Policy iteration alternates an exact linear solve for the current policy with greedy improvement. The full transition model of a slippery grid is known.

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
    policy = np.zeros(S, dtype=int)
    residuals = []
    for sweep in range(100):
        v = exact_values(policy)
        q = q_from_v(v)
        residuals.append(abs(q.max(1) - v).max())
        improved = q.argmax(axis=1)
        if np.array_equal(improved, policy):
            break
        policy = improved
    print(f"Policy evaluations {len(residuals)} | Bellman optimality residual {residuals[-1]:.3g}")

    policy_report([("exact", policy)], [residuals], "Bellman residual", "Sweep")
    plt.show()


if __name__ == "__main__":
    main()
