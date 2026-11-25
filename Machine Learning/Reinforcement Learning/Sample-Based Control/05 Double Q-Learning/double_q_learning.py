"""Double Q-Learning

Double Q-learning uses one table to select a next action and the other to evaluate it. The table that receives the update is chosen at random.

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


def train(seed):
    rng = np.random.default_rng(seed)
    left = np.zeros((S, A))
    right = np.zeros_like(left)
    trace = []
    for episode in range(650):
        epsilon = max(0.07, 0.7 / (1 + episode / 100))
        state = START
        total = 0.0
        for _ in range(60):
            action = choose((left + right)[state], epsilon, rng)
            nxt, r, done = step(state, action, rng)
            source, other = (left, right) if rng.random() < 0.5 else (right, left)
            best = choose(source[nxt], 0.0, rng)
            target = r if done else r + GAMMA * other[nxt, best]
            source[state, action] += 0.16 * (target - source[state, action])
            total += r
            state = nxt
            if done:
                break
        trace.append(total)
    return (left + right).argmax(1), trace


def main():
    policies, curves = [], []
    for seed in (17, 29, 53):
        policy, curve = train(seed)
        policies.append((seed, policy))
        curves.append(curve)
    policy_report(policies, curves, "Return, 25-episode average", "Episode")
    plt.show()


if __name__ == "__main__":
    main()
