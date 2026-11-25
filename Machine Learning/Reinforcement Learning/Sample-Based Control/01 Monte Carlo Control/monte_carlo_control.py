"""Monte Carlo Control

First-visit Monte Carlo control estimates action values from complete sampled returns. Epsilon-soft exploration keeps more than one action in play.

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
    q = np.zeros((S, A))
    counts = np.zeros((S, A))
    trace = []
    for episode in range(650):
        epsilon = max(0.07, 0.7 / (1 + episode / 100))
        state = START
        path = []
        for _ in range(60):
            action = choose(q[state], epsilon, rng)
            nxt, reward, done = step(state, action, rng)
            path.append((state, action, reward))
            state = nxt
            if done:
                break
        trace.append(sum(r for _, _, r in path))
        returns = np.zeros(len(path))
        value = 0.0
        for t in reversed(range(len(path))):
            value = path[t][2] + GAMMA * value
            returns[t] = value
        seen = set()
        for (state, action, _), value in zip(path, returns):
            pair = (state, action)
            if pair in seen:
                continue
            seen.add(pair)
            counts[pair] += 1
            q[pair] += (value - q[pair]) / counts[pair]
    return q.argmax(1), trace


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
