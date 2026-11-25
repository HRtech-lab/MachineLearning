"""A small slippery grid with known dynamics, sampling and policy evaluation."""

import matplotlib.pyplot as plt
import numpy as np

ROWS, COLS = 4, 5
S, A = ROWS * COLS, 4
START, GOAL, PIT = 15, 4, 12
WALLS = {6, 11}
GAMMA = 0.95
STEPS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
ARROWS = ["↑", "→", "↓", "←"]


def outcomes(s, a):
    if s in WALLS or s in (GOAL, PIT):
        return [(1.0, s, 0.0, True)]
    row, col = divmod(s, COLS)
    result = []
    for p, action in ((0.84, a), (0.08, (a - 1) % A), (0.08, (a + 1) % A)):
        dr, dc = STEPS[action]
        nr, nc = row + dr, col + dc
        nxt = nr * COLS + nc
        if not (0 <= nr < ROWS and 0 <= nc < COLS) or nxt in WALLS:
            nxt = s
        reward = 1.0 if nxt == GOAL else -1.0 if nxt == PIT else -0.04
        result.append((p, nxt, reward, nxt in (GOAL, PIT)))
    return result


TRANSITIONS = [[outcomes(s, a) for a in range(A)] for s in range(S)]
REWARD = np.zeros((S, A))
CONTINUATION = np.zeros((S, A, S))
for s in range(S):
    for a in range(A):
        for p, n, r, done in TRANSITIONS[s][a]:
            REWARD[s, a] += p * r
            if not done:
                CONTINUATION[s, a, n] += p


def step(s, a, rng):
    entries = TRANSITIONS[s][a]
    draw = rng.random()
    k = 0 if draw < 0.84 or len(entries) == 1 else 1 if draw < 0.92 else 2
    _, n, r, done = entries[k]
    return n, r, done


def q_from_v(v):
    return REWARD + GAMMA * (CONTINUATION @ v)


def exact_values(policy):
    rows = np.arange(S)
    return np.linalg.solve(np.eye(S) - GAMMA * CONTINUATION[rows, policy], REWARD[rows, policy])


def choose(q, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(A))
    best = np.flatnonzero(np.isclose(q, q.max()))
    return int(rng.choice(best))


def evaluate(policy, episodes=240, seed=137):
    rng = np.random.default_rng(seed)
    records = []
    for _ in range(episodes):
        s = START
        total = 0.0
        for length in range(1, 61):
            a = int(rng.integers(A)) if policy is None else int(policy[s])
            s, r, done = step(s, a, rng)
            total += r
            if done:
                break
        records.append([total, s == GOAL, s == PIT, length])
    return np.array(records, dtype=float).mean(0)


def draw_policy(ax, policy, values):
    im = ax.imshow(values.reshape(ROWS, COLS), cmap="RdBu", vmin=-1, vmax=1)
    for s in range(S):
        row, col = divmod(s, COLS)
        label = (
            "Wall"
            if s in WALLS
            else "Goal" if s == GOAL else "Pit" if s == PIT else ARROWS[policy[s]]
        )
        if s == START:
            label = "Start " + label
        ax.text(col, row, label, ha="center", va="center", fontsize=8)
    ax.set(xticks=[], yticks=[], title="Greedy policy and discounted value")
    return im


def policy_report(policies, curves, curve_label, unit):
    base = evaluate(None)
    values = []
    for seed, policy in policies:
        ev = evaluate(policy)
        v = exact_values(policy)
        print(
            f"Seed {seed}: mean return {ev[0]:.3f}, goal {ev[1]:.1%}, pit {ev[2]:.1%}, mean steps {ev[3]:.1f}, exact discounted start value {v[START]:.3f}"
        )
        values.append(ev)
    print(
        f"Random policy: mean return {base[0]:.3f}, goal {base[1]:.1%}, pit {base[2]:.1%}, mean steps {base[3]:.1f}"
    )
    if len(values) > 1:
        totals = np.stack(values)[:, 0]
        print(
            f"Across training seeds: mean return {totals.mean():.3f}, sample SD {totals.std(ddof=1):.3f}"
        )
    fig, (a, b) = plt.subplots(1, 2, figsize=(8, 3.1), constrained_layout=True)
    for (seed, _), curve in zip(policies, curves):
        curve = np.asarray(curve)
        window = min(25, len(curve)) if unit == "Episode" else 1
        a.plot(
            np.arange(window, len(curve) + 1),
            np.convolve(curve, np.ones(window) / window, "valid"),
            label=str(seed),
            lw=1.2,
        )
    a.set(xlabel=unit, ylabel=curve_label, title="Training trace")
    if len(policies) > 1:
        a.legend(title="Seed", fontsize=7)
    p = policies[0][1]
    draw_policy(b, p, exact_values(p))
