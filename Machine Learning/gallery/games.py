"""Payoff and equilibrium diagnostics for two-player games."""

import matplotlib.pyplot as plt
import numpy as np

# The row player maximises this payoff and the column player minimises it.
ZERO_SUM = np.array([[2.0, -1.0, 0.0], [-2.0, 2.0, 1.0], [0.0, -1.0, 2.0]])


def gap(matrix, p, q):
    upper = float((matrix @ q).max())
    lower = float((p @ matrix).min())
    return upper - lower


def show_mixes(p, q, axis):
    index = np.arange(len(p))
    axis.bar(index - 0.18, p, 0.36, label="Row", color="#377793")
    axis.bar(index + 0.18, q, 0.36, label="Column", color="#b66439")
    axis.set(
        xticks=index,
        xticklabels=[str(x + 1) for x in index],
        xlabel="Action",
        ylabel="Probability",
        ylim=(0, 1),
        title="Mixed strategies",
    )
    axis.legend(fontsize=8)


def trace_plot(history, p, q):
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.6, 3.1), constrained_layout=True)
    t, g = np.asarray(history).T
    a.plot(t, g, color="#396f89")
    a.set(xlabel="Round", ylabel="Best-response gap", title="Average strategies", xscale="log")
    show_mixes(p, q, b)
