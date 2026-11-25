"""Minimax Linear Programming

What expected payoff can a player guarantee against every opposing action? Two linear programmes solve opposite sides of the same zero-sum matrix game.

Nash, Equilibrium Points in N-Person Games (1950)
https://doi.org/10.1073/pnas.36.1.48
SciPy, linear programming with HiGHS
https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html"""

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
from gallery.games import ZERO_SUM, gap, show_mixes, trace_plot

from scipy.optimize import linprog


def maximin(A):
    m, n = A.shape
    payoff_constraints = np.column_stack([-A.T, np.ones(n)])
    normalise = np.r_[np.ones(m), 0][None, :]
    result = linprog(
        np.r_[np.zeros(m), -1],
        A_ub=payoff_constraints,
        b_ub=np.zeros(n),
        A_eq=normalise,
        b_eq=[1],
        bounds=[(0, None)] * m + [(None, None)],
        method="highs",
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result.x[:-1], result.x[-1]


def main():
    A = ZERO_SUM.copy()
    p, lo = maximin(A)
    q, negative_hi = maximin(-A.T)
    print("Row mix:", np.round(p, 4), "Column mix:", np.round(q, 4))
    print(
        f"Value {p@A@q:.5f} | Primal–dual gap {-negative_hi-lo:.3g} | Best-response gap {gap(A,p,q):.3g}"
    )
    assert np.isclose(p.sum(), 1) and np.isclose(q.sum(), 1) and gap(A, p, q) < 1e-7
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.7, 3), constrained_layout=True)
    im = a.imshow(A, cmap="RdBu", vmin=-2, vmax=2)
    a.set(
        title="Row player’s payoffs",
        xlabel="Column action",
        ylabel="Row action",
        xticks=range(3),
        yticks=range(3),
    )
    for (i, j), v in np.ndenumerate(A):
        a.text(j, i, str(int(v)), ha="center", va="center")
    show_mixes(p, q, b)
    plt.show()


if __name__ == "__main__":
    main()
