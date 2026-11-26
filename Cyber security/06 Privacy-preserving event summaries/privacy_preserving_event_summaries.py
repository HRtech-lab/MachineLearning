"""Privacy-preserving event summaries

Produce a diagnostic summary from an allowlist of fields.

OWASP, Logging Cheat Sheet
https://cheatsheetseries.owasp.org/cheatsheets/Logging_Cheat_Sheet.html"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


def main():
    rows = [
        {"component": "reader", "status": "ok", "account": "demo-a", "comment": "sample"},
        {"component": "writer", "status": "retry", "account": "demo-b"},
        {"component": "reader", "status": "ok", "extra": {"token": "fixture"}},
        {"component": "writer", "status": "ok"},
    ]
    allowed = {"reader", "writer"}
    states = {"ok", "retry"}
    public = [
        {"component": r["component"], "status": r["status"]}
        for r in rows
        if r.get("component") in allowed and r.get("status") in states
    ]
    summary = Counter((r["component"], r["status"]) for r in public)
    print("aggregate counts", sorted(summary.items()))
    assert all(set(r) == {"component", "status"} for r in public)
    fig, ax = plt.subplots(figsize=(5.7, 3), constrained_layout=True)
    names = sorted(allowed)
    bottom = np.zeros(len(names))
    for status in sorted(states):
        vals = np.array([summary[(n, status)] for n in names])
        ax.bar(names, vals, bottom=bottom, label=status)
        bottom += vals
    ax.set(ylabel="Events", title="Diagnostic summary after projection")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
