"""Software inventory coverage

Audit a small bill of materials for missing package metadata.

CycloneDX, specification overview
https://cyclonedx.org/specification/overview/
NIST Cybersecurity Framework 2.0
https://doi.org/10.6028/NIST.CSWP.29"""

import numpy as np
import matplotlib.pyplot as plt


def main():
    components = [
        {"name": "chart", "version": "2.1", "licence": "MIT"},
        {"name": "parser", "version": "1.4", "licence": "BSD-3-Clause"},
        {"name": "table", "version": "1.0"},
        {"name": "render", "licence": "Apache-2.0"},
    ]
    fields = ["name", "version", "licence"]
    present = np.array([[bool(r.get(k)) for k in fields] for r in components])
    coverage = present.mean(0)
    print("field coverage", dict(zip(fields, coverage)))
    print("complete records", present.all(1).sum(), "of", len(components))
    assert np.array_equal(present.sum(0), [4, 3, 3])
    fig, ax = plt.subplots(figsize=(5, 3.1), constrained_layout=True)
    ax.imshow(present, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set(
        xticks=range(len(fields)),
        xticklabels=fields,
        yticks=range(len(components)),
        yticklabels=[r["name"] for r in components],
        title="Recorded inventory fields",
    )
    plt.show()


if __name__ == "__main__":
    main()
