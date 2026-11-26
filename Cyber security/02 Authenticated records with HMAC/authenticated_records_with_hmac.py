"""Authenticated records with HMAC

Verify the origin and integrity of a small record shared by two trusted components.

RFC 2104, HMAC
https://www.rfc-editor.org/rfc/rfc2104
Python, HMAC and constant-time comparison
https://docs.python.org/3/library/hmac.html"""

import numpy as np
import matplotlib.pyplot as plt
import hmac, json, secrets


def main():
    key = secrets.token_bytes(32)

    def payload(event):
        return json.dumps(event, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

    event = {"event": "archive-complete", "count": 12}
    tag = hmac.digest(key, payload(event), "sha256")
    cases = [
        ("Original", event, tag),
        ("Reordered", dict(reversed(list(event.items()))), tag),
        ("Corrected count", {**event, "count": 13}, tag),
        ("Incomplete tag", event, tag[:-1]),
    ]
    out = [
        hmac.compare_digest(hmac.digest(key, payload(e), "sha256"), provided)
        for _, e, provided in cases
    ]
    print(dict(zip([c[0] for c in cases], out)))
    assert out == [True, True, False, False]
    fig, ax = plt.subplots(figsize=(6.8, 3), constrained_layout=True)
    ax.barh([c[0] for c in cases], [1] * 4, color=["#268173" if x else "#bd693e" for x in out])
    ax.set(xticks=[], title="Record verification")
    [
        ax.text(0.5, i, "accepted" if x else "rejected", ha="center", color="white")
        for i, x in enumerate(out)
    ]
    plt.show()


if __name__ == "__main__":
    main()
