"""Salted password verifiers

Create and verify independent salted records for a single demonstration passphrase held only in memory.

OWASP, Password Storage Cheat Sheet
https://cheatsheetseries.owasp.org/cheatsheets/Password_Storage_Cheat_Sheet.html
Python, hashlib
https://docs.python.org/3/library/hashlib.html"""

import numpy as np
import matplotlib.pyplot as plt
import hashlib, hmac, secrets


def main():
    n, r, p = 2**17, 8, 1

    def derive(value, salt):
        return hashlib.scrypt(value, salt=salt, n=n, r=r, p=p, maxmem=256 * 1024**2, dklen=32)

    phrase = secrets.token_bytes(24)
    s1, s2 = secrets.token_bytes(16), secrets.token_bytes(16)
    a, b = derive(phrase, s1), derive(phrase, s2)
    checks = [
        hmac.compare_digest(a, derive(phrase, s1)),
        hmac.compare_digest(a, b),
        hmac.compare_digest(a, derive(secrets.token_bytes(24), s1)),
    ]
    print("same record / different salt / different value", checks)
    assert checks == [True, False, False]
    ns = np.array([2**15, 2**16, 2**17, 2**18])
    fig, ax = plt.subplots(figsize=(6, 3), constrained_layout=True)
    ax.bar([f"2^{int(np.log2(x))}" for x in ns], 128 * ns * r / (1024**2))
    ax.set(
        xlabel="scrypt cost N (r=8)",
        ylabel="Nominal ROMix memory (MiB)",
        title="Approximate memory per derivation",
    )
    plt.show()


if __name__ == "__main__":
    main()
