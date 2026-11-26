# Cyber security

A defensive computing course built around local, reproducible checks. Begin by checking a file against a trusted record, then study authenticated records, password verifiers, access reviews, logging, read-only reporting, tested restores and signed releases. All examples operate on generated keys, temporary files, in-memory databases or constructed event records.

## Three questions about a file

Has the file changed? A digest and a trusted reference answer this for the bytes that were checked. Who could have authenticated it? HMAC relies on a shared secret, whereas a digital signature permits verification with a public key. Can a service recover the file? A restore test needs an independent destination and a manifest of the objects expected there. Each question has a separate experiment and observable success condition. [RFC 2104](https://www.rfc-editor.org/rfc/rfc2104) · [RFC 8032](https://www.rfc-editor.org/rfc/rfc8032) · [NIST CSF 2.0](https://doi.org/10.6028/NIST.CSWP.29)

## Counting evidence

An alert threshold trades sensitivity against review workload. The monitoring lesson sets a threshold from the past and uses known later events to count true and false alerts. The access review compares expected and observed grants, where an extra grant and a missing grant require different follow-up. The inventory lesson measures metadata coverage by field so that a nearly complete register cannot hide an entirely absent field.

![Monitoring event-volume shifts](05%20Monitoring%20event-volume%20shifts/figure.svg)

## Worked lectures

| Lecture | Question or experiment |
|---|---|
| [01. File integrity manifests](01%20File%20integrity%20manifests/README.md) | Check a directory snapshot against a previously recorded SHA-256 manifest. |
| [02. Authenticated records with HMAC](02%20Authenticated%20records%20with%20HMAC/README.md) | Verify the origin and integrity of a small record shared by two trusted components. |
| [03. Salted password verifiers](03%20Salted%20password%20verifiers/README.md) | Create and verify independent salted records for a single demonstration passphrase held only in memory. |
| [04. Role-based access reviews](04%20Role-based%20access%20reviews/README.md) | Compare an application permission matrix with a least-privilege baseline. |
| [05. Monitoring event-volume shifts](05%20Monitoring%20event-volume%20shifts/README.md) | Detect unusual volumes in a synthetic operational event series and measure false alerts. |
| [06. Privacy-preserving event summaries](06%20Privacy-preserving%20event%20summaries/README.md) | Produce a diagnostic summary from an allowlist of fields. |
| [07. Parameterised SQLite reads](07%20Parameterised%20SQLite%20reads/README.md) | Retrieve approved local inventory rows with bound query parameters. |
| [08. Testing a backup restore](08%20Testing%20a%20backup%20restore/README.md) | Measure whether a local restore reproduces every expected object. |
| [09. Verifying signed release artefacts](09%20Verifying%20signed%20release%20artefacts/README.md) | Use a public key to verify a release manifest and report which objects match it. |
| [10. Software inventory coverage](10%20Software%20inventory%20coverage/README.md) | Audit a small bill of materials for missing package metadata. |

## Running the labs

Install [requirements.txt](requirements.txt). Each folder contains a Python script, executed notebook, figure and references. The password-verifier exercise needs at least 128 MiB of working memory plus runtime overhead. Generated keys last only for the duration of the example. [MIT licence](../LICENSE)
