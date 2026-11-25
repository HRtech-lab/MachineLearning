# Representation Learning

JEPA, I-JEPA, V-JEPA, SimCLR and a denoising autoencoder, with frozen probes.

| Example | Question or method |
|---|---|
| [JEPA](01%20JEPA/jepa.ipynb) · [Python](01%20JEPA/jepa.py) | A predictor receives a partially observed digit and predicts the clean view in an embedding space. This small paired-view JEPA uses a moving-average target and a variance penalty to discourage collapse. |
| [I-JEPA](02%20I-JEPA/i_jepa.ipynb) · [Python](02%20I-JEPA/i_jepa.py) | An 8×8 digit is divided into sixteen 2×2 patches. A transformer predicts two held-out spatial blocks in the embedding space of a full-image target encoder. |
| [V-JEPA](03%20V-JEPA/v_jepa.ipynb) · [Python](03%20V-JEPA/v_jepa.py) | A moving disk is divided into non-overlapping 2-frame tubelets of 4×4 pixels. The target is a contextual feature vector for each hidden tubelet. |
| [SimCLR](04%20SimCLR/simclr.ipynb) · [Python](04%20SimCLR/simclr.py) | Two independently corrupted versions of the same digit form a positive pair. Embeddings from other digits in the batch supply negatives without consulting their labels. |
| [Denoising Autoencoder](05%20Denoising%20Autoencoder/denoising_autoencoder.ipynb) · [Python](05%20Denoising%20Autoencoder/denoising_autoencoder.py) | An encoder maps a corrupted digit to sixteen numbers. Its decoder reconstructs clean pixel intensities from that bottleneck. |

## Frozen-probe results

Each row uses the same linear classifier settings across its three feature sets. Accuracy is the fraction of correct held-out labels. The video example has four motion classes, and the digit examples have ten classes.

| Encoder | Test task | Raw inputs | Random encoder | Trained encoder | Effective rank after training |
|---|---|---:|---:|---:|---:|
| JEPA | Digits | 0.969 | 0.956 | 0.791 | 2.09 |
| I-JEPA | Digits | 0.969 | 0.980 | 0.980 | 19.19 |
| V-JEPA | Motion | 0.362 | 0.300 | 0.338 | 100.59 |
| SimCLR | Digits | 0.969 | 0.951 | 0.947 | 7.67 |
| Denoising Autoencoder | Digits | 0.969 | 0.880 | 0.916 | 6.91 |

The paired-view JEPA encoder had 32 coordinates and an effective rank of 2.09 on held-out digits. Many coordinates varied together, and its probe lost accuracy. The I-JEPA probe tied its random-encoder baseline on this split. These results describe the small experiments in this collection.
