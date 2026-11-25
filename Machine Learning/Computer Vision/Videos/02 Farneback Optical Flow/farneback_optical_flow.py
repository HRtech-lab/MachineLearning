"""Farneback Optical Flow

A textured surface translates by a known integer displacement between two frames. Estimate a displacement vector at each interior pixel.

Farnebäck, Two-Frame Motion Estimation Based on Polynomial Expansion (2003)
https://doi.org/10.1007/3-540-45103-X_50
OpenCV, motion analysis and object tracking
https://docs.opencv.org/4.x/dc/d6b/group__video__track.html"""

from pathlib import Path
import sys

start = Path(__file__).resolve().parent
for parent in (start, *start.parents):
    if (parent / "gallery" / "__init__.py").is_file():
        sys.path.insert(0, str(parent))
        break
else:
    raise RuntimeError("Open this example inside the Machine Learning collection.")


import cv2
import numpy as np
import matplotlib.pyplot as plt
from gallery.vision import translated_texture


def estimate(before, after):
    dense = cv2.calcOpticalFlowFarneback(before, after, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    yy, xx = np.mgrid[8 : before.shape[0] - 8, 8 : before.shape[1] - 8]
    return np.column_stack((xx.ravel(), yy.ravel())), dense[yy, xx].reshape(-1, 2), xx.size


def main():
    errors = []
    available = 0
    for seed in range(61, 85):
        before, after, target = translated_texture(seed)
        points, motion, count = estimate(before, after)
        errors.extend(np.linalg.norm(motion - target, axis=1))
        available += count
    errors = np.array(errors)
    print(f"Mean endpoint error: {errors.mean():.3f} px | Median: {np.median(errors):.3f} px")
    print(f"Within 1 px: {(errors<1).mean():.1%} | Valid fraction: {len(errors)/available:.1%}")
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3), constrained_layout=True)
    a.imshow(before, cmap="gray")
    step = max(1, len(points) // 95)
    p, v = points[::step], motion[::step]
    a.quiver(
        p[:, 0], p[:, 1], v[:, 0], v[:, 1], color="orange", angles="xy", scale_units="xy", scale=1
    )
    a.set_title(f"Flow, reference ({target[0]}, {target[1]}) px")
    a.axis("off")
    b.hist(errors, bins=24, color="#41718b")
    b.set(xlabel="Endpoint error (px)", ylabel="Count", title="Interior motion estimates")
    plt.show()


if __name__ == "__main__":
    main()
