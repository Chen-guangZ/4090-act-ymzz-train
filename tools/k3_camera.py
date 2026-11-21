"""Projection helpers for K3 D405 image measurements."""

import numpy as np


def optical_rays(pixels, camera):
    """Convert D405 plumb_bob image coordinates to rays with optical z equal to one."""
    k = np.asarray(camera["k"]).reshape(3, 3)
    distorted = (np.asarray(pixels) - k[:2, 2]) / k.diagonal()[:2]
    k1, k2, p1, p2, k3 = camera["d"]
    points = distorted.copy()
    for _ in range(8):
        x, y = points.T
        r2 = x*x + y*y
        radial = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
        tangential = np.column_stack((2*p1*x*y + p2*(r2 + 2*x*x), p1*(r2 + 2*y*y) + 2*p2*x*y))
        points = (distorted - tangential) / radial[:, None]
    return np.column_stack((points, np.ones(len(points))))
