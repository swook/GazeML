"""Utility methods for generating and visualizing heatmaps."""
import numpy as np


def gaussian_2d(shape, centre, sigma=1.0):
    """Generate heatmap with single 2D gaussian."""
    xs = np.arange(0.5, shape[1] + 0.5, step=1.0, dtype=np.float32)
    ys = np.expand_dims(np.arange(0.5, shape[0] + 0.5, step=1.0, dtype=np.float32), -1)
    alpha = -0.5 / (sigma**2)
    heatmap = np.exp(alpha * ((xs - centre[0])**2 + (ys - centre[1])**2))
    return heatmap
