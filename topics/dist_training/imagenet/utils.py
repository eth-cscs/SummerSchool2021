import os
import math
import numpy as np
import matplotlib.pyplot as plt

def minmax_norm(img, axis=(0,1)):
    img -= img.min(axis=axis, keepdims=True)
    img /= img.max(axis=axis, keepdims=True)
    return img

def plot_grid(images, title=None, rows=None, norm_axis=None, figsize=None):
    ''' plot a grid of images '''
    images = minmax_norm(images.copy(), axis=norm_axis)
    n = images.shape[0]
    if rows is None:
        rows = math.ceil(math.sqrt(n))
    cols = math.ceil(n / rows)

    fig, axs = plt.subplots(rows, cols, figsize=figsize, constrained_layout=True)
    fig.suptitle(title)
    for r in range(rows):
        for c in range(cols):
            idx = c + r * cols
            ax = axs[c] if rows == 1 else axs[r] if cols == 1 else axs[r,c]
            if idx < n:
                ax.imshow(images[idx])
            ax.set_axis_off()
    return fig, axs
