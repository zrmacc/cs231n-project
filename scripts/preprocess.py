"""
Purpose: Prepare images for modeling.

@author: zmccaw
"""

import numpy as np
import os

X_SLICE = 128
Y_SLICE = 128
Z_SLICE = 64

# -----------------------------------------------------------------------------


# Prepare image.
def prepare_image(img: np.ndarray) -> np.ndarray:
    """Prepare image.

    Create normalize and create crosshair slices.

    """
    # Create 3x 2D slices.
    xy_slice = img[:, :, Z_SLICE]
    yz_slice = np.pad(img[X_SLICE, :, :], ((0, 0), (64, 64)))
    xz_slice = np.pad(img[:, Y_SLICE, :], ((0, 0), (64, 64)))
    out = np.stack((xy_slice, yz_slice, xz_slice), axis=2)

    # Min-max normalize.
    if (np.max(out) == np.min(out)):
        return None
    out = (out - np.min(out)) / (np.max(out) - np.min(out))

    out = np.transpose(out, (2, 0, 1))
    return out


# Prepare images.
def prepare_images(blocks: np.ndarray, out_dir: str, out_prefix: str) -> None:
    """Prepare all images."""
    n = blocks.shape[0]
    for i in range(n):

        # Check if export already exists.
        out_file = os.path.join(out_dir, f"{out_prefix}_{i}.npy")
        if os.path.exists(out_file):
            continue

        # Prepare image. Continue if empty.
        out = prepare_image(blocks[i, :, :, :])
        if out is not None:
            np.save(out_file, out)

    return None


# -----------------------------------------------------------------------------

# Positive examples.
pos = np.load("col_pos_blocks.npy")
prepare_images(pos, "data/pos", "pos")

# Negative examples.
neg = np.load("col_neg_blocks.npy")
prepare_images(neg, "data/neg", "neg")
