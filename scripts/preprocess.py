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
def prepare_image(img: np.ndarray, d=1) -> np.ndarray:
    """Prepare image.

    Takes 3 orthogonal slices through the center of the input image, each
    having depth d. Collapses across d using max projection, then stacks
    the orthogonal slices along the channel dimension. Because the Z
    dimension has half the extent of the X and Y dimensions, slices parallel
    have double depth (s.t. the same proportion of the dimension is projected),
    and are zero padded after projection to reach size (256, 256, 1).

    Args
    ----
        img: 3D image of shape (256, 256, 128).
        d: Slices across which to max project when forming orthogonal sections.
            Should be an odd integer.

    Returns
    -------
        Normalized image of shape (256, 256, 3).

    """
    # Ensure d is odd.
    if (d % 2 == 0):
        d += 1
    h = (d - 1) // 2  # Half width.

    # Create 3x 2D slices.
    # XY plane.
    xy_slice = img[:, :, (Z_SLICE - h):(Z_SLICE + h + 1)]
    assert xy_slice.shape[2] == (2 * h + 1)
    xy_slice = np.max(xy_slice, axis=2)
    assert xy_slice.shape == (256, 256)

    # YZ plane.
    yz_slice = img[(X_SLICE - 2 * h):(X_SLICE + 2 * h + 1), :, :]
    assert yz_slice.shape[0] == (2 * 2 * h + 1)
    yz_slice = np.max(yz_slice, axis=0)
    yz_slice = np.pad(yz_slice, ((0, 0), (64, 64)))
    assert xy_slice.shape == (256, 256)

    # XZ plane.
    xz_slice = img[:, (Y_SLICE - 2 * h):(Y_SLICE + 2 * h + 1), :]
    assert xz_slice.shape[1] == (2 * 2 * h + 1)
    xz_slice = np.max(xz_slice, axis=1)
    xz_slice = np.pad(xz_slice, ((0, 0), (64, 64)))
    assert xz_slice.shape == (256, 256)

    # Stack slices.
    out = np.stack((xy_slice, yz_slice, xz_slice), axis=2)

    # Min-max normalize.
    if (np.max(out) == np.min(out)):
        return None
    out = (out - np.min(out)) / (np.max(out) - np.min(out))
    return out


# Prepare images.
def prepare_images(blocks: np.ndarray, out_dir: str, out_prefix: str) -> None:
    """Prepare all images.

    Loops over batch axis of input tensor, preparing and saving each image.

    """
    n = blocks.shape[0]
    for i in range(n):

        # Check if export already exists.
        out_file = os.path.join(out_dir, f"{out_prefix}_{i}.npy")
        if os.path.exists(out_file):
            continue

        # Prepare and save image. Continue if empty.
        out = prepare_image(blocks[i, :, :, :])
        if out is not None:
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            np.save(out_file, out)

    return None


# -----------------------------------------------------------------------------

# Load examples.
pos = np.load("col_pos_blocks.npy")
neg = np.load("col_neg_blocks.npy")

depths = [1, 3, 5, 7]
# Loop of depths.
for d in depths:
    prepare_images(pos, f"data/pos_d{d}", "pos")
    prepare_images(pos, f"data/neg_d{d}", "neg")
