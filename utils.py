import os

import czifile
import tifffile
import numpy as np


def load_czi_image(filename: str) -> np.array:
    """Load the czi image, and remove useless dimensions if needed.

    :param filename: path to the file to load.
    :return: image array.
    """
    image = czifile.imread(filename)
    print(
        f"Loaded image with dimensions {image.shape}, "
        "removing useless dimensions"
    )
    return np.squeeze(image)


def convert_czi_to_tiff(filename: str):
    """
    Convert a `.czi` file into 3 `.tiff` files coresponding to each channel.
    Files are saved in the same directory as the `.czi` with a sufix "_r", "_g"
    or "_b" according to their channel (RGB).

    :param filename: path to the file to load.
    """
    # Convert the `.czi` to `.tiff` and saved a channel stacked file
    czifile.czi2tif(filename)

    # Load the stacked `.tiff` file into arrays
    stacked_filename = "".join([filename, ".tif"])
    stacked_image = tifffile.imread(stacked_filename)

    # Save an image for each channel
    no_extension_filename, _ = os.path.splitext(filename)
    suffix = ["_b", "_g", "_r"]
    for k, channel_array in enumerate(stacked_image):
        channel_filename = "".join([no_extension_filename, suffix[k], ".tiff"])
        tifffile.imwrite(channel_filename, channel_array)
        print(f"{channel_filename} saved.")

    # Delete the stacked `.tiff` file
    os.remove(stacked_filename)
