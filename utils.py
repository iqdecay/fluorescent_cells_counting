import os

import czifile
import tifffile
import numpy as np
from sympy.ntheory import factorint


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
    Convert a `.czi` file into 3 `.tiff` files corresponding to each channel.
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
        print(f"{channel_filename} saved")

    # Delete the stacked `.tiff` file
    os.remove(stacked_filename)


def split_image(filename: str, n_chunks: int):
    """
    Split an image into `n` chunks and save them in a directory with a
    sufix "_ij" corresponding to the ij-th chunk.

    :param filename: path to the file to load.
    :param n_chunks: number (not prime) of chunks to split the initial image.
    """
    # Load the large image to chunk
    large_image = tifffile.imread(filename)

    # Find the optimal number of rows and colums to split the initial image in
    # `n_chunks` by finding the closest prime factor of `n_chunks`
    # from `np.sqrt(n_chunks)`
    prime_factors = np.array([k for k in factorint(n_chunks)])
    n_rows = prime_factors[np.argmin(prime_factors - np.sqrt(n_chunks))]
    n_columns = n_chunks // n_rows
    print(
        f"Split the image into {n_chunks} chunks: "
        f"{n_rows} rows and {n_columns} columns"
    )

    # Compute dimensions of chunks and a padding size to add to extrem chunks
    # in order to have chunks of same dimensions
    large_height, large_width = large_image.shape
    chunk_height = large_width // n_columns
    chunk_width = large_height // n_rows

    # Create a directory to save chunks
    saving_directory, _ = os.path.splitext(filename)
    if not os.path.exists(saving_directory):
        os.makedirs(saving_directory)
    # Get the chunk filename to save without suffix and extension
    saving_filename = os.path.join(
        saving_directory, os.path.basename(saving_directory)
    )

    for i in range(n_rows):
        width_min = chunk_height * i
        width_max = width_min + chunk_height
        for j in range(n_columns):
            height_min = chunk_width * j
            height_max = height_min + chunk_width
            # Select chunk inside the large image
            chunk = large_image[width_min:width_max, height_min:height_max]

            # Pad the chunk with zeros if it is too small (border chunks)
            pad_height = chunk_height - chunk.shape[0]
            pad_width = chunk_width - chunk.shape[1]
            chunk = np.pad(chunk, ((0, pad_height), (0, pad_width)))

            # Save the chunk
            chunk_filename = "".join(
                [saving_filename, "_", str(i), str(j), ".tiff"]
            )
            tifffile.imwrite(chunk_filename, chunk)

    print(f"All chunks saved in {saving_directory}")
