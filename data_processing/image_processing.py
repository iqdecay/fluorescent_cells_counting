import errno
import os

import cv2
import czifile
import numpy as np
import tifffile


def load_czi_image(filename: str) -> np.array:
    """Load the czi image, and remove useless dimensions if needed.

    :param filename: path to the file to load.
    :return: the image array.
    """
    image = czifile.imread(filename)
    return np.squeeze(image)


def convert_czi_to_tiff(filename: str) -> str:
    """
    Convert a `.czi` file into a `.tiff` file and return the filename.
    File is saved in the same directory as the `.czi`.

    :param filename: path to the file to load.
    :return the `.tiff` filename.
    """
    tiff_filename = os.path.splitext(filename)[0] + ".tiff"
    if not os.path.exists(tiff_filename):
        czifile.czi2tif(filename, tiff_filename)
    else:
        print(f"A .tiff file for {filename} already exists in this directory.")
    return tiff_filename


def convert_czi_to_three_grayscale_tiff(filename: str) -> None:
    """
    Convert a `.czi` file into 3 `.tiff` grayscale files corresponding to each
    channel. Files are saved in the same directory as the `.czi` with a suffix
    "_r", "_g" or "_b" according to their channel (RGB).
    It is assumed that the .czi file is of shape (3, x, y) and NOT (3,3,x,y).

    :param filename: path to the file to load.
    """
    # Convert .czi to .tiff
    tiff_filename = convert_czi_to_tiff(filename)
    tiff_array = tifffile.imread(tiff_filename)

    # Check the generated image

    # Because the array is already squeezed, the image should have 3 dimensions
    # only : (channels, height, width). Otherwise we don't know how to convert
    # it to grayscale.
    n_dimensions = len(tiff_array.shape)
    if not n_dimensions == 3:
        print(f"File {filename} has {n_dimensions} dimensions, expected 3")
        raise
    # The image is supposed to be RGB (no alpha channel) or grayscale (intensity)
    n_channels = tiff_array.shape[0]
    if n_channels not in [1, 3]:
        print(f"File {filename} has {n_channels} channels, expected 1 or 3")
        raise
    if n_channels == 1:
        print(
            f"File {filename} has only one channel, {tiff_filename} "
            "is its grayscale `.tiff` version"
        )
        raise

    # Save an image for each channel
    no_extension_filename, _ = os.path.splitext(filename)
    suffix = ["_b", "_g", "_r"]
    for k, channel_array in enumerate(tiff_array):
        channel_filename = "".join([no_extension_filename, suffix[k], ".tiff"])
        if not os.path.exists(channel_filename):
            tifffile.imwrite(
                channel_filename, channel_array, photometric="minisblack"
            )
            print(f"Created a .tiff file for {filename} on channel {'BGR'[k]}")
        else:
            print(
                f"A .tiff file for {filename} on channel {'BGR'[k]} already"
                " exists in this directory."
            )


def load_tiff_image(filename: str) -> np.array:
    """Load a `.tiff` file and return it as a numpy array in grayscale (0 to 255).

    :param filename: path to the file to load.
    :return: the image as a numpy array.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), filename
        )

    return cv2.imread(filename, cv2.IMREAD_GRAYSCALE)


def split_image(filename: str, n_chunks: int) -> str:
    """
    Split an image into `n_chunks` chunks and save them in a directory with a
    suffix "_ij" corresponding to the ij-th chunk.

    :param filename: path to the file to load.
    :param n_chunks: number (not prime) of chunks to split the initial image.
    :return : the directory in which the files were saved.
    """
    # Load the large image to chunk
    large_image = load_tiff_image(filename)

    # Find the optimal number of rows and columns to split the initial image
    n_rows = int(np.sqrt(n_chunks))
    n_columns = n_chunks // n_rows
    print(
        f"Split the image into {n_chunks} chunks: "
        f"{n_rows} rows and {n_columns} columns"
    )

    # Compute dimensions of chunks and a padding size to add to border chunks
    # in order to have chunks of same dimensions
    large_height, large_width = large_image.shape
    chunk_height = large_width // n_columns
    chunk_width = large_height // n_rows

    # Create a directory to save chunks
    saving_directory, _ = os.path.splitext(filename)
    saving_directory = saving_directory + f"_{n_chunks}parts"
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
    return saving_directory
