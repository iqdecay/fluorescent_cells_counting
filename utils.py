import czifile
import numpy as np


def load_czi_image(filename: str) -> np.array:
    """Load the czi image, and remove useless dimensions if needed"""
    image = czifile.imread(filename)
    print(f"Loaded image with dimensions {image.shape}, removing useless dimensions")
    return np.squeeze(image)
