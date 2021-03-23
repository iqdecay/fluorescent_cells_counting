import os

import cv2
import numpy as np
from .image_processing import load_tiff_image


def count_cells_in_mask(mask: np.array):
    """
    Given a binary mask in argument, return the number of cells found
    in that mask
    :param mask: the mask of the cells
    :return: the number of cells in the mask
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    n_contours = len(contours)
    return n_contours


def test_cell_counting():
    image_list = [("aaa_0.tif", 2),
                  ("aaa_1.tif", 3),
                  ("aaa_2.tif", 2),
                  ("aaa_3.tif", 3),
                  ("aaa_4.tif", 2),
                  ("aaa_5.tif", 3),
                  ("aaa_6.tif", 3),
                  ("aaa_7.tif", 3),
                  ("aaa_8.tif", 1),
                  ("aaa_9.tif", 1),
                  ("aaa_10.tif", 3),
                  ]
    for (fname, n_cells) in image_list:
        image = load_tiff_image(os.path.join("labels", fname))
        contours_found = count_cells_in_mask(image)
        if contours_found != n_cells:
            raise ValueError(f"File {fname} : found {contours_found} cells, expected {n}")
        else:
            print(f"File {fname} : found the right number of cells")
