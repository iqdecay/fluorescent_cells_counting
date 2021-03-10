import os
from itertools import product
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy
from scipy.spatial.distance import squareform, pdist
from sklearn.linear_model import LinearRegression

from image_processing import load_tiff_image


def extract_features_centers(image: np.array) -> Tuple[
    List[Tuple[float, float]], np.array]:
    """
    Extract areas in the centers of each square of the noise grid: those areas
    should be less noisy. Due to the irregularity of acquisition gathering we
    need to be smart, to do so, we process in multiple steps:

        - Step 1: extract contours of each noise circles in the grid. Cells
          area can hide those circles, thus we won't detect every area;

        - Step 2: find coordinates of each center of detected circles;

        - Step 3: group centers that should be on the same vertical and then
          horizontal line;

        - Step 4: interpolate lines described by those points

        - Step 5: find each intersection between those lines. Thanks to this we
          can have coordinates of hidden circles;

    :param filename: path to the file to load.
    :return: coordinates of each feature centers, the image with normalized
    histogram
    """
    # Step 1
    # Equalize the histogram of the loaded image
    equalized_lut = equalize_histogram(image)
    equalized_image = equalized_lut[image]
    # Inverse the LUT (blacks become whites) in order to highlight noise
    inverse_lut = 255 - np.arange(0, 256, 1)
    inversed_image = inverse_lut[equalized_image]
    # Emphasize noise areas and detect their centers
    emphasized_image = emphasize_noise_areas(inversed_image)
    noise_areas_df = detect_noise_areas(emphasized_image)

    # Step 2 and 3
    # Group points that should describe the same vertical and horizontal line
    X_groups = get_point_groups(noise_areas_df["X"].to_numpy().reshape(-1, 1))
    Y_groups = get_point_groups(noise_areas_df["Y"].to_numpy().reshape(-1, 1))
    # Store it in the dataframe
    noise_areas_df["X_groups"] = X_groups
    noise_areas_df["Y_groups"] = Y_groups

    # Step 4
    # Interpolate the grid
    horizontal_lines = interpolate_lines(noise_areas_df, "X", "Y")
    vertical_lines = interpolate_lines(noise_areas_df, "Y", "X")

    # Step 5
    # Find intersection points of the grid
    centers_coordinates = get_intersection_points(
        horizontal_lines, vertical_lines
    )

    return centers_coordinates, equalized_image


def get_intersection_points(
        horizontal_lines: List[float], vertical_lines: List[float]
) -> List[Tuple[float, float]]:
    """Find coordinates of intersection between horizontal and vertical lines.

    :param horizontal_lines: slopes and intercepts of horizontal lines.
    :param vertical_lines: slopes and intercepts of vertical lines.
    :return: coordinates of intersections.
    """
    centers_coordinates = []
    for l1, l2 in product(horizontal_lines, vertical_lines):
        # Get the slope and the intercept of current lines
        a1, b1 = l1
        c2, d2 = l2

        # Inverse the basis of the vertical line parameters
        # x = c2 * y + d2 => y = (x / c2) - (d2 / c2) = a2 * x + b2
        a2 = 1 / c2
        b2 = -d2 / c2

        # Compute the intersection coordinates and store it
        x = (b2 - b1) / (a1 - a2)
        centers_coordinates.append((x, a1 * x + b1))

    return centers_coordinates


def interpolate_lines(
        noise_areas_df: pd.DataFrame, x_name: str, y_name: str
) -> List[Tuple[float, float]]:
    """Interpolate a line passing through each group of points.

    :param noise_areas_df: point coordinates and their attributed group.
    :param x_name: name of the column of x-axis coordinates.
    :param y_name: name of the column of y-axis coordinates.
    :return: slopes and intercepts of lines.
    """
    # Get attributed group indices
    groups = noise_areas_df.groupby(f"{y_name}_groups")

    line_parameters = []
    for name, group in groups:
        # Check whether the group has more than 1 point (its name is -1) or not
        if name >= 0:
            # Get x and y coordinates of points
            X = group[x_name].to_numpy().reshape(-1, 1)
            Y = group[y_name].to_numpy().reshape(-1, 1)
            # Initialize and fit a line passing through each point of the group
            regressor = LinearRegression()
            regressor.fit(X, Y)
            # Store the parameters
            line_parameters.append(
                (regressor.coef_[0][0], regressor.intercept_[0])
            )

    return line_parameters


def get_point_groups(single_dim_coordinates: np.array) -> List[int]:
    """Gather point that seems to be on the same line.

    :param single_dim_coordinates: x or y coordinates of points.
    :return: list of group indices of each point.
    """
    n_points = single_dim_coordinates.size

    # Compute pairwise distance between each coordinate
    distances = squareform(pdist(single_dim_coordinates))

    # Gather points that are close from each others (will be on the same line)
    visited, groups = [], []
    # Gather points separated from a distance inferior than the threshold
    distance_threshold = 500
    for i in range(n_points):
        # Check whether the point has been visited or not
        if i in visited:
            pass

        else:
            # Mark the point as visited and add it in the current group
            visited.append(i)
            current_group = [i]
            for j in range(n_points):
                # Check whether the point is close from the current point
                if (distances[i][j] < distance_threshold) and (
                        j not in visited
                ):
                    # Mark the point as visited and add it in the current group
                    visited.append(j)
                    current_group.append(j)

            groups.append(current_group)

    # Fill a list: i-th place gives the group index of the i-th point
    sorted_group = np.zeros(n_points)
    for group_name, group_points in enumerate(groups):
        # Mark single element with -1 to avoid outliers
        if len(group_points) == 1:
            sorted_group[group_points[0]] = -1
        # Otherwise fill the list with group points
        else:
            for point in group_points:
                sorted_group[point] = group_name

    return sorted_group


def emphasize_noise_areas(image: np.array) -> np.array:
    """Erode and dilate the image in order to emphasize noise areas.

    :param image: image to process.
    :return: processed image.
    """
    # Apply a binary threshold to separate noise areas and background
    _, processed_image = cv2.threshold(
        image.astype(np.uint8), 200, 255, cv2.THRESH_BINARY
    )

    # Define erosion parameters and performs it
    erosion_kernel = np.ones((5, 5), np.uint8)
    erosion_iterations = 3
    eroded_image = cv2.erode(
        processed_image, erosion_kernel, iterations=erosion_iterations
    )
    # Define dilatation parameters and performs it
    dilatation_kernel = np.ones((20, 20), np.uint8)
    dilatation_iterations = 5
    emphasized_image = cv2.dilate(
        eroded_image, dilatation_kernel, iterations=dilatation_iterations
    )

    return emphasized_image


def detect_noise_areas(image: np.array) -> pd.DataFrame:
    """Detect centers of noise areas of an image by extracting their contours.

    :param image: image to process.
    :return: dataframe with coordinates of centers and surface of these areas.
    """
    # Detect contours of noises areas
    contours, _ = cv2.findContours(
        image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    # Compute moments of contours to calculate their areas and centroids
    contours_areas = []
    X_centroid, Y_centroid = [], []
    # Only contours with an area larger than a threshold are taken into account
    area_threshold = 10 ** 5
    for current_contour in contours:
        moments = cv2.moments(current_contour)
        # Compute countour's area with the moment `m00`, and store it
        area = moments["m00"]
        # Check if the area is larger than `area_threshold` to stor its data
        if area > area_threshold:
            # Compute and store the contour's centroid coordinates
            X_centroid.append(int(moments["m10"] / area))
            Y_centroid.append(int(moments["m01"] / area))
            # Store the contour's area
            contours_areas.append(area)

    noise_areas_df = pd.DataFrame.from_dict(
        {"X": X_centroid, "Y": Y_centroid, "area": contours_areas}
    )

    return noise_areas_df


def equalize_histogram(image: np.array) -> np.array:
    """Compute a lookup table to equalised a given image.

    :param image: image to process, equalize its histogram.
    :return: a lookup table to convert raw histogram to equalised one.
    """
    # Compute the histogram
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    # Compute the cumulative histogram
    cumulative_histogram = histogram.cumsum()

    # Mask zero pixels in order to not take them into account when
    # equalising the histogram
    masked_cumulative_histogram = np.ma.masked_equal(cumulative_histogram, 0)
    # Equalise the histogram
    masked_cumulative_histogram = (
            (masked_cumulative_histogram - masked_cumulative_histogram.min())
            * 255 / (masked_cumulative_histogram.max() -
                     masked_cumulative_histogram.min())
    )

    equalised_histogram = np.ma.filled(masked_cumulative_histogram, 0).astype(
        "uint8"
    )

    return equalised_histogram


def save_centers(filename: str, cropped: np.array, center_n: int) -> None:
    """
    Given a filename and a crop made around the nth center of that image
    save the crop under the corresponding directory
    :param filename: path to the .tiff file (red channel is recommended)
    :param cropped: a part of the tiff file
    :param center_n: number of the center the crop was made around
    """
    directory = os.path.dirname(filename)
    base = os.path.basename(filename)
    name, _ = os.path.splitext(base)
    height, width = cropped.shape[0], cropped.shape[1]
    crop_dir = name + f"_cropped_{height}x{width}"
    crop_dir = os.path.join(directory, crop_dir)
    if not os.path.exists(crop_dir):
        os.mkdir(crop_dir)
        print(f"Created {crop_dir} to save the cropped images for {filename}")
    else:
        print(f"Directory {crop_dir} already exists")
    path = os.path.join(crop_dir, f"center_{i}")
    np.save(path, cropped)
    print(f"Finding centers of {filename}")


def crop_centers(image: np.array, height: int, width: int) -> None:
    """
    Given an image, extract the noise centers of that image,
    and around each center, crop a sub-image with size height*width.
    :param image : the image, as a numpy array
    :param height: height of the crop around each center
    :param width: width of the crop around each center
    :return: cropped_images : list of cropped images
    """
    centers_coordinates, img = extract_features_centers(image)
    print(f"Found {len(centers_coordinates)} centers for image {img}")
    contour_mask,cell_contour = find_contour_image(image)
    h, w = img.shape
    cropped_images = []
    for i, (column, row) in enumerate(centers_coordinates):
        column = int(round(column))
        row = int(round(row))
        if 0 <= row < h and 0 <= column < w:
            left = max(0, column - width // 2)
            right = min(w, column + width // 2)
            top = max(0, row - height // 2)
            bottom = min(h, row + height // 2)
            cropped = image[top: bottom, left:right]
            contour_mask_cropped = contour_mask[top: bottom, left:right]
            cropped_cells = cv2.bitwise_and(cropped,cropped, mask = contour_mask_cropped)
            cropped_images.append(cropped_cells)
        else:
            print(
                f"Center {i} is out of bounds for image of size {h}x{w} "
                f"with coordinates {column, row}")
    #print(f"Saved {len(cropped_images)} cropped images")
    return cropped_images


def contract_edge(
        cell_contour: np.array, image: np.array,
        erosion_kernel: np.array = np.ones((9, 9), np.uint8),
        iterations: int = 50
) -> np.array:
    """
    Given an image and the contour of the cell, return the contracted
    contour.
    :param cell_contour: contour before erosion
    :param image: image on which we make pre-processing
    :param erosion_kernel: kernel which should be used for erosion
    :param iterations: number of iterations in the erosion process
    :return: cell_contour_eroded: contracted contour
    """
    cimg = np.zeros_like(image)
    cimg = cv2.fillPoly(cimg, pts=[cell_contour], color=(255, 255, 255))
    cimg = cv2.erode(cimg, erosion_kernel, iterations=iterations)
    cell_contour_eroded, _ = cv2.findContours(
        cimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )
    return cell_contour_eroded


def find_contour_image(
        image: np.array,
        erosion_kernel: np.array = np.ones((3, 3), np.uint8),
        dilatation_kernel: np.array = np.ones((30, 30), np.uint8),
        n_erosions: int = 2,
        n_dilatations: int = 5
) -> List[np.array,np.array]:
    """
    Given an image return the contour of the cells membrane
    :param image: image on which we make pre-processing
    :param erosion_kernel: kernel which should be used for erosion
    :param dilatation_kernel: kernel which should be used for dilatation
    :param n_erosionss: number of iterations in the erosion process
    :param n_dilatations: number of iterations in the dilatation process
    :return: cell_contour: contour of the cells membrane
    :return: contour_mask: mask corresponding to the contour
    """
    
    emphasized_image = emphasize_noise_areas(
    image,
    erosion_kernel=erosion_kernel,
    dilatation_kernel=dilatation_kernel,
    n_erosions=n_erosions,
    n_dilations=n_dilatations,
    )
    contours, _ = cv2.findContours(
    emphasized_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cell_contour = max(contours, key=cv2.contourArea)
    
    contour_mask = np.zeros_like(emphasized_image)
    contour_mask = cv2.fillPoly(
    contour_mask, pts =[cell_contour], color=(255,255,255)
    )
    return (contour_mask,cell_contour)

