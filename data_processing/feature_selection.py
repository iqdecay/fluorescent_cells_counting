from itertools import product
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy
from scipy.spatial.distance import squareform, pdist
from sklearn.linear_model import LinearRegression

from image_processing import load_tiff_image


def get_features_areas(filename: str):
    """"""
    # Load the image
    raw_image = load_tiff_image(filename)

    # Equalize the histogram of the loaded image
    equalized_lut = equalize_histogram(raw_image)
    equalized_image = equalized_lut[raw_image]

    # Inverse the LUT (blacks become whites) in order to highlight noise
    inverse_lut = 255 - np.arange(0, 256, 1)
    inversed_image = inverse_lut[equalized_image]

    # Emphasize noise areas and detect their centers
    emphasized_image = emphasize_noise_areas(inversed_image)
    noise_areas_df = find_noise_areas(inversed_image)

    # Group points that should describe the same vertical and horizontal line
    X_groups = get_same_line_points(
        noise_areas_df["X"].to_numpy().reshape(-1, 1)
    )
    Y_groups = get_same_line_points(
        noise_areas_df["Y"].to_numpy().reshape(-1, 1)
    )
    # Store it in the dataframe
    noise_areas_df["X_groups"] = X_groups
    noise_areas_df["Y_groups"] = Y_groups

    # Interpolate the grid
    horizontal_lines = get_lines(noise_areas_df, "X", "Y")
    vertical_lines = get_lines(noise_areas_df, "Y", "X")

    # Find intersection points of the grid
    x_intersections, y_intersections = get_intersection_points(
        horizontal_lines, vertical_lines
    )

    return x_intersections, y_intersections


def get_intersection_points(
    horizontal_lines: List[float], vertical_lines: List[float]
) -> Tuple[List[float], List[float]]:
    x_intersections, y_intersections = [], []
    for l1, l2 in product(horizontal_lines, vertical_lines):
        a1, b1 = l1
        c2, d2 = l2

        a2 = 1 / c2
        b2 = -d2 / c2

        x_intersections.append((b2 - b1) / (a1 - a2))
        y_intersections.append(a1 * x + b1)

    return x_intersections, y_intersections


def get_lines(
    noise_areas_df: pd.DataFrame, x_name: str, y_name: str
) -> List[float]:
    """"""
    line_parameters = []
    groups = noise_areas_df.groupby(f"{y_name}_groups")
    for name, group in groups:
        if name >= 0:
            X = group[x_name].to_numpy().reshape(-1, 1)
            Y = group[y_name].to_numpy().reshape(-1, 1)

            regressor = LinearRegression()
            regressor.fit(X, Y)

            line_parameters.append(
                (regressor.coef_[0][0], regressor.intercept_[0])
            )

    return line_parameters


def get_point_groups(single_dim_coordinates: np.array) -> np.array:
    """"""
    n_points = single_dim_coordinates.size

    # Compute pairwise distance between each coordinate
    distances = squareform(pdist(single_dim_coordinates))

    # Group points that are close together: they will be on the same line
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

    # Get a list where at the ith place there is group number of the ith point
    sorted_group = np.zeros(n_points)
    for group_name, group_points in enumerate(groups):
        # Mark single element with -1 to avoid outliers
        if len(group_points) == 1:
            sorted_group[group_points[0]] = -1
        else:
            for point in group_points:
                sorted_group[point] = group_name

    return sorted_group


def emphasize_noise_areas(image: np.array) -> pd.DataFrame:
    """Erode and dilate the image in order to emphasize noise areas"""
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


def find_noise_areas(image: np.array) -> pd.DataFrame:
    """"""
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
    # Compute the cummulative histogram
    cumulative_histogram = histogram.cumsum()

    # Mask zero pixels in order to not take them into account when
    # equalising the histogram
    masked_cumulative_histogram = np.ma.masked_equal(cumulative_histogram, 0)
    # Equalise the histogram
    masked_cumulative_histogram = (
        (masked_cumulative_histogram - masked_cumulative_histogram.min())
        * 255
        / (
            masked_cumulative_histogram.max()
            - masked_cumulative_histogram.min()
        )
    )

    equalised_histogram = np.ma.filled(masked_cumulative_histogram, 0).astype(
        "uint8"
    )

    return equalised_histogram
