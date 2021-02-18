import os

import cv2
import numpy as np


def split_image_pairs(
    sample: np.array,
    label: np.array,
    height: int,
    width: int,
    remove_black_images: bool = True,
    threshold: float = 360,
):
    """"""
    couples = []
    assert (
        sample.shape == label.shape
    ), f"Non-matching dimensions between image {sample.shape} and label {label.shape}"
    h, w, _ = sample.shape
    steps_x = w // width
    steps_y = h // height
    for row in range(0, steps_y):
        for column in range(0, steps_x):
            left = column * width
            right = (column + 1) * width
            top = row * height
            bottom = (row + 1) * height
            cropped_sample = sample[top:bottom, left:right]
            non_zero = np.count_nonzero(cropped_sample)
            if non_zero < threshold and remove_black_images:
                continue
            cropped_label = label[top:bottom, left:right]
            couples.append((cropped_sample, cropped_label))
    return couples


def split_dataset(
    root_dir: str,
    dataset_name: str,
    split_height: int = 256,
    split_width: int = 256,
):
    """"""
    dataset_folder = os.path.join(root_dir, dataset_name + "_dataset")
    raw_labels_folder = os.path.join(
        dataset_folder, dataset_name + "_raw_labels"
    )
    raw_samples_folder = os.path.join(
        dataset_folder, dataset_name + "_raw_samples"
    )

    assert os.path.exists(
        raw_labels_folder
    ), f"{raw_labels_folder} not in {dataset_folder}"
    assert os.path.exists(
        raw_samples_folder
    ), f"{raw_samples_folder} not in {dataset_folder}"

    labels_folder = os.path.join(dataset_folder, dataset_name + "_labels")
    samples_folder = os.path.join(dataset_folder, dataset_name + "_samples")

    assert not os.path.exists(
        labels_folder
    ), f"{labels_folder} already in {dataset_folder}"
    assert not os.path.exists(
        samples_folder
    ), f"{samples_folder} already in {dataset_folder}"

    os.mkdir(labels_folder)
    os.mkdir(samples_folder)

    # Gather sample and its label
    samples_files = sorted(os.listdir(raw_labels_folder))
    labels_files = sorted(os.listdir(raw_labels_folder))
    assert len(samples_files) == len(labels_files), "Samples do not match"

    for sample_name, label_name in zip(samples_files, labels_files):

        sample_array = cv2.imread(os.path.join(raw_labels_folder, sample_name))
        label_array = cv2.imread(os.path.join(raw_labels_folder, label_name))
        # Split each pair sample / label into smaller parts
        split_pairs = split_image_pairs(
            sample_array, label_array, split_height, split_width
        )
        # Save each smaller parts
        for k, (small_sample, small_label) in enumerate(split_pairs):
            s_name, s_extension = os.path.splitext(sample_name)
            l_name, l_extension = os.path.splitext(label_name)
            cv2.imwrite(
                os.path.join(samples_folder, s_name + f"_{k}" + s_extension),
                small_sample,
            )
            cv2.imwrite(
                os.path.join(labels_folder, l_name + f"_{k}" + l_extension),
                small_label,
            )


if __name__ == "__main__":
    split_dataset("./data", "EM")
