import os
import subprocess
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset


class MixedDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        datasets: List[str],
    ):
        """Gather all train, validation and test data.

        :param root_dir: directory with all the images.
        :param datasets: list of datasets used to train (B5, B39, ...).
        """
        super(MixedDataset, self).__init__()
        self.root_dir = root_dir
        self.datasets = datasets

        # Download datasets
        subprocess.call(["sh", "./scripts/download_datasets"])

        self._load_images()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> Tuple[int, int]:
        """"""
        if torch.is_tensor(index):
            index = index.tolist()

        sample_path, label_path = self.data[index]
        sample_image = cv2.imread(sample_path)
        label_image = cv2.imread(label_path)

        return sample_image, label_image

    def _load_images(self):
        """Load path to images of each selected datasets."""
        self.data = []
        for dataset_name in self.datasets:
            data_directory = os.path.join(
                self.root_dir, dataset_name + "_dataset"
            )
            samples_directory = os.path.join(
                data_directory, dataset_name + "_samples"
            )
            labels_directory = os.path.join(
                data_directory, dataset_name + "_labels"
            )
            # Gather sample and its label
            samples_files = sorted(os.listdir(samples_directory))
            labels_files = sorted(os.listdir(labels_directory))
            # Store them
            self.data.extend(
                [
                    (
                        os.path.join(samples_directory, sample),
                        os.path.join(labels_directory, label),
                    )
                    for sample, label in zip(samples_files, labels_files)
                ]
            )


if __name__ == "__main__":
    dataset = MixedDataset("./data", ["EM", "TNBC"])
    for i in range(len(dataset)):
        sample, label = dataset[i]

    print(i, sample.shape, label.shape)

    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(sample)
    axarr[1].imshow(label)
    plt.show()
