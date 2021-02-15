# import subprocess
import os
import random
from typing import List, Tuple

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF


class MixedDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        datasets: List[str],
        transform: bool,
    ):
        """Gather all train, validation and test data.

        :param root_dir: directory with all the images.
        :param datasets: list of datasets used to train (B5, B39, ...).
        :param transformer: whether to apply data transformation or not.
        """
        super(MixedDataset, self).__init__()
        self.root_dir = root_dir
        self.datasets = datasets
        self.transform = transform

        self._load_images()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> Tuple[int, int]:
        """"""
        if torch.is_tensor(index):
            index = index.tolist()

        sample_path, label_path = self.data[index]
        sample_image = Image.open(sample_path).resize((256, 256)).convert("L")
        label_mask = Image.open(label_path).resize((256, 256)).convert("L")

        if self.transform:
            sample_image, label_mask = self._transform(
                sample_image, label_mask
            )

        return sample_image, label_mask

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
            assert len(samples_files) == len(
                labels_files
            ), "Samples do not match"
            # Store them
            for sample_name, label_name in zip(samples_files, labels_files):
                assert os.path.splitext(sample_name) == os.path.splitext(
                    sample_name
                ), "Samples do not match"
                self.data.append(
                    (
                        os.path.join(samples_directory, sample_name),
                        os.path.join(labels_directory, label_name),
                    )
                )

    def _transform(self, image: np.array, mask: np.array):
        """Perform same transformations on image and its mask."""
        # Transform to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=(256, 256)
        )
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        return image, mask


if __name__ == "__main__":
    # # Download datasets
    # subprocess.call(["sh", "./scripts/download_datasets"])

    dataset = MixedDataset("./data", ["B39", "EM", "TNBC", "ssTEM"], True)
    for i in range(len(dataset)):
        sample, label = dataset[i]

    print(i, sample.shape, label.shape)

    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(sample.transpose(0, 2), cmap="gray")
    axarr[1].imshow(label.transpose(0, 2), cmap="gray")
    plt.show()
