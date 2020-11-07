import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy
from scipy.io import loadmat
import sklearn
from sklearn.feature_extraction import image


def display_image(path: str):
    """
    Display the image of the MicroNet dataset (matlab) dataset.

    path: the path of the image which needs to be loaded.
    """
    images = loadmat(path)
    plt.imshow(images["GTCellSegmentdata"])
    plt.show()
