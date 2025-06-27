import numpy as np
import scipy.ndimage
from skimage.feature import hog
import cv2


def compute_hog_features(gray_images: np.ndarray) -> np.ndarray:
    """
    Compute features based on histograms of oriented gradients (HOG) for a given set of images.

    :param gray_images: input images.
    :return: features for each image.
    """
    hog_features = np.stack([hog(im, cells_per_block=(1, 1), pixels_per_cell=(10, 10)) for im in gray_images])
    hog_features = (hog_features - np.min(hog_features)) / (np.max(hog_features) - np.min(hog_features))
    return hog_features


def compute_center_of_mass(gray_images: np.ndarray) -> np.ndarray:
    """
    Compute centers of mass for all input images.

    :param gray_images: input images.
    :return: center of mass for each image.
    """
    centers_of_mass = np.zeros((len(gray_images), 2))
    for i in range(len(gray_images)):
        centers_of_mass[i] = scipy.ndimage.center_of_mass(gray_images[i])
    centers_of_mass /= np.array(gray_images.shape[1:])
    return centers_of_mass


def compute_center_of_mass_per_component(gray_images: np.ndarray, threshold: float = 1e-5) -> np.ndarray:
    """
    Compute centers of mass for all input images.
    For each image, we first binarize it, determine its components, then compute the center mass for each component.

    :param gray_images: input images.
    :param threshold: binarization threshold.
    :return: center of mass for each component of each image.
    """
    movement_mask = (np.var(gray_images, axis=0) > threshold).astype(np.uint8)
    n_components, label_mask, stats, centroids = cv2.connectedComponentsWithStats(movement_mask)
    centers = np.zeros((len(gray_images), 2 * n_components - 2))
    for j, image in enumerate(gray_images):
        for i in range(1, n_components):
            center_of_mass = scipy.ndimage.center_of_mass(image, label_mask, i)
            centers[j, i * 2 - 2:i * 2] = center_of_mass
    centers /= np.tile(np.array(gray_images.shape[1:]), n_components - 1)
    return centers
