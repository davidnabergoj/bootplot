import abc
from typing import List, Tuple
import scipy.spatial
import scipy.ndimage
import skimage.transform
import cv2
import numpy as np
from sklearn.decomposition import PCA
import scipy
import networkx as nx

from bootplot.image_features import compute_hog_features, compute_center_of_mass, compute_center_of_mass_per_component


class Sorter(abc.ABC):
    def __init__(self, verbose: bool = False):
        """
        Abstract class for sorting.

        :param verbose: if True, print additional information during sorting.
        """
        self.verbose = verbose

    @abc.abstractmethod
    def sort(self, images: np.ndarray, **kwargs) -> List[int]:
        """
        Sort images.

        :param images: input images.
        :param kwargs: keyword arguments for the sorting function.
        :return:
        """
        pass

    def verbose_print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)


class DefaultSorter(Sorter):
    def sort(self, images: np.ndarray, **kwargs) -> List[int]:
        """
        Does not sort images.

        :param images: input images.
        :param kwargs: unused.
        :return: indices of images in the original order.
        """
        return list(range(len(images)))  # Does not sort


class PCASorter(Sorter):
    def sort(self, gray_images: np.ndarray, features: str = "center_mass", **kwargs) -> List[int]:
        """
        Sort images with a PCA-based method.
        We first compute image features, then project these features into 1D.
        The order of the projection is the order of the images.

        :param gray_images: input images.
        :param features: type of features to be used for PCA. Must be one of "selected" (HOG, center mass, per component
            center mass), "center_mass" (center mass), "full" (flattened images).
        :param kwargs: unused.
        :return: image indices in for the new ordering.
        """
        if features == "selected":
            hf = compute_hog_features(gray_images)
            cm = compute_center_of_mass(gray_images)
            cmc = compute_center_of_mass_per_component(gray_images)
            features = np.c_[hf, cm, cmc]
        elif features == "center_mass":
            features = compute_center_of_mass(gray_images)
        elif features == "full":
            # Sort images according to 1D PCA projections of their centers of mass
            features = gray_images.reshape(len(gray_images), -1)
        else:
            raise ValueError(f"Unsupported features: '{features}'")
        model = PCA(n_components=1)
        projections = model.fit_transform(features).ravel()
        order = list(np.argsort(projections))
        return order


class TravelingSalesmanSorter(Sorter):
    def distance_matrix(self, gray_images: np.ndarray, features: str = "full") -> np.ndarray:
        """
        Compute distances for all pairs input images. The distances are based on image representations, given by
        the features parameter.

        :param gray_images: input images.
        :param features: type of features to be used for distance computation. Must be one of "selected" (HOG,
            center mass, per component center mass), "center_mass" (center mass), "full" (flattened images).
        :return: square pairwise distance matrix.
        """
        m = len(gray_images)
        if features == 'full':
            ksize = list(gray_images[0].shape[:2])
            ksize[0] += ksize[0] % 2 - 1
            ksize[1] += ksize[1] % 2 - 1
            ksize = ksize[::-1]

            gray_images = np.array([
                cv2.GaussianBlur(im, ksize=tuple(ksize), sigmaX=ksize[0] / 6, sigmaY=ksize[1] / 6)
                for im in gray_images
            ])
            # convolve with a large kernel so that elements are "spread out"
            bootstrapped_vectors = gray_images.reshape(m, -1)
            distance_matrix = scipy.spatial.distance.cdist(bootstrapped_vectors, bootstrapped_vectors)
        elif features == "hog":
            hf = compute_hog_features(gray_images)
            distance_matrix = scipy.spatial.distance.cdist(hf, hf)
        else:
            raise NotImplementedError(f"Features '{features}' are not implemented.")
        return distance_matrix

    def sort(self, gray_images: np.ndarray, features: str = "full", **kwargs) -> List[int]:
        """
        Sort images by posing the task as solving the Traveling Salesman Problem (TSP).
        We first compute pairwise distances between images. These are treated as a matrix which gives rise to an
        undirected weighted graph. By solving TSP on this graph, we obtain a Hamiltonian path (i.e. each vertex is
        theoretically visited only once), which is equal to the order of the sorted images. Note that an approximation
        of TSP is used and it is possible (likely) that an image may be encountered more than once.

        :param gray_images: input images.
        :param features: types of features to use when computing pairwise distances.
        :param kwargs: unused.
        :return: image indices in the final ordering.
        """
        self.verbose_print("Computing distance matrix")
        distance_matrix = self.distance_matrix(gray_images, features=features)
        self.verbose_print('Solving TSP')
        image_graph = nx.from_numpy_array(distance_matrix)
        order = nx.algorithms.approximation.traveling_salesman_problem(image_graph, cycle=False)
        return order


class HorizontalMassSorter(Sorter):
    def sort(self, gray_images: np.ndarray, **kwargs) -> List[int]:
        """
        Sort images using a horizontal center mass algorithm.
        For each image, we compute its center mass.
        Images are sorted according to the horizontal (x) component of the center mass.

        :param gray_images: input images.
        :param kwargs: unused.
        :return: image indices in the final ordering.
        """
        centers_x = []
        for image in gray_images:
            cy, cx = scipy.ndimage.center_of_mass(image)
            centers_x.append(cx)
        order = list(np.argsort(centers_x))
        return order


def resize_images(images: np.ndarray, target_size: Tuple[int, int] = (128, 128)) -> np.ndarray:
    """
    Resize images to a specified size.

    :param images: input images.
    :param target_size: target size as (height, width).
    :return: array of resized images.
    """
    return np.array([skimage.transform.resize(im, target_size) for im in images])


def rgb_to_gray(images):
    """
    Convert images from RGB to grayscale.

    :param images: input images with shape (num_images, height, width, 3).
    :return: grayscale images with shape (num_images, height, width).
    """
    return np.clip(
        (images[..., 0] * 0.299 + images[..., 1] * 0.587 + images[..., 2] * 0.114) / 255,
        0,
        1
    ).astype(np.float32)


def sort_images(images: np.ndarray,
                sort_type: str = "tsp",
                verbose: bool = False,
                working_size: Tuple[int, int] = None,
                **kwargs) -> List[int]:
    """
    Sort images with a specified sorting method.

    :param images: input images to sort.
    :param sort_type: method to sort images when constructing the animation. Should be one of the following:
        "tsp" (traveling salesman method on the image similarity graph), "pca" (image projection onto the real line
        using PCA), "hm" (order using center mass in the horizontal direction), "none" (no sorting; random order).
    :param verbose: if True, print additional information during sorting.
    :param working_size: optional (height, width) tuple that determines working image size. Images are resized to this
        size before being sorted.
    :param kwargs: keyword arguments for the sorting method.
    :return: image indices in the final ordering.
    """
    if sort_type == "tsp":
        sorter = TravelingSalesmanSorter(verbose=verbose)
    elif sort_type == "pca":
        sorter = PCASorter(verbose=verbose)
    elif sort_type == "hm":
        sorter = HorizontalMassSorter(verbose=verbose)
    elif sort_type == "none":
        sorter = DefaultSorter(verbose=verbose)
    else:
        raise NotImplementedError(f"Sort type '{sort_type}' not implemented")

    gray_images = rgb_to_gray(images)
    if working_size is not None:
        gray_images = resize_images(gray_images, working_size)

    order = sorter.sort(gray_images, **kwargs)
    return order
