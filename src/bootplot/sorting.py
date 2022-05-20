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
        self.verbose = verbose

    @abc.abstractmethod
    def sort(self, images, **kwargs) -> List[int]:
        pass

    def verbose_print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)


class DefaultSorter(Sorter):
    def sort(self, images, **kwargs) -> List[int]:
        return list(range(len(images)))  # Does not sort


class PCASorter(Sorter):
    def sort(self, gray_images, features="center_mass", **kwargs) -> List[int]:
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
    def distance_matrix(self, gray_images, features="full"):
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

    def sort(self, gray_images, features="full", **kwargs) -> List[int]:
        self.verbose_print("Computing distance matrix")
        distance_matrix = self.distance_matrix(gray_images, features=features)
        self.verbose_print('Solving TSP')
        image_graph = nx.from_numpy_array(distance_matrix)
        order = nx.algorithms.approximation.traveling_salesman_problem(image_graph, cycle=False)
        return order


class HorizontalMassSorter(Sorter):
    def sort(self, gray_images, **kwargs) -> List[int]:
        # Each grayscale image is proportional to a bivariate density on a rectangle.
        # We can compute the mean of that density.
        # Images are sorted according to the x component of their means.
        # This could be generalized by picking the direction with the highest variance.
        # The latter is equivalent to PCA when projecting to 1D
        centers_x = []
        for image in gray_images:
            cy, cx = scipy.ndimage.center_of_mass(image)
            centers_x.append(cx)
        order = list(np.argsort(centers_x))
        return order


def resize_images(images, target_size=(128, 128)):
    return np.array([skimage.transform.resize(im, target_size) for im in images])


def rgb_to_gray(images):
    return np.clip(
        (images[..., 0] * 0.299 + images[..., 1] * 0.587 + images[..., 2] * 0.114) / 255,
        0,
        1
    ).astype(np.float32)


def sort_images(images: np.ndarray,
                sort_type: str = "tsp",
                verbose: bool = False,
                working_size: Tuple[int, int] = None,
                **kwargs):
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
