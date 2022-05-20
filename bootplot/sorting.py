import abc
from typing import List, Tuple, Optional
import scipy.spatial
import scipy.ndimage
import skimage.transform
import cv2
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm
import pygad
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


class GeneticAlgorithmSorter(Sorter):
    def __init__(self,
                 verbose: bool = False,
                 population_size: int = 20,
                 num_generations: int = 100,
                 crossover_probability: float = 0.5,
                 crossover_type: str = "two_points"):
        """
        Experimental.

        :param verbose:
        :param population_size:
        :param num_generations:
        :param crossover_probability:
        :param crossover_type:
        """
        super().__init__(verbose=verbose)
        self.gray_images: Optional[np.ndarray] = None
        self.population_size = population_size
        self.num_generations = num_generations
        self.crossover_probability = crossover_probability
        self.num_parents_mating = population_size // 2
        self.crossover_type = crossover_type

    @abc.abstractmethod
    def fitness(self, order, order_index):
        pass

    def sort(self, gray_images, **kwargs):
        self.gray_images = gray_images  # Make images accessible to the fitness function

        initial_population = [list(np.random.permutation(len(self.gray_images))) for _ in range(self.population_size)]
        with tqdm(total=self.num_generations, desc="Sorting images", disable=not self.verbose) as pbar:
            ga_instance = pygad.GA(
                fitness_func=lambda order, order_index: self.fitness(order, order_index),
                initial_population=initial_population,
                on_generation=lambda _: pbar.update(1),
                crossover_probability=self.crossover_probability,
                num_parents_mating=self.num_parents_mating,
                crossover_type=self.crossover_type,
                num_generations=self.num_generations,
                gene_type=int,
                **kwargs
            )
            ga_instance.run()

        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        solution = list(solution)
        return solution


class LucasKanadeSorter(GeneticAlgorithmSorter):
    def __init__(self,
                 corner_detection_params: dict = None,
                 lucas_kanade_params: dict = None,
                 verbose: bool = False,
                 **kwargs):
        """
        Experimental.
        Sort images using an optical flow approach.

        Given an arbitrary frame, we use Shi-Tomasi corner detection to identify features to track.
        We use a genetic algorithm to find the ordering of images, such that the changes between feature locations are
        minimal. The location changes are computed using Lucas Kanade optical flow.

        :param corner_detection_params: parameters for Shi-Tomasi corner detection.
        :param lucas_kanade_params: parameters for Lucas Kanade optical flow computations.
        """
        super().__init__(verbose=verbose, **kwargs)
        if corner_detection_params is None:
            corner_detection_params = dict(
                maxCorners=100,
                qualityLevel=0.3,
                minDistance=7,
                blockSize=7,
                mask=None
            )
        if lucas_kanade_params is None:
            lucas_kanade_params = dict(
                winSize=(15, 15),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            )
        self.corner_detection_params = corner_detection_params
        self.lucas_kanade_params = lucas_kanade_params

    def fitness(self, order, order_index):
        p0 = cv2.goodFeaturesToTrack(
            self.gray_images[order[0]],
            **self.corner_detection_params
        )
        loss = 0
        for i in range(0, len(order)):  # Start the loop from 0 for circular traversal
            # Calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(
                (self.gray_images[order[i - 1]] * 255).astype(np.uint8),
                (self.gray_images[order[i]] * 255).astype(np.uint8),
                p0,
                None,
                **self.lucas_kanade_params
            )
            # Select good points
            if p1 is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]
                if np.all((good_old - good_new) == 0):
                    loss = np.infty
                    break
                loss += np.sum(np.square(good_new - good_old))
            else:
                # Tracking error, infinite loss
                loss = np.infty
                break

            p0 = good_new.reshape(-1, 1, 2)
        if loss == 0:
            # if features to track are not detected correctly and everything is static
            loss = np.infty
        return -loss


class FarnebackSorter(GeneticAlgorithmSorter):
    def __init__(self,
                 population_size: int = 100,
                 farneback_params: dict = None,
                 verbose: bool = False,
                 **kwargs):
        """
        Experimental.
        Sorts plots.

        This done using a genetic algorithm that treats an ordering as an individual.
        For each ordering, we compute the optical flow vector field between every pair of consecutive images.
        This gives us the velocity at every point in the image.
        For each pair of consecutive velocity fields, we subtract the next from the previous one, obtaining a numerical
        derivative that corresponds to an acceleration field. Acceleration needs to be as small as possible in magnitude.
        The fitness of the individual is the sum of squared acceleration magnitudes over all fields and points in the field.
        Mutation means swapping two images at random.
        Crossover means splitting the ordering at some point and concatenating alternating ends for two individuals.
        """
        super().__init__(verbose=verbose, **kwargs)
        if farneback_params is None:
            farneback_params = dict(
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0
            )
        self.population_size = population_size
        self.farneback_params = farneback_params
        self.cached_flows = dict()

    def fitness(self, order, order_index):
        # Compute the optical flow fields and store them in a dictionary
        for i in range(0, len(self.gray_images)):  # Start the loop from 0 for circular traversal
            image_pair = (order[i - 1], order[i])
            if image_pair not in self.cached_flows:
                flow = cv2.calcOpticalFlowFarneback(
                    self.gray_images[image_pair[0]],
                    self.gray_images[image_pair[1]],
                    None,
                    **self.farneback_params
                )
                self.cached_flows[image_pair] = flow

        # Compute acceleration fields
        loss = 0
        for i in range(0, len(self.gray_images) - 1):  # Start the loop from 0 for circular traversal
            image_pair_0 = (order[i - 1], order[i])
            image_pair_1 = (order[i], order[i + 1])

            next_flow = self.cached_flows[image_pair_1]
            prev_flow = self.cached_flows[image_pair_0]

            # Penalty for changing directions
            angle_difference = np.dot(prev_flow.ravel(), next_flow.ravel())
            loss += np.sum(np.square(angle_difference))

        return -loss


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
    elif sort_type == "lk":
        sorter = LucasKanadeSorter(verbose=verbose)
    elif sort_type == "hm":
        sorter = HorizontalMassSorter(verbose=verbose)
    elif sort_type == "fb":
        sorter = FarnebackSorter(verbose=verbose)
    elif sort_type == "none":
        sorter = DefaultSorter(verbose=verbose)
    else:
        raise NotImplementedError(f"Sort type '{sort_type}' not implemented")

    gray_images = rgb_to_gray(images)
    if working_size is not None:
        gray_images = resize_images(gray_images, working_size)

    order = sorter.sort(gray_images, **kwargs)
    return order
