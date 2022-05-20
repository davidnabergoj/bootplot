import abc
from typing import List
import scipy.spatial
import scipy.ndimage
import cv2
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm
import pygad
import scipy
import networkx as nx

from src.image_features import compute_hog_features, compute_center_of_mass, compute_center_of_mass_per_component


class Sorter(abc.ABC):
    @abc.abstractmethod
    def sort(self, images, **kwargs) -> List[int]:
        pass


class DefaultSorter(Sorter):
    def sort(self, images, **kwargs) -> List[int]:
        return list(range(len(images)))  # Does not sort


class LucasKanadeSorter(Sorter):
    def __init__(self,
                 corner_detection_params: dict = None,
                 lucas_kanade_params: dict = None,
                 genetic_algorithm_params: dict = None):
        """
        Sort images using an optical flow approach.

        Given an arbitrary frame, we use Shi-Tomasi corner detection to identify features to track.
        We use a genetic algorithm to find the ordering of images, such that the changes between feature locations are
        minimal. The location changes are computed using Lucas Kanade optical flow.

        :param corner_detection_params: parameters for Shi-Tomasi corner detection.
        :param lucas_kanade_params: parameters for Lucas Kanade optical flow computations.
        """
        if corner_detection_params is None:
            corner_detection_params = dict(
                maxCorners=100,
                qualityLevel=0.3,
                minDistance=7,
                blockSize=7
            )
        if lucas_kanade_params is None:
            lucas_kanade_params = dict(
                winSize=(15, 15),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            )
        if genetic_algorithm_params is None:
            genetic_algorithm_params = dict(
                num_generations=200,
                num_parents_mating=4,
                crossover_type="two_points",
                crossover_probability=0.2,
            )
        self.corner_detection_params = corner_detection_params
        self.lucas_kanade_params = lucas_kanade_params
        self.genetic_algorithm_params = genetic_algorithm_params

    def sort(self, gray_images, **kwargs) -> List[int]:
        def fitness(order, order_index):
            p0 = cv2.goodFeaturesToTrack(gray_images[order[0]], mask=None, **self.corner_detection_params)
            loss = 0
            for i in range(0, len(order)):  # Start the loop from 0 for circular traversal
                # Calculate optical flow
                p1, st, err = cv2.calcOpticalFlowPyrLK(
                    gray_images[order[i - 1]],
                    gray_images[order[i]],
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

        initial_population = [list(np.random.permutation(len(gray_images))) for _ in range(50)]
        with tqdm(
                total=self.genetic_algorithm_params["num_generations"],
                desc="Sorting images with GA"
        ) as pbar:
            ga_instance = pygad.GA(
                **{
                    "fitness_func": fitness,
                    "initial_population": initial_population,
                    "on_generation": lambda _: pbar.update(1),
                    **self.genetic_algorithm_params
                }
            )
            ga_instance.run()

        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        solution = list(solution)

        return solution


class FarnebackSorter(Sorter):
    def __init__(self,
                 population_size: int = 100,
                 farneback_params: dict = None,
                 genetic_algorithm_params: dict = None):
        """
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
        if genetic_algorithm_params is None:
            genetic_algorithm_params = dict(
                crossover_type="scattered",
                num_generations=300,
                num_parents_mating=25
            )
        self.population_size = population_size
        self.farneback_params = farneback_params
        self.genetic_algorithm_params = genetic_algorithm_params

    def sort(self, gray_images, **kwargs) -> List[int]:
        cached_flows = {}

        def fitness(order, order_index):
            # Compute the optical flow fields and store them in a dictionary
            for i in range(0, len(gray_images)):  # Start the loop from 0 for circular traversal
                image_pair = (order[i - 1], order[i])
                if image_pair not in cached_flows:
                    flow = cv2.calcOpticalFlowFarneback(
                        gray_images[image_pair[0]],
                        gray_images[image_pair[1]],
                        None,
                        **self.farneback_params
                    )
                    cached_flows[image_pair] = flow

            # Compute acceleration fields
            loss = 0
            for i in range(0, len(gray_images) - 1):  # Start the loop from 0 for circular traversal
                image_pair_0 = (order[i - 1], order[i])
                image_pair_1 = (order[i], order[i + 1])

                next_flow = cached_flows[image_pair_1]
                prev_flow = cached_flows[image_pair_0]

                # Penalty for changing directions
                angle_difference = np.dot(prev_flow.ravel(), next_flow.ravel())
                loss += np.sum(np.square(angle_difference))

            return -loss

        initial_population = [list(np.random.permutation(len(gray_images))) for _ in range(self.population_size)]
        ga_instance = pygad.GA(
            fitness_func=fitness,
            gene_type=int,
            initial_population=initial_population,
            **self.genetic_algorithm_params
        )

        ga_instance.run()
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        solution = list(solution)

        return solution


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
        print("Computing distance matrix")
        distance_matrix = self.distance_matrix(gray_images, features=features)
        print('Solving TSP')
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


def sort_images(images, sort_type="tsp", **kwargs):
    if sort_type == "tsp":
        sorter = TravelingSalesmanSorter()
    elif sort_type == "pca":
        sorter = PCASorter()
    elif sort_type == "lk":
        sorter = LucasKanadeSorter()
    elif sort_type == "hm":
        sorter = HorizontalMassSorter()
    elif sort_type == "fb":
        sorter = FarnebackSorter()
    elif sort_type == "none":
        sorter = DefaultSorter()
    else:
        raise NotImplementedError(f"Sort type '{sort_type}' not implemented")
    gray_images = np.array([
        cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
        for im in images
    ])
    order = sorter.sort(gray_images, **kwargs)
    return order
