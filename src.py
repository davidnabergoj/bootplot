import time

import imageio
from skimage.feature import hog
import matplotlib.pyplot as plt
import numpy as np
import io

import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm
import scipy.spatial.distance
import scipy.misc
import scipy.ndimage
import networkx as nx
from PIL import Image
import pygad
import cv2

from swd import swd


def fig_to_array(fig, ax):
    with io.BytesIO() as buff:
        fig.savefig(buff, format='raw')
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    im = data.reshape((int(h), int(w), -1))
    ax.cla()  # Clear axis
    return im


def merge_matrices_(matrices) -> np.ndarray:
    # matrices.shape == (batch_size, width, height, channels)
    # Overwrites source images
    matrices = matrices.astype(np.float32) / 255  # Cast to float
    merged = np.mean(matrices, axis=0)
    merged = (merged * 255).astype(np.uint8)
    return merged


def plot_to_array(plot_function, data, fig, ax, **kwargs):
    # plot_function should plot onto ax
    plot_function(data, ax, **kwargs)
    data = fig_to_array(fig, ax)
    return data


def sort_images_optical_flow_lk(images):
    # We track elements using LK OF. Genetic algorithm sorts images so that changes between elements from frame to frame
    # are minimal.

    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)
    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    gray_images = [cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) for image in images]

    def fitness(order, order_index):
        p0 = cv2.goodFeaturesToTrack(gray_images[order[0]], mask=None, **feature_params)
        loss = 0

        # mask = np.zeros_like(images[0])
        # color = np.random.randint(0, 255, (100, 3))
        for i in range(0, len(order)):  # Start the loop from 0 for circular traversal
            # calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(gray_images[order[i - 1]], gray_images[order[i]], p0, None,
                                                   **lk_params)
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

            # draw the tracks
            # frame = np.copy(images[order[i]])
            # for i, (new, old) in enumerate(zip(good_new, good_old)):
            #     a, b = new.ravel()
            #     c, d = old.ravel()
            #     mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            #     frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
            # img = cv2.add(frame, mask)
            # cv2.imshow('frame', img)
            # k = cv2.waitKey(30) & 0xff
            # if k == 27:
            #     break

            p0 = good_new.reshape(-1, 1, 2)
        # cv2.destroyAllWindows()
        if loss == 0:
            loss = np.infty  # if features to track are not detected correctly and everything is static
        return -loss

    initial_population = [list(np.random.permutation(len(images))) for _ in range(50)]
    num_generations = 200
    with tqdm(total=num_generations, desc="Sorting images with GA") as pbar:
        ga_instance = pygad.GA(
            fitness_func=fitness,
            num_generations=num_generations,
            num_parents_mating=4,
            initial_population=initial_population,
            gene_type=int,
            crossover_type="two_points",
            crossover_probability=0.2,
            on_generation=lambda _: pbar.update(1)
        )
        ga_instance.run()

    plt.figure()
    ga_instance.plot_fitness()
    plt.show()
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    solution = list(solution)

    return solution


def sort_images_optical_flow_dense(images):
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

    :param images: list of np.ndarray objects.
    :return: image order as a list of integers.
    """
    gray_images = [cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) for image in images]
    cached_flows = {}

    def fitness(order, order_index):
        # Compute the optical flow fields and store them in a dictionary
        for i in range(0, len(gray_images)):  # Start the loop from 0 for circular traversal
            image_pair = (order[i - 1], order[i])
            if image_pair not in cached_flows:
                flow = cv2.calcOpticalFlowFarneback(
                    gray_images[image_pair[0]], gray_images[image_pair[1]], None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                # mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                cached_flows[image_pair] = flow

        # Compute acceleration fields
        loss = 0
        for i in range(0, len(gray_images) - 1):  # Start the loop from 0 for circular traversal
            image_pair_0 = (order[i - 1], order[i])
            image_pair_1 = (order[i], order[i + 1])

            next_flow = cached_flows[image_pair_1]
            prev_flow = cached_flows[image_pair_0]

            # prev_mag, prev_ang = cv2.cartToPolar(prev_flow[..., 0], prev_flow[..., 1])
            # next_mag, next_ang = cv2.cartToPolar(next_flow[..., 0], next_flow[..., 1])

            # Penalty for changing directions
            angle_difference = np.dot(prev_flow.ravel(), next_flow.ravel())
            loss += np.sum(np.square(angle_difference))

        # loss_jerk = 0
        # # Compute jerk fields
        # for i in range(0, len(gray_images) - 2):  # Start the loop from 0 for circular traversal
        #     image_pair_0 = (order[i - 1], order[i])
        #     image_pair_1 = (order[i], order[i + 1])
        #     image_pair_2 = (order[i + 1], order[i + 2])
        #     acceleration0 = cached_flows[image_pair_1] - cached_flows[image_pair_0]
        #     acceleration1 = cached_flows[image_pair_2] - cached_flows[image_pair_1]
        #     jerk = acceleration1 - acceleration0
        #     loss_jerk += np.sqrt(np.sum(np.square(jerk)))

        return -loss

    initial_population = [list(np.random.permutation(len(images))) for _ in range(100)]
    ga_instance = pygad.GA(
        fitness_func=fitness,
        num_generations=300,
        num_parents_mating=25,
        initial_population=initial_population,
        gene_type=int,
        crossover_type="scattered"
    )

    ga_instance.run()
    plt.figure()
    ga_instance.plot_fitness()
    plt.show()
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    solution = list(solution)

    return solution


def compute_hog_features(gray_images, threshold=1e-5):
    # movement_mask = (np.var(gray_images, axis=0) > threshold).astype(np.uint8)
    # hog_features = np.stack([
    #     hog(im, cells_per_block=(1, 1), pixels_per_cell=(10, 10), visualize=True)[1][movement_mask.astype(bool)]
    #     for im in gray_images
    # ])
    hog_features = np.stack([hog(im, cells_per_block=(1, 1), pixels_per_cell=(10, 10)) for im in gray_images])
    hog_features = (hog_features - np.min(hog_features)) / (np.max(hog_features) - np.min(hog_features))
    return hog_features


def compute_center_of_mass(gray_images):
    centers_of_mass = np.zeros((len(gray_images), 2))
    for i in range(len(gray_images)):
        centers_of_mass[i] = scipy.ndimage.center_of_mass(gray_images[i])
    centers_of_mass /= np.array(gray_images.shape[1:])
    return centers_of_mass


def compute_center_of_mass_per_component(gray_images, threshold=1e-5):
    movement_mask = (np.var(gray_images, axis=0) > threshold).astype(np.uint8)
    n_components, label_mask, stats, centroids = cv2.connectedComponentsWithStats(movement_mask)
    centers = np.zeros((len(gray_images), 2 * n_components - 2))
    for j, image in enumerate(gray_images):
        for i in range(1, n_components):
            center_of_mass = scipy.ndimage.center_of_mass(image, label_mask, i)
            centers[j, i * 2 - 2:i * 2] = center_of_mass
    centers /= np.tile(np.array(gray_images.shape[1:]), n_components - 1)
    return centers


def sort_images_pca_2(images):
    gray_images = np.array([
        cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
        for im in images
    ])
    hf = compute_hog_features(gray_images)
    cm = compute_center_of_mass(gray_images)
    cmc = compute_center_of_mass_per_component(gray_images)

    features = np.c_[hf, cm, cmc]

    # distance_matrix = scipy.spatial.distance.cdist(features, features, "cityblock")
    model = PCA(n_components=1)
    # model = TSNE(n_components=1)
    projections = model.fit_transform(features).ravel()
    order = list(np.argsort(projections))
    # print('Solving TSP')
    # image_graph = nx.from_numpy_array(distance_matrix)
    # order = nx.algorithms.approximation.traveling_salesman_problem(
    #     image_graph,
    #     cycle=False,
    #     method=nx.algorithms.approximation.traveling_salesman.greedy_tsp
    # )
    return order
    #
    # movement_mask = np.var(gray_images, axis=0)
    # ret, labels = cv2.connectedComponents(movement_mask)
    #
    # mean_mask = np.mean(gray_images, axis=0)
    # centroids = np.zeros((len(images), 2 * len(labels)))
    # for i in range(len(gray_images)):
    #     velocity_magnitude = np.abs(gray_images[i] - mean_mask)
    #     for label in labels:
    #         # movement_mask[]
    #         pass

    # gray_images = np.array([
    #     gray_images[i] - np.mean(gray_images, axis=0) for i in range(len(gray_images))
    # ])
    # gray_images = np.array([
    #     cv2.GaussianBlur(im, ksize=tuple(ksize), sigmaX=ksize[0] / 6, sigmaY=ksize[1] / 6)
    #     for im in gray_images
    # ])


def compute_n_components(images, threshold=1e-5):
    gray_images = np.array([cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255 for im in images])
    movement_mask = (np.var(gray_images, axis=0) > threshold).astype(np.uint8)
    n_components, label_mask, stats, centroids = cv2.connectedComponentsWithStats(movement_mask)
    return n_components - 1


def sort_images_tsp(images, m: int, distance='hog'):
    # Compute similarity between matrices
    if distance == 'euc':
        print('Computing pixelwise image similarity')
        ksize = list(images[0].shape[:2])
        ksize[0] += ksize[0] % 2 - 1
        ksize[1] += ksize[1] % 2 - 1
        ksize = ksize[::-1]
        gray_images = np.array([
            cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
            for im in images
        ])
        # gray_images = np.array([
        #     gray_images[i] - np.mean(gray_images, axis=0) for i in range(len(gray_images))
        # ])
        gray_images = np.array([
            cv2.GaussianBlur(im, ksize=tuple(ksize), sigmaX=ksize[0] / 6, sigmaY=ksize[1] / 6)
            for im in gray_images
        ])
        # convolve with a large kernel so that elements are "spread out"
        bootstrapped_vectors = gray_images.reshape(m, -1)
        distance_matrix = scipy.spatial.distance.cdist(bootstrapped_vectors, bootstrapped_vectors)
        # Not sure if normalization does anything
        distance_matrix = (distance_matrix - np.min(distance_matrix)) / (
                np.max(distance_matrix) - np.min(distance_matrix))
        # plt.figure()
        # plt.matshow(np.var(gray_images, axis=0))
        # plt.show()
    elif distance == 'swd':
        distance_matrix = np.zeros((len(images), len(images)), dtype=np.float32)
        for i in (pbar := tqdm(range(len(images)), desc="Computing SWD")):
            for j in range(i + 1, len(images)):
                with torch.no_grad():
                    im0 = torch.tensor(images[i].astype(np.float32)).view(1, *images[i].shape)
                    im1 = torch.tensor(images[j].astype(np.float32)).view(1, *images[j].shape)
                    im0 = torch.swapaxes(im0, 1, 3)
                    im1 = torch.swapaxes(im1, 1, 3)
                    im0 = im0[:, :3]
                    im1 = im1[:, :3]
                    dist = float(swd(im0, im1, device="cpu"))
                    distance_matrix[i, j] = dist
                    distance_matrix[j, i] = dist
    elif distance == "hog":
        gray_images = np.array([
            cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
            for im in images
        ])
        hf = compute_hog_features(gray_images)
        distance_matrix = scipy.spatial.distance.cdist(hf, hf)
    else:
        raise NotImplementedError(f"Distance {distance} is not implemented.")
    print('Solving TSP')
    image_graph = nx.from_numpy_array(distance_matrix)
    order = nx.algorithms.approximation.traveling_salesman_problem(
        image_graph,
        cycle=False,
        # method=nx.algorithms.approximation.traveling_salesman.greedy_tsp
    )

    # order can be longer than the number of vertices in the graph because of the TSP approximation
    # order = list(dict.fromkeys(order))
    return order


def sort_images_pca(images, center_mass=True):
    # Sort images according to 1D PCA projections of their centers of mass
    gray_images = np.array([cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float) / 255 for im in images])
    pca = PCA(n_components=1)
    if center_mass:
        centers_of_mass = np.zeros((len(images), 2))
        for i in range(len(images)):
            centers_of_mass[i] = scipy.ndimage.center_of_mass(gray_images[i])
        projections = pca.fit_transform(centers_of_mass).ravel()
    else:
        projections = pca.fit_transform(gray_images.reshape(len(gray_images), -1)).ravel()
    order = list(np.argsort(projections))
    return order


def sort_images_horizontal_mass(images):
    # Each grayscale image is proportional to a bivariate density on a rectangle.
    # We can compute the mean of that density.
    # Images are sorted according to the x component of their means.
    # This could be generalized by picking the direction with the highest variance.
    # The latter is equivalent to PCA when projecting to 1D
    gray_images = np.array([cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) for im in images])
    centers_x = []
    for image in gray_images:
        cy, cx = scipy.ndimage.center_of_mass(image)
        centers_x.append(cx)
    order = list(np.argsort(centers_x))
    return order


def bootstrapped_plot(plot_function, data, m=100, out_file: str = None, resample_in_advance=True):
    # plot function receives data as the first argument and ax as the second one
    fig, ax = plt.subplots()
    if resample_in_advance:
        bootstrapped_matrices = np.stack([
            plot_to_array(plot_function, data[np.random.randint(low=0, high=len(data), size=len(data))], fig, ax)
            for _ in tqdm(range(m), desc='Generating bootstrapped plots')
        ])
    else:
        bootstrapped_matrices = np.stack([
            plot_to_array(plot_function, data, fig, ax) for _ in tqdm(range(m), desc='Generating bootstrapped plots')
        ])
    merged_matrices = merge_matrices_(bootstrapped_matrices)
    plt.close(fig)

    if out_file is not None:
        out_im = Image.fromarray(merged_matrices)
        out_im.save(out_file)

    return merged_matrices


def bootstrapped_animation(plot_function, data, m=100, out_file: str = None, fps=60, resize=True, sort=True,
                           decay=False, decay_length=15, resample_in_advance=True, sort_type: str = "tsp",
                           animation_duration=3):
    fig, ax = plt.subplots()
    if resample_in_advance:
        bootstrapped_matrices = np.stack([
            plot_to_array(plot_function, data[np.random.randint(low=0, high=len(data), size=len(data))], fig, ax)
            for _ in tqdm(range(m), desc='Generating bootstrapped plots')
        ])
    else:
        bootstrapped_matrices = np.stack([
            plot_to_array(plot_function, data, fig, ax) for _ in tqdm(range(m), desc='Generating bootstrapped plots')
        ])
    plt.close(fig)

    if sort:
        if resize:
            resized_images = np.array([
                np.array(Image.fromarray(bm).resize((128, 128))) for bm in bootstrapped_matrices
            ])
        else:
            resized_images = (bootstrapped_matrices.astype(np.float32) / 255)

        if sort_type == "tsp":
            order = sort_images_tsp(images=resized_images, m=m)
        elif sort_type == "of":
            order = sort_images_optical_flow_lk(images=resized_images)
        elif sort_type == "hm":
            order = sort_images_horizontal_mass(images=resized_images)
        elif sort_type == "pca":
            order = sort_images_pca(images=resized_images, center_mass=True)
        elif sort_type == "pca2":
            order = sort_images_pca_2(images=resized_images)
        else:
            raise NotImplementedError(f"Sort type '{sort_type}' not implemented")
        order.extend(order[:-1][::-1])  # go in reverse
        order = np.array(order)
        bootstrapped_matrices = bootstrapped_matrices[order]

    if decay:
        print('Applying decay')
        decayed_bootstrapped_matrices = []

        for i in range(1, m):
            # matrix_indices = np.arange(max(i - decay_length, 0), i + 1)
            matrix_indices = np.arange(i - decay_length, i)  # Getting frames at the end makes the gif loop smoothly
            weights = np.arange(1, decay_length + 1)
            weights = weights ** 2
            weights = weights / np.sum(weights)
            weights = weights.reshape(-1, 1, 1, 1)
            decayed_bootstrapped_matrices.append(
                np.sum(bootstrapped_matrices[matrix_indices].astype(np.float32) * weights, axis=0)
            )

        bootstrapped_matrices = np.array(decayed_bootstrapped_matrices)
        bootstrapped_matrices = bootstrapped_matrices.astype(np.uint8)

    if out_file is not None:
        print('Saving animation')
        if fps * animation_duration > len(bootstrapped_matrices):
            image_mask = np.arange(len(bootstrapped_matrices))
        else:
            image_mask = np.linspace(0, len(bootstrapped_matrices) - 1, fps * animation_duration).astype(np.int)
        # print(bootstrapped_matrices[image_mask].shape)
        imageio.mimwrite(out_file, bootstrapped_matrices[image_mask], format="GIF", fps=fps)
        # bootstrapped_images = [Image.fromarray(bm) for bm in bootstrapped_matrices]
        # bootstrapped_images = [bootstrapped_images[i] for i in image_mask]
        # bootstrapped_images[0].save(
        #     out_file,
        #     save_all=True,
        #     append_images=bootstrapped_images[1:],
        #     duration=1000 / fps,  # 1000 / len(bootstrapped_images),  # This will make the gif 1 second long
        #     loop=0
        # )

    return bootstrapped_matrices
