import matplotlib.pyplot as plt
import numpy as np
import io
from tqdm import tqdm
import scipy.spatial.distance
import scipy.misc
import networkx as nx
from PIL import Image


def fig_to_array(fig, ax):
    with io.BytesIO() as buff:
        fig.savefig(buff, format='raw')
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    im = data.reshape((int(h), int(w), -1))
    # fig.clf()  # Clear figure (does not work)
    ax.cla()  # Clear axis
    return im


def merge_matrices_(matrices) -> np.ndarray:
    # matrices.shape == (batch_size, width, height, channels)
    # Overwrites source images
    matrices = matrices.astype(np.float32) / 255  # Cast to float
    merged = np.mean(matrices, axis=0)
    merged = (merged - np.min(merged)) / (np.max(merged) - np.min(merged))
    merged = (merged * 255).astype(np.uint8)
    return merged


def plot_to_array(plot_function, data, fig, ax, **kwargs):
    # plot_function should plot onto ax
    plot_function(data, ax, **kwargs)
    data = fig_to_array(fig, ax)
    return data


def bootstrapped_plot(plot_function, data, m=100, out_file: str = None):
    # plot function receives data as the first argument and ax as the second one
    fig, ax = plt.subplots()
    bootstrapped_matrices = np.stack([
        plot_to_array(plot_function, data[np.random.randint(low=0, high=len(data), size=len(data))], fig, ax)
        for _ in tqdm(range(m), desc='Generating bootstrapped plots')
    ])
    merged_matrices = merge_matrices_(bootstrapped_matrices)
    plt.close(fig)

    if out_file is not None:
        out_im = Image.fromarray(merged_matrices)
        out_im.save(out_file)

    return merged_matrices


def bootstrapped_animation(plot_function, data, m=100, out_file: str = None, fps=60, resize=True, sort=True,
                           decay=True, decay_length=30):
    fig, ax = plt.subplots()
    bootstrapped_matrices = np.stack([
        plot_to_array(plot_function, data[np.random.randint(low=0, high=len(data), size=len(data))], fig, ax)
        for _ in tqdm(range(m), desc='Generating bootstrapped plots')
    ])
    plt.close(fig)

    if sort:
        # Compute similarity between matrices
        print('Computing image similarity')
        if resize:
            bootstrapped_vectors = np.array([
                np.array(Image.fromarray(bm).resize((50, 50))) for bm in bootstrapped_matrices
            ]).reshape(m, -1)
        else:
            bootstrapped_vectors = (bootstrapped_matrices.astype(np.float32) / 255).reshape(m, -1)
        distance_matrix = scipy.spatial.distance.cdist(bootstrapped_vectors, bootstrapped_vectors)

        print('Solving TSP')
        image_graph = nx.from_numpy_array(distance_matrix)
        order = np.array(nx.algorithms.approximation.traveling_salesman_problem(image_graph))
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
        bootstrapped_images = [Image.fromarray(bm) for bm in bootstrapped_matrices]
        bootstrapped_images[0].save(
            out_file,
            save_all=True,
            append_images=bootstrapped_images[1:],
            duration=1000 / fps,
            loop=0
        )

    return bootstrapped_matrices
