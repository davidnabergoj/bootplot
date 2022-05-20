from pathlib import Path
from typing import Union, Tuple

import numpy as np
import imageio
import io
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

from src.sorting import sort_images


def fig_to_array(fig, ax):
    with io.BytesIO() as buff:
        fig.savefig(buff, format='raw')
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    im = data.reshape((int(h), int(w), -1))
    ax.cla()  # Clear axis
    return im


def plot_to_array(plot_function, data, indices, fig, ax, **kwargs):
    # plot_function should plot onto ax
    plot_function(data[indices], data, ax, **kwargs)
    data = fig_to_array(fig, ax)
    return data


def merge_matrices(matrices) -> np.ndarray:
    # matrices.shape == (batch_size, width, height, channels)
    # Overwrites source images
    matrices = matrices.astype(np.float32) / 255  # Cast to float
    merged = np.mean(matrices, axis=0)
    merged = (merged * 255).astype(np.uint8)
    return merged


def decay_images(images, m: int, decay_length: int):
    decayed_images = np.zeros(m - 1, images[0].shape[1:], dtype=np.uint8)
    for i in range(1, m):
        # matrix_indices = np.arange(max(i - decay_length, 0), i + 1)
        matrix_indices = np.arange(i - decay_length, i)  # Getting frames at the end makes the gif loop smoothly
        weights = np.arange(1, decay_length + 1)
        weights = weights ** 2
        weights = weights / np.sum(weights)
        weights = weights.reshape(-1, 1, 1, 1)
        decayed_images[i] = (np.sum(images[matrix_indices].astype(np.float32) * weights, axis=0)).astype(np.uint8)
    return decayed_images


def bootstrapped_plot(f: callable,
                      data: np.ndarray,
                      m: int = 100,
                      output_size_px: Tuple[int, int] = (512, 512),
                      output_image_path: Union[str, Path] = None,
                      output_animation_path: Union[str, Path] = None,
                      sort_type: str = 'tsp',
                      sort_kwargs: dict = None,
                      decay: bool = False,
                      decay_length: int = 1,
                      fps: int = 60,
                      verbose:bool=False):
    px_size_inches = 1 / plt.rcParams['figure.dpi']
    fig, ax = plt.subplots(figsize=(output_size_px[0] * px_size_inches, output_size_px[1] * px_size_inches))
    bootstrapped_matrices = np.stack([
        plot_to_array(f, data, np.random.randint(low=0, high=len(data), size=len(data)), fig, ax)
        for _ in tqdm(range(m), desc='Generating plots', disable=not verbose)
    ])
    merged_matrices = merge_matrices(bootstrapped_matrices)
    plt.close(fig)

    if output_image_path is not None:
        Image.fromarray(merged_matrices).save(output_image_path)
    if output_animation_path is not None:
        sort_kwargs = dict() if sort_kwargs is None else sort_kwargs
        order = sort_images(bootstrapped_matrices, sort_type, verbose=verbose, **sort_kwargs)
        order.extend(order[:-1][::-1])  # go in reverse
        order = np.array(order)
        bootstrapped_matrices = bootstrapped_matrices[order]

        # Apply decay
        if decay:
            bootstrapped_matrices = decay_images(bootstrapped_matrices, m=m, decay_length=decay_length)

        imageio.mimwrite(output_animation_path, bootstrapped_matrices, fps=fps)

    return merged_matrices
