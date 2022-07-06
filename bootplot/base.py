from pathlib import Path
from typing import Union, Tuple

import numpy as np
import imageio
import io
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

from bootplot.sorting import sort_images


def fig_to_array(fig: plt.Figure,
                 ax: plt.Axes) -> np.array:
    """
    Retrieve array of pixels from a figure.
    The Axis is cleared afterwards.

    :param fig: figure with the plot. The figure should contain a single Axes object.
    :param ax: axis with the plot.
    :return: numpy array of pixels of the plot.
    """
    with io.BytesIO() as buff:
        fig.savefig(buff, format='raw')
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    im = data.reshape((int(h), int(w), -1))
    ax.cla()
    return im


def plot_to_array(plot_function: callable,
                  data: np.ndarray,
                  indices: np.ndarray,
                  fig: plt.Figure,
                  ax: plt.Axes,
                  xlim: Tuple[float, float] = (None, None),
                  ylim: Tuple[float, float] = (None, None),
                  **kwargs) -> np.ndarray:
    """
    Plot data and obtain image.

    :param plot_function: function to do the plotting. The function should receive the data subset, original data, Axes
        object and optional keyword arguments.
    :param data: full data to be used in plotting.
    :param indices: bootstrap resampled indices of the data.
    :param fig: Figure object with a single Axes.
    :param ax: Axes object.
    :param xlim: x axis limits.
    :param ylim: y axis limits.
    :param kwargs: keyword arguments to plot_function.
    :return: image.
    """
    plot_function(data[indices], data, ax, **kwargs)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    data = fig_to_array(fig, ax)
    return data


def merge_images(images: np.ndarray) -> np.ndarray:
    """
    Merge images into a static image (averaged image).
    The shape of images is (batch_size, width, height, channels).
    This operation overwrites input images.

    :param images: images corresponding to different bootstrap resamples.
    :return: merged image.
    """
    images = images.astype(np.float32) / 255  # Cast to float
    merged = np.mean(images, axis=0)
    merged = (merged * 255).astype(np.uint8)
    return merged


def decay_images(images: np.ndarray,
                 m: int,
                 decay_length: int) -> np.ndarray:
    """
    Apply visual decay to images.
    Once applied, images[t] will contain a weighted sum of images from t - decay_length to t.

    :param images: array of images corresponding to different bootstrap samples.
    :param m: number of bootstrap samples.
    :param decay_length: consider this many preceding images when creating a decayed image.
    :return: decayed images with the same shape as input images.
    """
    decayed_images = np.zeros((m, *images[0].shape), dtype=np.uint8)
    for i in range(m):
        matrix_indices = np.arange(i - decay_length, i)  # Getting frames at the end makes the gif loop smoothly
        weights = np.arange(1, decay_length + 1)
        weights = weights ** 2
        weights = weights / np.sum(weights)
        weights = weights.reshape(-1, 1, 1, 1)
        decayed_images[i] = (np.sum(images[matrix_indices].astype(np.float32) * weights, axis=0)).astype(np.uint8)
    return decayed_images


def bootplot(f: callable,
             data: np.ndarray,
             m: int = 100,
             output_size_px: Tuple[int, int] = (512, 512),
             output_image_path: Union[str, Path] = None,
             output_animation_path: Union[str, Path] = None,
             sort_type: str = 'tsp',
             sort_kwargs: dict = None,
             decay: int = 0,
             fps: int = 60,
             xlim: Tuple[float, float] = (None, None),
             ylim: Tuple[float, float] = (None, None),
             verbose: bool = False) -> np.ndarray:
    """
    Create a bootstrapped plot.

    :param f: function to perform the plotting. The function should receive the data subset, original data, Axes object.
    :param data: data to be used in plotting.
    :param m: number of boostrap resamples.
    :param output_size_px: output size (height, width) in pixels.
    :param output_image_path: path where the image should be stored. If None, the image is not stored.
    :param output_animation_path: path where the animation should be stored. If None, the animation is not created.
    :param sort_type: method to sort images when constructing the animation. Should be one of the following:
        "tsp" (traveling salesman method on the image similarity graph), "pca" (image projection onto the real line
        using PCA), "hm" (order using center mass in the horizontal direction), "none" (no sorting; random order).
    :param sort_kwargs: keyword arguments for the sorting method. See bootplot.sorting.sort_images for details.
    :param decay: decay length when creating the animation. If 0, no decay is applied.
    :param fps: desired output framerate for the animation.
    :param xlim: x axis limits.
    :param ylim: y axis limits.
    :param verbose: if True, print progress messages.
    :return: bootstrapped plot as a numpy array.
    """
    px_size_inches = 1 / plt.rcParams['figure.dpi']
    fig, ax = plt.subplots(figsize=(output_size_px[0] * px_size_inches, output_size_px[1] * px_size_inches))
    image_samples = np.stack([
        plot_to_array(f, data, np.random.randint(low=0, high=len(data), size=len(data)), fig, ax, xlim=xlim, ylim=ylim)
        for _ in tqdm(range(m), desc='Generating plots', disable=not verbose)
    ])
    merged_image = merge_images(image_samples)[..., :3]  # Do not use the alpha channel
    plt.close(fig)

    if output_image_path is not None:
        if verbose:
            print(f'Saving bootstrapped image to {output_image_path}')
        Image.fromarray(merged_image).save(output_image_path)
    if output_animation_path is not None:
        sort_kwargs = dict() if sort_kwargs is None else sort_kwargs
        order = sort_images(image_samples, sort_type, verbose=verbose, **sort_kwargs)
        order.extend(order[:-1][::-1])  # go in reverse
        order = np.array(order)
        image_samples = image_samples[order]

        # Apply decay
        if decay > 0:
            image_samples = decay_images(image_samples, m=m, decay_length=decay)

        imageio.mimwrite(output_animation_path, image_samples, fps=fps)
        if verbose:
            print(f'Saving bootstrapped animation to {output_animation_path}')

    return merged_image
