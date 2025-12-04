from pathlib import Path
from typing import Union, Tuple

import numpy as np
import imageio
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
from PIL import Image, ImageFilter
from scipy.stats import beta

from bootplot.backend.base import Backend, create_backend
from bootplot.sorting import sort_images

import jax.numpy as jnp
from jax import jit, vmap, device_get
from jax.scipy.special import betainc


def symmetric_transformation_new(x: float,
                                 k: float,
                                 threshold: float) -> float:
    y = betainc(k, k, x)
    return (1 - 2 * threshold) * y + threshold

def adjust_freqs(freqs: jnp.ndarray,
                 k: float,
                 threshold: float) -> jnp.ndarray:
    dom_idx = jnp.argmax(freqs)
    dom = freqs[dom_idx]

    t_dom = symmetric_transformation_new(dom, k, threshold)
    sum_other = 1.0 - dom
    scale = (1.0 - t_dom) / sum_other

    out = freqs * scale
    return out.at[dom_idx].set(t_dom)


def process_pixel(pixel_stack: jnp.ndarray,
                  k: float,
                  threshold: float) -> jnp.ndarray:
    mn = pixel_stack.shape[0]

    r = pixel_stack[:, 0].astype(jnp.int32)
    g = pixel_stack[:, 1].astype(jnp.int32)
    b = pixel_stack[:, 2].astype(jnp.int32)

    idx = (r << 16) + (g << 8) + b

    uniq, counts = jnp.unique(idx, size=mn, fill_value=0, return_counts=True)

    n_unique = jnp.sum(counts > 0)

    ur = ((uniq >> 16) & 255).astype(jnp.float32)
    ug = ((uniq >> 8) & 255).astype(jnp.float32)
    ub = (uniq & 255).astype(jnp.float32)
    
    colors = jnp.stack([ur, ug, ub], axis=1)

    freqs = counts.astype(jnp.float32) / mn

    only_one = (n_unique == 1)
    one_color = colors[0].astype(jnp.uint8)

    freqs_adj = adjust_freqs(freqs, k, threshold)

    rgb = jnp.sum(colors * freqs_adj[:, None], axis=0)
    rgb = jnp.clip(rgb, 0, 255).astype(jnp.uint8)

    return jnp.where(only_one, one_color, rgb)


    
@jit
def merge_images(images: np.ndarray,
                 k: float,
                 threshold: float) -> jnp.ndarray:
    mn, rows, cols, _ = images.shape 

    pixels = images.transpose(1, 2, 0, 3)
    
    #each of the rows * cols elements is a list of RGB pixels from all images at the same location:
    pixels = pixels.reshape(rows * cols, mn, 3)

    fused = vmap(process_pixel, in_axes=(0, None, None))(pixels, k, threshold)
    return fused.reshape(rows, cols, 3)


def merge_images_original(images: np.ndarray) -> np.ndarray:
    """
    Merge images into a static image (averaged image) without transformation.
    The shape of images is (batch_size, width, height, channels).
    This operation overwrites input images.

    :param images: images corresponding to different bootstrap resamples.
    :param images: images corresponding to different bootstrap samples.
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
             data: Union[np.ndarray, pd.DataFrame],
             m: int = 100,
             k: int = 2.5,
             threshold: int = 0.3,
             output_size_px: Tuple[int, int] = (512, 512),
             single_sample: bool = False,
             output_image_path: Union[str, Path] = None,
             transformation: bool = True,
             output_animation_path: Union[str, Path] = None,
             sort_type: str = 'tsp',
             sort_kwargs: dict = None,
             decay: int = 0,
             animation_duration: float = 5.0,
             backend: Union[Backend, str] = 'matplotlib',
             verbose: bool = False) -> np.ndarray:
    """
    Create a bootstrapped plot or animation.

    This function internally creates ``m`` samples with replacement from the provided ``data``. Each sample has the same
    number of rows as the input. The samples are then plotted using the function handle ``f`` and the images stored as
    `numpy.ndarray` objects. The output is a weighted sum of these images. If specified, this function can also create
    an animation where images are sorted according to ``sort_type`` and the output animation is written to disk.

    :param f: function handle to perform the plotting. The handle should have the form ``f(data_subset, data_full, ax)``
        where ``data_subset``, ``data_full`` are `numpy.ndarray` or `pandas.DataFrame` objects and ``ax`` is a
        `matplotlib.axes.Axes` object.
    :type f: callable

    :param data: data to be used in plotting.
    :type data: numpy.ndarray or pandas.DataFrame

    :param m: number of boostrap resamples. Default: ``100``.
    :type m: int

    :param k: input beta cdf transformation parameter. Controls the shape Default: ``2.5``.
    :type k: int
    
    :param threshold: input transformation parameter. Controls the codomain of the transformation. It lies between 0 and 0.5. Default: ``0,3``.
    :type threshold: int

    :param output_size_px: output size (width, heigth) in pixels. Default: ``(512, 512)``.
    :type output_size_px: tuple[int, int]

    :param single_sample: if true data_subset consists of a single sample. Default: ``False``.
    :type single_sample: bool

    :param output_image_path: path where the image should be stored. The image format is inferred from the filename
        extension. If None, the image is not stored. Default: ``None``.
    :type output_image_path: str or pathlib.Path

    :param transformation: if True transformation is applied, else images are just averaged. Default: ``True``.
    :type transformation: bool

    :param output_animation_path: path where the animation should be stored. The animation format is inferred from the
        filename extension. If None, the animation is not created. Default: ``None``.
    :type output_animation_path: str or pathlib.Path

    :param sort_type: method to sort images when constructing the animation. Should be one of the following:
        "tsp" (traveling salesman method on the image similarity graph), "pca" (image projection onto the real line
        using PCA), "hm" (order using center mass in the horizontal direction), "none" (no sorting; random order).
        Default: ``"tsp"``.
    :type sort_type: str

    :param sort_kwargs: keyword arguments for the sorting method. If None, no keyword arguments are passed to the
        sorting method. See ``bootplot.sorting.sort_images`` for details. Default: ``None``.
    :type sort_kwargs: dict

    :param decay: decay length when creating the animation. If 0, no decay is applied. Default: ``0``.
    :type decay: int

    :param animation_duration: desired output animation duration in seconds. Default: ``5.0``.
    :type animation_duration: float

    :param xlim: x axis limits representing the minimum and maximum. If a limit is ``None``, the plot is unbounded
        horizontally and the user is warned. Default: ``(None, None)``.
    :type xlim: tuple[float, float]

    :param ylim: y axis limits representing the minimum and maximum. If a limit is ``None``, the plot is unbounded
        vertically and the user is warned. Default: ``(None, None)``.
    :type ylim: tuple[float, float]

    :param verbose: if True, print progress messages. Default: ``False``.
    :type verbose: bool

    :param warn_limits: if True, warns the user when a limit is not specified. Default: ``True``.
    :type warn_limits: bool

    :return: bootstrapped plot.
    :rtype: numpy.ndarray

    Examples:
        Consider the task of estimating the uncertainty of a regression model.
        In this example, we use linear regression model to fit data drawn from a bivariate normal distribution.
        Instead of manually deriving and writing uncertainty estimation code, we only need to know how to plot our data.

        We define a function that plots our data of interest and pass it to ``bootplot``. In this case, we show a
        scatterplot of the entire dataset and a regression line based on the bootstrapped sample. We also provide axis
        limits to constrain our region of interest. ``bootplot`` generates the static image and saves it to disk.
        We can also continue to work with the returned image as a numpy.ndarray.

        >>> import numpy as np
        >>> from bootplot import bootplot
        >>> from sklearn.linear_model import LinearRegression
        >>> np.random.seed(0)
        >>>
        >>> def make_plot(data_subset, data_full, ax):
        ...     ax.scatter(data_full[:, 0], data_full[:, 1])
        ...     lr = LinearRegression()
        ...     lr.fit(data_subset[:, 0].reshape(-1, 1), data_subset[:, 1])
        ...     xs = np.linspace(-10, 10, 1000)
        ...     ax.plot(xs, lr.predict(xs.reshape(-1, 1)), c='r')
        >>>
        >>> dataset = np.random.multivariate_normal(mean=[0, 0], cov=[[5, 1.5], [1.5, 1]], size=(25, ))
        >>> dataset.shape
        (25, 2)
        >>> image = bootplot(make_plot, dataset, output_image_path='bootstrapped_linear_regression.png')
        >>> image.shape
        (512, 512, 3)
    """

    
    if isinstance(backend, str):
        backend_class = create_backend(backend, f, data, m, output_size_px=output_size_px, single_sample=single_sample)

    backend_class.create_figure()
    images = []
    for _ in tqdm(range(m), desc='Generating plots', disable=not verbose):
        backend_class.plot()
        image = backend_class.plot_to_array()
        images.append(image)
        backend_class.clear_figure()
    backend_class.close_figure()
    images = np.stack(images)


    if transformation:
        merged_image = np.array(merge_images(images[..., :3], k, threshold))

    else:
        merged_image = merge_images_original(images[..., :3])

    if output_image_path is not None:
        if verbose:
            print(f'> Saving bootstrapped image to {output_image_path}')
        if isinstance(backend, str) and backend.lower() == "matplotlib":
            dpi = plt.rcParams['figure.dpi']
            Image.fromarray(merged_image).save(output_image_path, dpi=(dpi, dpi))
        else:
            Image.fromarray(merged_image).save(output_image_path)
    if output_animation_path is not None:
        sort_kwargs = dict() if sort_kwargs is None else sort_kwargs
        order = sort_images(images, sort_type, verbose=verbose, **sort_kwargs)
        order.extend(order[:-1][::-1])  # go in reverse
        order = np.array(order)
        images = images[order]

        # Apply decay
        if decay > 0:
            images = decay_images(images, m=m, decay_length=decay)

        animation_speed = max(int(len(images) / animation_duration), 1)
        imageio.mimwrite(output_animation_path, images, fps=animation_speed)
        if verbose:
            print(f'> Saving bootstrapped animation to {output_animation_path}')

    return merged_image
