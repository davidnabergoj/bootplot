import warnings
from pathlib import Path
from typing import Union, Tuple

import numpy as np
import imageio
import io
import matplotlib.pyplot as plt
import pandas as pd
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
                  data: Union[np.ndarray, pd.DataFrame],
                  indices: np.ndarray,
                  fig: plt.Figure,
                  ax: plt.Axes,
                  xlim: Tuple[float, float] = (None, None),
                  ylim: Tuple[float, float] = (None, None),
                  **kwargs) -> np.ndarray:
    """
    Plot data and obtain image.

    :param plot_function: function handle to perform the plotting. The handle should have the form ``f(data_subset,
        data_full, ax)`` where ``data_subset``, ``data_full`` are `numpy.ndarray` or `pandas.DataFrame` objects and
        ``ax`` is a `matplotlib.axes.Axes` object.
    :type plot_function: callable

    :param data: full data to be used in plotting.
    :type data: numpy.ndarray or pandas.DataFrame

    :param indices: bootstrap resampled indices of the data.
    :type indices: numpy.ndarray

    :param fig: figure object with a single Axes.
    :type fig: matplotlib.figure.Figure

    :param ax: Axes object.
    :type ax: matplotlib.axes.Axes

    :param xlim: x axis limits.
    :type xlim: tuple[float, float]

    :param ylim: y axis limits.
    :type ylim: tuple[float, float]

    :param kwargs: keyword arguments to plot_function.

    :return: image.
    :rtype: numpy.ndarray
    """
    if isinstance(data, pd.DataFrame):
        plot_function(data.iloc[indices], data, ax, **kwargs)
    else:
        plot_function(data[indices], data, ax, **kwargs)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    data = fig_to_array(fig, ax)
    return data


def merge_images(images: np.ndarray, power: float = 1.0) -> np.ndarray:
    """
    Merge images into a static image (averaged image).
    The shape of images is (batch_size, width, height, channels).
    This operation overwrites input images.

    :param images: images corresponding to different bootstrap samples.
    :param power: raise the merged image pixels to the specified power to increase contrast.
    :return: merged image.
    """
    images = images.astype(np.float32) / 255  # Cast to float
    merged = np.mean(images, axis=0) ** power
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
             output_size_px: Tuple[int, int] = (512, 512),
             output_image_path: Union[str, Path] = None,
             output_animation_path: Union[str, Path] = None,
             contrast_modifier: float = 1.0,
             sort_type: str = 'tsp',
             sort_kwargs: dict = None,
             decay: int = 0,
             fps: int = 60,
             xlim: Tuple[float, float] = (None, None),
             ylim: Tuple[float, float] = (None, None),
             verbose: bool = False,
             warn_limits: bool = True) -> np.ndarray:
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

    :param output_size_px: output size (height, width) in pixels. Default: ``(512, 512)``.
    :type output_size_px: tuple[int, int]

    :param output_image_path: path where the image should be stored. The image format is inferred from the filename
        extension. If None, the image is not stored. Default: ``None``.
    :type output_image_path: str or pathlib.Path

    :param output_animation_path: path where the animation should be stored. The animation format is inferred from the
        filename extension. If None, the animation is not created. Default: ``None``.
    :type output_animation_path: str or pathlib.Path

    :param contrast_modifier: modify the contrast in the static image. Setting this to 1 keeps the same contrast,
        setting this to less than 1 reduces contrast, setting this to greater than 1 increases contrast. Default: ``1``.
    :type contrast_modifier: float

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

    :param fps: desired output framerate for the animation. Default: ``60``.
    :type fps: int

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
        >>> image = bootplot(
        ...     make_plot,
        ...     dataset,
        ...     output_image_path='bootstrapped_linear_regression.png',
        ...     xlim=(-10, 10),
        ...     ylim=(-10, 10)
        ... )
        >>> image.shape
        (512, 512, 3)
    """
    if None in xlim and warn_limits:
        warnings.warn("One or both x limits are None. "
                      "This may cause the results to be blurry. "
                      "We recommend setting both x limits.")
    if None in ylim and warn_limits:
        warnings.warn("One or both y limits are None. "
                      "This may cause the results to be blurry. "
                      "We recommend setting both y limits.")

    px_size_inches = 1 / plt.rcParams['figure.dpi']
    fig, ax = plt.subplots(figsize=(output_size_px[0] * px_size_inches, output_size_px[1] * px_size_inches))
    image_samples = np.stack([
        plot_to_array(f, data, np.random.randint(low=0, high=len(data), size=len(data)), fig, ax, xlim=xlim, ylim=ylim)
        for _ in tqdm(range(m), desc='Generating plots', disable=not verbose)
    ])
    merged_image = merge_images(image_samples, power=contrast_modifier)[..., :3]  # Do not use the alpha channel
    plt.close(fig)

    if output_image_path is not None:
        if verbose:
            print(f'> Saving bootstrapped image to {output_image_path}')
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
            print(f'> Saving bootstrapped animation to {output_animation_path}')

    return merged_image
