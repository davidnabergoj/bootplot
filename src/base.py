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


def plot_to_array(plot_function, data, fig, ax, **kwargs):
    # plot_function should plot onto ax
    plot_function(data, ax, **kwargs)
    data = fig_to_array(fig, ax)
    return data


def resize_images(images, target_size=(128, 128)):
    return np.array([
        np.array(Image.fromarray(im).resize(target_size)) for im in images
    ])


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
    merged_matrices = merge_matrices(bootstrapped_matrices)
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
        order = sort_images(
            resize_images(bootstrapped_matrices) if resize else bootstrapped_matrices,
            sort_type=sort_type
        )
        order.extend(order[:-1][::-1])  # go in reverse
        order = np.array(order)
        bootstrapped_matrices = bootstrapped_matrices[order]
    if decay:
        bootstrapped_matrices = decay_images(bootstrapped_matrices, m=m, decay_length=decay_length)

    if out_file is not None:
        print('Saving animation')
        if fps * animation_duration > len(bootstrapped_matrices):
            image_mask = np.arange(len(bootstrapped_matrices))
        else:
            image_mask = np.linspace(0, len(bootstrapped_matrices) - 1, fps * animation_duration).astype(np.int)
        imageio.mimwrite(out_file, bootstrapped_matrices[image_mask], format="GIF", fps=fps)

    return bootstrapped_matrices
