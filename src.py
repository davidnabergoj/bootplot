import matplotlib.pyplot as plt
import numpy as np
import io
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
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
