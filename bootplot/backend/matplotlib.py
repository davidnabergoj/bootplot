import io
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt


def plot_to_array(fig):
    with io.BytesIO() as buff:
        fig.savefig(buff, format='raw')
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    im = data.reshape((int(h), int(w), -1))
    return im


def create_figure(output_size_px: Tuple[int, int]):
    px_size_inches = 1 / plt.rcParams['figure.dpi']
    fig, ax = plt.subplots(
        figsize=(output_size_px[0] * px_size_inches, output_size_px[1] * px_size_inches)
    )
    return fig, ax


def clear_figure(ax):
    ax.cla()


def close_figure(fig):
    plt.close(fig)
