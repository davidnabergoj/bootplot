import pathlib
from bootplot import bootplot
from bootplot.base import merge_images
import numpy as np
import shutil


def make_dataset():
    return np.array([11, 7, 7, 5, 6, 5, 4, 4, 7, 9])


def make_bar_chart(data_subset, data_full, ax):
    n_classes = 15
    bar_ticks, bar_counts = np.unique(data_subset, return_counts=True)
    bar_ticks_adjusted = np.arange(n_classes)
    bar_counts_adjusted = np.zeros_like(bar_ticks_adjusted, dtype=np.float32)
    bar_counts_adjusted[bar_ticks] = bar_counts
    ax.bar(bar_ticks_adjusted, bar_counts_adjusted)


def make_directory():
    directory = pathlib.Path('./tests_tmp').absolute()
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir()
    return directory


def test_png():
    # should generate png
    directory = make_directory()
    image_path = directory / 'test.png'
    x = make_dataset()
    bootplot(make_bar_chart, x, m=5, output_image_path=image_path)
    assert image_path.exists()
    shutil.rmtree(directory)


def test_jpg():
    # should generate jpg
    directory = make_directory()
    image_path = directory / 'test.jpg'
    x = make_dataset()
    bootplot(make_bar_chart, x, m=5, output_image_path=image_path)
    assert image_path.exists()
    shutil.rmtree(directory)


def test_gif():
    # should generate gif
    directory = make_directory()
    animation_path = directory / 'test.gif'
    x = make_dataset()
    bootplot(make_bar_chart, x, m=5, output_animation_path=animation_path)
    assert animation_path.exists()
    shutil.rmtree(directory)


def test_mp4():
    # should generate mp4
    directory = make_directory()
    animation_path = directory / 'test.mp4'
    x = make_dataset()
    bootplot(make_bar_chart, x, m=5, output_animation_path=animation_path)
    assert animation_path.exists()
    shutil.rmtree(directory)


def test_bmp():
    # should generate bmp
    directory = make_directory()
    image_path = directory / 'test.bmp'
    x = make_dataset()
    bootplot(make_bar_chart, x, m=5, output_image_path=image_path)
    assert image_path.exists()
    shutil.rmtree(directory)


def test_merge():
    np.random.seed(0)
    images = np.random.randint(low=0, high=256, size=(25, 100, 100, 3))
    merged = merge_images(images)
    assert merged.shape == (100, 100, 3)
    assert np.min(merged) >= 0
    assert np.max(merged) <= 255
    assert merged.dtype == np.uint8


def test_pandas():
    import pandas as pd
    from sklearn.linear_model import LinearRegression

    np.random.seed(0)
    x = np.random.randn(20) * 5
    noise = np.random.randn(20) * 5
    y = 2 * x + 3 + noise

    df = pd.DataFrame({'x': x, 'y': y})

    def plot_regression(data_subset, data_full, ax):
        ax.scatter(data_full['x'], data_full['y'])
        lr = LinearRegression()
        lr.fit(data_subset[['x']].values, data_subset['y'])
        ax.plot([-10, 10], lr.predict([[-10], [10]]), c='r')
        ax.set_xlim(-10, 10)
        ax.set_ylim(-30, 30)

    directory = make_directory()
    image_path = directory / 'pandas_test.png'
    bootplot(
        plot_regression,
        df,
        output_image_path=image_path,
    )
    assert image_path.exists()
    shutil.rmtree(directory)
