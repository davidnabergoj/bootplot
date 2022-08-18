![logo](https://raw.githubusercontent.com/davidnabergoj/bootplot/master/logo.png)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/davidnabergoj/bootplot/blob/main/LICENSE)
[![Documentation Status](https://readthedocs.org/projects/bootplot/badge/?version=latest)](https://bootplot.readthedocs.io/en/latest/?badge=latest)
[![build](https://github.com/davidnabergoj/bootplot/actions/workflows/build.yml/badge.svg)](https://github.com/davidnabergoj/bootplot/actions/workflows/build.yml)
[![tests](https://github.com/davidnabergoj/bootplot/actions/workflows/tests.yml/badge.svg)](https://github.com/davidnabergoj/bootplot/actions/workflows/tests.yml)

**bootplot** is a package for black-box uncertainty visualization. 
By providing a dataset and a plotting function, **bootplot** automatically generates a static image and an animation of your uncertainty.

The method works by resampling the original dataset using bootstrap and plotting each bootstrapped sample.
The plots are then combined into a single image or an animation.
**bootplot** is also especially useful when dealing with small datasets, since it
relies on the bootstrap method which robustly estimates uncertainty using resampling.

**bootplot** supports datasets represented as numpy arrays or pandas dataframes. 
Supported image output formats include popular formats such as JPG, PNG, BMP. Supported animation formats include popular formats such as GIF and MP4.
<!--For a complete list of formats, see the [Pillow documentation](https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html) and the [FFMPEG documentation](https://ffmpeg.org/ffmpeg-formats.html).-->

## Installation

**bootplot** requires Python version 3.8 or greater. You can install it using:

```
pip install bootplot
```

Alternatively, you can install **bootplot** using:

```
git clone https://github.com/davidnabergoj/bootplot
cd bootplot
python setup.py install
```

## Example

Suppose we have some data and their corresponding targets. We can model our targets with a regression
line and visualize the uncertainty with the following code:

```python 
import numpy as np
from sklearn.linear_model import LinearRegression

from bootplot import bootplot


def plot_regression(data_subset, data_full, ax):
    # Plot full dataset
    ax.scatter(data_full[:, 0], data_full[:, 1])

    # Plot regression line trained on the subset
    lr = LinearRegression()
    lr.fit(data_subset[:, 0].reshape(-1, 1), data_subset[:, 1])
    ax.plot([-10, 10], lr.predict([[-10], [10]]), c='r')
    
    # Show root mean squared error in a text box
    rmse = np.sqrt(np.mean(np.square(data_subset[:, 1] - lr.predict(data_subset[:, 0].reshape(-1, 1)))))
    bbox_kwargs = dict(facecolor='none', edgecolor='black', pad=10.0)
    ax.text(x=0, y=-8, s=f'RMSE: {rmse:.4f}', fontsize=12, ha='center', bbox=bbox_kwargs)

    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

if __name__ == '__main__':
    np.random.seed(0)

    # Dataset to be modeled
    dataset = np.random.randn(100, 2)
    noise = np.random.randn(len(dataset)) * 2.5
    dataset[:, 1] = dataset[:, 0] * 1.5 + 2 + noise

    # Create image and animation that show uncertainty
    bootplot(
        plot_regression,
        dataset,
        output_image_path='demo_image.png',
        output_animation_path='demo_animation.gif',
        verbose=True
    )
```

This will generate a static image and an animation, as shown below.
The static image on points shows the full scattered dataset in blue and regression lines that correspond to each
bootstrapped sample of the dataset in red.
The spread of regression lines represents uncertainty according to the bootstrap process.
We can also see the uncertainty in root mean squared error (RMSE).
We see that only the first digit of RMSE is significant, since the decimal part is blurred.
The animation on the right displays uncertainty by iterating over a sequence of plots containing regression lines.

<table>
    <tr>
        <td><img src="https://raw.githubusercontent.com/davidnabergoj/bootplot/master/demo_image.png"></td>
        <td><img src="https://raw.githubusercontent.com/davidnabergoj/bootplot/master/demo_animation.gif"></td>
    </tr>
</table>

See the [examples](examples) folder for more examples, including bar charts, point plots, polynomial regression models, pie charts, text plots and pandas dataframes.

## Documentation

Read the documentation and check out tutorials at https://bootplot.readthedocs.io/en/latest/
