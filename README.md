# Bootplot: bootstrap your plot

**Bootplot** is a package that lets you easily visualize uncertainty. You only need to provide a function
that generates a plot from your data and pass it to `bootplot`. This will generate a static image and an
animation of your data uncertainty.

The method works by resampling the original dataset using bootstrap and plotting each bootstrapped sample.
The plots are then combined into a single image or an animation.

As an example, suppose we have some data and their corresponding targets. We can model our targets with a regression
line and visualize the uncertainty with the following code:

```python 
import numpy as np
from sklearn.linear_model import LinearRegression

from bootplot import bootplot


def make_linear_regression(data_subset, data_full, ax):
    # Plot full dataset
    ax.scatter(data_full[:, 0], data_full[:, 1])

    # Plot regression line trained on the subset
    lr = LinearRegression()
    lr.fit(data_subset[:, 0].reshape(-1, 1), data_subset[:, 1])
    xs = np.linspace(-10, 10, 1000)
    ax.plot(xs, lr.predict(xs.reshape(-1, 1)), c='r')


if __name__ == '__main__':
    np.random.seed(0)

    # Dataset to be modeled
    dataset = np.random.randn(100, 2)
    noise = np.random.randn(len(dataset)) * 2.5
    dataset[:, 1] = dataset[:, 0] * 1.5 + 2 + noise

    # Create image and animation that show uncertainty
    bootplot(
        make_linear_regression,
        dataset,
        output_image_path='bootstrapped_linear_regression.png',
        output_animation_path='bootstrapped_linear_regression.gif',
        xlim=(-10, 10),
        ylim=(-10, 10),
        verbose=True
    )
```

See the `examples` folder for more examples, including bar charts, point plots, polynomial regression models, pie charts and text plots.

## Installation

**Bootplot** requires Python version 3.7 or greater. You can install **Bootplot** using: 
```
pip install bootplot
```

Alternatively, you can install **Bootplot** locally:
```
git clone https://github.com/davidnabergoj/bootplot
cd bootplot
pip install .
```
