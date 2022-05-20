# Bootstrap your plot

**Bootstrap your plot** is a library that lets you easily visualize uncertainty. You only need to provide a function
that generates a plot from your data and pass it to `bootstrapped_plot`. This will generate a static image and an
animation of your data uncertainty.

The method works by resampling the original dataset and plotting each bootstrapped sample.
The plots are then combined into a single image or an animation.

As an example, suppose we have some data and their corresponding targets. We can model our targets with a regression
line and visualize the uncertainty with the following code:

```python 
import numpy as np
from sklearn.linear_model import LinearRegression

from src import bootstrapped_plot


def make_linear_regression(data_subset, data_full, ax):
    # Plot full dataset
    ax.scatter(data_full[:, 0], data_full[:, 1])

    # Plot regression line trained on the subset
    lr = LinearRegression()
    lr.fit(data_subset[:, 0].reshape(-1, 1), data_subset[:, 1])
    xs = np.linspace(-10, 10, 1000)
    ax.plot(xs, lr.predict(xs.reshape(-1, 1)), c='r')

    # Define global axis settings
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)


if __name__ == '__main__':
    np.random.seed(0)

    # Dataset to be modeled
    dataset = np.random.randn(100, 2)
    noise = np.random.randn(len(dataset)) * 2.5
    dataset[:, 1] = dataset[:, 0] * 1.5 + 2 + noise

    # Create image and animation that show uncertainty
    bootstrapped_plot(
        make_linear_regression,
        dataset,
        output_image_path='bootstrapped_linear_regression.png',
        output_animation_path='bootstrapped_linear_regression.gif',
        verbose=True
    )
```

See the `demo` folder for more examples, including bar charts, point plots, polynomial regression models, pie charts and text plots.