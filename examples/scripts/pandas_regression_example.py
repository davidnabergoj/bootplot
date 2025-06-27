import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from bootplot import bootplot

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


img = bootplot(
    plot_regression,
    df,
    output_image_path='pandas_example.png',
    verbose=True
)
