import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from bootplot import bootplot


def plot_regression(data_subset, data_full, ax):
    # Plot full dataset
    ax.scatter(data_full['x'], data_full['y'])

    # Plot regression line trained on the subset
    lr = LinearRegression()
    lr.fit(data_subset[['x']].values, data_subset['y'])
    ax.plot([-10, 10], lr.predict([[-10], [10]]), c='r')

    # Show root mean squared error in a text box
    rmse = np.sqrt(np.mean((data_subset['y'] - lr.predict(data_subset[['x']].values)) ** 2))
    bbox_kwargs = dict(facecolor='none', edgecolor='black', pad=10.0)
    ax.text(x=0, y=-8, s=f'RMSE: {rmse:.4f}', fontsize=12, ha='center', bbox=bbox_kwargs)


if __name__ == '__main__':
    np.random.seed(0)

    # Dataset to be modeled
    df = pd.DataFrame(data=np.random.randn(100, 2), columns=['x', 'y'])
    df['y'] = df['x'] * 1.5 + 2 + np.random.randn(len(df)) * 2.5

    # Create image and animation that show uncertainty
    bootplot(
        plot_regression,
        df,
        output_image_path='demo_image.png',
        output_animation_path='demo_animation.gif',
        xlim=(-10, 10),
        ylim=(-10, 10),
        verbose=True
    )
