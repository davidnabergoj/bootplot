from sklearn.datasets import make_regression

x, y = make_regression(n_samples=20, n_features=1, random_state=0, noise=5.0)

import matplotlib.pyplot as plt

plt.scatter(x, y)
plt.savefig('quickstart_scatter.png')
plt.show()

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x, y)

import numpy as np

test_x = np.linspace(-2, 3).reshape(-1, 1)
lr.predict(test_x)

fig, ax = plt.subplots()
ax.scatter(x, y)
ax.plot(test_x, lr.predict(test_x), c='r')

ax.set_xlim(-2, 3)
ax.set_ylim(-20, 40)

plt.savefig('quickstart_regression_basic.png')

plt.show()

from bootplot import bootplot


def plot_regression(data_subset, data_full, ax):
    lr = LinearRegression()
    lr.fit(data_subset[:, 0].reshape(-1, 1), data_subset[:, 1])

    test_x = np.linspace(-2, 3).reshape(-1, 1)
    ax.scatter(data_full[:, 0], data_full[:, 1])

    ax.plot(test_x, lr.predict(test_x), c='r')
    ax.set_xlim(-2, 3)
    ax.set_ylim(-20, 40)


bootplot(
    plot_regression,
    data=np.column_stack([x, y]),
    output_image_path='quickstart_regression.png',
    output_animation_path='quickstart_regression.gif',
    verbose=True
)
