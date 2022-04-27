import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from src import bootstrapped_plot, bootstrapped_animation


def make_plot_demo_poly_reg(data, ax, xlim=(-10, 10), ylim=(-10, 10), degree=2):
    poly = PolynomialFeatures(degree=degree)
    features = poly.fit_transform(data[:, 0].reshape(-1, 1))

    lr = LinearRegression()
    lr.fit(features, data[:, 1])

    xs = np.linspace(xlim[0], xlim[1], 100)
    xs_features = poly.transform(xs.reshape(-1, 1))
    ax.plot(xs, lr.predict(xs_features), c='r')
    ax.scatter(data[:, 0], data[:, 1])
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    # ax.axis('off')


if __name__ == '__main__':
    np.random.seed(0)

    # You can nicely see which part of the polynomial is an uncertain fit

    plt.rcParams["figure.figsize"] = (5, 5)

    dataset = np.random.randn(100, 2)
    noise = np.random.randn(len(dataset)) * 2.5
    dataset[:, 1] = dataset[:, 0] * 1.5 + 2 + noise
    mat = bootstrapped_plot(make_plot_demo_poly_reg, dataset, m=100, out_file='bootstrapped_poly_reg_deg2.png')

    plt.figure()
    plt.matshow(mat)
    plt.axis('off')
    plt.show()

    mat = bootstrapped_plot(
        lambda *args, **kwargs: make_plot_demo_poly_reg(*args, **kwargs, degree=3),
        dataset, m=100, out_file='bootstrapped_poly_reg_deg3.png'
    )

    plt.figure()
    plt.matshow(mat)
    plt.axis('off')
    plt.show()

    bootstrapped_animation(make_plot_demo_poly_reg, dataset, m=100, out_file='bootstrapped_poly_reg_deg2.gif')
    bootstrapped_animation(
        lambda *args, **kwargs: make_plot_demo_poly_reg(*args, **kwargs, degree=3),
        dataset, m=100, out_file='bootstrapped_poly_reg_deg3.gif'
    )
