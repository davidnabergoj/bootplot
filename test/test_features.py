import numpy as np
from bootplot.image_features import compute_hog_features, compute_center_of_mass, compute_center_of_mass_per_component


def make_gray_images():
    np.random.seed(0)
    return np.random.uniform(low=0, high=1, size=(25, 100, 100))


def test_hog():
    images = make_gray_images()
    features = compute_hog_features(images)
    assert len(features.shape) == 2
    assert len(features) == len(images)

    small_images = images[:, :10, :10]
    features = compute_hog_features(small_images)
    assert len(features.shape) == 2
    assert len(features) == len(small_images)


def test_center_of_mass():
    images = make_gray_images()
    features = compute_center_of_mass(images)
    assert len(features.shape) == 2
    assert len(features) == len(images)

    small_images = images[:, :10, :10]
    features = compute_center_of_mass(small_images)
    assert len(features.shape) == 2
    assert len(features) == len(small_images)


def test_center_of_mass_per_component():
    images = make_gray_images()
    features = compute_center_of_mass_per_component(images)
    assert len(features.shape) == 2
    assert len(features) == len(images)

    small_images = images[:, :10, :10]
    features = compute_center_of_mass_per_component(small_images)
    assert len(features.shape) == 2
    assert len(features) == len(small_images)
