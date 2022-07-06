from bootplot.sorting import PCASorter, DefaultSorter, HorizontalMassSorter, TravelingSalesmanSorter
import numpy as np


def make_gray_images():
    np.random.seed(0)
    return np.random.uniform(low=0, high=1, size=(25, 100, 100))


def test_pca():
    images = make_gray_images()
    sorter = PCASorter()

    order = sorter.sort(images, features="selected")
    assert len(order) == len(images)
    assert len(set(order)) == len(images)
    assert set(order) == set(range(len(images)))

    order = sorter.sort(images, features="center_mass")
    assert len(order) == len(images)
    assert len(set(order)) == len(images)
    assert set(order) == set(range(len(images)))

    order = sorter.sort(images, features="full")
    assert len(order) == len(images)
    assert len(set(order)) == len(images)
    assert set(order) == set(range(len(images)))


def test_default():
    images = make_gray_images()
    sorter = DefaultSorter()
    order = sorter.sort(images)
    assert order == list(range(len(images)))


def test_hm():
    images = make_gray_images()
    sorter = HorizontalMassSorter()

    order = sorter.sort(images)
    assert len(order) == len(images)
    assert len(set(order)) == len(images)
    assert set(order) == set(range(len(images)))


def test_tsp():
    images = make_gray_images()
    sorter = TravelingSalesmanSorter()

    order = sorter.sort(images)
    assert len(order) == len(images)
    assert len(set(order)) == len(images)
    assert set(order) == set(range(len(images)))
