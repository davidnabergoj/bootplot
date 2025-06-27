from abc import abstractmethod, ABC
from typing import Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

import bootplot.backend.matplotlib


class Backend(ABC):
    def __init__(self,
                 f: callable,
                 data: Union[np.ndarray, pd.DataFrame],
                 m: int,
                 output_size_px: Tuple[int, int]):
        self.output_size_px = output_size_px
        self.f = f
        self.data = data
        self.m = m

    @abstractmethod
    def create_figure(self):
        raise NotImplemented

    def plot(self):
        indices = np.random.randint(low=0, high=len(self.data), size=len(self.data))
        if isinstance(self.data, pd.DataFrame):
            return self.f(self.data.iloc[indices], self.data, *self.plot_args)
        elif isinstance(self.data, np.ndarray):
            return self.f(self.data[indices], self.data, *self.plot_args)

    @abstractmethod
    def plot_to_array(self) -> np.ndarray:
        raise NotImplemented

    @abstractmethod
    def clear_figure(self):
        raise NotImplemented

    @abstractmethod
    def close_figure(self):
        raise NotImplemented

    @property
    @abstractmethod
    def plot_args(self):
        raise NotImplemented


class Basic(Backend):
    def __init__(self, f: callable, data: Union[np.ndarray, pd.DataFrame], m: int, output_size_px: Tuple[int, int]):
        super().__init__(f, data, m, output_size_px)
        self.cached_image = None

    def plot(self):
        self.cached_image = super().plot()

    def create_figure(self):
        pass

    def plot_to_array(self) -> np.ndarray:
        return self.cached_image

    def clear_figure(self):
        pass

    def close_figure(self):
        pass

    @property
    def plot_args(self):
        return []


class Matplotlib(Backend):
    def __init__(self,
                 f: callable,
                 data: Union[np.ndarray, pd.DataFrame],
                 m: int,
                 output_size_px: Tuple[int, int] = (512, 512)):
        self.fig = None
        self.ax = None
        super().__init__(f, data, m, output_size_px)

    def create_figure(self):
        self.fig, self.ax = bootplot.backend.matplotlib.create_figure(self.output_size_px)

    def plot_to_array(self) -> np.ndarray:
        return bootplot.backend.matplotlib.plot_to_array(self.fig)

    def clear_figure(self):
        bootplot.backend.matplotlib.clear_figure(self.ax)

    def close_figure(self):
        bootplot.backend.matplotlib.close_figure(None)

    @property
    def plot_args(self):
        return [self.ax]


class GGPlot2(Backend):
    def __init__(self,
                 f: callable,
                 data: Union[np.ndarray, pd.DataFrame],
                 m: int,
                 output_size_px: Tuple[int, int] = (512, 512)):
        super().__init__(f, data, m, output_size_px)

    def create_figure(self):
        raise NotImplemented

    def plot_to_array(self) -> np.ndarray:
        raise NotImplemented

    def clear_figure(self):
        raise NotImplemented

    def close_figure(self):
        raise NotImplemented

    @property
    def plot_args(self):
        raise NotImplemented


class Plotly(Backend):
    def __init__(self,
                 f: callable,
                 data: Union[np.ndarray, pd.DataFrame],
                 m: int,
                 output_size_px: Tuple[int, int] = (512, 512)):
        super().__init__(f, data, m, output_size_px)

    def create_figure(self):
        raise NotImplemented

    def plot_to_array(self) -> np.ndarray:
        raise NotImplemented

    def clear_figure(self):
        raise NotImplemented

    def close_figure(self):
        raise NotImplemented

    @property
    def plot_args(self):
        raise NotImplemented


def create_backend(backend_string, f, data, m, **kwargs):
    if backend_string == 'matplotlib':
        return Matplotlib(f=f, data=data, m=m, **kwargs)
    elif backend_string == 'basic':
        return Basic(f=f, data=data, m=m, **kwargs)
    else:
        raise NotImplemented
