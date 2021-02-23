import numpy as np

import functools as ft

import matplot


class Visualizer:
    """

    """

    def __init__(self, time_evolution_matrix: np.ndarray, number_of_grid_points: int):
        self.time_evolution_matrix = time_evolution_matrix
        self.number_of_grid_points = number_of_grid_points

    @property
    def number_of_grid_points(self) -> int:
        """

        :return:
        """
        return self._number_of_grid_points

    @number_of_grid_points.setter
    def number_of_grid_points(self, new_number_of_grid_points: int):
        """

        :return:
        """
        self._number_of_grid_points = new_number_of_grid_points

    @property
    def time_evolution_matrix(self) -> np.ndarray:
        """

        :return:
        """
        return self._time_evolution_matrix

    @time_evolution_matrix.setter
    def time_evolution_matrix(self, new_time_evolution_matrix: np.ndarray):
        """

        :param new_time_evolution_matrix:
        :return:
        """
        number_of_time_steps = len(new_time_evolution_matrix)
        if isinstance(new_time_evolution_matrix, np.ndarray):
            if new_time_evolution_matrix.shape == (self.number_of_grid_points, number_of_time_steps):
                if ft.reduce(lambda x, y: x and isinstance(y, float),
                             new_time_evolution_matrix.reshape((self.number_of_grid_points * number_of_time_steps)),
                             True):
                    self._time_evolution_matrix = new_time_evolution_matrix
                else:
                    raise TypeError("The entries of the evolution matrix must be of type float.")
            else:
                raise ValueError("The evolution must be a 2D numpy array. Its rows must have the same length as the "
                                 "number of grid points.")
        else:
            raise TypeError("The evolution has to be a numpy array.")

    def visualizes_matrix(self):

