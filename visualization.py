import numpy as np

import functools as ft

import matplotlib.pyplot as plt


class Visualizer:
    # todo: Add documentation for everything.
    """

    """
    def __init__(self, time_evolution_matrix: np.ndarray, number_of_grid_points: int, grid_spacing: float) -> None:
        self.time_evolution_matrix = time_evolution_matrix
        self.number_of_grid_points = number_of_grid_points
        self.grid_spacing = grid_spacing
        self.x_coordinates = np.linspace(0., (self.number_of_grid_points - 1) * self.grid_spacing,
                                         num=self.number_of_grid_points, endpoint=True)

    @property
    def number_of_grid_points(self) -> int:
        """

        :return:
        """
        return self._number_of_grid_points

    @number_of_grid_points.setter
    def number_of_grid_points(self, new_number_of_grid_points: int) -> None:
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
    def time_evolution_matrix(self, new_time_evolution_matrix: np.ndarray) -> None:
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
                    self.time_evolution_matrix = new_time_evolution_matrix
                else:
                    raise TypeError("The entries of the evolution matrix must be of type float.")
            else:
                raise ValueError("The evolution must be a 2D numpy array. Its rows must have the same length as the "
                                 "number of grid points.")
        else:
            raise TypeError("The evolution has to be a numpy array.")

    @property
    def grid_spacing(self) -> float:
        """

        :return:
        """
        return self._grid_spacing

    @grid_spacing.setter
    def grid_spacing(self, new_grid_spacing: float) -> None:
        """

        :return:
        """
        if isinstance(new_grid_spacing, (float, int)):
            if new_grid_spacing > 0:
                self._grid_spacing = float(new_grid_spacing)
            else:
                raise ValueError("The grid spacing must be greater than zero.")
        else:
            raise TypeError("The grid spacing must be of type float.")

    def visualize_amplitudes(self, amplitudes: np.ndarray) -> None:
        """

        :return:
        """
        # Create figure and add axes
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        ax.plot(self.x_coordinates, amplitudes, linewidth=0.25)
