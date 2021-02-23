import numpy as np

import functools as ft

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class Visualizer:
    # todo: Add documentation for everything.
    """

    """

    _number_of_time_steps = None

    def __init__(self, time_evolution_matrix: np.ndarray, number_of_grid_points: int, grid_spacing: float) -> None:
        self.number_of_grid_points = number_of_grid_points
        self.grid_spacing = grid_spacing
        self.time_evolution_matrix = time_evolution_matrix
        self._x_coordinates = np.linspace(0., (self.number_of_grid_points - 1) * self.grid_spacing,
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
        self._number_of_time_steps = len(new_time_evolution_matrix)
        if isinstance(new_time_evolution_matrix, np.ndarray):
            if new_time_evolution_matrix.shape[1] == self.number_of_grid_points:
                if ft.reduce(lambda x, y: x and isinstance(y, float),
                             new_time_evolution_matrix.reshape(
                                 (self.number_of_grid_points * self._number_of_time_steps)),
                             True):
                    self._time_evolution_matrix = new_time_evolution_matrix
                else:
                    raise TypeError("The entries of the evolution matrix must be of type float.")
            else:
                raise ValueError(
                    "The evolution must be a 2D numpy array. Its rows must have the same length as the number of grid points.")
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

    def visualize_amplitude(self, index: int) -> None:
        """

        :return:
        """
        if isinstance(index, int):
            if 0 <= index <= len(self.time_evolution_matrix):
                # Create figure and add axes
                fig, ax = plt.subplots()
                ax.plot(self._x_coordinates, self.time_evolution_matrix[index], linewidth=0.25, marker=".")
                ax.set(xlabel="position", ylabel="amplitude", title="Wave simulation")
                ax.grid()
                plt.show()
            else:
                raise ValueError("The index must be between 0 and the number of time steps.")
        else:
            raise TypeError("The index must be of type int.")

    def animate_wave(self):
        """

        :return:
        """
        # Create variable reference to plot
        fig, ax = plt.subplots()
        ax = plt.axis([self._x_coordinates[0], self._x_coordinates[-1], -1, 1])
        f_d, = plt.plot([], [], "ro")  # Add text annotation and create variable reference

        x_coord = self._x_coordinates
        mat = self.time_evolution_matrix

        print(x_coord)
        print(mat[5])

        # Animation function
        def animate(i):
            """

            :param i:
            :return:
            """
            # f_d.set_data(self._x_coordinates, self.time_evolution_matrix[i])
            f_d.set_data(x_coord, mat[i])
            return f_d,
            # temp.set_text(str(int(T[i])) + ' K')
            # temp.set_color(colors(i))

        # Create animation
        FuncAnimation(fig, animate, frames=np.arange(0.0, self._number_of_time_steps, 1), interval=500,
                      blit=True, repeat=True)
        # Ensure the entire plot is visible
        plt.show()
