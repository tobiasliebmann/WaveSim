import numpy as np

import datetime as dt


class Numeric1DWaveSimulator:
    # todo: Add class doc string.

    # Counter for the time steps taken int he algorithm.
    time_step = 0

    def __init__(self, delta_x, delta_t, speed_of_sound, number_of_grid_points, number_of_time_steps,
                 initial_amplitudes, initial_velocities):

        # Distance between individual grid points.
        self.delta_x = delta_x
        # Time steps taken in the simulation.
        self.delta_t = delta_t
        # Speed of sound in the medium.
        self.speed_of_sound = speed_of_sound
        # Number of grid points in the grid.
        self.number_of_grid_points = number_of_grid_points
        # Number of time steps after which the simulation will terminate.
        self.number_of_time_steps = number_of_time_steps
        # Initial positions of the points.
        self.initial_amplitudes = initial_amplitudes
        # Initial velocities of the points.
        self.initial_velocities = initial_velocities
        # grid constant.
        self.courant_number = (self.delta_t * self.speed_of_sound / self.delta_x) ** 2
        # Defines the first position as the entered initial position.
        self.current_amplitudes = self.initial_amplitudes
        # There are no former amplitudes at t = 0.
        self.former_amplitudes = None
        # Creates the time step matrix.
        self.time_step_matrix = self.create_time_step_matrix(self.number_of_grid_points, self.courant_number)
        # This array saves the time evolution of the amplitudes.
        self.amplitudes_time_evolution = np.array([self.initial_amplitudes])

    @classmethod
    def init_from_file(cls, link_to_file):
        """
        This method functions as a second constructor, which can be used to initialize a simulator via a file.
        :param link_to_file: This is the link to the file in which the data is stored. It should be a .npy file.
        :return: Numeric1DWaveSimulator, Returns a Numeric1DWaveSimulator object with the variables that are declared in
        the file.
        """
        if isinstance(link_to_file, str):
            # Load the data.
            loaded_data = np.load(link_to_file, allow_pickle=True)
            # print(loaded_data)
            print(loaded_data[-1])
            # Save the time evolution matrix for later use.
            temp = loaded_data[-1]
            # Delete the time evolution matrix since it is not needed in the initializer.
            loaded_data = np.delete(loaded_data, -1)
            # Make new instance of the class using the loaded data.
            new_obj = cls(*loaded_data)
            # Set the time evolution matrix.
            new_obj.amplitudes_time_evolution = temp
            return new_obj
        else:
            raise ValueError("The provided link must be a string.")

    @property
    def delta_x(self) -> float:
        """
        Getter method for the distance between the grid points
        :return: The distance between the grid points.
        """
        return self._delta_x

    @delta_x.setter
    def delta_x(self, new_delta_x: float) -> None:
        """
        Setter method for the distance between the grid points delta_x. The function only takes floats and ints
        which are greater than zero. The method will raise errors if this is not the case.
        :param new_delta_x: New distance between grid points.
        :return: -
        """
        if isinstance(new_delta_x, (float, int)):
            if new_delta_x > 0:
                # Cast the new grid spacing as a float.
                self._delta_x = float(new_delta_x)
            else:
                raise ValueError("The distance between the grid points must be greater than zero.")
        else:
            raise TypeError("delta_x must be of type float or int.")

    @property
    def delta_t(self) -> float:
        """
        Getter method for the time steps in the simulation.
        :return: Size of the time steps.
        """
        return self._delta_t

    @delta_t.setter
    def delta_t(self, new_delta_t: float) -> None:
        """
        Setter method for the time steps. The function only takes floats which are greater
        than zero and will raise errors if this is not the case.
        :param new_delta_t: size of the time steps in the simulation.
        :return: -
        """
        if isinstance(new_delta_t, (int, float)):
            if new_delta_t > 0:
                # Cast the new grid spacing as a float.
                self._delta_t = float(new_delta_t)
            else:
                raise ValueError("The distance between the grid points must be greater than zero.")
        else:
            raise TypeError("delta_x must be of type float or int.")

    @property
    def speed_of_sound(self) -> float:
        """
        Getter method for the speed of sound used in the simulation.
        :return: Speed of sound.
        """
        return self._speed_of_sound

    @speed_of_sound.setter
    def speed_of_sound(self, new_speed_of_sound: float) -> None:
        """
        Setter method for the speed of sound. The new speed of sound must be of type float or int.
        If this is not the case, the program will raise an according error.
        :return: -
        """
        if isinstance(new_speed_of_sound, (int, float)):
            # Cast the new grid spacing as a float.
            self._speed_of_sound = float(new_speed_of_sound)
        else:
            raise TypeError("The speed of sound must be of type int or float.")

    @property
    def number_of_grid_points(self) -> int:
        """
        Getter method for the number of grid points in the simulation. Returns the number of grid points used in the
        simulation.
        :return: Number of grid points used in the 1D simulation.
        """
        return self._number_of_grid_points

    @number_of_grid_points.setter
    def number_of_grid_points(self, new_number_of_grid_points: int) -> None:
        """
        Setter method for the number of grid points used in the 1D simulation. This number must be of type int and
        greater than zero. If this is not the case this method will raise an error.
        :param new_number_of_grid_points: Number of grid points in the 1D simulation.
        :return: -
        """
        if isinstance(new_number_of_grid_points, int):
            if new_number_of_grid_points > 0:
                self._number_of_grid_points = new_number_of_grid_points
            else:
                raise ValueError("The number of grid point must be greater than zero.")
        else:
            raise TypeError("The number of grid points must be of type int.")

    @property
    def number_of_time_steps(self) -> int:
        """
        Getter method for the number of time steps used in the !D simulation.
        :return: Number of time steps used in the simulation.
        """
        return self._number_of_time_steps

    @number_of_time_steps.setter
    def number_of_time_steps(self, new_number_of_time_steps: int) -> None:
        """
        Setter method for the number of time steps used in the 1D simulation. This number must be of type int and
        greater than zero. If this is not the case this method will raise an error.
        :param new_number_of_time_steps:
        :return: -
        """
        if isinstance(new_number_of_time_steps, int):
            if new_number_of_time_steps > 0:
                self._number_of_time_steps = new_number_of_time_steps
            else:
                raise ValueError("The number of time steps must be greater than zero.")
        else:
            raise TypeError("The number of grid points must be of type int.")

    @property
    def initial_amplitudes(self) -> np.ndarray:
        """
        Getter method for the initial positions of the grid points at t = 0.
        :return: The initial positions.
        """
        return self._initial_amplitudes

    @initial_amplitudes.setter
    def initial_amplitudes(self, new_initial_amplitudes: np.ndarray) -> None:
        """
        The setter method for the initial positions at t = 0. The new initial positions must be a numpy array and its
        length has to coincide with the number of grid points. Further the boundary condition has to be fulfilled.
        The methods tests if these conditions are met and raises according errors.
        :param new_initial_amplitudes: New initial amplitudes for the initial condition.
        :return: None
        """
        # Threshold value under which a variable will be taken as zero
        threshold_value = 10 ** (-10)
        if isinstance(new_initial_amplitudes, np.ndarray):
            if len(new_initial_amplitudes) == self.number_of_grid_points:
                if new_initial_amplitudes[0] <= threshold_value and new_initial_amplitudes[-1] <= threshold_value:
                    self._initial_amplitudes = new_initial_amplitudes
                else:
                    raise ValueError("The first and last entry of the amplitudes have to be 0 to respect the boundary "
                                     "conditions")
            else:
                raise ValueError("The number of grid points and the length of the initial amplitudes must "
                                 "coincide.")
        else:
            raise TypeError("The initial amplitudes must be a numpy array.")

    @property
    def initial_velocities(self) -> np.ndarray:
        """
        Getter method for the initial positions of the grid points at t = 0.
        :return: The initial positions.
        """
        return self._initial_velocities

    @initial_velocities.setter
    def initial_velocities(self, new_initial_velocities: np.ndarray) -> None:
        """
        The setter method for the initial velocities at t = 0. The new initial velocities must be a numpy array and its
        length has to coincide with the number of grid points. The methods tests if these conditions are met and raises
        according errors.
        :param new_initial_velocities:
        :return: None
        """
        # Threshold value under which a variable will be taken as zero
        threshold_value = 10 ** (-10)
        if isinstance(new_initial_velocities, np.ndarray):
            if len(new_initial_velocities) == self.number_of_grid_points:
                if new_initial_velocities[0] <= threshold_value and new_initial_velocities[-1] <= threshold_value:
                    self._initial_velocities = new_initial_velocities
                else:
                    raise ValueError("The first and last entry of the velocities have to be 0 to respect the boundary "
                                     "conditions")
            else:
                raise ValueError("The number of grid points and the length of the new initial velocities must coincide."
                                 )
        else:
            raise TypeError("The new initial velocities must be a numpy array.")

    def stability_test(self) -> None:
        """
        Checks if the entered values of the grid spacing, time spacing and speed of sound is between 0  and 1. If this
        is not the case a warning message will be displayed.
        :return: None
        """
        if not 0 <= self.courant_number <= 1:
            print("The scheme may be unstable since the Courant number is ", self.courant_number, ". It should be "
                                                                                                  "between 0  and 1.")
        else:
            print("Scheme is stable. The Courant number is", str(self.courant_number) + ".")

    @staticmethod
    def create_time_step_matrix(dim: int, courant_number: float) -> np.ndarray:
        """
        Returns the matrix connecting the time steps. This matrix is a quadratic matrix with dimension
        dim x dim. The Matrix has only zeros in the first and last row and only diagonal and off diagonals are
        populated.
        :param dim: Dimension N of the NxN matrix.
        :param courant_number: grid constant corresponding to the grid of the simulation.
        :return: The matrix used to calculate the next time step.
        """
        if courant_number != float:
            courant_number = float(courant_number)
        temp = np.zeros((dim, dim))
        rearrange_array = np.arange(dim - 1)
        temp[rearrange_array, rearrange_array + 1] = 1
        temp = 2 * (1 - courant_number) * np.identity(dim) + courant_number * temp + courant_number * temp.T
        temp[0, 0] = 0
        temp[0, 1] = 0
        temp[1, 0] = 0
        temp[dim - 2, dim - 1] = 0
        temp[dim - 1, dim - 1] = 0
        temp[dim - 1, dim - 2] = 0
        return temp

    def update(self) -> None:
        """
        Updates the amplitudes to the next time step and sets the current and former state accordingly. The counter for
        the time steps is then increased by one.
        :return: None
        """
        if self.time_step == 0:
            # print(self.time_step_matrix)
            self.former_amplitudes = self.current_amplitudes
            # The first is given by this equation.
            self.current_amplitudes = np.dot((1 / 2) * self.time_step_matrix,
                                             self.current_amplitudes) + self.delta_t * self.initial_velocities
        else:
            temp = np.dot(self.time_step_matrix, self.current_amplitudes) - self.former_amplitudes
            self.former_amplitudes = self.current_amplitudes
            self.current_amplitudes = temp
        self.amplitudes_time_evolution = np.vstack(
            [self.amplitudes_time_evolution, np.array([self.current_amplitudes])])
        self.time_step += 1

    def run(self) -> np.ndarray:
        """
        Checks the stability of the scheme and then runs the simulation using the according formula. This is done
        updating the simulation until the number of time steps is reached at which point the method will return the
        result of the simulation.
        :return: The result of the simulation. Each row in the matrix corresponds to a time step.
        """
        self.stability_test()
        while self.time_step <= self.number_of_time_steps:
            self.update()
        return self.amplitudes_time_evolution

    def save_data(self, link_to_file=None) -> None:
        """
        This method saves the current attributes of a simulator object in a npy-file. If no link is provided in the form
        of a string, the method will create a file using the current date and time.
        :param link_to_file: Optional variable which is a link to a npy-file, where the data is saved.
        :return: None
        """
        utc_time = dt.datetime.utcnow().replace(microsecond=0)
        obj_to_save = np.array([self.delta_x,
                                self.delta_t,
                                self.speed_of_sound,
                                self.number_of_grid_points,
                                self.number_of_time_steps,
                                self.initial_amplitudes,
                                self.initial_velocities,
                                self.amplitudes_time_evolution], dtype=object)
        if link_to_file is not None:
            if isinstance(link_to_file, str):
                np.save(link_to_file, obj_to_save, allow_pickle=True)
            else:
                raise ValueError("The provided link must be a string.")
        else:
            np.save(("wave_sim_1D"+str(utc_time)+".npy").replace(" ", "_"), obj_to_save, allow_pickle=True)
