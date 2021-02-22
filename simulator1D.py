import numpy as np


class Numeric1DWaveSimulator:
    def __init__(self, _delta_x, _delta_t, _speed_of_sound, _number_of_grid_points, _number_of_time_steps,
                 _initial_positions, _initial_velocities):
        # Distance between grid points.
        self._delta_x = _delta_x
        # Time steps taken in the simulation.
        self._delta_t = _delta_t
        # Speed of sound in the medium.
        self._speed_of_sound = _speed_of_sound
        # Number of grid points in the grid.
        self._number_of_grid_points = _number_of_grid_points
        # Number of time steps after which the simulation will terminate.
        self._number_of_time_steps = _number_of_time_steps
        # Initial positions of the points.
        self._initial_positions = _initial_positions
        # Initial velocities of the points.
        self._initial_velocities = _initial_velocities
        # grid constant.
        self.grid_constant = self.delta_t * self.speed_of_sound / self.delta_x
        # Defines the first position as the entered initial position.
        self.current_positions = self.initial_positions
        # Defines the former position lso as the initial position.
        # todo: I don't know if this right physically. Look this up.
        self.former_position = self.initial_positions
        # creates the time step matrix.
        # todo: Check if this part blows up.
        self.time_step_matrix = self.create_time_step_matrix(self.number_of_grid_points, self.grid_constant)

    def set_delta_x(self, new_delta_x) -> None:
        """
        Setter method for the distance between the grid points delta_x. The function only takes floats and ints
        which are greater than zero. The method will raise errors if this is not the case.
        :param new_delta_x: New distance between grid points.
        :return: -
        """
        if isinstance(new_delta_x, (float, int)):
            if new_delta_x > 0:
                self._delta_x = new_delta_x
            else:
                raise ValueError("The distance between the grid points must be greater than zero.")
        else:
            raise TypeError("delta_x must be of type float or int.")

    def get_delta_x(self):
        """
        Getter method for the distance between the grid points
        :return: The distance between the grid points.
        """
        return self._delta_x

    def set_delta_t(self, new_delta_t) -> None:
        """
        Setter method for the time steps. The function only takes floats which are greater
        than zero and will raise errors if this is not the case.
        :param new_delta_t: size of the time steps in the simulation.
        :return: -
        """
        if isinstance(new_delta_t, (int, float)):
            if new_delta_t > 0:
                self._delta_x = new_delta_t
            else:
                raise ValueError("The distance between the grid points must be greater than zero.")
        else:
            raise TypeError("delta_x must be of type float or int.")

    def get_delta_t(self):
        """
        Getter method for the time steps in the simulation.
        :return: Size of the time steps.
        """
        return self._delta_t

    def set_speed_of_sound(self, new_speed_of_sound) -> None:
        """
        Setter method for the speed of sound. The new speed of sound must be of type float or int.
        If this is not the case, the program will raise an according error.
        :return: -
        """
        if isinstance(new_speed_of_sound, (int, float)):
            self._speed_of_sound = new_speed_of_sound
        else:
            raise TypeError("The speed of sound must be of type int or float.")

    def get_speed_of_sound(self):
        """
        Getter method for the speed of sound used in the simulation.
        :return: Speed of sound.
        """
        return self._speed_of_sound

    def set_number_of_grid_points(self, new_number_of_grid_points: int) -> None:
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

    def get_number_of_grid_points(self) -> int:
        """
        Getter method for the number of grid points in the simulation. Returns the number of grid points used in the
        simulation.
        :return: Number of grid points used in the 1D simulation.
        """
        return self._number_of_grid_points

    def set_number_of_time_steps(self, new_number_of_time_steps: int) -> None:
        """
        Setter method for the number of time steps used in the 1D simulation. This number must be of type int and
        greater than zero. If this is not the case this method will raise an error.
        :param new_number_of_time_steps:
        :return: -
        """
        if isinstance(new_number_of_time_steps, int):
            if new_number_of_time_steps > 0:
                self._number_of_grid_points = new_number_of_time_steps
            else:
                raise ValueError("The number of time steps must be greater than zero.")
        else:
            raise TypeError("The number of grid points must be of type int.")

    def get_number_of_time_steps(self) -> int:
        """
        Getter method for the number of time steps used in the !D simulation.
        :return: Number of time steps used in the simulation.
        """
        return self._number_of_time_steps

    def set_initial_positions(self, new_initial_positions: np.ndarray) -> None:
        """
        The setter method for the initial positions at t = 0. The new initial positions must be a numpy array and its
        length has to coincide with the number of grid points. The methods tests if these conditions are met and raises
        according errors.
        :param new_initial_positions: New initial positions of the
        :return: None
        """
        if isinstance(new_initial_positions, np.ndarray):
            if len(new_initial_positions) == self._number_of_grid_points:
                self._initial_positions = new_initial_positions
            else:
                raise ValueError("The number of grid points and the length of the new initial positions must coincide.")
        else:
            raise TypeError("The new initial position must be a numpy array.")

    def get_initial_positions(self) -> np.ndarray:
        """
        Getter method for the initial positions of the grid points at t = 0.
        :return: The initial positions.
        """
        return self._initial_positions

    def set_initial_velocities(self, new_initial_velocities: np.ndarray) -> None:
        """
        The setter method for the initial velocities at t = 0. The new initial velocities must be a numpy array and its
        length has to coincide with the number of grid points. The methods tests if these conditions are met and raises
        according errors.
        :param new_initial_velocities:
        :return: None
        """
        if isinstance(new_initial_velocities, np.ndarray):
            if len(new_initial_velocities) == self._number_of_grid_points:
                self._initial_velocities = new_initial_velocities
            else:
                raise ValueError("The number of grid points and the length of the new initial velocities must coincide."
                                 )
        else:
            raise TypeError("The new initial velocities must be a numpy array.")

    def get_initial_velocities(self) -> np.ndarray:
        """
        Getter method for the initial positions of the grid points at t = 0.
        :return: The initial positions.
        """
        return self._initial_velocities

    # Make all the instance variable properties.
    delta_x = property(get_delta_x, set_delta_x)
    delta_t = property(get_delta_t, set_delta_t)
    speed_of_sound = property(get_speed_of_sound, set_speed_of_sound)
    number_of_grid_points = property(get_number_of_grid_points, set_number_of_grid_points)
    number_of_time_steps = property(get_number_of_time_steps, set_number_of_time_steps)
    initial_positions = property(get_initial_positions, set_initial_positions)
    get_initial_velocities = property(get_initial_velocities, set_initial_velocities)

    def stability_test(self):
        """
        Checks if the entered values of the grid spacing, time spacing and speed of sound is between 0  and 1. If this
        is not the case a warning message will be displayed.
        :return: None
        """
        if not 0 <= self.grid_constant <= 1:
            print("The scheme may be unstable since the grid constant is " + str(self.grid_constant) + ". It should be"
                                                                                                       "between 0  and 1.")

    @staticmethod
    def create_time_step_matrix(dim: int, grid_cons) -> np.ndarray:
        """
        Should calculate the matrix connecting the time steps.
        :param dim: Dimension N of the NxN matrix.
        :param grid_cons: grid constant corresponding to the grid of the simulation.
        :return:
        """
        temp = np.zeros((dim, dim))
        rearrange_array = np.arange(dim - 1)
        temp[rearrange_array, rearrange_array + 1] = 1
        temp = 2 * (1 - grid_cons) * np.identity(dim) + grid_cons * temp + grid_cons * temp.T
        temp[0, 0] = 0
        temp[0, 1] = 0
        temp[dim - 1, dim - 1] = 0
        temp[dim - 1, dim - 2] = 0
        return temp
