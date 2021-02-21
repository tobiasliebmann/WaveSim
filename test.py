import numpy as np


class Numeric1DWaveSimulator:

    def __init__(self, _delta_x, _delta_t, _speed_of_sound, _number_of_grid_points, _number_of_time_steps):
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
        Setter method for the speed of sound. The new speed of sound must be of type float or int and greater than
        zero. If this is not the case, the program will raise an according error.
        :return: -
        """
        if isinstance(new_speed_of_sound, (int, float)):
            if new_speed_of_sound > 0:
                self._speed_of_sound = new_speed_of_sound
            else:
                raise ValueError("The speed of sound must be greater than zero.")
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

    # Make all the instance variable properties.
    _delta_x = property(get_delta_x, set_delta_x)
    _delta_t = property(get_delta_t, set_delta_t)
    _speed_of_sound = property(get_speed_of_sound, set_speed_of_sound)
    _number_of_grid_points = property(get_number_of_grid_points, set_number_of_grid_points)
    _number_of_time_steps = property(get_number_of_time_steps, set_number_of_time_steps)
