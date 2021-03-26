import numpy as np
import datetime as dt
from abc import ABC, abstractmethod
from typing import List


class NumericWaveSimulator(ABC):
    def __init__(self, delta_x: float, delta_t: float, speed_of_sound: float, number_of_grid_points,
                 number_of_time_steps: int, initial_amplitude_function: callable,
                 initial_velocities_function: callable) -> None:
        """
        Initializer for an abstract wave simulator object.
        :param delta_x: Distance between two neighbouring grid points.
        :param delta_t: Time difference between two time steps in the simulation.
        :param speed_of_sound: Speed of sound of the medium in which the wave equation is solved.
        :param number_of_grid_points: Number of grid points used in the simulation
        :param number_of_time_steps: Number of time steps after which the simulation terminates.
        :param initial_amplitude_function:
        :param initial_velocities_function:
        """
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
        # Initial condition for the amplitudes.
        self.initial_amplitude_function = initial_amplitude_function
        # Initial condition for the amplitude's velocities.
        self.initial_velocities_function = initial_velocities_function
        # Defines the first position as the entered initial position.
        self.current_amplitudes = self.initial_amplitudes
        # There are no former amplitudes at t = 0.
        self.former_amplitudes = None
        # This array saves the time evolution of the amplitudes.
        self.amplitudes_time_evolution = None

    @classmethod
    def init_from_file(cls, link_to_file: str):
        """
        This method functions as a second constructor, which can be used to initialize a simulator via a file.
        :param link_to_file: This is the link to the file in which the data is stored. It should be a .npy file.
        :return: Returns a Numeric1DWaveSimulator object with the variables that are declared in
        the file.
        """
        if isinstance(link_to_file, str):
            # Load the data.
            loaded_data = np.load(link_to_file, allow_pickle=True)
            # Save the time evolution matrix for later use.
            temp = loaded_data[-1]
            # Delete the time evolution matrix since it is not needed in the initializer.
            del loaded_data[-1]
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
        :return: None.
        """
        # Check if new grid spacing is an int or float.
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
        # Check if the the new time delta is an int or float.
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
        # Check if speed of sound is an int or float.
        if isinstance(new_speed_of_sound, (int, float)):
            # Cast the new grid spacing as a float.
            self._speed_of_sound = float(new_speed_of_sound)
        else:
            raise TypeError("The speed of sound must be of type int or float.")

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
        :return: None
        """
        if isinstance(new_number_of_time_steps, int):
            if new_number_of_time_steps > 0:
                self._number_of_time_steps = new_number_of_time_steps
            else:
                raise ValueError("The number of time steps must be greater than zero.")
        else:
            raise TypeError("The number of time steps must be of type int.")

    def save_data(self, link_to_file: str = None) -> None:
        """
        This method saves the current attributes of a simulator object in a npy-file. If no link is provided in the form
        of a string, the method will create a file using the current date and time.
        :param link_to_file: Optional variable which is a link to a npy-file, where the data is saved.
        :return: None
        """
        # Save the UTC time.
        utc_time = dt.datetime.utcnow().replace(microsecond=0)
        # Save the important values of the simulator in numpy array.
        obj_to_save = np.array([self.delta_x,
                                self.delta_t,
                                self.speed_of_sound,
                                self.number_of_grid_points,
                                self.number_of_time_steps,
                                self.initial_amplitude_function,
                                self.initial_velocities_function,
                                self.amplitudes_time_evolution], dtype=object)
        # Check if a link was provided
        if link_to_file is not None:
            # Check if the provided link is a string
            if isinstance(link_to_file, str):
                # Save the data in the npy format.
                np.save(link_to_file, obj_to_save, allow_pickle=True)
            else:
                raise ValueError("The provided link must be a string.")
        else:
            # If no link was provided save the file with the following name
            file_name = ("wave_sim_1D" + str(utc_time) + ".npy").replace(" ", "_")
            np.save(file_name, obj_to_save, allow_pickle=True)

    @property
    @abstractmethod
    def number_of_grid_points(self):
        """
        Getter method for the number of grid points in the simulation. Returns the number of grid points used in the
        simulation.
        :return: Number of grid points used in the 1D simulation.
        """
        pass

    @number_of_grid_points.setter
    @abstractmethod
    def number_of_grid_points(self, new_number_of_grid_points):
        """
        Setter method for the number of grid points used in the 1D simulation. This number must be of type int and
        greater than zero. If this is not the case this method will raise an error.
        :param new_number_of_grid_points: Number of grid points in the simulation.
        :return: None
        """
        pass

    @property
    @abstractmethod
    def initial_amplitude_function(self) -> callable:
        """
        Getter method for the initial condition of the amplitude of the wave.
        :return: The initial condition for the amplitude.
        """
        pass

    @initial_amplitude_function.setter
    @abstractmethod
    def initial_amplitude_function(self, new_initial_amplitude_function: callable) -> None:
        """
        Setter method for the initial condition of the amplitude of the wave.
        :return: None
        """
        pass

    @property
    @abstractmethod
    def initial_velocities_function(self) -> callable:
        """
        Getter method for the initial condition of the amplitude of the wave.
        :return: The function describing the velocity of the amplitude of the wave.
        """
        pass

    @initial_velocities_function.setter
    @abstractmethod
    def initial_velocities_function(self, new_initial_velocities_function: callable) -> None:
        """
        Setter method for the initial condition of the amplitude of the wave.
        :return:
        """
        pass

    @property
    @abstractmethod
    def initial_amplitudes(self) -> np.ndarray:
        """
        Getter method for the initial amplitudes of the grid points at t = 0.
        :return: The initial positions.
        """
        pass

    @initial_amplitudes.setter
    @abstractmethod
    def initial_amplitudes(self, new_initial_amplitudes: np.ndarray) -> None:
        """
        The setter method for the initial positions at t = 0. The new initial positions must be a numpy array and its
        length has to coincide with the number of grid points. Further the boundary condition has to be fulfilled.
        The methods tests if these conditions are met and raises according errors.
        :param new_initial_amplitudes: New initial amplitudes for the initial condition.
        :return: None
        """
        pass

    @property
    @abstractmethod
    def initial_velocities(self) -> np.ndarray:
        """
        Getter method for the initial positions of the grid points at t = 0.
        :return: The initial positions.
        """
        pass

    @initial_velocities.setter
    @abstractmethod
    def initial_velocities(self, new_initial_velocities: np.ndarray) -> None:
        """
        The setter method for the initial velocities at t = 0. The new initial velocities must be a numpy array and its
        length has to coincide with the number of grid points. The methods tests if these conditions are met and raises
        according errors.
        :param new_initial_velocities: New initial velocities.
        :return: None
        """
        pass

    @abstractmethod
    def stability_test(self) -> None:
        """
        Checks if the entered values of the grid spacing, time spacing and speed of sound is between 0  and 1. If this
        is not the case a warning message will be displayed.
        :return: None
        """
        pass

    @abstractmethod
    def create_time_step_matrix(self, dim: int) -> np.ndarray:
        """
        Returns the matrix connecting the time steps. This matrix is a quadratic matrix with dimension
        dim x dim. The Matrix has only zeros in the first and last row and only diagonal and off diagonals are
        populated.
        :return: The matrix used to calculate the next time step.
        """
        pass

    @abstractmethod
    def update_first_time(self) -> np.ndarray:
        """
        The first time step is given by the initial conditions and is given by this method.
        :return: Returns the current amplitudes after the first time step.
        """
        pass

    @abstractmethod
    def update(self) -> np.ndarray:
        """
        Updates the amplitudes to the next time step and sets the current and former state accordingly. The counter for
        the time steps is then increased by one.
        :return: Current amplitudes.
        """
        pass

    @abstractmethod
    def run(self) -> List[np.ndarray]:
        """
        Checks the stability of the scheme and then runs the simulation using the according formula. This is done
        updating the simulation until the number of time steps is reached at which point the method will return the
        result of the simulation.
        :return: The result of the simulation. Each row in the list corresponds to the amplitudes in one time step.
        """
        pass


# ---------------------------
# 1D wave equation simulation
# ---------------------------


class Numeric1DWaveSimulator(NumericWaveSimulator):

    def __init__(self, delta_x: float, delta_t: float, speed_of_sound: float, number_of_grid_points: int,
                 number_of_time_steps: int, initial_amplitude_function: callable,
                 initial_velocities_function: callable) -> None:
        """
        Initializer for the 1D wave equation simulator. After the variables are passed the initializer calculates the
        courant number and the sets the current amplitudes to the initial amplitudes. Further, the time step matrix
        to transition from one time step to another is calculated. Lastly, the time evolution matrix, which saves the
        amplitudes for all the time steps, is initialized.
        :param delta_x: Distance between two neighbouring grid points.
        :param delta_t: Time difference between two time steps in the simulation.
        :param speed_of_sound: Speed of sound of the medium in which the wave equation is solved.
        :param number_of_grid_points: Number of grid points used in the simulation
        :param number_of_time_steps: Number of time steps after which the simulation terminates.
        :param initial_amplitude_function: Amplitudes corresponding to the first initial condition.
        :param initial_velocities_function: Velocities of the amplitudes corresponding to the second initial condition.
        """
        super().__init__(delta_x, delta_t, speed_of_sound, number_of_grid_points, number_of_time_steps,
                         initial_amplitude_function, initial_velocities_function)
        # Courant number of the problem.
        self.courant_number = float((self.delta_t * self.speed_of_sound / self.delta_x) ** 2)
        # Creates the time step matrix.
        self.time_step_matrix = self.create_time_step_matrix(self.number_of_grid_points)

    def calculate_grid_coordinates(self) -> np.ndarray:
        """
        This method returns the grid coordinates in x- and y-direction for the grid represented by the distance dx and
        the dimension given by the number_of_grid_points attribute.
        :return: A tuple of numpy arrays containing the grid coordinates in x- and y-direction.
        """
        return np.arange(0., self.number_of_grid_points * self.delta_x, self.delta_x)

    @property
    def number_of_grid_points(self) -> int:
        return self._number_of_grid_points

    @number_of_grid_points.setter
    def number_of_grid_points(self, new_number_of_grid_points: int) -> None:
        if isinstance(new_number_of_grid_points, int):
            if new_number_of_grid_points > 0:
                self._number_of_grid_points = new_number_of_grid_points
            else:
                raise ValueError("The number of grid point must be greater than zero.")
        else:
            raise TypeError("The number of grid points must be of type int or tuple.")

    @property
    def initial_amplitude_function(self) -> callable:
        return self._initial_amplitude_function

    @initial_amplitude_function.setter
    def initial_amplitude_function(self, new_initial_amplitude_function: callable) -> None:
        if callable(new_initial_amplitude_function):
            self._initial_amplitude_function = new_initial_amplitude_function
            self.initial_amplitudes = new_initial_amplitude_function(self.calculate_grid_coordinates())
        else:
            raise TypeError("The new function has to be a callable.")

    @property
    def initial_velocities_function(self) -> callable:
        return self._initial_velocities_function

    @initial_velocities_function.setter
    def initial_velocities_function(self, new_initial_velocities_function: callable) -> None:
        if callable(new_initial_velocities_function):
            self._initial_velocities_function = new_initial_velocities_function
            self.initial_velocities = new_initial_velocities_function(self.calculate_grid_coordinates())
        else:
            raise TypeError("The new function has to be a callable.")

    @property
    def initial_amplitudes(self) -> np.ndarray:
        return self._initial_amplitudes

    @initial_amplitudes.setter
    def initial_amplitudes(self, new_initial_amplitudes: np.ndarray) -> None:
        # Threshold value under which a variable will be taken as zero.
        threshold_value = 10 ** (-10)
        # Check if the initial amplitude is a numpy array.
        if isinstance(new_initial_amplitudes, np.ndarray):
            # Check if the length of the initial conditions coincides with the number of grid points.
            if len(new_initial_amplitudes) == self.number_of_grid_points:
                # Check if the initial amplitudes respect the boundary conditions.
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
        return self._initial_velocities

    @initial_velocities.setter
    def initial_velocities(self, new_initial_velocities: np.ndarray) -> None:
        # Threshold value under which a variable will be taken as zero
        threshold_value = 10 ** (-10)
        # Check if the initial velocity is a numpy array.
        if isinstance(new_initial_velocities, np.ndarray):
            # Check initial velocities and the number of grid points coincide.
            if len(new_initial_velocities) == self.number_of_grid_points:
                # Check if the initial velocity fort the first and last grid point are zero.
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
        if not 0 <= self.courant_number <= 1:
            print("The scheme may be unstable since the Courant number is ", self.courant_number, ". It should be "
                                                                                                  "between 0  and 1.")
        else:
            print("Scheme is stable. The Courant number is", str(self.courant_number) + ".")

    def create_time_step_matrix(self, dim: int) -> np.ndarray:
        # Define a temporary matrix to fill the off diagonals.
        temp = np.zeros((dim, dim))
        rearrange_array = np.arange(dim - 1)
        temp[rearrange_array, rearrange_array + 1] = 1
        temp = 2 * (1 - self.courant_number) * np.identity(dim) + self.courant_number * temp + \
               self.courant_number * temp.T
        # Set these elements to zero, so that the boundary conditions are fulfilled.
        temp[0, 0] = 0
        temp[0, 1] = 0
        temp[1, 0] = 0
        temp[dim - 2, dim - 1] = 0
        temp[dim - 1, dim - 1] = 0
        temp[dim - 1, dim - 2] = 0
        return temp

    def update_first_time(self) -> np.ndarray:
        # The first is given by this equation.
        self.current_amplitudes = 0.5 * np.dot(self.time_step_matrix, self.initial_amplitudes) + self.delta_t * \
                                  self.initial_velocities
        self.former_amplitudes = self.initial_amplitudes
        return self.current_amplitudes

    def update(self) -> np.ndarray:
        temp = self.current_amplitudes
        # Save the next time step as a temporary value. The next time step is calculated via a linear equation.
        self.current_amplitudes = np.dot(self.time_step_matrix, self.current_amplitudes) - self.former_amplitudes
        # Set the former and current amplitude accordingly.
        self.former_amplitudes = temp
        return self.current_amplitudes

    def run(self) -> List[np.ndarray]:
        # Check if the length of the initial amplitudes and initial velocities coincide with the number grid points.
        if self.number_of_grid_points != len(self.initial_amplitudes):
            self.initial_amplitudes = self.initial_amplitude_function(self.number_of_grid_points)
        elif self.number_of_grid_points != len(self.initial_velocities):
            self.initial_velocities = self.initial_velocities_function(self.number_of_grid_points)
        # Test if the scheme is stable.
        self.stability_test()
        # Use list comprehension to construct the time_evolution of the amplitudes.
        self.amplitudes_time_evolution = [self.update_first_time()] + [self.update() for _ in
                                                                       range(self.number_of_time_steps - 1)]
        return self.amplitudes_time_evolution


# ---------------------------
# 2D wave equation simulation
# ---------------------------


class Numeric2DWaveSimulator(NumericWaveSimulator):
    allowed_boundary_conditions = {"cyclical", "fixed edges", "loose edges"}

    constructor_call_flag = True

    def __init__(self, delta_x: float, delta_t: float, speed_of_sound: float, number_of_grid_points: tuple,
                 number_of_time_steps: int, initial_amplitude_function: callable, initial_velocities_function: callable,
                 boundary_condition: str) -> None:
        """
        Initializer for the 2D wave equation simulator. After the variables are passed the initializer calculates the
        courant number and the initializer sets the current amplitudes to the initial amplitudes. Further, the time
        step matrix is calculated. Lastly, the time evolution matrix, which saves the amplitudes for all the time steps,
        is initialized.
        :param delta_x: Distance between two neighbouring grid points.
        :param delta_t: Time difference between two time steps in the simulation.
        :param speed_of_sound: Speed of sound of the medium in which the wave equation is solved.
        :param number_of_grid_points: Number of grid points used in the simulation
        :param number_of_time_steps: Number of time steps after which the simulation terminates.
        :param initial_amplitude_function: Amplitudes corresponding to the first initial condition.
        :param initial_velocities_function: Velocities of the amplitudes corresponding to the second initial condition.
        :param boundary_condition: Boundary condition for the wave simulation. It can be cyclical, fixed edges or
        loose edges.
        """
        # Call the initializer of the parent class
        super().__init__(delta_x, delta_t, speed_of_sound, number_of_grid_points, number_of_time_steps,
                         initial_amplitude_function, initial_velocities_function)
        # Courant number of the problem.
        self.courant_number = float((self.delta_t * self.speed_of_sound / self.delta_x) ** 2)
        # Set the boundary condition.
        # A boundary condition, should be one of the the options in the allowed_boundary_conditions attribute.
        self.boundary_condition = boundary_condition
        # Creates the time step matrix which is multiplied by the state matrix on the left.
        self.time_step_matrix_left = self.create_time_step_matrix(self.number_of_grid_points[0])
        self.time_step_matrix_right = self.create_time_step_matrix(self.number_of_grid_points[1])
        # Set the function for the first initial condition
        self.initial_amplitude_function = initial_amplitude_function
        # Set the second initial condition.
        self.initial_velocities_function = initial_velocities_function

        self.constructor_call_flag = False

    def calculate_grid_coordinates(self) -> tuple:
        """
        This method returns the grid coordinates in x- and y-direction for the grid represented by the distance dx and
        the dimension given by the number_of_grid_points attribute.
        :return: A tuple of numpy arrays containing the grid coordinates in x- and y-direction.
        """
        x_dim = self.number_of_grid_points[0]
        y_dim = self.number_of_grid_points[1]
        x_coord = np.arange(0., x_dim * self.delta_x, self.delta_x)
        y_coord = np.arange(0., y_dim * self.delta_x, self.delta_x)
        x_mat, y_mat = np.meshgrid(x_coord, y_coord, sparse=True)
        return x_mat, y_mat

    @property
    def delta_x(self):
        return self._delta_x

    @delta_x.setter
    def delta_x(self, new_delta_x):
        # Check if new grid spacing is an int or float.
        if isinstance(new_delta_x, (float, int)):
            if new_delta_x > 0:
                if self.constructor_call_flag:
                    # Cast the new grid spacing as a float.
                    self._delta_x = float(new_delta_x)
                else:
                    self._delta_x = float(new_delta_x)
                    self.courant_number = (self.delta_t * self.speed_of_sound / self.delta_x) ** 2
                    self.time_step_matrix_left = self.create_time_step_matrix(self.number_of_grid_points[0])
                    self.time_step_matrix_right = self.create_time_step_matrix(self.number_of_grid_points[1])
            else:
                raise ValueError("The distance between the grid points must be greater than zero.")
        else:
            raise TypeError("delta_x must be of type float or int.")

    @property
    def delta_t(self):
        return self._delta_t

    @delta_t.setter
    def delta_t(self, new_delta_t):
        # Check if new grid spacing is an int or float.
        if isinstance(new_delta_t, (float, int)):
            if new_delta_t > 0:
                if self.constructor_call_flag:
                    # Cast the new grid spacing as a float.
                    self._delta_t = float(new_delta_t)
                else:
                    self._delta_t = float(new_delta_t)
                    self.courant_number = (self.delta_t * self.speed_of_sound / self.delta_x) ** 2
                    self.time_step_matrix_left = self.create_time_step_matrix(self.number_of_grid_points[0])
                    self.time_step_matrix_right = self.create_time_step_matrix(self.number_of_grid_points[1])
            else:
                raise ValueError("The distance between the grid points must be greater than zero.")
        else:
            raise TypeError("delta_x must be of type float or int.")

    @property
    def speed_of_sound(self):
        return self._speed_of_sound

    @speed_of_sound.setter
    def speed_of_sound(self, new_speed_of_sound):
        # Check if new grid spacing is an int or float.
        if isinstance(new_speed_of_sound, (float, int)):
            if self.constructor_call_flag:
                # Cast the new grid spacing as a float.
                self._speed_of_sound = float(new_speed_of_sound)
            else:
                self._speed_of_sound = float(new_speed_of_sound)
                self.courant_number = (self.delta_t * self.speed_of_sound / self.delta_x) ** 2
                self.time_step_matrix_left = self.create_time_step_matrix(self.number_of_grid_points[0])
                self.time_step_matrix_right = self.create_time_step_matrix(self.number_of_grid_points[1])
        else:
            raise TypeError("delta_x must be of type float or int.")

    @property
    def boundary_condition(self) -> str:
        return self._boundary_condition

    @boundary_condition.setter
    def boundary_condition(self, new_boundary_condition: str) -> None:
        """
        Setter method for the boundary condition. The boundary condition can either be cyclical, fixed edges or loose
        edges.
        :param new_boundary_condition: New boundary condition.
        :return: None
        """
        if isinstance(new_boundary_condition, str):
            if new_boundary_condition in self.allowed_boundary_conditions:
                if self.constructor_call_flag:
                    self._boundary_condition = new_boundary_condition
                else:
                    self._boundary_condition = new_boundary_condition
                    self.time_step_matrix_left = self.create_time_step_matrix(self.number_of_grid_points[0])
                    self.time_step_matrix_left = self.create_time_step_matrix(self.number_of_grid_points[1])
            else:
                raise ValueError("The boundary condition has to be: cyclical, fixed edges or loose edges.")
        else:
            raise TypeError("The boundary condition has to be a string.")

    @property
    def initial_amplitude_function(self) -> callable:
        return self._initial_amplitude_function

    @initial_amplitude_function.setter
    def initial_amplitude_function(self, new_initial_amplitude_function: callable) -> None:
        if callable(new_initial_amplitude_function):
            self._initial_amplitude_function = new_initial_amplitude_function
            x_mat, y_mat = self.calculate_grid_coordinates()
            self.initial_amplitudes = new_initial_amplitude_function(x_mat, y_mat)
        else:
            raise TypeError("The new function has to be a callable.")

    @property
    def initial_velocities_function(self) -> callable:
        return self._initial_velocities_function

    @initial_velocities_function.setter
    def initial_velocities_function(self, new_initial_velocities_function: callable) -> None:
        if callable(new_initial_velocities_function):
            self._initial_velocities_function = new_initial_velocities_function
            x_mat, y_mat = self.calculate_grid_coordinates()
            self.initial_velocities = self._initial_velocities_function(x_mat, y_mat)
        else:
            raise TypeError("The new function has to be a callable.")

    @property
    def number_of_grid_points(self) -> tuple:
        return self._number_of_grid_points

    @number_of_grid_points.setter
    def number_of_grid_points(self, new_number_of_grid_points: tuple) -> None:
        if isinstance(new_number_of_grid_points, tuple):
            if len(new_number_of_grid_points) == 2:
                if all(isinstance(n, int) for n in new_number_of_grid_points):
                    if self.constructor_call_flag:
                        self._number_of_grid_points = new_number_of_grid_points
                    else:
                        self._number_of_grid_points = new_number_of_grid_points
                        # Creates the time step matrix which is multiplied by the state matrix on the left.
                        self.time_step_matrix_left = self.create_time_step_matrix(new_number_of_grid_points[0])
                        # Creates the time step matrix which is multiplied by the state matrix on the right.
                        self.time_step_matrix_right = self.create_time_step_matrix(new_number_of_grid_points[1])
                        # Set the initial conditions.
                        x_mat, y_mat = self.calculate_grid_coordinates()
                        self.initial_amplitudes = self.initial_amplitude_function(x_mat, y_mat)
                        self.initial_velocities = self.initial_velocities_function(x_mat, y_mat)
                else:
                    raise ValueError("The number of grid point must be greater than zero.")
            else:
                raise ValueError("The tuple representing the number of grid points must be of length two.")
        else:
            raise TypeError("The number of grid points must be of type int or tuple.")

    @property
    def initial_amplitudes(self) -> np.ndarray:
        return self._initial_amplitudes

    @initial_amplitudes.setter
    def initial_amplitudes(self, new_initial_amplitudes: np.ndarray) -> None:
        # Check if the initial amplitude is a numpy array.
        if isinstance(new_initial_amplitudes, np.ndarray):
            # Check if the length of the initial conditions coincides with the number of grid points.
            if new_initial_amplitudes.shape == self.number_of_grid_points:
                if new_initial_amplitudes.dtype == "float64" or new_initial_amplitudes.dtype == "int64":
                    self._initial_amplitudes = new_initial_amplitudes
                else:
                    raise TypeError("The numpy array containing the initial velocities must have the type float64 or"
                                    "int64.")
            else:
                raise ValueError("The number of grid points and the length of the initial amplitudes must "
                                 "coincide.")
        else:
            raise TypeError("The initial amplitudes must be a numpy array.")

    @property
    def initial_velocities(self) -> np.ndarray:
        return self._initial_velocities

    @initial_velocities.setter
    def initial_velocities(self, new_initial_velocities: np.ndarray) -> None:
        # Check if the initial velocity is a numpy array.
        if isinstance(new_initial_velocities, np.ndarray):
            # Check initial velocities and the number of grid points coincide.
            if new_initial_velocities.shape == self.number_of_grid_points:
                # Check the data type of the numpy array initial velocities.
                if new_initial_velocities.dtype == "float64" or new_initial_velocities.dtype == "int64":
                    self._initial_velocities = new_initial_velocities
                else:
                    raise TypeError("The numpy array containing the initial velocities must have the type float64 or"
                                    "int64.")
            else:
                raise ValueError("The number of grid points and the length of the new initial velocities must coincide."
                                 )
        else:
            raise TypeError("The new initial velocities must be a numpy array.")

    # todo: Look up how to implement this in a 2D scheme.
    def stability_test(self) -> None:
        if not 0 <= self.courant_number <= 1:
            print("The scheme may be unstable since the Courant number is ", self.courant_number, ". It should be "
                                                                                                  "between 0  and 1.")
        else:
            print("Scheme is stable. The Courant number is", str(self.courant_number) + ".")

    def create_time_step_matrix(self, dim: int) -> np.ndarray:
        # Define a temporary matrix to fill the off diagonals.
        temp = np.zeros((dim, dim))
        rearrange_array = np.arange(dim - 1)
        temp[rearrange_array, rearrange_array + 1] = 1
        temp = (1 - 2 * self.courant_number) * np.identity(dim) + self.courant_number * temp \
               + self.courant_number * temp.T
        if self.boundary_condition == "cyclical":
            # Set the off diagonal edges to fulfill he cyclical boundary conditions.
            temp[dim - 1, 0] = self.courant_number
            temp[0, dim - 1] = self.courant_number
        elif self.boundary_condition == "fixed edges":
            # Set these elements to zero, so that the boundary conditions are fulfilled.
            temp[0, 0] = 0.
            temp[0, 1] = 0.
            temp[1, 0] = 0.
            temp[dim - 2, dim - 1] = 0.
            temp[dim - 1, dim - 1] = 0.
            temp[dim - 1, dim - 2] = 0.
        # Boundary condition of fixed edges. The matrix remains the same.
        else:
            pass
        # Return a sparse matrix.
        return temp

    def update_first_time(self) -> np.ndarray:
        # Save the initial amplitudes as the former amplitudes.
        self.former_amplitudes = self.current_amplitudes
        # The first is given by this equation.
        self.current_amplitudes = 0.5 * (np.dot(self.time_step_matrix_left, self.current_amplitudes) +
                                         np.dot(self.current_amplitudes, self.time_step_matrix_right)) + \
                                  self.delta_t * self.initial_velocities
        return self.current_amplitudes

    def update(self) -> np.ndarray:
        temp = self.current_amplitudes
        self.current_amplitudes = np.dot(self.time_step_matrix_left, self.current_amplitudes) + \
                                  np.dot(self.current_amplitudes, self.time_step_matrix_right) - self.former_amplitudes
        self.former_amplitudes = temp
        return self.current_amplitudes

    def run(self) -> List[np.ndarray]:
        self.stability_test()
        self.amplitudes_time_evolution = [self.update_first_time()] + \
                                         [self.update() for _ in range(self.number_of_time_steps - 1)]
        return self.amplitudes_time_evolution
