import unittest as ut

import numpy as np

from .. import simulator1D as sim


class Sim1DTest(ut.TestCase):
    initial_positions = np.array([0, 0, 0, 0, 0, 0])

    initial_velocities = np.array([0, 1, 1, 1, 1, 0])

    # Initialize simulator
    my_sim = sim.Numeric1DWaveSimulator(1, 1, 0.5, 6, 6, initial_positions, initial_velocities)

    def test_create_time_step_matrix(self):
        """
        Tests if the time_step_matrix is created correctly. It should have a frame of zeros and only the diagonals and
        off-diagonals should be filled with numbers.
        :return: None
        """
        # Create matrix to compare our result to.
        compare_matrix = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                   [0., 0.6, 0.7, 0., 0., 0., 0., 0., 0., 0.],
                                   [0., 0.7, 0.6, 0.7, 0., 0., 0., 0., 0., 0.],
                                   [0., 0., 0.7, 0.6, 0.7, 0., 0., 0., 0., 0.],
                                   [0., 0., 0., 0.7, 0.6, 0.7, 0., 0., 0., 0.],
                                   [0., 0., 0., 0., 0.7, 0.6, 0.7, 0., 0., 0.],
                                   [0., 0., 0., 0., 0., 0.7, 0.6, 0.7, 0., 0.],
                                   [0., 0., 0., 0., 0., 0., 0.7, 0.6, 0.7, 0.],
                                   [0., 0., 0., 0., 0., 0., 0., 0.7, 0.6, 0.],
                                   [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
        # Construct time step matrix.
        constructed_matrix = self.my_sim.create_time_step_matrix(10, 0.7)
        # Use a numpy test here since I couldn't find a fitting test in the unittest package. The method
        # assert_almost_equal is used since we are dealing with floats.
        np.testing.assert_almost_equal(constructed_matrix, compare_matrix)

    def test_delta_x(self):
        """
        Tests if the setter function for delta_x raise the right errors when no int or floats are entered. Further, it
        is also tested if ints are correctly casted as floats by the setter.
        :return: None
        """
        # Tests if a TypeError is raised when wrong type is entered.
        with self.assertRaises(TypeError):
            self.my_sim.delta_x = "some string"

        # Tests if ValueError is  raised for invalid ints or floats.
        with self.assertRaises(ValueError):
            self.my_sim.delta_x = -1.5

        with self.assertRaises(ValueError):
            self.my_sim.delta_x = -1

        with self.assertRaises(ValueError):
            self.my_sim.delta_x = 0

        # Test if int is correctly casted as float.
        self.my_sim.delta_x = 1
        self.assertTrue(isinstance(self.my_sim.delta_x, float))

    def test_delta_t(self):
        """
        Tests if the setter function for delta_t raise the right errors when no int or floats are entered. Further, it
        is also tested if ints are correctly casted as floats by the setter.
        :return: None
        """
        # Tests if a TypeError is raised when wrong type is entered.
        with self.assertRaises(TypeError):
            self.my_sim.delta_t = "some string"

        # Tests if ValueError is  raised for invalid ints or floats.
        with self.assertRaises(ValueError):
            self.my_sim.delta_t = -1.5

        with self.assertRaises(ValueError):
            self.my_sim.delta_t = -1

        with self.assertRaises(ValueError):
            self.my_sim.delta_t = 0

        # Test if int is correctly casted as float.
        self.my_sim.delta_t = 1
        self.assertTrue(isinstance(self.my_sim.delta_t, float))

    def test_speed_of_sound(self):
        """
        Tests if the setter for the speed of sound raises a TypeError when something else than an int or float is
        entered. Further it is tested if the entered ints are correctly casted to floats.
        :return: None
        """
        with self.assertRaises(TypeError):
            self.my_sim.speed_of_sound = "Test string"

        self.speed_of_sound = 1
        self.assertTrue(isinstance(self.my_sim.speed_of_sound, float))

    def test_number_of_grid_points(self):
        """
        Tests if a a TypeError is raised if anything else than an int is entered as the number of grid points for the
        simulation. Further, it is tested if a ValueError is raised if an int <= 0 is entered.
        :return: None
        """
        # Test for TypeError if the entered variable is not an int.
        with self.assertRaises(TypeError):
            self.my_sim.number_of_grid_points = 1.5

        # Test for ValueError.
        with self.assertRaises(ValueError):
            self.my_sim.number_of_grid_points = 0

    def test_number_of_time_steps(self):
        """
        Tests if a a TypeError is raised if anything else than an int is entered as the number of time steps for the
        simulation. Further, it is tested if a ValueError is raised if an int <= 0 is entered.
        :return: None
        """
        # Test for TypeError if the entered variable is not an int.
        with self.assertRaises(TypeError):
            self.my_sim.number_of_time_steps = 1.5

        # Test for ValueError.
        with self.assertRaises(ValueError):
            self.my_sim.number_of_time_steps = 0

    def initial_amplitudes(self):
        """
        Test if an error is raised if the initial amplitudes do not have the right data type, length (which should
        coincide with the number of grid points) or fulfill the boundary conditions, which means that there start and
        end point must be equal to zero.
        :return: None
        """
        with self.assertRaises(TypeError):
            self.my_sim.initial_amplitudes = 1

        with self.assertRaises(TypeError):
            self.my_sim.initial_amplitudes = "Hello"

        with self.assertRaises(ValueError):
            # Set the number of grid points and the number of grid points in the initial condition to different values.
            self.my_sim.number_of_grid_points = 4
            self.my_sim.initial_amplitudes = np.array([1, 1, 1])

        with self.assertRaises(ValueError):
            # Test if an error is if the end points of the initial amplitudes are not zero.
            self.my_sim.number_of_grid_points = 5
            self.my_sim.initial_amplitudes = np.array([1, 1, 1, 1, 1])

    def test_initial_velocities(self):
        """
        Test if an error is raised if the initial velocities do not have the right data type, length (which should
        coincide with the number of grid points) or fulfill the boundary conditions, which means that there start and
        end point must be equal to zero.
        :return:
        """
        with self.assertRaises(TypeError):
            self.my_sim.initial_velocities = 1

        with self.assertRaises(TypeError):
            self.my_sim.initial_velocities = "Test string"

        with self.assertRaises(ValueError):
            # Set the number of grid points and the number of grid points in the initial condition to different values.
            self.my_sim.number_of_grid_points = 4
            self.my_sim.initial_velocities = np.array([1, 1, 1])

        with self.assertRaises(ValueError):
            # Test if an error is if the end points of the initial amplitudes are not zero.
            self.my_sim.number_of_grid_points = 5
            self.my_sim.initial_velocities = np.array([1, 1, 1, 1, 1])

    def test_update1(self):
        """
        Tests if the update method is correct by checking the latest entry in the attribute amplitudes_time_evolution
        after calling the update method. The simulator is reset and the update formula is called two times to test the
        correctness of the method for the first and any following time steps.
        :return: None
        """
        # Reset the simulator with new values.
        self.my_sim = sim.Numeric1DWaveSimulator(1., 1., 1., 5, 1, np.array([0, 0, 1, 0, 0]),
                                                 np.array([0, 0, 0, 0, 0]))
        # Update the simulator manually.
        self.my_sim.update()
        # Test if the update formula is correct by calling the current_amplitudes attribute of the simulator.
        np.testing.assert_almost_equal(self.my_sim.amplitudes_time_evolution[-1], np.array([0, 0.5, 0, 0.5, 0]))

        # Update the simulator manually.
        self.my_sim.update()
        # Test update formula.
        np.testing.assert_almost_equal(self.my_sim.amplitudes_time_evolution[-1], np.array([0, 0., 0., 0., 0.]))


if __name__ == "__main__":
    ut.main()
