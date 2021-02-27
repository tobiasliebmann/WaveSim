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

    


if __name__ == "__main__":
    ut.main()
