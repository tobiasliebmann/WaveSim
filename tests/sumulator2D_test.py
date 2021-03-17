import unittest as ut

import numpy as np
import simulator as sim


class TestCase2DSim(ut.TestCase):
    """
    This test class will test all the methods which are exclusively necessary for the 2D wave equation simulator.
    All the methods which are inherited and already tested by the 1D wave equation simulator will not be tested.
    The tests include the methods:
    - Setter for boundary conditions.
    - Setter for the number of grid points.
    - Setter for initial amplitude.
    - Setter for initial velocity.
    - Creating the time step matrix.
    - Updating the simulation.
    """

    # Initial amplitudes
    init_amps = np.full((10, 10), 1.)

    # Initial velocities.
    init_vel = np.zeros((10, 10))

    my_sim = sim.Numeric2DWaveSimulator(1., 1., 0.5, (10, 10), 10, init_amps, init_vel, "cyclical")

    def test_boundary_condition(self):
        """
        Tests if the the setter method for the boundary condition of the 2D wave simulator correctly raises errors and
        if the getter function returns the correct values.
        :return: None.
        """
        self.assertEqual(self.my_sim.boundary_condition, "cyclical")

        with self.assertRaises(TypeError):
            self.my_sim.boundary_condition = 5.

        with self.assertRaises(ValueError):
            self.my_sim.boundary_condition = "Hello"

    def test_number_of_grid_points(self):
        """
        Tests the getter and setter methods for the number of grid points attribute. It is tested if the correct errors
        are raised if the wrong type is entered and if the getter returns the correct value.
        :return: None.
        """
        self.assertEqual((10, 10), self.my_sim.number_of_grid_points)

        with self.assertRaises(TypeError):
            self.my_sim.number_of_grid_points = 9.

        with self.assertRaises(ValueError):
            self.my_sim.number_of_grid_points = (10, 10, 10)

        with self.assertRaises(ValueError):
            self.my_sim.number_of_grid_points = ("Hello", 1.)

    def test_initial_amplitudes(self):
        """

        :return: None.
        """
        np.testing.assert_almost_equal(self.init_amps, self.my_sim.initial_amplitudes)

        with self.assertRaises(TypeError):
            self.my_sim.initial_amplitudes = "String"

        with self.assertRaises(ValueError):
            self.my_sim.initial_amplitudes = np.array([[3., 4.], [2., 3.]])

        with self.assertRaises(TypeError):
            self.my_sim.initial_amplitudes = np.full((10, 10), "String")

    def test_initial_velocities(self):
        """

        :return: None.
        """
        np.testing.assert_almost_equal(self.init_vel, self.my_sim.initial_velocities)

        with self.assertRaises(TypeError):
            self.my_sim.initial_velocities = "String"

        with self.assertRaises(ValueError):
            self.my_sim.initial_velocities = np.array([[3., 4.], [2., 3.]])

        with self.assertRaises(TypeError):
            self.my_sim.initial_velocities = np.full((10, 10), "Hello")


if __name__ == '__main__':
    ut.main()
