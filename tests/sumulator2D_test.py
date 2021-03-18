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

    def setUp(self) -> None:
        """
        Is evaluated before each test function call.
        :return: None
        """
        self.dim = (8, 10)

        # Initial amplitudes
        self.init_amps = np.full(self.dim, 1.)

        # Initial velocities.
        self.init_vel = np.zeros(self.dim)

        self.my_sim = sim.Numeric2DWaveSimulator(1., 1., 0.5, self.dim, 10, self.init_amps, self.init_vel, "loose edges"
                                                 )

    def tearDown(self) -> None:
        """
        is evaluated after each test function call.
        :return: None
        """
        pass

    def test_courant_number(self):
        """

        :return:
        """
        np.testing.assert_almost_equal(self.my_sim.courant_number, 0.25)

    def test_boundary_condition(self):
        """
        Tests if the the setter method for the boundary condition of the 2D wave simulator correctly raises errors and
        if the getter function returns the correct values.
        :return: None.
        """
        self.assertEqual(self.my_sim.boundary_condition, "loose edges")

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
        np.testing.assert_almost_equal(self.dim, self.my_sim.number_of_grid_points)

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
            self.my_sim.initial_amplitudes = np.full(self.dim, "String")

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
            self.my_sim.initial_velocities = np.full(self.dim, "Hello")

    def test_create_time_step_matrix(self):
        """
        Tests if the time step matrices are created correctly.
        :return:
        """
        # Tests if the left matrix is implemented correctly for the boundary condition of loose edges.
        np.testing.assert_almost_equal(self.my_sim.time_step_matrix_left,
                                       np.array([[0.5, 0.25, 0., 0., 0., 0., 0., 0.],
                                                 [0.25, 0.5, 0.25, 0., 0., 0., 0., 0.],
                                                 [0., 0.25, 0.5, 0.25, 0., 0., 0., 0.],
                                                 [0., 0., 0.25, 0.5, 0.25, 0., 0., 0.],
                                                 [0., 0., 0., 0.25, 0.5, 0.25, 0., 0.],
                                                 [0., 0., 0., 0., 0.25, 0.5, 0.25, 0.],
                                                 [0., 0., 0., 0., 0., 0.25, 0.5, 0.25],
                                                 [0., 0., 0., 0., 0., 0., 0.25, 0.5]]))

        # Tests if the right matrix was created correctly for the boundary condition of loose edges.
        np.testing.assert_almost_equal(self.my_sim.time_step_matrix_right,
                                       np.array([[0.5, 0.25, 0., 0., 0., 0., 0., 0., 0., 0.],
                                                 [0.25, 0.5, 0.25, 0., 0., 0., 0., 0., 0., 0.],
                                                 [0., 0.25, 0.5, 0.25, 0., 0., 0., 0., 0., 0.],
                                                 [0., 0., 0.25, 0.5, 0.25, 0., 0., 0., 0., 0.],
                                                 [0., 0., 0., 0.25, 0.5, 0.25, 0., 0., 0., 0.],
                                                 [0., 0., 0., 0., 0.25, 0.5, 0.25, 0., 0., 0.],
                                                 [0., 0., 0., 0., 0., 0.25, 0.5, 0.25, 0., 0.],
                                                 [0., 0., 0., 0., 0., 0., 0.25, 0.5, 0.25, 0.],
                                                 [0., 0., 0., 0., 0., 0., 0., 0.25, 0.5, 0.25],
                                                 [0., 0., 0., 0., 0., 0., 0., 0., 0.25, 0.5]]))

        # Define a new simulator with cyclical boundary conditions.
        self.my_sim = sim.Numeric2DWaveSimulator(1., 1., 0.5, self.dim, 10, self.init_amps, self.init_vel, "cyclical")

        # Tests if the left matrix is implemented correctly for a cyclical boundary condition.
        np.testing.assert_almost_equal(self.my_sim.time_step_matrix_left,
                                       np.array([[0.5, 0.25, 0., 0., 0., 0., 0., 0.25],
                                                 [0.25, 0.5, 0.25, 0., 0., 0., 0., 0.],
                                                 [0., 0.25, 0.5, 0.25, 0., 0., 0., 0.],
                                                 [0., 0., 0.25, 0.5, 0.25, 0., 0., 0.],
                                                 [0., 0., 0., 0.25, 0.5, 0.25, 0., 0.],
                                                 [0., 0., 0., 0., 0.25, 0.5, 0.25, 0.],
                                                 [0., 0., 0., 0., 0., 0.25, 0.5, 0.25],
                                                 [0.25, 0., 0., 0., 0., 0., 0.25, 0.5]]))

        # Tests if the right matrix was created correctly for a cyclical boundary condition.
        np.testing.assert_almost_equal(self.my_sim.time_step_matrix_right,
                                       np.array([[0.5, 0.25, 0., 0., 0., 0., 0., 0., 0., 0.25],
                                                 [0.25, 0.5, 0.25, 0., 0., 0., 0., 0., 0., 0.],
                                                 [0., 0.25, 0.5, 0.25, 0., 0., 0., 0., 0., 0.],
                                                 [0., 0., 0.25, 0.5, 0.25, 0., 0., 0., 0., 0.],
                                                 [0., 0., 0., 0.25, 0.5, 0.25, 0., 0., 0., 0.],
                                                 [0., 0., 0., 0., 0.25, 0.5, 0.25, 0., 0., 0.],
                                                 [0., 0., 0., 0., 0., 0.25, 0.5, 0.25, 0., 0.],
                                                 [0., 0., 0., 0., 0., 0., 0.25, 0.5, 0.25, 0.],
                                                 [0., 0., 0., 0., 0., 0., 0., 0.25, 0.5, 0.25],
                                                 [0.25, 0., 0., 0., 0., 0., 0., 0., 0.25, 0.5]]))

        # Create a new simulator object with fixed edges as boundary condition.
        self.my_sim = sim.Numeric2DWaveSimulator(1., 1., 0.5, self.dim, 10, self.init_amps, self.init_vel, "fixed edges"
                                                 )
        # Tests if the left matrix is implemented correctly with the boundary condition of fixed edges.
        np.testing.assert_almost_equal(self.my_sim.time_step_matrix_left,
                                       np.array([[0., 0., 0., 0., 0., 0., 0., 0.],
                                                 [0., 0.5, 0.25, 0., 0., 0., 0., 0.],
                                                 [0., 0.25, 0.5, 0.25, 0., 0., 0., 0.],
                                                 [0., 0., 0.25, 0.5, 0.25, 0., 0., 0.],
                                                 [0., 0., 0., 0.25, 0.5, 0.25, 0., 0.],
                                                 [0., 0., 0., 0., 0.25, 0.5, 0.25, 0.],
                                                 [0., 0., 0., 0., 0., 0.25, 0.5, 0.],
                                                 [0., 0., 0., 0., 0., 0., 0., 0.]]))

        # Tests if the right matrix was created correctly with the boundary condition of fixed edges.
        np.testing.assert_almost_equal(self.my_sim.time_step_matrix_right,
                                       np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                                 [0., 0.5, 0.25, 0., 0., 0., 0., 0., 0., 0.],
                                                 [0., 0.25, 0.5, 0.25, 0., 0., 0., 0., 0., 0.],
                                                 [0., 0., 0.25, 0.5, 0.25, 0., 0., 0., 0., 0.],
                                                 [0., 0., 0., 0.25, 0.5, 0.25, 0., 0., 0., 0.],
                                                 [0., 0., 0., 0., 0.25, 0.5, 0.25, 0., 0., 0.],
                                                 [0., 0., 0., 0., 0., 0.25, 0.5, 0.25, 0., 0.],
                                                 [0., 0., 0., 0., 0., 0., 0.25, 0.5, 0.25, 0.],
                                                 [0., 0., 0., 0., 0., 0., 0., 0.25, 0.5, 0.],
                                                 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]))

    def test_update(self):
        """

        :return:
        """
        pass


if __name__ == '__main__':
    ut.main()
