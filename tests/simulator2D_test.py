import unittest as ut
import numpy as np
import simulator as sim


# Initial amplitudes.
def init_amps_func(x, y):
    return 1. + 0. * x + 0. * y


# Initial velocity of the amplitudes.
def init_vel_func(x, y):
    return 0. * x + 0. * y


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
        # Dimension of the grid used in the simulation.
        self.dim = (8, 10)

        # Initialize the simulator object.
        self.my_sim = sim.Numeric2DWaveSimulator(1., 1., .5, self.dim, 10, init_amps_func, init_vel_func, "loose edges")

    def tearDown(self) -> None:
        """
        is evaluated after each test function call.

        :return: None
        """
        pass

    def test_delta_t(self):
        """
        Tests if the delta_t-setter raises errors correctly and if the time step matrices are changed correctly when
        delta_t is changed.

        :return: None.
        """
        with self.assertRaises(TypeError):
            self.my_sim.delta_t = "test string"

        with self.assertRaises(ValueError):
            self.my_sim.delta_t = -2.

        self.my_sim.delta_t = 2.

        np.testing.assert_almost_equal(self.my_sim.time_step_matrix_left, np.array([[-1., 1., 0., 0., 0., 0., 0., 0.],
                                                                                    [1, -1., 1, 0., 0., 0., 0.,
                                                                                     0.],
                                                                                    [0., 1., -1., 1., 0., 0., 0.,
                                                                                     0.],
                                                                                    [0., 0., 1., -1., 1., 0., 0.,
                                                                                     0.],
                                                                                    [0., 0., 0., 1., -1., 1., 0.,
                                                                                     0.],
                                                                                    [0., 0., 0., 0., 1., -1., 1.,
                                                                                     0.],
                                                                                    [0., 0., 0., 0., 0., 1., -1.,
                                                                                     1.],
                                                                                    [0., 0., 0., 0., 0., 0., 1.,
                                                                                     -1.]]))

        np.testing.assert_almost_equal(self.my_sim.time_step_matrix_right, np.array([[-1., 1., 0., 0., 0., 0., 0., 0.,
                                                                                      0., 0.],
                                                                                     [1, -1., 1, 0., 0., 0., 0., 0., 0.,
                                                                                      0.],
                                                                                     [0., 1., -1., 1., 0., 0., 0.,
                                                                                      0., 0., 0.],
                                                                                     [0., 0., 1., -1., 1., 0., 0.,
                                                                                      0., 0., 0.],
                                                                                     [0., 0., 0., 1., -1., 1., 0.,
                                                                                      0., 0., 0.],
                                                                                     [0., 0., 0., 0., 1., -1., 1.,
                                                                                      0., 0., 0.],
                                                                                     [0., 0., 0., 0., 0., 1., -1.,
                                                                                      1., 0., 0.],
                                                                                     [0., 0., 0., 0., 0., 0., 1.,
                                                                                      -1., 1., 0.],
                                                                                     [0., 0., 0., 0., 0., 0., 0.,
                                                                                      1., -1., 1.],
                                                                                     [0., 0., 0., 0., 0., 0., 0.,
                                                                                      0., 1., -1.]]))

    def test_delta_x(self):
        """
        Tests if the delta_x-setter raises errors correctly and if the time step matrices are changed correctly if
        delta_x is changed.

        :return: None.
        """
        with self.assertRaises(TypeError):
            self.my_sim.delta_x = "test string"

        with self.assertRaises(ValueError):
            self.my_sim.delta_x = -1.

        self.my_sim.delta_x = .5

        np.testing.assert_almost_equal(self.my_sim.time_step_matrix_left, np.array([[-1., 1., 0., 0., 0., 0., 0., 0.],
                                                                                    [1, -1., 1, 0., 0., 0., 0.,
                                                                                     0.],
                                                                                    [0., 1., -1., 1., 0., 0., 0.,
                                                                                     0.],
                                                                                    [0., 0., 1., -1., 1., 0., 0.,
                                                                                     0.],
                                                                                    [0., 0., 0., 1., -1., 1., 0.,
                                                                                     0.],
                                                                                    [0., 0., 0., 0., 1., -1., 1.,
                                                                                     0.],
                                                                                    [0., 0., 0., 0., 0., 1., -1.,
                                                                                     1.],
                                                                                    [0., 0., 0., 0., 0., 0., 1.,
                                                                                     -1.]]))

        np.testing.assert_almost_equal(self.my_sim.time_step_matrix_right, np.array([[-1., 1., 0., 0., 0., 0., 0., 0.,
                                                                                      0., 0.],
                                                                                     [1, -1., 1, 0., 0., 0., 0., 0., 0.,
                                                                                      0.],
                                                                                     [0., 1., -1., 1., 0., 0., 0.,
                                                                                      0., 0., 0.],
                                                                                     [0., 0., 1., -1., 1., 0., 0.,
                                                                                      0., 0., 0.],
                                                                                     [0., 0., 0., 1., -1., 1., 0.,
                                                                                      0., 0., 0.],
                                                                                     [0., 0., 0., 0., 1., -1., 1.,
                                                                                      0., 0., 0.],
                                                                                     [0., 0., 0., 0., 0., 1., -1.,
                                                                                      1., 0., 0.],
                                                                                     [0., 0., 0., 0., 0., 0., 1.,
                                                                                      -1., 1., 0.],
                                                                                     [0., 0., 0., 0., 0., 0., 0.,
                                                                                      1., -1., 1.],
                                                                                     [0., 0., 0., 0., 0., 0., 0.,
                                                                                      0., 1., -1.]]))

    def test_speed_of_sound(self):
        """
        Tests if the speed_of_sound-setter raises errors correctly and if the time step matrices are changed correctly
        when speed_of_sound is changed.

        :return: None.
        """
        with self.assertRaises(TypeError):
            self.my_sim.speed_of_sound = "test string"

        self.my_sim.speed_of_sound = 1.

        np.testing.assert_almost_equal(self.my_sim.time_step_matrix_left, np.array([[-1., 1., 0., 0., 0., 0., 0., 0.],
                                                                                    [1, -1., 1, 0., 0., 0., 0.,
                                                                                     0.],
                                                                                    [0., 1., -1., 1., 0., 0., 0.,
                                                                                     0.],
                                                                                    [0., 0., 1., -1., 1., 0., 0.,
                                                                                     0.],
                                                                                    [0., 0., 0., 1., -1., 1., 0.,
                                                                                     0.],
                                                                                    [0., 0., 0., 0., 1., -1., 1.,
                                                                                     0.],
                                                                                    [0., 0., 0., 0., 0., 1., -1.,
                                                                                     1.],
                                                                                    [0., 0., 0., 0., 0., 0., 1.,
                                                                                     -1.]]))

        np.testing.assert_almost_equal(self.my_sim.time_step_matrix_right, np.array([[-1., 1., 0., 0., 0., 0., 0., 0.,
                                                                                      0., 0.],
                                                                                     [1, -1., 1, 0., 0., 0., 0., 0., 0.,
                                                                                      0.],
                                                                                     [0., 1., -1., 1., 0., 0., 0.,
                                                                                      0., 0., 0.],
                                                                                     [0., 0., 1., -1., 1., 0., 0.,
                                                                                      0., 0., 0.],
                                                                                     [0., 0., 0., 1., -1., 1., 0.,
                                                                                      0., 0., 0.],
                                                                                     [0., 0., 0., 0., 1., -1., 1.,
                                                                                      0., 0., 0.],
                                                                                     [0., 0., 0., 0., 0., 1., -1.,
                                                                                      1., 0., 0.],
                                                                                     [0., 0., 0., 0., 0., 0., 1.,
                                                                                      -1., 1., 0.],
                                                                                     [0., 0., 0., 0., 0., 0., 0.,
                                                                                      1., -1., 1.],
                                                                                     [0., 0., 0., 0., 0., 0., 0.,
                                                                                      0., 1., -1.]]))

    def test_courant_number(self):
        """
        Tests if the courant number is calculated correctly by the initializer.

        :return: None.
        """
        np.testing.assert_almost_equal(self.my_sim.courant_number, 0.25)

    def test_calculate_couant_number(self):
        """
        Tests the method which calculates the courant number using the formula courant number = ((c * dt)/dx)^2.

        :return: None
        """
        self.assertEqual(self.my_sim.calculate_courant_number(), 0.25)

        self.my_sim.speed_of_sound = 1.

        self.assertEqual(self.my_sim.calculate_courant_number(), 1.)

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

        self.my_sim.boundary_condition = "fixed edges"

        np.testing.assert_almost_equal(self.my_sim.time_step_matrix_left, np.array([[0., 0., 0., 0., 0., 0., 0., 0.],
                                                                                    [0., 0.5, 0.25, 0., 0., 0., 0.,
                                                                                     0.],
                                                                                    [0., 0.25, 0.5, 0.25, 0., 0., 0.,
                                                                                     0.],
                                                                                    [0., 0., 0.25, 0.5, 0.25, 0., 0.,
                                                                                     0.],
                                                                                    [0., 0., 0., 0.25, 0.5, 0.25, 0.,
                                                                                     0.],
                                                                                    [0., 0., 0., 0., 0.25, 0.5, 0.25,
                                                                                     0.],
                                                                                    [0., 0., 0., 0., 0., 0.25, 0.5,
                                                                                     0.],
                                                                                    [0., 0., 0., 0., 0., 0., 0., 0.]]))

        self.my_sim.boundary_condition = "cyclical"

        np.testing.assert_almost_equal(self.my_sim.time_step_matrix_left, np.array([[0.5, 0.25, 0., 0., 0., 0., 0.,
                                                                                     0.25],
                                                                                    [0.25, 0.5, 0.25, 0., 0., 0., 0.,
                                                                                     0.],
                                                                                    [0., 0.25, 0.5, 0.25, 0., 0., 0.,
                                                                                     0.],
                                                                                    [0., 0., 0.25, 0.5, 0.25, 0., 0.,
                                                                                     0.],
                                                                                    [0., 0., 0., 0.25, 0.5, 0.25, 0.,
                                                                                     0.],
                                                                                    [0., 0., 0., 0., 0.25, 0.5, 0.25,
                                                                                     0.],
                                                                                    [0., 0., 0., 0., 0., 0.25, 0.5,
                                                                                     0.25],
                                                                                    [0.25, 0., 0., 0., 0., 0., 0.25,
                                                                                     0.5]]))

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

        self.my_sim.number_of_grid_points = (5, 5)

        np.testing.assert_almost_equal(self.my_sim.time_step_matrix_left, np.array([[0.5, 0.25, 0., 0., 0.],
                                                                                    [0.25, 0.5, 0.25, 0., 0.],
                                                                                    [0., 0.25, 0.5, 0.25, 0.],
                                                                                    [0., 0., 0.25, 0.5, 0.25],
                                                                                    [0., 0., 0., 0.25, 0.5]]))

        np.testing.assert_almost_equal(self.my_sim.time_step_matrix_right, np.array([[0.5, 0.25, 0., 0., 0.],
                                                                                    [0.25, 0.5, 0.25, 0., 0.],
                                                                                    [0., 0.25, 0.5, 0.25, 0.],
                                                                                    [0., 0., 0.25, 0.5, 0.25],
                                                                                    [0., 0., 0., 0.25, 0.5]]))

    def test_initial_amplitudes(self):
        """
        Tests the getter and setter methods for the initial amplitudes. It is tested if the getter returns the right
        value and if exceptions are thrown when incorrect types are entered.

        :return: None.
        """
        # np.testing.assert_almost_equal(self.init_amps, self.my_sim.initial_amplitudes)

        with self.assertRaises(TypeError):
            self.my_sim.initial_amplitudes = "String"

        with self.assertRaises(ValueError):
            self.my_sim.initial_amplitudes = np.array([[3., 4.], [2., 3.]])

        with self.assertRaises(TypeError):
            self.my_sim.initial_amplitudes = np.full(self.dim, "String")

    def test_initial_velocities(self):
        """
        Tests the getter and setter methods for the initial velocities. It is tested if the getter returns the right
        value and if exceptions are thrown when incorrect types are entered.

        :return: None.
        """
        # np.testing.assert_almost_equal(self.init_vel, self.my_sim.initial_velocities)

        with self.assertRaises(TypeError):
            self.my_sim.initial_velocities = "String"

        with self.assertRaises(ValueError):
            self.my_sim.initial_velocities = np.array([[3., 4.], [2., 3.]])

        with self.assertRaises(TypeError):
            self.my_sim.initial_velocities = np.full(self.dim, "Hello")

    def test_create_time_step_matrix(self):
        """
        Tests if the time step matrices are created correctly. Teh time step matrices are tested for all three boundary
        conditions.

        :return: None
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
        self.my_sim = sim.Numeric2DWaveSimulator(1., 1., 0.5, self.dim, 10, init_amps_func, init_vel_func, "cyclical")

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
        self.my_sim = sim.Numeric2DWaveSimulator(1., 1., 0.5, self.dim, 10, init_amps_func, init_vel_func, "fixed edges"
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
        Tests the update method which is applied to get from one time step to another. The update method is tested two
        times to see if the update equation on the first and the following steps are correct.

        :return: None
        """
        # Update the simulator for the first time.
        np.testing.assert_almost_equal(self.my_sim.update_first_time(), np.array(
            [[0.75, 0.875, 0.875, 0.875, 0.875, 0.875, 0.875, 0.875, 0.875, 0.75],
             [0.875, 1., 1., 1., 1., 1., 1., 1., 1., 0.875],
             [0.875, 1., 1., 1., 1., 1., 1., 1., 1., 0.875],
             [0.875, 1., 1., 1., 1., 1., 1., 1., 1., 0.875],
             [0.875, 1., 1., 1., 1., 1., 1., 1., 1., 0.875],
             [0.875, 1., 1., 1., 1., 1., 1., 1., 1., 0.875],
             [0.875, 1., 1., 1., 1., 1., 1., 1., 1., 0.875],
             [0.75, 0.875, 0.875, 0.875, 0.875, 0.875, 0.875, 0.875, 0.875, 0.75]]))

        # Update the simulator another time to check the update equation not only for the first step.
        np.testing.assert_almost_equal(self.my_sim.update(), np.array(
            [[0.1875, 0.53125, 0.5625, 0.5625, 0.5625, 0.5625, 0.5625, 0.5625, 0.53125, 0.1875],
             [0.53125, 0.9375, 0.96875, 0.96875, 0.96875, 0.96875, 0.96875, 0.96875, 0.9375, 0.53125],
             [0.5625, 0.96875, 1., 1., 1., 1., 1., 1., 0.96875, 0.5625],
             [0.5625, 0.96875, 1., 1., 1., 1., 1., 1., 0.96875, 0.5625],
             [0.5625, 0.96875, 1., 1., 1., 1., 1., 1., 0.96875, 0.5625],
             [0.5625, 0.96875, 1., 1., 1., 1., 1., 1., 0.96875, 0.5625],
             [0.53125, 0.9375, 0.96875, 0.96875, 0.96875, 0.96875, 0.96875, 0.96875, 0.9375, 0.53125],
             [0.1875, 0.53125, 0.5625, 0.5625, 0.5625, 0.5625, 0.5625, 0.5625, 0.53125, 0.1875]]))


if __name__ == '__main__':
    ut.main()
