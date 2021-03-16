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
    - Checking the boundary condition.
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
        todo: Add documentation.
        """
        self.assertEqual(self.my_sim.boundary_condition, "cyclical")

        with self.assertRaises(TypeError):
            self.my_sim.boundary_condition = 5.

        with self.assertRaises(ValueError):
            self.my_sim.boundary_condition = "Hello"


if __name__ == '__main__':
    ut.main()
