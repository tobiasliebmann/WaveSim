import unittest as ut

import numpy as np

from .. import simulator1D as sim


class Sim1DTest(ut.TestCase):

    initial_positions = np.array([0, 0, 0, 0, 0, 0])

    initial_velocities = np.array([1, 1, 1, 1, 1, 1])

    # Initialize simulator
    my_sim = sim.Numeric1DWaveSimulator(1, 1, 0.5, 6, 6, initial_positions, initial_velocities)

    def test_create_time_step_matrix(self):
        compare_matrix = np.ndarray([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0. ],
                                         [0.7, 0.6, 0.7, 0., 0., 0., 0., 0., 0., 0.],
                                         [0., 0.7, 0.6, 0.7, 0., 0., 0., 0., 0., 0.],
                                         [0., 0., 0.7, 0.6, 0.7, 0., 0., 0., 0., 0.],
                                         [0., 0., 0., 0.7, 0.6, 0.7, 0., 0.,  0., 0.],
                                         [0., 0., 0., 0., 0.7, 0.6, 0.7, 0., 0., 0.],
                                         [0., 0., 0., 0., 0., 0.7, 0.6, 0.7, 0., 0.],
                                         [0., 0., 0., 0., 0., 0., 0.7, 0.6, 0.7, 0.],
                                         [0., 0., 0., 0., 0., 0., 0., 0.7, 0.6, 0.7],
                                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])

        self.assertEqual(self.my_sim.create_time_step_matrix(10, 0.7), compare_matrix)


if __name__ == "__main__":
    ut.main()